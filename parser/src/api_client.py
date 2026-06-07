"""
Назначение файла: клиент для Blockchain.com Explorer API.
Основные шаги: выполнение HTTP-запросов, retry с backoff, rate limit и обработка ошибок.
Зависимости или источники данных: Blockchain.com API latestblock, rawblock, block-height.
"""

from __future__ import annotations

import time
from typing import Any

import requests
from loguru import logger
from pydantic import BaseModel, Field, HttpUrl
from requests import Response


class APIClientConfig(BaseModel):
    """Конфигурация HTTP-клиента."""

    base_url: HttpUrl = Field(default="https://blockchain.info")
    timeout_seconds: int = Field(default=30, ge=1)
    request_delay_seconds: float = Field(default=1.0, ge=0)
    max_retries: int = Field(default=5, ge=1)
    backoff_factor: float = Field(default=1.5, ge=1.0)


class BlockchainAPIClient:
    """Клиент Blockchain.com API с retry и ограничением частоты запросов."""

    def __init__(
        self,
        base_url: str = "https://blockchain.info",
        timeout_seconds: int = 30,
        request_delay_seconds: float = 1.0,
        max_retries: int = 5,
        backoff_factor: float = 1.5,
    ) -> None:
        self.config = APIClientConfig(
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            request_delay_seconds=request_delay_seconds,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )
        self.session = requests.Session()
        self._last_request_ts = 0.0

    def get_latest_block(self) -> dict[str, Any]:
        """Возвращает последний блок."""
        return self._get_json("/latestblock")

    def get_raw_block(self, block_hash: str) -> dict[str, Any]:
        """Возвращает raw JSON блока по hash."""
        return self._get_json(f"/rawblock/{block_hash}")

    def get_blocks_by_height(self, height: int) -> dict[str, Any]:
        """Возвращает блоки по высоте."""
        return self._get_json(f"/block-height/{height}", params={"format": "json"})

    def _get_json(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Выполняет GET-запрос и возвращает JSON."""
        url = f"{str(self.config.base_url).rstrip('/')}{endpoint}"
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            self._rate_limit()
            try:
                logger.info("Запрос Blockchain.com API: {}", url)
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.config.timeout_seconds,
                )
                self._raise_for_status(response)
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("API вернул JSON не в формате object")
                return data
            except (requests.RequestException, ValueError) as error:
                last_error = error
                logger.warning(
                    "Ошибка запроса, попытка {} из {}: {}",
                    attempt,
                    self.config.max_retries,
                    error,
                )
                if attempt == self.config.max_retries:
                    break
                sleep_seconds = self.config.backoff_factor ** (attempt - 1)
                time.sleep(sleep_seconds)

        raise RuntimeError(f"Не удалось получить данные API: {last_error}") from last_error

    def _rate_limit(self) -> None:
        """Добавляет паузу между запросами."""
        elapsed = time.monotonic() - self._last_request_ts
        delay = self.config.request_delay_seconds
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_ts = time.monotonic()

    @staticmethod
    def _raise_for_status(response: Response) -> None:
        """Обрабатывает HTTP-ошибки с коротким текстом ответа."""
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            text = response.text[:500]
            raise requests.HTTPError(
                f"HTTP {response.status_code}: {text}",
                response=response,
            ) from error
