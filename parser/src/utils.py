"""
Назначение файла: общие утилиты модуля parser.
Основные шаги: загрузка конфигурации, создание директорий, работа с JSON и датами.
Зависимости или источники данных: config.yaml, .env, локальные JSON-файлы.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def setup_logging() -> None:
    """Настраивает единый формат логов."""
    logger.remove()
    logger.add(
        sink=sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO",
    )


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Загружает config.yaml и применяет значения из .env."""
    load_dotenv(PROJECT_ROOT / ".env")
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    api_config = config.setdefault("api", {})
    env_mapping = {
        "base_url": "BLOCKCHAIN_API_BASE_URL",
        "timeout_seconds": "BLOCKCHAIN_API_TIMEOUT_SECONDS",
        "request_delay_seconds": "BLOCKCHAIN_API_REQUEST_DELAY_SECONDS",
        "max_retries": "BLOCKCHAIN_API_MAX_RETRIES",
    }
    for key, env_name in env_mapping.items():
        if os.getenv(env_name) is not None:
            api_config[key] = os.getenv(env_name)

    api_config["timeout_seconds"] = int(api_config.get("timeout_seconds", 30))
    api_config["request_delay_seconds"] = float(
        api_config.get("request_delay_seconds", 1.0)
    )
    api_config["max_retries"] = int(api_config.get("max_retries", 5))
    api_config["backoff_factor"] = float(api_config.get("backoff_factor", 1.5))
    return config


def resolve_project_path(path_value: str | Path) -> Path:
    """Возвращает абсолютный путь внутри проекта."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_directories(paths: list[str | Path]) -> None:
    """Создает директории, если они отсутствуют."""
    for path in paths:
        resolve_project_path(path).mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> Any:
    """Читает JSON-файл."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, data: Any) -> None:
    """Сохраняет данные в JSON-файл."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def parse_date_to_timestamp(value: str | None) -> int | None:
    """Преобразует дату ISO в Unix timestamp."""
    if not value:
        return None

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def timestamp_to_utc(timestamp: int | float | None) -> str | None:
    """Преобразует Unix timestamp в ISO-строку UTC."""
    if timestamp is None:
        return None
    return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat()


def satoshi_to_btc(value: int | float | None) -> float:
    """Переводит satoshi в BTC."""
    if value is None:
        return 0.0
    return float(value) / 100_000_000


def get_block_file_path(raw_blocks_dir: str | Path, height: int, block_hash: str) -> Path:
    """Формирует путь для raw JSON блока."""
    safe_hash = block_hash.replace("/", "_")
    return resolve_project_path(raw_blocks_dir) / f"{height}_{safe_hash}.json"
