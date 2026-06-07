"""
Назначение файла: загрузка Bitcoin-блоков назад по цепочке.
Основные шаги: выбор стартового блока, сохранение raw JSON, checkpoint и фильтрация по датам.
Зависимости или источники данных: Blockchain.com API, data/raw/blocks, data/interim/checkpoint.json.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from tqdm import tqdm

from src.api_client import BlockchainAPIClient
from src.utils import (
    get_block_file_path,
    parse_date_to_timestamp,
    read_json,
    resolve_project_path,
    write_json,
)


class BitcoinBlockCollector:
    """Собирает raw JSON Bitcoin-блоков через Blockchain.com API."""

    def __init__(
        self,
        api_client: BlockchainAPIClient,
        raw_blocks_dir: str | Path,
        checkpoint_path: str | Path,
    ) -> None:
        self.api_client = api_client
        self.raw_blocks_dir = resolve_project_path(raw_blocks_dir)
        self.checkpoint_path = resolve_project_path(checkpoint_path)
        self.raw_blocks_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def collect_backwards(
        self,
        start_hash: str | None = None,
        max_blocks: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Path]:
        """Собирает блоки назад от start_hash или latestblock."""
        current_hash = start_hash or self._get_latest_hash()
        start_ts = parse_date_to_timestamp(start_date)
        end_ts = parse_date_to_timestamp(end_date)
        collected_paths: list[Path] = []
        collected_count = 0
        visited_count = 0

        progress = tqdm(total=max_blocks, desc="Collecting blocks", unit="block")
        try:
            while current_hash:
                existing_path = self._find_existing_block(current_hash)
                if existing_path:
                    block = read_json(existing_path)
                    logger.info("Используется сохраненный блок: {}", existing_path)
                else:
                    block = self.api_client.get_raw_block(current_hash)
                visited_count += 1
                block_ts = int(block.get("time", 0))
                height = int(block.get("height", -1))
                block_hash = str(block.get("hash", current_hash))
                prev_block = block.get("prev_block")

                if start_ts is not None and block_ts < start_ts:
                    logger.info("Достигнута нижняя граница start_date")
                    break

                if end_ts is not None and block_ts > end_ts:
                    self._save_checkpoint(block_hash, prev_block, collected_count)
                    current_hash = prev_block
                    continue

                path = existing_path or get_block_file_path(
                    self.raw_blocks_dir,
                    height,
                    block_hash,
                )
                if existing_path:
                    logger.info("Повторное скачивание не требуется: {}", path)
                else:
                    write_json(path, block)
                    logger.info("Сохранен блок: {}", path)

                collected_paths.append(path)
                collected_count += 1
                progress.update(1)
                self._save_checkpoint(block_hash, prev_block, collected_count)

                if max_blocks is not None and collected_count >= max_blocks:
                    logger.info("Достигнут лимит max_blocks")
                    break

                current_hash = prev_block
        finally:
            progress.close()

        logger.info(
            "Сбор завершен: сохранено {}, посещено {}",
            collected_count,
            visited_count,
        )
        return collected_paths

    def load_checkpoint(self) -> dict[str, object]:
        """Возвращает текущий checkpoint, если он есть."""
        if not self.checkpoint_path.exists():
            return {}
        data = read_json(self.checkpoint_path)
        if isinstance(data, dict):
            return data
        return {}

    def _get_latest_hash(self) -> str:
        latest = self.api_client.get_latest_block()
        block_hash = latest.get("hash")
        if not block_hash:
            raise ValueError("latestblock не содержит поле hash")
        return str(block_hash)

    def _find_existing_block(self, block_hash: str) -> Path | None:
        """Ищет уже сохраненный блок по hash в имени файла."""
        matches = list(self.raw_blocks_dir.glob(f"*_{block_hash}.json"))
        if not matches:
            return None
        return matches[0]

    def _save_checkpoint(
        self,
        current_hash: str,
        next_hash: str | None,
        collected_count: int,
    ) -> None:
        checkpoint = {
            "current_hash": current_hash,
            "next_hash": next_hash,
            "collected_count": collected_count,
        }
        write_json(self.checkpoint_path, checkpoint)
