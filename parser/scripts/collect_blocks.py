"""
Назначение файла: CLI для загрузки Bitcoin-блоков.
Основные шаги: чтение config.yaml, запуск BlockchainAPIClient и BitcoinBlockCollector.
Зависимости или источники данных: Blockchain.com API, data/raw/blocks, data/interim/checkpoint.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api_client import BlockchainAPIClient
from src.collector import BitcoinBlockCollector
from src.utils import ensure_directories, load_config, resolve_project_path, setup_logging


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Collect Bitcoin blocks")
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--start-hash", type=str, default=None)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Запускает сбор блоков."""
    setup_logging()
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    ensure_directories(
        [
            paths["raw_blocks_dir"],
            paths["interim_dir"],
            paths["processed_dir"],
        ]
    )

    api_config = config["api"]
    client = BlockchainAPIClient(**api_config)
    collector = BitcoinBlockCollector(
        api_client=client,
        raw_blocks_dir=paths["raw_blocks_dir"],
        checkpoint_path=paths["checkpoint_path"],
    )

    max_blocks = args.max_blocks
    if max_blocks is None and args.start_date is None:
        max_blocks = 100

    collected_paths = collector.collect_backwards(
        start_hash=args.start_hash,
        max_blocks=max_blocks,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(f"Сбор завершен файлов {len(collected_paths)}")
    print(f"Raw blocks {resolve_project_path(paths['raw_blocks_dir'])}")


if __name__ == "__main__":
    main()
