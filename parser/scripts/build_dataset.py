"""
Назначение файла: CLI для сборки Elliptic-like датасета.
Основные шаги: парсинг raw JSON, построение графа, расчет признаков и экспорт CSV/Parquet.
Зависимости или источники данных: data/raw/blocks, data/interim, data/processed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.export import export_elliptic_like_dataset, export_interim_tables
from src.features import build_transaction_features
from src.graph_builder import (
    build_transaction_edges,
    build_transaction_graph,
    export_graph_edges,
)
from src.parser import parse_all_blocks
from src.utils import ensure_directories, load_config, resolve_project_path, setup_logging


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Build Elliptic-like dataset")
    parser.add_argument("--time-window-hours", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Запускает сборку датасета."""
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

    raw_blocks_dir = resolve_project_path(paths["raw_blocks_dir"])
    interim_dir = resolve_project_path(paths["interim_dir"])
    processed_dir = resolve_project_path(paths["processed_dir"])
    feature_config = config.get("features", {})
    time_window_hours = (
        args.time_window_hours
        if args.time_window_hours is not None
        else int(feature_config.get("time_window_hours", 24))
    )

    parsed_tables = parse_all_blocks(raw_blocks_dir)
    export_interim_tables(parsed_tables, interim_dir)

    transactions_df = parsed_tables["transactions"]
    inputs_df = parsed_tables["inputs"]
    outputs_df = parsed_tables["outputs"]
    edges_df = build_transaction_edges(inputs_df)
    graph = build_transaction_graph(transactions_df, edges_df)
    export_graph_edges(edges_df, processed_dir)

    features_df = build_transaction_features(
        transactions_df=transactions_df,
        inputs_df=inputs_df,
        outputs_df=outputs_df,
        graph=graph,
        time_window_hours=time_window_hours,
        many_inputs_threshold=int(feature_config.get("many_inputs_threshold", 10)),
        many_outputs_threshold=int(feature_config.get("many_outputs_threshold", 10)),
        high_fee_quantile=float(feature_config.get("high_fee_quantile", 0.95)),
        small_output_satoshi=int(feature_config.get("small_output_satoshi", 10_000)),
    )
    export_elliptic_like_dataset(features_df, edges_df, processed_dir)

    print(f"Датасет собран транзакций {len(features_df)}")
    print(f"Processed {processed_dir}")


if __name__ == "__main__":
    main()
