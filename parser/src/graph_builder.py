"""
Назначение файла: построение directed transaction graph.
Основные шаги: создание ребер prev_tx_hash -> tx_hash и экспорт edge list.
Зависимости или источники данных: таблицы transactions и inputs после парсинга блоков.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
from loguru import logger


EDGE_COLUMNS = ["txId1", "txId2"]


def build_transaction_edges(inputs_df: pd.DataFrame) -> pd.DataFrame:
    """Формирует edge list транзакционного графа."""
    if inputs_df.empty or "prev_tx_hash" not in inputs_df.columns:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    edges = inputs_df.loc[
        inputs_df["prev_tx_hash"].notna() & inputs_df["tx_hash"].notna(),
        ["prev_tx_hash", "tx_hash"],
    ].copy()
    edges = edges.rename(columns={"prev_tx_hash": "txId1", "tx_hash": "txId2"})
    edges = edges.drop_duplicates().reset_index(drop=True)
    return edges[EDGE_COLUMNS]


def build_transaction_graph(
    transactions_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> nx.DiGraph:
    """Строит networkx.DiGraph на уровне транзакций."""
    graph = nx.DiGraph()
    if not transactions_df.empty and "tx_hash" in transactions_df.columns:
        graph.add_nodes_from(transactions_df["tx_hash"].dropna().astype(str).tolist())

    if not edges_df.empty:
        graph.add_edges_from(
            edges_df[["txId1", "txId2"]]
            .dropna()
            .astype(str)
            .itertuples(index=False, name=None)
        )
    return graph


def export_graph_edges(edges_df: pd.DataFrame, processed_dir: str | Path) -> None:
    """Сохраняет edge list в CSV и Parquet."""
    target_dir = Path(processed_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / "parsed_txs_edgelist.csv"
    parquet_path = target_dir / "graph_edges.parquet"

    edges_df.to_csv(csv_path, index=False)
    edges_df.to_parquet(parquet_path, index=False)
    logger.info("Сохранен edge list: {}", csv_path)
    logger.info("Сохранен Parquet edge list: {}", parquet_path)
