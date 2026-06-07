"""
Назначение файла: расчет Elliptic-like признаков транзакций.
Основные шаги: локальные, временные, адресные, графовые признаки и агрегации соседей.
Зависимости или источники данных: transactions, inputs, outputs и directed transaction graph.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd


FEATURE_COLUMNS = [
    "tx_hash",
    "time_step",
    "block_height",
    "timestamp",
    "input_count",
    "output_count",
    "total_input_value",
    "total_output_value",
    "fee",
    "fee_rate",
    "avg_input_value",
    "avg_output_value",
    "max_output_value",
    "min_output_value",
    "is_coinbase",
    "in_degree",
    "out_degree",
    "total_degree",
    "pagerank",
    "weak_component_size",
    "num_unique_input_addresses",
    "num_unique_output_addresses",
    "has_many_inputs",
    "has_many_outputs",
    "has_high_fee",
    "has_small_outputs",
    "avg_neighbor_fee",
    "avg_neighbor_total_output_value",
    "sum_neighbor_total_output_value",
]


def build_transaction_features(
    transactions_df: pd.DataFrame,
    inputs_df: pd.DataFrame,
    outputs_df: pd.DataFrame,
    graph: nx.DiGraph,
    time_window_hours: int = 24,
    many_inputs_threshold: int = 10,
    many_outputs_threshold: int = 10,
    high_fee_quantile: float = 0.95,
    small_output_satoshi: int = 10_000,
) -> pd.DataFrame:
    """Собирает витрину признаков на уровне транзакций."""
    if transactions_df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    features = transactions_df.copy()
    features["tx_hash"] = features["tx_hash"].astype(str)
    features = features.sort_values(["timestamp", "tx_hash"]).reset_index(drop=True)

    min_timestamp = int(features["timestamp"].min())
    window_seconds = max(int(time_window_hours), 1) * 3600
    features["time_step"] = (
        (features["timestamp"].astype(int) - min_timestamp) // window_seconds
    ) + 1

    features["fee_rate"] = _safe_divide(features["fee"], features["size"])
    features["avg_input_value"] = _safe_divide(
        features["total_input_value"],
        features["input_count"],
    )
    features["avg_output_value"] = _safe_divide(
        features["total_output_value"],
        features["output_count"],
    )

    output_stats = _build_output_stats(outputs_df)
    input_address_stats = _build_address_stats(inputs_df, "input_address")
    output_address_stats = _build_address_stats(outputs_df, "output_address")

    features = features.merge(output_stats, on="tx_hash", how="left")
    features = features.merge(input_address_stats, on="tx_hash", how="left")
    features = features.merge(output_address_stats, on="tx_hash", how="left")

    graph_features = _build_graph_features(graph)
    neighbor_features = _build_neighbor_features(graph, features)
    features = features.merge(graph_features, on="tx_hash", how="left")
    features = features.merge(neighbor_features, on="tx_hash", how="left")

    high_fee_threshold = features["fee"].fillna(0).quantile(high_fee_quantile)
    small_output_btc = small_output_satoshi / 100_000_000

    features["has_many_inputs"] = (
        features["input_count"].fillna(0) >= many_inputs_threshold
    ).astype(int)
    features["has_many_outputs"] = (
        features["output_count"].fillna(0) >= many_outputs_threshold
    ).astype(int)
    features["has_high_fee"] = (features["fee"].fillna(0) >= high_fee_threshold).astype(
        int
    )
    features["has_small_outputs"] = (
        features["min_output_value"].fillna(float("inf")) <= small_output_btc
    ).astype(int)

    fill_zero_columns = [
        "max_output_value",
        "min_output_value",
        "num_unique_input_addresses",
        "num_unique_output_addresses",
        "in_degree",
        "out_degree",
        "total_degree",
        "pagerank",
        "weak_component_size",
        "avg_neighbor_fee",
        "avg_neighbor_total_output_value",
        "sum_neighbor_total_output_value",
    ]
    features[fill_zero_columns] = features[fill_zero_columns].fillna(0)
    features["is_coinbase"] = features["is_coinbase"].astype(int)

    return features[FEATURE_COLUMNS]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Делит Series с защитой от нуля."""
    denominator = denominator.replace(0, pd.NA)
    return (numerator / denominator).fillna(0)


def _build_output_stats(outputs_df: pd.DataFrame) -> pd.DataFrame:
    """Считает min/max выходов транзакции."""
    if outputs_df.empty:
        return pd.DataFrame(
            columns=["tx_hash", "max_output_value", "min_output_value"]
        )
    return (
        outputs_df.groupby("tx_hash", as_index=False)["output_value"]
        .agg(max_output_value="max", min_output_value="min")
    )


def _build_address_stats(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Считает число уникальных адресов."""
    output_column = f"num_unique_{column.replace('_address', '')}_addresses"
    if df.empty:
        return pd.DataFrame(columns=["tx_hash", output_column])
    stats = (
        df.dropna(subset=[column])
        .groupby("tx_hash", as_index=False)[column]
        .nunique()
        .rename(columns={column: output_column})
    )
    return stats


def _build_graph_features(graph: nx.DiGraph) -> pd.DataFrame:
    """Считает степени, PageRank и размер weak component."""
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(
            columns=[
                "tx_hash",
                "in_degree",
                "out_degree",
                "total_degree",
                "pagerank",
                "weak_component_size",
            ]
        )

    pagerank = _calculate_pagerank(graph)
    component_sizes: dict[str, int] = {}
    for component in nx.weakly_connected_components(graph):
        size = len(component)
        for node in component:
            component_sizes[str(node)] = size

    rows = []
    for node in graph.nodes:
        node_str = str(node)
        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)
        rows.append(
            {
                "tx_hash": node_str,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "total_degree": in_degree + out_degree,
                "pagerank": pagerank.get(node, 0.0),
                "weak_component_size": component_sizes.get(node_str, 1),
            }
        )
    return pd.DataFrame(rows)


def _calculate_pagerank(
    graph: nx.DiGraph,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1.0e-6,
) -> dict[object, float]:
    """Считает PageRank без обязательной зависимости от scipy."""
    nodes = list(graph.nodes())
    node_count = len(nodes)
    if node_count == 0:
        return {}
    if graph.number_of_edges() == 0:
        return {node: 0.0 for node in nodes}

    rank = {node: 1.0 / node_count for node in nodes}
    base_rank = (1.0 - alpha) / node_count

    for _ in range(max_iter):
        previous_rank = rank
        dangling_rank = sum(
            previous_rank[node]
            for node in nodes
            if graph.out_degree(node) == 0
        )
        rank = {
            node: base_rank + alpha * dangling_rank / node_count
            for node in nodes
        }

        for source in nodes:
            out_degree = graph.out_degree(source)
            if out_degree == 0:
                continue
            contribution = alpha * previous_rank[source] / out_degree
            for target in graph.successors(source):
                rank[target] += contribution

        error = sum(abs(rank[node] - previous_rank[node]) for node in nodes)
        if error < node_count * tol:
            break

    return rank


def _build_neighbor_features(
    graph: nx.DiGraph,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Агрегирует признаки соседних транзакций."""
    value_map = (
        features_df.set_index("tx_hash")[["fee", "total_output_value"]]
        .fillna(0)
        .to_dict(orient="index")
    )
    rows = []
    for node in features_df["tx_hash"].astype(str):
        neighbors = set(graph.predecessors(node)) | set(graph.successors(node))
        neighbor_values = [value_map[item] for item in neighbors if item in value_map]
        if not neighbor_values:
            rows.append(
                {
                    "tx_hash": node,
                    "avg_neighbor_fee": 0.0,
                    "avg_neighbor_total_output_value": 0.0,
                    "sum_neighbor_total_output_value": 0.0,
                }
            )
            continue

        fees = [item["fee"] for item in neighbor_values]
        outputs = [item["total_output_value"] for item in neighbor_values]
        rows.append(
            {
                "tx_hash": node,
                "avg_neighbor_fee": sum(fees) / len(fees),
                "avg_neighbor_total_output_value": sum(outputs) / len(outputs),
                "sum_neighbor_total_output_value": sum(outputs),
            }
        )
    return pd.DataFrame(rows)
