"""
Назначение файла: экспорт Elliptic-like датасета.
Основные шаги: сохранение features, edgelist, classes и шаблона risk scores.
Зависимости или источники данных: рассчитанные признаки и граф транзакций.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


def export_elliptic_like_dataset(
    features_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    processed_dir: str | Path,
) -> dict[str, Path]:
    """Сохраняет итоговые CSV-файлы в Elliptic-like формате."""
    target_dir = Path(processed_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    named_features_path = target_dir / "parsed_txs_features_named.csv"
    features_path = target_dir / "parsed_txs_features.csv"
    edgelist_path = target_dir / "parsed_txs_edgelist.csv"
    classes_path = target_dir / "parsed_txs_classes.csv"
    scores_path = target_dir / "parsed_txs_scores_template.csv"

    named_features = features_df.copy()
    named_features.to_csv(named_features_path, index=False)

    elliptic_features = _to_numbered_features(features_df)
    elliptic_features.to_csv(features_path, index=False)
    edges_df.to_csv(edgelist_path, index=False)

    classes = pd.DataFrame(
        {
            "txId": features_df.get("tx_hash", pd.Series(dtype=str)),
            "class": "unknown",
        }
    )
    classes.to_csv(classes_path, index=False)

    scores = pd.DataFrame(
        {
            "txId": features_df.get("tx_hash", pd.Series(dtype=str)),
            "risk_score": pd.NA,
            "risk_label": pd.NA,
        }
    )
    scores.to_csv(scores_path, index=False)

    paths = {
        "features": features_path,
        "features_named": named_features_path,
        "edgelist": edgelist_path,
        "classes": classes_path,
        "scores_template": scores_path,
    }
    for path in paths.values():
        logger.info("Сохранен файл: {}", path)
    return paths


def export_interim_tables(
    parsed_tables: dict[str, pd.DataFrame],
    interim_dir: str | Path,
) -> dict[str, Path]:
    """Сохраняет промежуточные таблицы в Parquet."""
    target_dir = Path(interim_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, df in parsed_tables.items():
        path = target_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        paths[name] = path
        logger.info("Сохранена промежуточная таблица: {}", path)
    return paths


def _to_numbered_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Преобразует понятные названия признаков в feature_1, feature_2."""
    if features_df.empty:
        return pd.DataFrame(columns=["txId", "time_step"])

    feature_columns = [
        column
        for column in features_df.columns
        if column not in {"tx_hash", "time_step"}
    ]
    renamed = features_df[["tx_hash", "time_step", *feature_columns]].copy()
    renamed = renamed.rename(columns={"tx_hash": "txId"})
    renamed = renamed.rename(
        columns={
            column: f"feature_{index}"
            for index, column in enumerate(feature_columns, start=1)
        }
    )
    return renamed
