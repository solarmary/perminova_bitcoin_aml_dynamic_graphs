"""
Назначение файла: преобразование raw JSON Bitcoin-блоков в таблицы.
Основные шаги: разбор blocks, transactions, inputs и outputs с обработкой coinbase.
Зависимости или источники данных: JSON-файлы data/raw/blocks из Blockchain.com API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import read_json, satoshi_to_btc, timestamp_to_utc


BLOCK_COLUMNS = [
    "block_hash",
    "height",
    "timestamp",
    "datetime_utc",
    "prev_block",
    "n_tx",
    "size",
]
TRANSACTION_COLUMNS = [
    "tx_hash",
    "tx_index",
    "block_hash",
    "block_height",
    "timestamp",
    "datetime_utc",
    "size",
    "vin_sz",
    "vout_sz",
    "input_count",
    "output_count",
    "total_input_value",
    "total_output_value",
    "fee",
    "is_coinbase",
]
INPUT_COLUMNS = [
    "tx_hash",
    "block_height",
    "input_index",
    "prev_tx_hash",
    "prev_output_index",
    "input_value",
    "input_address",
]
OUTPUT_COLUMNS = [
    "tx_hash",
    "block_height",
    "output_index",
    "output_value",
    "output_address",
    "spent",
    "script",
]


def parse_block_json(
    path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разбирает один raw JSON блока в четыре DataFrame."""
    block = read_json(path)
    if not isinstance(block, dict):
        raise ValueError(f"Некорректный JSON блока: {path}")

    block_hash = str(block.get("hash", ""))
    height = int(block.get("height", -1))
    timestamp = int(block.get("time", 0))
    transactions = block.get("tx", []) or []

    blocks_df = pd.DataFrame(
        [
            {
                "block_hash": block_hash,
                "height": height,
                "timestamp": timestamp,
                "datetime_utc": timestamp_to_utc(timestamp),
                "prev_block": block.get("prev_block"),
                "n_tx": int(block.get("n_tx", len(transactions))),
                "size": int(block.get("size", 0)),
            }
        ],
        columns=BLOCK_COLUMNS,
    )

    tx_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []

    for tx in transactions:
        tx_hash = str(tx.get("hash", ""))
        inputs = tx.get("inputs", []) or []
        outputs = tx.get("out", []) or []
        is_coinbase = _is_coinbase_transaction(inputs)
        total_input_value = 0.0
        total_output_value = 0.0

        for input_index, tx_input in enumerate(inputs):
            prev_out = tx_input.get("prev_out") or {}
            input_value = None if is_coinbase else prev_out.get("value")
            input_value_btc = satoshi_to_btc(input_value)
            total_input_value += input_value_btc
            input_rows.append(
                {
                    "tx_hash": tx_hash,
                    "block_height": height,
                    "input_index": input_index,
                    "prev_tx_hash": _extract_prev_tx_hash(prev_out),
                    "prev_output_index": prev_out.get("n"),
                    "input_value": input_value_btc,
                    "input_address": prev_out.get("addr"),
                }
            )

        for output_index, tx_output in enumerate(outputs):
            output_value_btc = satoshi_to_btc(tx_output.get("value"))
            total_output_value += output_value_btc
            output_rows.append(
                {
                    "tx_hash": tx_hash,
                    "block_height": height,
                    "output_index": output_index,
                    "output_value": output_value_btc,
                    "output_address": tx_output.get("addr"),
                    "spent": bool(tx_output.get("spent", False)),
                    "script": tx_output.get("script"),
                }
            )

        fee = max(total_input_value - total_output_value, 0.0)
        if is_coinbase:
            fee = 0.0

        tx_rows.append(
            {
                "tx_hash": tx_hash,
                "tx_index": tx.get("tx_index"),
                "block_hash": block_hash,
                "block_height": height,
                "timestamp": timestamp,
                "datetime_utc": timestamp_to_utc(timestamp),
                "size": int(tx.get("size", 0)),
                "vin_sz": int(tx.get("vin_sz", len(inputs))),
                "vout_sz": int(tx.get("vout_sz", len(outputs))),
                "input_count": len(inputs),
                "output_count": len(outputs),
                "total_input_value": total_input_value,
                "total_output_value": total_output_value,
                "fee": fee,
                "is_coinbase": bool(is_coinbase),
            }
        )

    transactions_df = pd.DataFrame(tx_rows, columns=TRANSACTION_COLUMNS)
    inputs_df = pd.DataFrame(input_rows, columns=INPUT_COLUMNS)
    outputs_df = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    return blocks_df, transactions_df, inputs_df, outputs_df


def parse_all_blocks(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Разбирает все JSON-блоки из директории."""
    paths = sorted(Path(raw_dir).glob("*.json"))
    parsed_blocks = [parse_block_json(path) for path in paths]

    if not parsed_blocks:
        return {
            "blocks": pd.DataFrame(columns=BLOCK_COLUMNS),
            "transactions": pd.DataFrame(columns=TRANSACTION_COLUMNS),
            "inputs": pd.DataFrame(columns=INPUT_COLUMNS),
            "outputs": pd.DataFrame(columns=OUTPUT_COLUMNS),
        }

    blocks, transactions, inputs, outputs = zip(*parsed_blocks)
    return {
        "blocks": pd.concat(blocks, ignore_index=True),
        "transactions": pd.concat(transactions, ignore_index=True),
        "inputs": pd.concat(inputs, ignore_index=True),
        "outputs": pd.concat(outputs, ignore_index=True),
    }


def _is_coinbase_transaction(inputs: list[dict[str, Any]]) -> bool:
    """Определяет coinbase-транзакцию."""
    if not inputs:
        return True
    return all(not tx_input.get("prev_out") for tx_input in inputs)


def _extract_prev_tx_hash(prev_out: dict[str, Any]) -> str | None:
    """Извлекает идентификатор предыдущей транзакции из вариантов API."""
    for key in ("hash", "tx_hash", "txid", "tx_index"):
        value = prev_out.get(key)
        if value:
            return str(value)
    return None
