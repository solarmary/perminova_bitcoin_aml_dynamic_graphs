# Bitcoin Parser

Модуль собирает Bitcoin-блоки через Blockchain.com API и формирует Elliptic-like витрины для анализа транзакционного графа.

## Основные шаги

1. Загрузка raw JSON блоков в `data/raw/blocks`.
2. Разбор блоков на таблицы `blocks`, `transactions`, `inputs`, `outputs`.
3. Построение directed transaction graph.
4. Расчёт локальных, временных, адресных и графовых признаков.
5. Экспорт CSV/Parquet в `data/processed` и `data/interim`.

## Установка

```powershell
cd parser
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск

```powershell
python scripts\collect_blocks.py --max-blocks 20
python scripts\build_dataset.py --time-window-hours 24
```

## Выходные файлы

- `data/processed/parsed_txs_features.csv`.
- `data/processed/parsed_txs_features_named.csv`.
- `data/processed/parsed_txs_edgelist.csv`.
- `data/processed/parsed_txs_classes.csv`.
- `data/processed/parsed_txs_scores_template.csv`.
- `data/processed/graph_edges.parquet`.

Все новые транзакции получают `class = unknown`, так как Blockchain.com API не предоставляет AML-разметку.
