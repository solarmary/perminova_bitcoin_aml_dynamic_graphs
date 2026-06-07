# Bitcoin AML Dynamic Graphs

Репозиторий содержит материалы выпускной квалификационной работы по теме применения технологий ИИ для решения оптимизационных задач на больших динамических графах и AML-анализа Bitcoin-транзакций.

Проект объединяет четыре блока:

- `experiments/optimization_tasks` — эксперимент на синтетическом динамическом графе для пяти задач оптимизации: MST/MSF, shortest path, MaxCut, vertex cover и поиск целевых подграфов.
- `aml/elliptic_patterns_and_classification` — ноутбуки по Elliptic Dataset: EDA, AML-паттерны fan-in/fan-out, peel chains, burst patterns, enrichment и классификация `licit` / `illicit`.
- `parser` — парсер Bitcoin-блоков через Blockchain.com API и подготовка Elliptic-like витрин.
- `site` — Streamlit-прототип мониторинга подозрительных Bitcoin-транзакций.

## Структура

```text
perminova_bitcoin_aml_dynamic_graphs/
  docs/
  experiments/optimization_tasks/
  aml/elliptic_patterns_and_classification/
  parser/
  site/
```

## Быстрый запуск парсера

```powershell
cd parser
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts\collect_blocks.py --max-blocks 20
python scripts\build_dataset.py --time-window-hours 24
```

## Быстрый запуск сайта

```powershell
cd site
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Перед запуском сайта нужно собрать данные в `parser/data/processed` и `parser/data/interim`.

## Данные

Большие датасеты не включены в репозиторий. Это касается raw Bitcoin-блоков, промежуточных Parquet/CSV, enriched Elliptic CSV и сгенерированных больших таблиц синтетического эксперимента. В репозитории оставлены код, ноутбуки, документация, небольшие итоговые таблицы и визуализации.

Исходный Elliptic Dataset необходимо получать отдельно из Kaggle. Новые Bitcoin-данные собираются командами из модуля `parser`.

## Ограничения

Новые Bitcoin-транзакции из парсера не имеют внешних AML-меток и получают `class = unknown`. Сайт использует объяснимый rule-based `risk_score`; это индикатор структурного риска, а не доказательство незаконной активности.
