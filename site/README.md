# Bitcoin AML Monitoring Site

`site` — Streamlit MVP для мониторинга подозрительных Bitcoin-транзакций на основе витрин, подготовленных модулем `parser`.

## Входные данные

Сайт ожидает файлы:

```text
parser/data/processed/parsed_txs_features_named.csv
parser/data/processed/parsed_txs_edgelist.csv
parser/data/processed/parsed_txs_scores_template.csv
parser/data/interim/inputs.parquet
parser/data/interim/outputs.parquet
```

## Установка

```powershell
cd site
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск

```powershell
streamlit run app.py
```

## Разделы интерфейса

- `Обзор` — агрегированные метрики и динамика suspicious-активности.
- `Динамический граф` — снимок или GIF эволюции транзакционного графа.
- `Проверка транзакции` — карточка `tx_id`, локальный ego-граф и объяснение риска.

В MVP используется rule-based `risk_score`. Он показывает структурный риск и не является доказательством незаконной активности.
