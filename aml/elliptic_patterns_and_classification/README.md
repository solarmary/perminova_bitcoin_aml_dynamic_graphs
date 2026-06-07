# Elliptic AML Patterns and Classification

Раздел содержит ноутбуки и результаты по AML-анализу Elliptic Bitcoin Dataset.

## Пайплайн

1. EDA Elliptic Dataset и анализ временного разбиения.
2. Baseline-классификация транзакций `licit` / `illicit`.
3. Выявление AML-паттернов fan-in/fan-out, peel chains и burst patterns.
4. Обогащение исходных признаков паттерновыми признаками.
5. Абляционный эксперимент по наборам `base`, `base+fan`, `base+fan+peel`, `base+fan+peel+burst`, `patterns_only`.

## Содержимое

- `notebooks/` — канонические ноутбуки по каждому этапу.
- `results/` — небольшие CSV с итогами pattern detection.
- `figures/` — визуализации, перенесённые из исходного блока.
- `NOTEBOOKS_SUMMARY.md`, `PATTERN_FEATURES_FOR_ELLIPTIC.md`, `CONCLUSION.md` — краткая документация по экспериментам.

Большой файл `elliptic_enriched_features_with_classes.csv` не включён. Его нужно воспроизводить через ноутбук enrichment или хранить вне GitHub.
