# краткий обзор ноутбуков в корне `BITOC_WORK/notebooks`

## `0-visualization-elliptic.ipynb`
- EDA Elliptic: классы, временные шаги, граф транзакций, связность, центральности, сообщества, t-SNE.
- идея: перед моделями проверить дисбаланс, temporal shift и графовую структуру.
- результат: обоснование, почему нужны temporal split и graph/pattern features.

## `1_kaggle_elliptic-data-set-node-classification.ipynb`
- классификация `licit / illicit` на честном temporal split: train до 29, val 29-34, test с 35.
- предобработка: class weights, threshold tuning по val, scaling, калибровка, SMOTE.
- признаки: исходные Elliptic, log/ratio features, time encoding, graph degree features.
- модели: Logistic Regression, RF, SVM, XGBoost, LightGBM, CatBoost, residual MLP.
- ансамбли: stacking и soft voting для стабилизации вероятностей.
- GNN-идеи из AML/GNN-статей: GraphSAGE, DGI/self-supervised embedding, temporal WaveGate.
- результат: основной notebook для сравнения табличных, ансамблевых, нейросетевых и GNN-подходов.

## `04-elliptic-fan-in-fan-out.ipynb`
- поиск fan-in, fan-out и bipartite hub-паттернов как AML-сигналов mixing/smurfing.
- предобработка: in/out degree, z-score, hub thresholds, ego-subgraph вокруг hub-узла.
- rule-based часть: degree-filter и FlowScope-like anomalousness score.
- ML/GNN часть: GAT с hub-признаками, temporal split, оценка all nodes / hub nodes / bipartite nodes.
- интерпретация: attention-анализ важных соседей hub-узла.
- результат: признаки `is_fanin_hub`, `is_fanout_hub`, `is_bipartite_hub`, `anomalousness_score`.

## `05-elliptic-peel-chains.ipynb`
- поиск peel-chain паттернов: основная сумма идет дальше, малые части отделяются на каждом шаге.
- rule-based часть: ограничения на длину цепочки, main ratio, peel share, delta time, branching.
- sequence idea: GRU/sequence autoencoder ищет нетипичную динамику шагов цепочки.
- tabular anomaly: Isolation Forest, LOF, PCA-error по агрегированным chain features.
- subgraph anomaly: признаки локального подграфа вокруг цепочки.
- результат: chain-level scores и node-level peel features для классификации.

## `06-elliptic-burst-patterns.ipynb`
- поиск burst-паттернов: резкие всплески входящей/исходящей активности во времени.
- предобработка: temporal edge events, robust z-score внутри timestep, neighbor activity.
- rule-based score: total events, degree, neighbor degree, feature volume proxy.
- unsupervised часть: Isolation Forest по burst z-признакам.
- subgraph часть: ego-subgraph score вокруг top burst-кандидатов.
- sequence часть: GRU forecasting error как признак нетипичного временного ряда.
- результат: burst features для последующего enrichment.

## `7-elliptic-patterns-feature-enrichment.ipynb`
- объединение исходных Elliptic features с fan, peel и burst признаками.
- внутри повторно считаются pattern features: hub, peel-chain anomaly, burst anomaly.
- добавляется каталог групп признаков: base, fan, peel, burst, target.
- результат: enriched dataset, отдельные csv по группам и ML-ready версия без текстовых категорий.

## `8-elliptic-pattern-features-ablation.ipynb`
- проверка, дают ли pattern features прирост к исходным Elliptic features.
- варианты: `base`, `base+fan`, `base+fan+peel`, `base+fan+peel+burst`, `patterns_only`.
- модели: LightGBM, RandomForest, tabular MLP, GraphSAGE на sparse adjacency.
- предобработка: median imputation, scaling, class imbalance weights, threshold tuning по val.
- анализ: F1 delta vs base, LightGBM feature importance, отдельно pattern importance.
- результат: ablation-доказательство полезности или бесполезности новых признаков.

## общий вывод
- корень папки содержит 7 notebooks.
- логика пайплайна: EDA, сильные baselines, паттерны из AML-идей, enrichment, ablation.
- главная цель: проверить, улучшают ли fan, peel и burst признаки AML-классификацию Elliptic.
