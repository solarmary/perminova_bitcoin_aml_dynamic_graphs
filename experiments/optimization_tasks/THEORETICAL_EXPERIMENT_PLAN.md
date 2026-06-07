# План теоретического эксперимента

Документ задает экспериментальную часть для темы: "Применение технологий и моделей ИИ в решении оптимизационных задач на больших динамических графах".

Цель эксперимента: сгенерировать большой синтетический динамический граф, решить на нем набор классических оптимизационных задач и сравнить классические алгоритмы с ИИ-подходами по качеству решения, времени, масштабируемости и устойчивости к динамике.

Ключевое требование: все выбранные оптимизационные задачи должны решаться на одном и том же большом динамическом графе или на его временных снимках `G_t`. Нельзя под каждую задачу генерировать отдельный маленький граф, потому что тогда сравнение перестанет соответствовать теме диплома. Допускаются малые подграфы только как вспомогательные тесты для получения exact/best-known решений и проверки корректности.

## 1. Научная опора

Основные статьи из `THEORY_WORK/papers`:

- `Dai_NIPS_2017.pdf`: единый подход `S2V-DQN` для `Minimum Vertex Cover`, `MaxCut`, `TSP` на синтетических `ER` и `BA` графах. Использовать как базовую работу про `RL + GNN` для нескольких задач.
- `5426_Primal_Dual_Graph_Neural_.pdf`: `PDGNN`, где GNN имитирует primal-dual алгоритмы и улучшает их на `Minimum Vertex Cover`, `Set Cover`, `Hitting Set`. Использовать как основу для задач покрытия и warm-start классических решателей.
- `2107.01188v2.pdf`: physics-inspired GNN через `QUBO` для `MaxCut`, `MIS`, `MVC`. Использовать как основу для unsupervised GNN-решателя без разметки оптимальных решений.
- `2406.11897v1.pdf`: `MaxCut-Bench`, критическое сравнение классических и learned эвристик. Использовать как аргумент, что в дипломе нужны сильные классические baselines: `Tabu Search`, `Reversible Greedy`, `Extremal Optimization`, а не только слабый greedy.

Связь с практикой `BITOC_WORK`: практическая часть уже использует fan-in/fan-out, peel chains и burst activity как AML-паттерны. Поэтому задача поиска подграфов должна быть не абстрактной кликой, а поиском аномальных временных подграфов, близких к burst/hub/chain структурам.

## 2. Синтетический динамический граф

Рекомендуемый тип графа: направленный, взвешенный, атрибутированный, динамический граф транзакционного типа.

Такой граф лучше всего связан с `BITOC_WORK`, потому что Elliptic Data Set является временным графом транзакций, где важны направление ребра, временной шаг, локальные паттерны и подозрительные подграфы.

Все эксперименты строятся вокруг единого объекта:

- общий граф `G_dynamic`;
- последовательность снимков `G_1, ..., G_T`;
- единая таблица узлов;
- единая таблица ребер;
- единые временные признаки;
- единые внедренные аномальные паттерны.

Для каждой оптимизационной задачи меняется не граф, а постановка задачи, целевая функция и набор алгоритмов.

### 2.1. Формальная структура

Граф задается как последовательность снимков:

`G_1, G_2, ..., G_T`, где `G_t = (V_t, E_t, X_t, W_t)`.

Сущности:

- `V_t`: узлы на временном шаге `t`;
- `E_t`: направленные ребра-транзакции;
- `X_t`: признаки узлов;
- `W_t`: веса ребер, например сумма, частота, стоимость перехода или риск;
- `time_step`: дискретный временной шаг;
- `pattern_label`: техническая метка для синтетически внедренных паттернов, используется только для проверки качества поиска.

### 2.2. Размеры

Эксперимент предусматривает три режима, чтобы сохранить воспроизводимость и проверить поведение методов на графах разного масштаба:

- `pilot`: `5 000` узлов, `50 000` ребер, `20` временных шагов;
- `main`: `100 000` узлов, `1 000 000` ребер, `50` временных шагов;
- `stress`: `300 000-1 000 000` узлов, `3 000 000-10 000 000` ребер, если хватит ресурсов.

Основной режим для диплома: `100 000` узлов, `1 000 000` ребер, `50` временных шагов. Это достаточно крупно для классических алгоритмов и реалистично для Python-пайплайна при использовании sparse-структур.

### 2.3. Генерация структуры

Базовый граф генерируется как смесь нескольких моделей:

- `Barabási-Albert`: дает hub-структуры и степенное распределение степеней;
- `Erdos-Renyi`: добавляет случайный шум;
- `Holme-Kim`: добавляет треугольники и локальную кластеризацию;
- directed edge orientation: направление ребер задается по времени или по типу узла;
- edge weights: веса из log-normal или Pareto распределения, чтобы имитировать heavy-tailed транзакционные суммы.

Динамика по временным шагам:

- добавление новых узлов с вероятностью роста сети;
- добавление новых ребер по preferential attachment;
- удаление или затухание части старых ребер;
- изменение весов ребер;
- локальные всплески активности в отдельных временных окнах;
- перенос части узлов из нормального режима в аномальный.

### 2.4. Внедренные AML-паттерны

Чтобы связать теоретический граф с практикой `BITOC_WORK`, в синтетический граф нужно явно внедрить паттерны:

- `fan-in`: много входящих ребер в один узел за короткое окно;
- `fan-out`: много исходящих ребер из одного узла за короткое окно;
- `bipartite hub`: узел одновременно собирает и распределяет поток;
- `peel chain`: цепочка `v1 -> v2 -> ... -> vk` с боковыми ответвлениями;
- `burst subgraph`: локальный подграф с резким ростом активности;
- `dense anomalous subgraph`: малый подграф с повышенной плотностью, весом и временной синхронностью.

Технически эти паттерны нужны не для всех задач, а прежде всего для задачи поиска подграфов и для проверки `precision@k`, `recall@k`, `AP`.

### 2.5. Инструменты генерации

Для реализации:

- `Python`: основной язык;
- `numpy`, `pandas`: генерация таблиц узлов, ребер, временных событий;
- `scipy.sparse`: хранение больших adjacency matrices;
- `networkx`: прототипирование и проверка корректности на `pilot`;
- `igraph` или `graph-tool`: более быстрые графовые операции на `main`;
- `PyTorch Geometric` или `DGL`: GNN-модели;
- `scikit-learn`: Isolation Forest, PCA, метрики, baselines;
- `scipy.optimize.milp` или `HiGHS`: точные решения на малых подграфах;
- `joblib`, `numba`: ускорение независимых экспериментов.

`NetworkX` не стоит использовать как основной инструмент для `main`, потому что он медленный на миллионах ребер. Его лучше оставить для проверки корректности и малых графов.

### 2.6. Структура генерируемых данных

Генератор должен сохранять данные не только как объект графа, но и как набор плоских таблиц. Это упростит повторяемость экспериментов, визуализации и запуск разных алгоритмов на одних и тех же входах.

Рекомендуемая структура папок:

```text
THEORY_WORK/
  data/
    synthetic_dynamic_graph/
      graph_config.json
      nodes.csv
      edges.csv
      node_features.csv
      edge_features.csv
      snapshots/
        edges_t_001.csv
        edges_t_002.csv
        ...
      planted_patterns.csv
      task_instances/
        mst_instances.csv
        shortest_path_queries.csv
        maxcut_instances.csv
        vertex_cover_instances.csv
        subgraph_candidates.csv
  results/
    classical/
    ai/
    metrics/
    visualizations/
```

`graph_config.json`:

- `seed`: random seed;
- `n_nodes`;
- `n_edges`;
- `n_timesteps`;
- `base_generators`: доли `BA`, `ER`, `HK`;
- `directed`;
- `weighted`;
- `weight_distribution`;
- `dynamic_update_rates`;
- `planted_pattern_counts`;
- `train_val_test_split`;
- `hardware_info`, если фиксируется окружение.

`nodes.csv`:

- `node_id`;
- `birth_time`;
- `node_type`: `normal`, `hub`, `bridge`, `pattern`;
- `base_risk`;
- `activity_level`;
- `community_id`;
- `is_planted_anomaly`;
- `planted_pattern_type`.

`edges.csv`:

- `edge_id`;
- `source`;
- `target`;
- `time_step`;
- `weight`;
- `amount`;
- `risk_cost`;
- `latency_cost`;
- `is_active`;
- `is_planted_pattern_edge`;
- `pattern_id`.

`node_features.csv`:

- `node_id`;
- `time_step`;
- `in_degree`;
- `out_degree`;
- `weighted_in_degree`;
- `weighted_out_degree`;
- `pagerank`;
- `local_clustering`;
- `burst_score`;
- `hub_score`;
- `chain_participation_score`.

`edge_features.csv`:

- `edge_id`;
- `time_step`;
- `weight`;
- `amount_log`;
- `age`;
- `weight_delta`;
- `is_new_edge`;
- `is_removed_edge`;
- `mst_cost`;
- `shortest_path_cost`;
- `maxcut_weight`.

`planted_patterns.csv`:

- `pattern_id`;
- `pattern_type`: `fan_in`, `fan_out`, `bipartite_hub`, `peel_chain`, `burst_subgraph`, `dense_subgraph`;
- `time_start`;
- `time_end`;
- `center_node`;
- `node_ids`;
- `edge_ids`;
- `n_nodes`;
- `n_edges`;
- `density`;
- `total_weight`;
- `expected_detection_task`.

`task_instances/*.csv`:

- для `MST/MSF`: `snapshot_id`, `n_nodes`, `n_edges`, `component_id`;
- для `Shortest Path`: `query_id`, `time_step`, `source`, `target`, `known_reachable`;
- для `MaxCut`: `snapshot_id`, `graph_projection`, `weight_type`;
- для `MVC`: `snapshot_id`, `subgraph_id`, `weight_type`;
- для `Subgraph Discovery`: `candidate_id`, `time_step`, `center_node`, `candidate_nodes`, `candidate_edges`, `label`.

## 3. Выбранные оптимизационные задачи

Рекомендуемый набор из 5 задач:

1. `Minimum Spanning Tree / Minimum Spanning Forest`.
2. `Shortest Path` на динамических весах.
3. `Maximum Cut`.
4. `Minimum Vertex Cover`.
5. `Top-k Anomalous Temporal Subgraph Discovery`.

Такой набор покрывает:

- полиномиальные задачи: `MST`, `Shortest Path`;
- NP-трудные задачи: `MaxCut`, `MVC`, поиск оптимального подграфа;
- задачи из статей: `MaxCut`, `MVC`, `MST`, `Shortest Path`;
- связь с практикой: поиск аномальных подграфов, fan/hub/chain/burst.

Важно: все 5 задач решаются на общей динамической сети. Для задач, где нужен неориентированный граф (`MST`, `MaxCut`, `MVC`), используется согласованная projection-функция из исходного directed-графа. Для задач, где важны направление и время (`Shortest Path`, `Top-k Anomalous Temporal Subgraph Discovery`), используется исходная directed temporal structure.

## 4. Задача 1: Minimum Spanning Tree / Forest

Смысл: найти минимальный остов на взвешенном графе или минимальный остовный лес, если граф несвязный.

Для направленного транзакционного графа использовать undirected projection:

- ребро `(u, v)` и `(v, u)` сворачиваются в одно;
- вес ребра можно задать как `risk_cost`, `1 / volume`, `time_delay` или комбинированную стоимость;
- если граф несвязный, решается `Minimum Spanning Forest`.

### Классические подходы

- `Kruskal`;
- `Prim`;
- пересчет MST на каждом снимке;
- incremental update: пересчитывать только компоненты, затронутые изменениями ребер.

### ИИ-подходы

- GNN edge classifier: предсказывает вероятность, что ребро входит в MST;
- NAR-подход: имитация шагов `Kruskal` или `Prim`;
- warm-start: GNN отбирает кандидаты ребер, затем классический алгоритм достраивает корректный MST.

### Метрики

- `total_tree_weight`: суммарный вес дерева или леса;
- `optimality_gap`: разница с точным `Kruskal/Prim`;
- `validity_rate`: доля запусков без циклов и с покрытием всех компонент;
- `runtime`;
- `update_time_per_snapshot`;
- `changed_edges_ratio`: доля ребер MST, изменившихся между `t` и `t+1`;
- `memory_peak`.

## 5. Задача 2: Shortest Path

Смысл: найти кратчайшие пути в графе, где веса ребер меняются во времени.

Постановка:

- выбрать набор пар `(source, target)` или набор источников;
- веса ребер зависят от времени: `cost_t(u, v)`;
- изменения весов имитируют задержки, перегрузку или риск.

### Классические подходы

- `Dijkstra` для неотрицательных весов;
- `Bellman-Ford`, если допускаются отрицательные штрафы;
- `A*`, если есть координаты или embedding-эвристика;
- incremental replanning: пересчет только для изменившихся областей.

### ИИ-подходы

- `Q-routing`: RL-агент выбирает следующий узел;
- `DQN`/`GraphSAGE-DQN`: состояние включает признаки текущего узла, цели и локальной окрестности;
- supervised GNN: предсказывает next-hop или расстояние до цели по обучающим ответам от `Dijkstra`.

### Метрики

- `path_cost`;
- `optimality_gap` относительно `Dijkstra`;
- `success_rate`: доля найденных достижимых маршрутов;
- `adaptation_lag`: сколько шагов нужно после изменения весов, чтобы качество восстановилось;
- `runtime_per_query`;
- `throughput_qps`: число запросов маршрутизации в секунду;
- `replanning_count`;
- `memory_peak`.

## 6. Задача 3: Maximum Cut

Смысл: разбить вершины на две группы так, чтобы максимизировать суммарный вес ребер между группами.

Эта задача хорошо покрыта статьями `Dai_NIPS_2017.pdf`, `2107.01188v2.pdf` и `2406.11897v1.pdf`.

### Классические подходы

- `Forward Greedy`;
- `Reversible Greedy`;
- `Tabu Search`;
- `Extremal Optimization`;
- `Goemans-Williamson` на малых графах как сильный, но дорогой baseline.

### ИИ-подходы

- `S2V-DQN`;
- `ECO-DQN` или упрощенный `SoftTabu`;
- `RUN-CSP` / `ANYCSP`, если делать CSP-формулировку;
- physics-inspired GNN через `QUBO` loss.

### Метрики

- `cut_value`;
- `approximation_ratio` к best-known решению;
- `relative_gap`;
- `runtime`;
- `steps_to_best`;
- `generalization_gap`: падение качества при смене генератора графа;
- `stability_over_snapshots`: насколько сильно меняется разрез при малых изменениях графа;
- `cpu_memory`, `gpu_memory`.

## 7. Задача 4: Minimum Vertex Cover

Смысл: выбрать минимальное множество вершин, покрывающее все ребра.

Для диплома задача полезна как мост между классической теорией, `PDGNN` и практическим смыслом "выбора важных узлов сети".

### Классические подходы

- 2-approximation через выбор непокрытого ребра и добавление обеих вершин;
- greedy по максимальной степени;
- weighted greedy, если узлы имеют стоимость;
- primal-dual approximation;
- exact `MILP` только на малых подграфах для оценки оптимума.

### ИИ-подходы

- `S2V-DQN` из Dai et al.;
- `PDGNN` как имитация primal-dual алгоритма;
- supervised GNN node classifier по оптимальным решениям малых графов;
- GNN warm-start для `MILP`.

### Метрики

- `cover_size` или `cover_weight`;
- `coverage_rate`: доля покрытых ребер;
- `uncovered_edges_count`;
- `wpred / walgo`;
- `wpred / woptm` на малых графах;
- `cleanup_rate`: как часто нужно достраивать решение;
- `runtime`;
- `warm_start_speedup`, если используется solver.

## 8. Задача 5: Top-k Anomalous Temporal Subgraph Discovery

Смысл: найти `k` наиболее подозрительных локальных подграфов во временном графе.

Это ключевая задача для связи с `BITOC_WORK`, потому что она соответствует уже реализованным идеям:

- fan-in/fan-out hub-паттерны;
- peel chains;
- burst activity;
- local ego-subgraph scoring;
- top-k ранжирование подозрительных транзакций.

### Формализация

Для каждого кандидата `S_t` на временном шаге `t` максимизировать score:

`score(S_t) = alpha * density + beta * burst + gamma * flow_asymmetry + delta * chain_score + eta * weight_anomaly`

Ограничения:

- размер подграфа: `min_size <= |S_t| <= max_size`;
- временное окно: `t - window <= edge_time <= t`;
- связность или weak connectivity;
- top-k выдача без сильного пересечения кандидатов.

### Классические подходы

- ego-subgraph enumeration вокруг top-degree/top-burst узлов;
- greedy expansion по приросту score;
- k-core / densest subgraph peeling;
- Louvain / Leiden для community candidates;
- rule-based fan/peel/burst scoring как в `BITOC_WORK`.

### ИИ-подходы

- Isolation Forest по признакам подграфа;
- GRU autoencoder для временного профиля активности;
- GNN/GraphSAGE/GAT для node или subgraph scoring;
- contrastive learning: нормальные подграфы против синтетически внедренных аномальных;
- learning-to-rank для top-k подграфов.

### Метрики

- `precision@k`;
- `recall@k`;
- `average_precision`;
- `roc_auc` по кандидатам, если есть synthetic labels;
- `mean_detected_score`;
- `overlap_jaccard` с внедренными паттернами;
- `time_to_detect`: задержка обнаружения после появления паттерна;
- `runtime_per_snapshot`;
- `candidate_count`;
- `subgraph_diversity`: средняя непохожесть найденных подграфов.

## 9. Общие метрики сравнения

Для всех задач фиксировать единый набор системных метрик:

- качество решения: objective value или normalized score;
- `relative_gap`: разница с точным или best-known baseline;
- `approximation_ratio`, если применимо;
- `runtime_total`;
- `runtime_per_snapshot`;
- `memory_peak`;
- `scalability_slope`: рост времени при увеличении числа узлов и ребер;
- `generalization_gap`: перенос с `pilot` на `main` и между генераторами `ER/BA/HK`;
- `dynamic_stability`: изменение качества при переходе от `G_t` к `G_{t+1}`;
- `adaptation_time`: время восстановления качества после резкого изменения графа;
- `implementation_complexity`: число существенных гиперпараметров и необходимость GPU.

Для ИИ-подходов дополнительно:

- `train_time`;
- `inference_time`;
- `gpu_memory`;
- `seed_std`: стандартное отклонение по random seed;
- `cleanup_rate`, если нейросеть выдает невалидное решение;
- `quality_after_classical_postprocessing`.

## 10. Визуализации

Визуализации нужны не только для презентации, но и для проверки, что синтетический граф действительно похож на большой динамический граф с транзакционными паттернами.

### 10.1. Визуализации структуры графа

- распределение входящих и исходящих степеней в log-log масштабе;
- распределение весов ребер и `risk_cost`;
- число узлов и ребер по `time_step`;
- доля новых, удаленных и измененных ребер по времени;
- размер крупнейшей компоненты по времени;
- распределение компонент связности;
- clustering coefficient по снимкам;
- PageRank / centrality top-k по временным шагам.

### 10.2. Визуализации динамики

- line plot `n_edges(t)`, `n_nodes(t)`, `density(t)`;
- heatmap активности по временным шагам и группам узлов;
- temporal edge churn: сколько ребер появилось, исчезло, изменило вес;
- график среднего и максимального веса ребер по времени;
- график burst-событий и внедренных аномалий;
- сравнение слабой, средней и резкой динамики.

### 10.3. Визуализации внедренных паттернов

- ego-subgraph для `fan-in`;
- ego-subgraph для `fan-out`;
- bipartite hub subgraph;
- peel-chain path с боковыми ребрами;
- burst-subgraph во временном окне;
- распределение размеров и плотностей внедренных подграфов;
- timeline появления planted-паттернов.

### 10.4. Визуализации результатов по задачам

Для `MST/MSF`:

- суммарный вес MST по времени;
- доля изменившихся ребер MST между снимками;
- runtime classical vs AI по размерам графа;
- gap GNN warm-start относительно `Kruskal`.

Для `Shortest Path`:

- распределение `path_cost`;
- `optimality_gap` по запросам;
- runtime per query;
- adaptation lag после изменения весов;
- примеры маршрутов до и после burst-события.

Для `MaxCut`:

- `cut_value` по подходам;
- `approximation_ratio` по размерам графа;
- time-quality trade-off;
- generalization gap между `BA`, `ER`, `HK`;
- boxplot по random seeds.

Для `Minimum Vertex Cover`:

- `cover_size` / `cover_weight` по подходам;
- `coverage_rate`;
- `uncovered_edges_count`;
- `wpred / walgo`;
- `wpred / woptm` на малых графах;
- warm-start speedup для solver.

Для `Top-k Anomalous Temporal Subgraph Discovery`:

- `precision@k`, `recall@k`, `average_precision`;
- PR-кривая по кандидатам подграфов;
- top-k найденных подграфов на примерах;
- overlap найденных подграфов с planted patterns;
- time-to-detect по типам паттернов;
- сравнение rule-based, Isolation Forest, GRU AE, GNN scoring.

### 10.5. Итоговые сравнительные визуализации

- общая диаграмма `quality vs runtime`;
- heatmap: задачи по строкам, подходы по столбцам, значение `relative_gap`;
- barplot потребления памяти;
- line plot масштабируемости при росте `n_nodes`, `n_edges`;
- radar chart для подходов: качество, время, память, устойчивость, сложность реализации;
- таблица-рейтинг подходов по каждой задаче.

## 11. Экспериментальный протокол

1. Сгенерировать `pilot`, проверить корректность всех задач.
2. Сгенерировать единый `main` динамический граф и сохранить:
   - `nodes.csv`;
   - `edges.csv`;
   - `snapshots/edges_t.csv`;
   - `planted_patterns.csv`;
   - `graph_config.json`.
3. Для каждой задачи построить вход из одного и того же `G_dynamic`:
   - `MST/MSF`: undirected weighted projection каждого снимка;
   - `Shortest Path`: directed weighted temporal graph;
   - `MaxCut`: undirected weighted projection каждого снимка;
   - `MVC`: undirected projection или подграфы снимков;
   - `Subgraph Discovery`: directed temporal graph с временными окнами.
4. Для каждой задачи запустить классические baselines на всех выбранных снимках `G_t`.
5. Для каждой задачи запустить ИИ-подходы на тех же снимках `G_t`.
6. Для малых sampled-подграфов получить exact/best-known решения, но использовать их только как контроль качества.
7. Для большого графа сравнивать подходы с лучшим классическим baseline, known bounds или стабильным best-known результатом.
8. Сравнивать результаты в двух разрезах:
   - внутри каждой задачи: classical vs AI;
   - между задачами: как разные классы задач ведут себя на одной динамической сети.
9. Сделать ablation по динамике:
   - слабая динамика;
   - средняя динамика;
   - резкие burst-события;
   - смена распределения графа.
10. Свести результаты в общие таблицы:
   - качество;
   - время;
   - память;
   - устойчивость к динамике;
   - применимость к большим графам.

Главный формат итогового сравнения: строки соответствуют задачам, столбцы соответствуют подходам, а каждая ячейка содержит качество, время и устойчивость подхода на одном и том же динамическом графе.

## 12. План реализации по ноутбукам

Рекомендуемая структура notebooks:

- `00_generate_synthetic_dynamic_graph.ipynb`: генерация `pilot` и `main`, сохранение таблиц;
- `01_graph_eda_and_visualizations.ipynb`: проверка структуры, динамики и planted-паттернов;
- `02_experiment_mst_msf.ipynb`: MST/MSF, classical и GNN warm-start;
- `03_experiment_shortest_path.ipynb`: динамические кратчайшие пути, Dijkstra и RL/GNN next-hop;
- `04_experiment_maxcut.ipynb`: MaxCut, greedy/local search/Tabu и learned heuristics;
- `05_experiment_vertex_cover.ipynb`: MVC, approximation/primal-dual и PDGNN-like подход;
- `06_experiment_anomalous_subgraphs.ipynb`: top-k поиск аномальных временных подграфов;
- `07_compare_all_results.ipynb`: общие таблицы, графики, выводы.

Минимальный порядок выполнения:

1. Сначала сделать генератор и EDA.
2. Затем реализовать классические методы для всех задач.
3. После этого добавить ИИ-подходы.
4. В конце собрать единый comparison notebook.

## 13. Рекомендуемый минимальный набор для реализации

Сокращенный вариант эксперимента включает 3 задачи:

1. `Shortest Path`.
2. `MaxCut`.
3. `Top-k Anomalous Temporal Subgraph Discovery`.

Если объем позволяет, добавить:

4. `Minimum Vertex Cover`.
5. `Minimum Spanning Forest`.

Итоговая рекомендация для диплома: делать все 5 задач, но ИИ-подходы реализовать в разной глубине. Для `MaxCut` и `MVC` достаточно воспроизвести упрощенные learned heuristics. Для поиска подграфов сделать наиболее сильную часть, потому что она напрямую связывает теорию с практикой `BITOC_WORK`.
