# Pattern-признаки для основного Elliptic Data Set

Документ фиксирует признаки из pattern-ноутбуков для объединения с основной таблицей транзакций Elliptic и дальнейшей классификации `licit / illicit`.

Базовый ключ объединения: `txId`.  
Для burst-признаков желательно использовать ключ `txId + time_step`, так как активность считается во временном шаге.

## 1. Fan-in / Fan-out

Ноутбук: `04-elliptic-fan-in-fan-out.ipynb`

Смысл паттерна: поиск hub-структур, похожих на smurfing, consolidation или распределение средств.

Использованные подходы:

- rule-based degree-filter по `in_degree` и `out_degree`;
- z-score по входящей и исходящей степени;
- FlowScope-inspired scoring через локальную структуру входящих и исходящих соседей;
- ego-subgraph вокруг hub-узла;
- GAT для классификации узлов с учетом hub-признаков.

Основные результаты:

- найдено `2504` fan-out узлов;
- найдено `2011` fan-in узлов;
- найден `191` bipartite hub;
- GAT на всех тестовых узлах: `F1 = 0.4796`, `ROC-AUC = 0.8873`;
- вывод: простые пороги плохо работают как самостоятельный классификатор, но hub-признаки полезны как дополнительные graph-features.

Признаки для добавления:

- `in_degree` — число входящих ребер;
- `out_degree` — число исходящих ребер;
- `z_in` — z-score входящей степени;
- `z_out` — z-score исходящей степени;
- `hub_type` — тип узла: `fanin`, `fanout`, `bipartite`, `normal`;
- `is_fanin_hub` — бинарный признак fan-in центра;
- `is_fanout_hub` — бинарный признак fan-out центра;
- `is_bipartite_hub` — бинарный признак одновременно высокого входа и выхода;
- `anomalousness_score` — нормированный score локальной hub-аномальности;
- `elliptic_time_step_feature` — технический temporal-признак для модели, дублирует `time_step`.

## 2. Peel Chains

Ноутбук: `05-elliptic-peel-chains.ipynb`

Смысл паттерна: поиск цепочек слоения, где средства последовательно проходят через цепь транзакций, а часть уходит в боковые выходы.

Использованные подходы:

- rule-based поиск цепочек по directed-графу;
- ограничения на длину цепочки, временной разрыв, main-flow и peel-flow;
- GRU sequence autoencoder для ошибки реконструкции цепочки;
- Isolation Forest, LOF и PCA reconstruction для табличной аномальности;
- subgraph-level scoring через признаки локального подграфа;
- ensemble score через объединение рангов нескольких методов.

Основные результаты:

- найдено `288` baseline peel chains;
- размер sequence tensor: `(288, 18, 5)`;
- обучен GRU autoencoder;
- экспортируются `peel_features.csv` и `peel_chain_scores.csv`;
- вывод: peel chains дают отдельный набор структурных признаков для downstream node classification.

Признаки для добавления:

- `peel_in_chain_any` — участвует ли узел хотя бы в одной peel-chain;
- `peel_chain_count` — число цепочек, где встречается узел;
- `peel_baseline_chain_participation` — участие в rule-based baseline цепочке;
- `peel_start_count` — сколько раз узел был началом цепочки;
- `peel_middle_count` — сколько раз узел был внутри цепочки;
- `peel_end_count` — сколько раз узел был концом цепочки;
- `peel_max_chain_length` — максимальная длина цепочки с участием узла;
- `peel_mean_ae_error` — средняя ошибка GRU autoencoder по цепочкам узла;
- `peel_max_ae_error` — максимальная ошибка GRU autoencoder по цепочкам узла;
- `peel_mean_tab_anom_score` — средний табличный anomaly score;
- `peel_mean_gnn_anom_score` — средний subgraph anomaly score.

## 3. Burst Patterns

Ноутбук: `06-elliptic-burst-patterns.ipynb`

Смысл паттерна: поиск резких всплесков активности узла или локальной окрестности во временном графе.

Использованные подходы:

- RUSH-style rule-based detector по временным событиям ребер;
- агрегация `in_events`, `out_events`, `total_events` по `txId` и `time_step`;
- robust z-score внутри каждого временного шага;
- локальный burst-subgraph scoring вокруг top-кандидатов;
- Isolation Forest как unsupervised ML baseline;
- GRU forecasting error как опциональный temporal neural baseline;
- поддержка `DEVICE` для запуска GRU на GPU.

Основные результаты:

- рассчитаны burst-признаки для `203769` узлов;
- rule-based ranking и Isolation Forest сравниваются через `precision@k`, `recall@k`, `average_precision`;
- GRU-блок строит activity matrix для top dynamic nodes;
- экспортируются `elliptic_burst_features.csv`, `elliptic_burst_subgraphs.csv`, `elliptic_burst_ranking_metrics.csv`;
- вывод: burst-признаки подходят для оптимизационной постановки top-k выбора подозрительных транзакций.

Признаки для добавления:

- `in_events` — число входящих событий в текущем временном шаге;
- `out_events` — число исходящих событий в текущем временном шаге;
- `total_events` — суммарная активность узла в текущем временном шаге;
- `in_events_z` — robust z-score входящей активности;
- `out_events_z` — robust z-score исходящей активности;
- `total_events_z` — robust z-score общей активности;
- `degree_z` — z-score степени узла;
- `neighbor_degree_sum_z` — z-score суммарной степени соседей;
- `feature_volume_proxy_z` — z-score proxy-объема по анонимизированным локальным признакам;
- `burst_score_rule` — итоговый rule-based burst score;
- `burst_score_iforest_norm` — нормированный anomaly score Isolation Forest;
- `is_burst_iforest` — бинарный флаг аномалии по Isolation Forest;
- `burst_score_gru_error_norm` — нормированная ошибка GRU-прогноза, если GRU-блок был запущен.

Отдельные признаки для анализа burst-подграфов:

- `center_txId` — центральный узел burst-подграфа;
- `center_time_step` — временной шаг центра;
- `n_nodes` — число узлов в локальном подграфе;
- `n_edges` — число ребер в локальном подграфе;
- `density` — плотность локального подграфа;
- `center_burst_score` — burst score центрального узла;
- `mean_local_burst_score` — средний burst score в окрестности;
- `burst_subgraph_score` — итоговый score локального burst-подграфа;
- `known_illicit_share` — доля известных illicit-узлов в подграфе, только для анализа.

## Итоговый набор для финальной модели

Минимальный набор для первого объединения:

- fan признаки: `is_fanin_hub`, `is_fanout_hub`, `is_bipartite_hub`, `anomalousness_score`, `z_in`, `z_out`;
- peel признаки: `peel_in_chain_any`, `peel_chain_count`, `peel_max_chain_length`, `peel_mean_ae_error`, `peel_mean_tab_anom_score`, `peel_mean_gnn_anom_score`;
- burst признаки: `total_events_z`, `neighbor_degree_sum_z`, `feature_volume_proxy_z`, `burst_score_rule`, `burst_score_iforest_norm`, `is_burst_iforest`.

Расширенный набор используется после проверки корреляций и важности признаков.

## Как использовать в основной классификации

1. Считать основную таблицу Elliptic features.
2. Присоединить fan-признаки по `txId`.
3. Присоединить peel-признаки по `txId`.
4. Присоединить burst-признаки по `txId` и `time_step`.
5. Заполнить пропуски нулями для признаков участия в паттернах.
6. Сравнить модели:
   - baseline без pattern-признаков;
   - baseline + fan;
   - baseline + fan + peel;
   - baseline + fan + peel + burst.

Главный проверяемый вопрос: повышают ли структурные pattern-признаки качество классификации и top-k ранжирования подозрительных транзакций.
