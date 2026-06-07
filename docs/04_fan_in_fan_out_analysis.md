# Анализ ноутбука `04-elliptic-fan-in-fan-out.ipynb`

## Цель ноутбука

В ноутбуке проверяется AML-паттерн `Fan-in / Fan-out` в графе Elliptic.

Цель не финальная классификация всех узлов, а отдельный эксперимент по паттерну:

1. Найти `Fan-in`, `Fan-out`, `Fan-in + Fan-out` структуры в графе.
2. Проверить, насколько хорошо разные подходы находят подозрительные hub-узлы.
3. Получить метрики.
4. Создать новые признаки для дальнейшей задачи классификации вершин в другом ноутбуке.

## Исходные данные

Используется стандартный `Elliptic Bitcoin Transaction Dataset`:

- `203769` узлов;
- `234355` рёбер;
- `46564` размеченных узлов.

Граф строится как directed-граф транзакций:

```text
txId1 -> txId2
```

## Искомые паттерны

- `Fan-in`: много входящих рёбер в один узел.
- `Fan-out`: много исходящих рёбер из одного узла.
- `Bipartite`: одновременно высокий вход и высокий выход.

## Классические подходы

### `Degree-filter`

Считаются:

- `in_degree`;
- `out_degree`.

Через пороги выделяются hub-узлы.

Результат:

- `fanout`: `2504`;
- `fanin`: `2011`;
- `bipartite`: `191`.

### `z-score` по степеням

Проверяется, насколько степень узла выше средней по графу.

Это лучше простого порога, потому что учитывает распределение степеней.

### `FlowScope-inspired anomalousness score`

Используется идея оценки аномальности локального потока.

Ограничение Elliptic:

- в датасете нет BTC-сумм на рёбрах;
- поэтому flow заменён proxy-оценкой через число входящих и исходящих соседей.

Score нормируется в диапазон `0..1`.

### `ego-subgraph visualization`

Для найденных hub-узлов строятся локальные фрагменты графа:

- центральный hub-узел;
- входящие соседи;
- исходящие соседи.

Визуализируются примеры:

- `fanin`;
- `fanout`;
- `bipartite`.

## ML / GNN подход

Используется `GAT` — graph attention network.

Модель получает:

- исходные признаки Elliptic;
- новые hub-признаки.

Unknown-узлы остаются в графе как структурный контекст.

Разбиение:

- train: `26904`;
- val: `2990`;
- test: `16670`.

Результаты `GAT`:

```text
all nodes: F1 = 0.4796, AUC = 0.8873
hub nodes only: F1 = 0.1818, AUC = 0.9110
bipartite only: F1 = 0.0000
```

## Сравнение методов

```text
Degree-filter threshold: Precision 0.0096, Recall 0.0083, F1 0.0089
FlowScope scoring:       Precision 0.0339, Recall 0.0018, F1 0.0035
GAT all nodes:           Precision 0.4451, Recall 0.5199, F1 0.4796
GAT hub nodes only:      Precision 0.5000, Recall 0.1111, F1 0.1818
```

Вывод:

- классические правила хорошо находят сам паттерн;
- классические правила плохо работают как самостоятельный классификатор illicit;
- `Degree-filter` и `FlowScope score` полезны как feature engineering;
- `GAT` лучше использует структуру графа и признаки;
- `Fan-in / Fan-out` реально присутствует в Elliptic.

## Новые признаки

В ноутбуке создаются признаки уровня узла:

```text
is_fanout_hub
is_fanin_hub
is_bipartite_hub
anomalousness_score
```

Эти признаки нужны не только для анализа паттерна, но и для дальнейшей классификации вершин.

## Сохранённые результаты

Ноутбук сохраняет:

```text
fanfan_detected.csv
predictions_fanfan.csv
fanfan_pipeline_comparison.csv
attention_analysis.csv
fanfan_ego_graph_examples.png
fanfan_methods_comparison.png
fanfan_f1_ranking.png
output_04_elliptic_fan_in_fan_out.zip
```

## Как использовать дальше

В следующем ноутбуке по основной классификации вершин нужно добавить hub-признаки к исходным данным Elliptic:

```text
исходные признаки Elliptic
+ temporal признаки
+ graph degree/PageRank признаки
+ Fan-in/Fan-out признаки
=> классификация illicit / licit
```

Новые признаки для переноса:

```text
is_fanout_hub
is_fanin_hub
is_bipartite_hub
anomalousness_score
```

## Главный вывод

`Fan-in / Fan-out` — рабочий AML-паттерн для Elliptic.

Сам по себе он не решает задачу классификации, но даёт интерпретируемые дополнительные признаки для финальной модели классификации узлов.
