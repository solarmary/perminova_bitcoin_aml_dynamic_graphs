"""
Назначение файла: Streamlit MVP для мониторинга Bitcoin-транзакций.
Основные шаги: загрузка витрин parser, расчет risk_score,
фильтрация периода, визуализация графа и проверка tx_id.
Зависимости или источники данных: CSV/Parquet из parser/data.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARSER_DIR = PROJECT_ROOT / "parser"
PROCESSED_DIR = PARSER_DIR / "data" / "processed"
INTERIM_DIR = PARSER_DIR / "data" / "interim"

FEATURES_PATH = PROCESSED_DIR / "parsed_txs_features_named.csv"
SCORES_PATH = PROCESSED_DIR / "parsed_txs_scores_template.csv"
EDGES_PATH = PROCESSED_DIR / "parsed_txs_edgelist.csv"
INPUTS_PATH = INTERIM_DIR / "inputs.parquet"
OUTPUTS_PATH = INTERIM_DIR / "outputs.parquet"
CONFIG_PATH = Path(__file__).resolve().parent / "config.txt"
TEST_GRAPH_PATH = Path(__file__).resolve().parent / "test_dynamic_graph.html"

HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.4
MAX_GRAPH_NODES = 250
MAX_GRAPH_EDGES = 600
MAX_GIF_NODES = 120
MAX_GIF_FRAMES = 30


st.set_page_config(
    page_title="Мониторинг Bitcoin-транзакций",
    page_icon=None,
    layout="wide",
)


def configure_compact_layout() -> None:
    """Настраивает компактный вид Streamlit-интерфейса."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 0.65rem;
            padding-bottom: 0.4rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
            max-width: 100%;
        }
        h1 {
            font-size: 1.35rem !important;
            margin-bottom: 0.1rem !important;
        }
        h2, h3 {
            font-size: 1.0rem !important;
            margin-top: 0.25rem !important;
            margin-bottom: 0.25rem !important;
        }
        [data-testid="stMetric"] {
            padding: 0.25rem 0.45rem;
            border: 1px solid #eeeeee;
            border-radius: 0.35rem;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.72rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.05rem !important;
        }
        [data-testid="stVerticalBlock"] {
            gap: 0.35rem;
        }
        [data-testid="stHorizontalBlock"] {
            gap: 0.6rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 2rem;
            padding: 0.25rem 0.75rem;
        }
        .stCaption, [data-testid="stCaptionContainer"] {
            font-size: 0.75rem !important;
        }
        .stDataFrame {
            font-size: 0.78rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_site_data() -> dict[str, pd.DataFrame]:
    """Загружает доступные таблицы сайта."""
    features = read_csv(FEATURES_PATH)
    scores = read_csv(SCORES_PATH)
    edges = read_csv(EDGES_PATH)
    inputs = read_table(INPUTS_PATH)
    outputs = read_table(OUTPUTS_PATH)

    features = prepare_features(features, scores)
    inputs = normalize_tx_column(inputs)
    outputs = normalize_tx_column(outputs)
    edges = normalize_edges(edges)

    return {
        "features": features,
        "edges": edges,
        "inputs": inputs,
        "outputs": outputs,
    }


def read_csv(path: Path) -> pd.DataFrame:
    """Читает CSV, если файл существует."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_table(path: Path) -> pd.DataFrame:
    """Читает Parquet или CSV-версию таблицы."""
    if path.exists():
        return pd.read_parquet(path)

    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)

    return pd.DataFrame()


def normalize_tx_column(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит идентификатор транзакции к строковому типу."""
    if df.empty or "tx_hash" not in df.columns:
        return df
    result = df.copy()
    result["tx_hash"] = result["tx_hash"].astype(str)
    return result


def normalize_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Нормализует edge list транзакционного графа."""
    if edges.empty:
        return pd.DataFrame(columns=["txId1", "txId2"])

    result = edges.copy()
    for column in ["txId1", "txId2"]:
        if column not in result.columns:
            result[column] = pd.Series(dtype=str)
        result[column] = result[column].astype(str)
    return result[["txId1", "txId2"]].dropna().drop_duplicates()


def prepare_features(features: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Добавляет risk_score, risk_label и дату."""
    if features.empty or "tx_hash" not in features.columns:
        return pd.DataFrame()

    result = features.copy()
    result["tx_hash"] = result["tx_hash"].astype(str)

    if "timestamp" in result.columns:
        result["datetime"] = pd.to_datetime(
            result["timestamp"],
            unit="s",
            errors="coerce",
            utc=True,
        )
    else:
        result["datetime"] = pd.NaT

    result = merge_existing_scores(result, scores)
    if "risk_score" not in result.columns or result["risk_score"].isna().all():
        result["risk_score"] = calculate_rule_based_score(result)

    result["risk_score"] = pd.to_numeric(result["risk_score"], errors="coerce")
    result["risk_score"] = result["risk_score"].fillna(0).clip(0, 1)
    result["risk_label"] = result["risk_score"].apply(label_risk)
    return result


def merge_existing_scores(
    features: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Подмешивает risk_score из внешнего файла, если он заполнен."""
    if scores.empty:
        return features

    score_id = "txId" if "txId" in scores.columns else "tx_hash"
    if score_id not in scores.columns or "risk_score" not in scores.columns:
        return features

    score_columns = [score_id, "risk_score"]
    if "risk_label" in scores.columns:
        score_columns.append("risk_label")

    prepared_scores = scores[score_columns].copy()
    prepared_scores = prepared_scores.rename(columns={score_id: "tx_hash"})
    prepared_scores["tx_hash"] = prepared_scores["tx_hash"].astype(str)
    prepared_scores["risk_score"] = pd.to_numeric(
        prepared_scores["risk_score"],
        errors="coerce",
    )

    if prepared_scores["risk_score"].isna().all():
        return features

    result = features.merge(prepared_scores, on="tx_hash", how="left")
    return result


def calculate_rule_based_score(features: pd.DataFrame) -> pd.Series:
    """Считает простой rule-based риск по структурным признакам."""
    score = pd.Series(0.0, index=features.index)
    weighted_columns = {
        "input_count": 0.12,
        "output_count": 0.16,
        "fee": 0.08,
        "total_output_value": 0.08,
        "total_degree": 0.14,
        "pagerank": 0.10,
        "weak_component_size": 0.08,
        "sum_neighbor_total_output_value": 0.08,
    }
    flag_columns = {
        "has_many_inputs": 0.10,
        "has_many_outputs": 0.12,
        "has_high_fee": 0.08,
        "has_small_outputs": 0.06,
    }

    for column, weight in weighted_columns.items():
        if column in features.columns:
            score += normalize_series(features[column]) * weight

    for column, weight in flag_columns.items():
        if column in features.columns:
            score += pd.to_numeric(features[column], errors="coerce").fillna(0) * weight

    return score.clip(0, 1).round(4)


def normalize_series(series: pd.Series) -> pd.Series:
    """Нормализует Series в диапазон 0..1 по 95-му перцентилю."""
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    if values.empty or values.max() == 0:
        return pd.Series(0.0, index=series.index)

    upper = values.quantile(0.95)
    if upper <= 0:
        upper = values.max()
    return (values / upper).clip(0, 1)


def label_risk(score: float) -> str:
    """Возвращает текстовый уровень риска."""
    if score >= HIGH_RISK_THRESHOLD:
        return "высокий"
    if score >= MEDIUM_RISK_THRESHOLD:
        return "средний"
    return "низкий"


def filter_features(features: pd.DataFrame) -> pd.DataFrame:
    """Рисует фильтры и возвращает отфильтрованную витрину."""
    st.sidebar.header("Фильтры")
    mode = st.sidebar.radio(
        "Режим проверки периода",
        ["По дате", "По номеру блока"],
    )

    if features.empty:
        return features

    filtered = features.copy()
    if mode == "По дате" and filtered["datetime"].notna().any():
        min_date = filtered["datetime"].min().date()
        max_date = filtered["datetime"].max().date()
        selected_dates = st.sidebar.date_input(
            "Дата от и дата до",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            date_from, date_to = selected_dates
        else:
            date_from = selected_dates
            date_to = selected_dates
        start = pd.Timestamp(date_from, tz="UTC")
        end = pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1)
        filtered = filtered[
            (filtered["datetime"] >= start) & (filtered["datetime"] < end)
        ]

    if mode == "По номеру блока" and "block_height" in filtered.columns:
        min_block = int(filtered["block_height"].min())
        max_block = int(filtered["block_height"].max())
        col_from, col_to = st.sidebar.columns(2)
        block_from = col_from.number_input(
            "Блок от",
            min_value=min_block,
            max_value=max_block,
            value=min_block,
            step=1,
        )
        block_to = col_to.number_input(
            "Блок до",
            min_value=min_block,
            max_value=max_block,
            value=max_block,
            step=1,
        )
        if block_from > block_to:
            block_from, block_to = block_to, block_from
        filtered = filtered[
            filtered["block_height"].between(block_from, block_to)
        ]

    min_risk = st.sidebar.slider(
        "Минимальный risk_score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
    )
    only_suspicious = st.sidebar.checkbox("Только suspicious", value=False)

    filtered = filtered[filtered["risk_score"] >= min_risk]
    if only_suspicious:
        filtered = filtered[filtered["risk_score"] >= HIGH_RISK_THRESHOLD]

    return filtered


def build_address_risks(
    features: pd.DataFrame,
    inputs: pd.DataFrame,
    outputs: pd.DataFrame,
) -> pd.DataFrame:
    """Строит агрегаты риска по адресам."""
    address_frames = []
    tx_scores = features[["tx_hash", "risk_score", "total_output_value"]].copy()

    if not inputs.empty and "input_address" in inputs.columns:
        frame = inputs[["tx_hash", "input_address"]].rename(
            columns={"input_address": "address"},
        )
        frame["direction"] = "input"
        address_frames.append(frame)

    if not outputs.empty and "output_address" in outputs.columns:
        frame = outputs[["tx_hash", "output_address"]].rename(
            columns={"output_address": "address"},
        )
        frame["direction"] = "output"
        address_frames.append(frame)

    if not address_frames:
        return pd.DataFrame(
            columns=[
                "address",
                "risk_score",
                "tx_count",
                "total_volume",
                "input_tx_count",
                "output_tx_count",
            ],
        )

    addresses = pd.concat(address_frames, ignore_index=True)
    addresses = addresses.dropna(subset=["address"])
    addresses["address"] = addresses["address"].astype(str)
    addresses = addresses.merge(tx_scores, on="tx_hash", how="inner")

    if addresses.empty:
        return pd.DataFrame()

    pivot = (
        addresses.pivot_table(
            index="address",
            columns="direction",
            values="tx_hash",
            aggfunc="nunique",
            fill_value=0,
        )
        .reset_index()
        .rename(columns={"input": "input_tx_count", "output": "output_tx_count"})
    )

    grouped = (
        addresses.groupby("address", as_index=False)
        .agg(
            risk_score=("risk_score", "max"),
            avg_risk_score=("risk_score", "mean"),
            tx_count=("tx_hash", "nunique"),
            total_volume=("total_output_value", "sum"),
        )
    )
    return grouped.merge(pivot, on="address", how="left").fillna(0)


def build_graph(edges: pd.DataFrame, tx_ids: set[str] | None = None) -> nx.DiGraph:
    """Строит граф транзакций с опциональным фильтром узлов."""
    graph = nx.DiGraph()
    if edges.empty:
        return graph

    filtered = edges
    if tx_ids is not None:
        filtered = edges[
            edges["txId1"].isin(tx_ids) | edges["txId2"].isin(tx_ids)
        ]

    graph.add_edges_from(filtered[["txId1", "txId2"]].itertuples(index=False))
    return graph


def count_suspicious_subgraphs(
    features: pd.DataFrame,
    edges: pd.DataFrame,
) -> int:
    """Считает компоненты среди транзакций высокого риска."""
    suspicious = set(
        features.loc[
            features["risk_score"] >= HIGH_RISK_THRESHOLD,
            "tx_hash",
        ].astype(str)
    )
    if not suspicious:
        return 0

    graph = build_graph(edges)
    subgraph = graph.subgraph(suspicious).to_undirected()
    return nx.number_connected_components(subgraph) if subgraph.number_of_nodes() else 0


def render_dashboard(
    features: pd.DataFrame,
    edges: pd.DataFrame,
    address_risks: pd.DataFrame,
) -> None:
    """Показывает главную страницу мониторинга."""
    suspicious = features[features["risk_score"] >= HIGH_RISK_THRESHOLD]
    suspicious_addresses = address_risks[
        address_risks["risk_score"] >= HIGH_RISK_THRESHOLD
    ]

    st.subheader("Обзор периода")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Всего транзакций", f"{len(features):,}".replace(",", " "))
    col2.metric("Подозрительных", f"{len(suspicious):,}".replace(",", " "))
    col3.metric("Доля suspicious", format_percent(len(suspicious), len(features)))
    col4.metric("Риск-адресов", f"{len(suspicious_addresses):,}".replace(",", " "))

    st.plotly_chart(
        compact_plotly_figure(build_activity_chart(features), height=215),
        use_container_width=True,
    )

    col_left, col_right = st.columns(2)
    with col_left:
        risk_distribution = (
            features["risk_label"]
            .value_counts()
            .rename_axis("risk_label")
            .reset_index(name="tx_count")
        )
        st.plotly_chart(
            compact_plotly_figure(
                px.bar(
                    risk_distribution,
                    x="risk_label",
                    y="tx_count",
                    color="risk_label",
                    labels={"risk_label": "Уровень риска", "tx_count": "Транзакции"},
                ),
                height=185,
            ),
            use_container_width=True,
        )

    with col_right:
        subgraph_count = count_suspicious_subgraphs(features, edges)
        st.metric("Подозрительные подграфы", subgraph_count)
        st.metric("Средний risk_score", f"{features['risk_score'].mean():.3f}")

    col_tx, col_addr = st.columns(2)
    with col_tx:
        st.caption("Топ транзакций")
        st.dataframe(
            top_transactions(features),
            use_container_width=True,
            hide_index=True,
            height=185,
        )

    with col_addr:
        st.caption("Топ адресов")
        st.dataframe(
            top_addresses(address_risks),
            use_container_width=True,
            hide_index=True,
            height=185,
        )

    with st.expander("Как считается risk_score"):
        render_model_page()


def build_activity_chart(features: pd.DataFrame):
    """Строит график suspicious-транзакций по времени или блокам."""
    suspicious = features[features["risk_score"] >= HIGH_RISK_THRESHOLD].copy()
    if suspicious.empty:
        return px.line(
            pd.DataFrame({"period": [], "tx_count": []}),
            x="period",
            y="tx_count",
        )

    period_column, period_label = choose_activity_period(suspicious)
    suspicious["period"] = period_column

    activity = (
        suspicious.groupby("period", as_index=False)
        .agg(tx_count=("tx_hash", "count"))
        .sort_values("period")
    )
    return px.line(
        activity,
        x="period",
        y="tx_count",
        markers=True,
        labels={"period": period_label, "tx_count": "Подозрительные транзакции"},
    )


def compact_plotly_figure(figure, height: int = 220):
    """Уплотняет Plotly-граф для отображения без прокрутки."""
    figure.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 18, "b": 10},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        font={"size": 10},
    )
    figure.update_xaxes(title_font={"size": 10}, tickfont={"size": 9})
    figure.update_yaxes(title_font={"size": 10}, tickfont={"size": 9})
    return figure


def choose_activity_period(features: pd.DataFrame) -> tuple[pd.Series, str]:
    """Выбирает детализацию для графика активности."""
    if "block_height" in features.columns and features["block_height"].nunique() > 1:
        return features["block_height"], "Блок"
    if "time_step" in features.columns and features["time_step"].nunique() > 1:
        return features["time_step"], "time_step"
    if "datetime" in features.columns and features["datetime"].notna().any():
        return features["datetime"].dt.date, "Дата"
    return pd.Series(1, index=features.index), "Период"


def top_transactions(features: pd.DataFrame) -> pd.DataFrame:
    """Возвращает топ транзакций по risk_score."""
    columns = [
        "tx_hash",
        "risk_score",
        "risk_label",
        "block_height",
        "input_count",
        "output_count",
        "total_output_value",
        "total_degree",
    ]
    available = [column for column in columns if column in features.columns]
    return features.sort_values("risk_score", ascending=False)[available].head(10)


def top_addresses(address_risks: pd.DataFrame) -> pd.DataFrame:
    """Возвращает топ адресов по risk_score."""
    if address_risks.empty:
        return pd.DataFrame()
    columns = [
        "address",
        "risk_score",
        "tx_count",
        "input_tx_count",
        "output_tx_count",
        "total_volume",
    ]
    available = [column for column in columns if column in address_risks.columns]
    return address_risks.sort_values("risk_score", ascending=False)[available].head(10)


def format_percent(numerator: int, denominator: int) -> str:
    """Форматирует долю в процентах."""
    if denominator == 0:
        return "0.0%"
    return f"{numerator / denominator * 100:.1f}%"


def render_transaction_graph(
    features: pd.DataFrame,
    edges: pd.DataFrame,
) -> None:
    """Показывает интерактивный граф транзакций."""
    st.subheader("Граф транзакций")

    if is_test_mode_enabled():
        render_test_dynamic_graph()
        return

    max_value = max(float(features["total_output_value"].max() or 0.0), 0.01)
    graph_mode = st.radio(
        "Режим графа",
        ["Снимок", "GIF за период"],
        horizontal=True,
        key="graph_mode",
    )

    col_value, col_risk, col_flag = st.columns([1, 1, 1])
    with col_value:
        min_value = st.slider(
            "Минимальная сумма перевода BTC",
            min_value=0.0,
            max_value=max_value,
            value=0.0,
            step=0.01,
        )
    with col_risk:
        risk_from = st.slider(
            "risk_score от",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="graph_risk_from",
        )
    with col_flag:
        only_suspicious = st.checkbox("Показать только suspicious", value=False)

    if graph_mode == "Снимок":
        render_static_graph_snapshot(
            features=features,
            edges=edges,
            min_value=min_value,
            risk_from=risk_from,
            only_suspicious=only_suspicious,
        )
        return

    render_graph_gif_mode(
        features=features,
        edges=edges,
        min_value=min_value,
        risk_from=risk_from,
        only_suspicious=only_suspicious,
    )


def is_test_mode_enabled() -> bool:
    """Проверяет флаг тестового режима."""
    if not CONFIG_PATH.exists():
        return False
    value = CONFIG_PATH.read_text(encoding="utf-8").strip().lower()
    return value == "true"


def render_test_dynamic_graph() -> None:
    """Показывает заранее подготовленный тестовый граф."""
    if not TEST_GRAPH_PATH.exists():
        st.error(f"Файл тестового графа не найден: {TEST_GRAPH_PATH}")
        return

    html = TEST_GRAPH_PATH.read_text(encoding="utf-8")
    st.caption(
        "Тестовый режим включен: отображается демонстрационный граф "
        "с подозрительными паттернами."
    )
    components.html(html, height=560, scrolling=False)


def render_static_graph_snapshot(
    features: pd.DataFrame,
    edges: pd.DataFrame,
    min_value: float,
    risk_from: float,
    only_suspicious: bool,
) -> None:
    """Показывает интерактивный снимок графа на момент времени."""
    show_available_datetime_hint(features)
    selected = select_graph_snapshot(features)
    selected = apply_graph_filters(selected, min_value, risk_from, only_suspicious)

    selected_tx_ids = set(selected["tx_hash"].astype(str))
    selected_edges = select_graph_edges(
        selected,
        edges,
        selected_tx_ids,
        allow_external_neighbors=False,
    )
    st.caption(
        f"Узлов: {len(selected):,}. Ребер: {len(selected_edges):,}. "
        "Граф можно приближать и двигать.".replace(",", " ")
    )

    if len(selected) > MAX_GRAPH_NODES:
        selected = selected.nlargest(MAX_GRAPH_NODES, "risk_score")
        st.info(f"Граф ограничен {MAX_GRAPH_NODES} узлами с максимальным риском.")

    html, graph_note = build_pyvis_html(
        selected,
        edges,
        allow_external_neighbors=False,
        height_px=430,
    )
    if html:
        if graph_note:
            st.info(graph_note)
        col_graph, col_side = st.columns([4, 1.15])
        with col_graph:
            components.html(html, height=455, scrolling=False)
        with col_side:
            render_graph_info_panel(
                node_count=len(selected),
                edge_count=len(selected_edges),
                suspicious_count=int(
                    (selected["risk_score"] >= HIGH_RISK_THRESHOLD).sum()
                ),
            )
    else:
        st.warning("Недостаточно ребер для построения графа.")


def render_graph_info_panel(
    node_count: int,
    edge_count: int,
    suspicious_count: int,
) -> None:
    """Показывает правую информационную панель графа."""
    st.markdown(
        f"""
        <div style="
            height: 431px;
            padding: 12px;
            border: 1px solid #d9e2ec;
            border-radius: 12px;
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
            font-size: 12px;
        ">
            <div style="font-weight: 700; font-size: 15px; margin-bottom: 8px;">
                Граф Bitcoin-паттернов
            </div>
            <div style="color: #52606d; line-height: 1.35; margin-bottom: 10px;">
                Узлы соответствуют транзакциям, ребра показывают связи
                <code>txId1 -> txId2</code>. Красный цвет означает повышенный риск.
            </div>
            <div style="margin: 8px 0;">
                <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#22c55e;"></span>
                обычный узел
            </div>
            <div style="margin: 8px 0;">
                <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ef4444;"></span>
                подозрительный узел
            </div>
            <div style="margin-top: 10px; padding: 8px; border-radius: 10px; background:#f8fafc; border:1px solid #e5e7eb;">
                <b style="font-size:18px;">{node_count}</b><br>транзакций
            </div>
            <div style="margin-top: 10px; padding: 8px; border-radius: 10px; background:#f8fafc; border:1px solid #e5e7eb;">
                <b style="font-size:18px;">{edge_count}</b><br>связей
            </div>
            <div style="margin-top: 10px; padding: 8px; border-radius: 10px; background:#f8fafc; border:1px solid #e5e7eb;">
                <b style="font-size:18px;">{suspicious_count}</b><br>подозрительных узлов
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_graph_gif_mode(
    features: pd.DataFrame,
    edges: pd.DataFrame,
    min_value: float,
    risk_from: float,
    only_suspicious: bool,
) -> None:
    """Показывает генерацию GIF по выбранному периоду."""
    if not features["datetime"].notna().any():
        st.warning("Для GIF по дате и часу нужна колонка datetime.")
        return

    show_available_datetime_hint(features)
    start_dt, end_dt = select_graph_datetime_period(features)
    cumulative = st.checkbox(
        "Показывать накопительно",
        value=True,
        key="gif_cumulative",
    )

    if start_dt >= end_dt:
        st.warning("Дата и время начала должны быть раньше даты и времени конца.")
        return

    period_features = features[
        (features["datetime"] >= start_dt) & (features["datetime"] <= end_dt)
    ]
    period_features = apply_graph_filters(
        period_features,
        min_value,
        risk_from,
        only_suspicious,
    )
    if period_features.empty:
        st.warning("В выбранном периоде нет транзакций после фильтрации.")
        return

    if st.button("Сформировать GIF", type="primary"):
        try:
            with st.spinner("Формируется GIF графа"):
                gif_bytes = build_graph_gif(
                    period_features,
                    edges,
                    cumulative=cumulative,
                )
        except ModuleNotFoundError as error:
            st.error(f"Не установлена зависимость для GIF: {error.name}")
            st.code("pip install -r requirements.txt", language="powershell")
            return

        if gif_bytes is None:
            st.warning("Недостаточно данных для GIF.")
            return

        col_gif, col_download = st.columns([4, 1])
        with col_gif:
            st.image(gif_bytes, use_container_width=True)
        with col_download:
            st.download_button(
                "Скачать GIF",
                data=gif_bytes,
                file_name="transaction_graph_dynamic.gif",
                mime="image/gif",
            )


def apply_graph_filters(
    features: pd.DataFrame,
    min_value: float,
    risk_from: float,
    only_suspicious: bool,
) -> pd.DataFrame:
    """Применяет фильтры графа."""
    if features.empty:
        return features

    selected = features[
        (features["risk_score"] >= risk_from)
        & (features["total_output_value"].fillna(0) >= min_value)
    ]
    if only_suspicious:
        selected = selected[selected["risk_score"] >= HIGH_RISK_THRESHOLD]
    return selected


def select_graph_snapshot(features: pd.DataFrame) -> pd.DataFrame:
    """Выбирает данные для статического снимка."""
    if features.empty or not features["datetime"].notna().any():
        return select_graph_scope(features)

    min_dt = features["datetime"].min()
    max_dt = features["datetime"].max()
    snapshot_default = max_dt.floor("h")
    col_date, col_time, col_mode = st.columns([1, 1, 1])
    with col_date:
        snapshot_date = st.date_input(
            "Дата снимка",
            value=snapshot_default.date(),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
            key="snapshot_date",
        )
    with col_time:
        snapshot_time = st.time_input(
            "Час снимка",
            value=snapshot_default.time().replace(minute=0, second=0, microsecond=0),
            key="snapshot_time",
        )
    with col_mode:
        cumulative = st.checkbox(
            "Состояние к моменту",
            value=True,
            key="snapshot_cumulative",
        )

    snapshot_dt = pd.Timestamp.combine(snapshot_date, snapshot_time).tz_localize("UTC")
    if cumulative:
        selected = features[features["datetime"] <= snapshot_dt]
    else:
        end_dt = snapshot_dt + pd.Timedelta(hours=1)
        selected = features[
            (features["datetime"] >= snapshot_dt) & (features["datetime"] < end_dt)
        ]

    if selected.empty:
        st.info(
            "На выбранный момент транзакций нет. "
            f"Доступный диапазон: {format_datetime_range(min_dt, max_dt)}. "
            "Показан весь выбранный период."
        )
        return features
    return selected


def show_available_datetime_hint(features: pd.DataFrame) -> None:
    """Показывает доступный диапазон дат для графа."""
    if features.empty or "datetime" not in features.columns:
        return
    if not features["datetime"].notna().any():
        return

    min_dt = features["datetime"].min()
    max_dt = features["datetime"].max()
    best_hour = (
        features.groupby(features["datetime"].dt.floor("h"))
        .size()
        .sort_values(ascending=False)
        .index[0]
    )
    st.caption(
        "Данные доступны: "
        f"{format_datetime_range(min_dt, max_dt)}. "
        f"Рекомендуемый час для снимка: {format_datetime(best_hour)} UTC."
    )


def format_datetime_range(min_dt: pd.Timestamp, max_dt: pd.Timestamp) -> str:
    """Форматирует диапазон дат UTC."""
    return f"{format_datetime(min_dt)} UTC - {format_datetime(max_dt)} UTC"


def format_datetime(value: pd.Timestamp) -> str:
    """Форматирует дату и время для подсказок."""
    return value.strftime("%Y-%m-%d %H:%M")


def select_graph_datetime_period(
    features: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Рисует выбор периода с точностью до часа."""
    min_dt = features["datetime"].min()
    max_dt = features["datetime"].max()
    start_default = min_dt.floor("h")
    end_default = max_dt.ceil("h")

    col_start_date, col_start_time, col_end_date, col_end_time = st.columns(4)
    with col_start_date:
        start_date = st.date_input(
            "Дата от",
            value=start_default.date(),
            min_value=start_default.date(),
            max_value=end_default.date(),
            key="gif_start_date",
        )
    with col_start_time:
        start_time = st.time_input(
            "Час от",
            value=start_default.time().replace(minute=0, second=0, microsecond=0),
            key="gif_start_time",
        )
    with col_end_date:
        end_date = st.date_input(
            "Дата до",
            value=end_default.date(),
            min_value=start_default.date(),
            max_value=end_default.date(),
            key="gif_end_date",
        )
    with col_end_time:
        end_time = st.time_input(
            "Час до",
            value=end_default.time().replace(minute=0, second=0, microsecond=0),
            key="gif_end_time",
        )

    start_dt = pd.Timestamp.combine(start_date, start_time).tz_localize("UTC")
    end_dt = pd.Timestamp.combine(end_date, end_time).tz_localize("UTC")
    return start_dt, end_dt


def select_graph_scope(features: pd.DataFrame) -> pd.DataFrame:
    """Выбирает временной срез для графа."""
    if features.empty:
        return features

    return select_dynamic_graph_slice(features)


def select_static_graph_slice(features: pd.DataFrame) -> pd.DataFrame:
    """Возвращает статичный срез по дате или time_step."""
    if features["datetime"].notna().any():
        min_date = features["datetime"].min().date()
        max_date = features["datetime"].max().date()
        selected_date = st.date_input(
            "Дата статичного среза",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="graph_static_date",
        )
        start = pd.Timestamp(selected_date, tz="UTC")
        end = start + pd.Timedelta(days=1)
        selected = features[
            (features["datetime"] >= start) & (features["datetime"] < end)
        ]
        if selected.empty:
            st.info("За выбранную дату транзакций нет. Используется весь выбранный период.")
            return features
        return selected

    if "time_step" not in features.columns:
        return features

    steps = sorted(pd.to_numeric(features["time_step"], errors="coerce").dropna().unique())
    if not steps:
        return features

    selected_step = st.select_slider(
        "time_step статичного среза",
        options=[int(step) for step in steps],
        value=int(steps[-1]),
        key="graph_static_step",
    )
    return features[features["time_step"] == selected_step]


def select_dynamic_graph_slice(features: pd.DataFrame) -> pd.DataFrame:
    """Возвращает текущий кадр динамического графа."""
    axis_column, axis_label = choose_graph_axis(features)
    if axis_column is None:
        st.info("В данных нет временной оси. Показан весь выбранный период.")
        return features

    axis_values = sorted(
        pd.to_numeric(features[axis_column], errors="coerce").dropna().unique()
    )
    if not axis_values:
        return features

    min_value = int(min(axis_values))
    max_value = int(max(axis_values))
    if min_value == max_value:
        st.info(f"В выбранном периоде только один {axis_label}. Показан весь граф.")
        return features

    st.plotly_chart(
        compact_plotly_figure(
            build_graph_timeline_chart(features, axis_column, axis_label),
            height=175,
        ),
        use_container_width=True,
    )

    col_current, col_mode = st.columns([2, 1])
    with col_current:
        current_value = st.slider(
            f"Кадр динамики по {axis_label}",
            min_value=min_value,
            max_value=max_value,
            value=min_value,
            step=1,
            key="graph_dynamic_current_value",
        )
    with col_mode:
        cumulative = st.checkbox(
            "Накопительно",
            value=True,
            key="graph_dynamic_cumulative",
        )

    if cumulative:
        selected = features[features[axis_column] <= current_value]
        st.caption(f"Показано развитие графа до {axis_label}: {current_value}.")
        return selected

    selected = features[features[axis_column] == current_value]
    st.caption(f"Показан отдельный кадр: {axis_label} {current_value}.")
    return selected


def choose_graph_axis(features: pd.DataFrame) -> tuple[str | None, str]:
    """Выбирает ось для динамического графа."""
    if "block_height" in features.columns and features["block_height"].nunique() > 1:
        return "block_height", "блок"
    if "time_step" in features.columns and features["time_step"].nunique() > 1:
        return "time_step", "time_step"
    return None, "периоду"


def build_graph_timeline_chart(
    features: pd.DataFrame,
    axis_column: str,
    axis_label: str,
):
    """Строит краткую динамику доступных time_step."""
    if features.empty or axis_column not in features.columns:
        return px.line(
            pd.DataFrame({axis_column: [], "tx_count": [], "high_risk_count": []}),
            x=axis_column,
            y="tx_count",
        )

    timeline = (
        features.assign(
            high_risk=features["risk_score"] >= HIGH_RISK_THRESHOLD,
        )
        .groupby(axis_column, as_index=False)
        .agg(
            tx_count=("tx_hash", "count"),
            high_risk_count=("high_risk", "sum"),
        )
        .sort_values(axis_column)
    )
    return px.line(
        timeline,
        x=axis_column,
        y=["tx_count", "high_risk_count"],
        markers=True,
        labels={
            axis_column: axis_label,
            "value": "Количество транзакций",
            "variable": "Метрика",
        },
    )


def build_pyvis_html(
    features: pd.DataFrame,
    edges: pd.DataFrame,
    allow_external_neighbors: bool = True,
    height_px: int = 430,
) -> tuple[str | None, str | None]:
    """Генерирует HTML для PyVis-графа."""
    if features.empty:
        return None, None

    tx_ids = set(features["tx_hash"].astype(str))
    graph_edges = select_graph_edges(
        features,
        edges,
        tx_ids,
        allow_external_neighbors=allow_external_neighbors,
    )
    graph_note = None

    if graph_edges.empty:
        graph_edges = build_demo_edges(features)
        graph_note = (
            "В исходном edge list нет связей для выбранных транзакций. "
            "Показан демонстрационный граф по порядку блоков."
        )
    elif not (
        graph_edges["txId1"].isin(tx_ids) & graph_edges["txId2"].isin(tx_ids)
    ).all():
        graph_note = (
            "Часть соседних транзакций находится вне выбранного периода. "
            "Они добавлены как внешние узлы."
        )

    if graph_edges.empty:
        return None, None

    score_map = features.set_index("tx_hash")["risk_score"].to_dict()
    value_map = features.set_index("tx_hash")["total_output_value"].to_dict()

    net = Network(
        height=f"{height_px}px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
    )
    net.barnes_hut()

    node_ids = set(graph_edges["txId1"]) | set(graph_edges["txId2"])
    for node_id in node_ids:
        is_known = node_id in score_map
        score = float(score_map.get(node_id, 0.0))
        value = float(value_map.get(node_id, 0.0) or 0.0)
        title = f"tx_id: {node_id}<br>risk_score: {score:.3f}"
        color = risk_color(score)
        if not is_known:
            title = f"tx_id: {node_id}<br>внешняя транзакция"
            color = "#8c8c8c"
        net.add_node(
            node_id,
            label=f"tx {short_hash(node_id)}",
            title=title,
            color=color,
            size=12 + min(value * 2, 30) + score * 15,
        )

    for edge in graph_edges.itertuples(index=False):
        weight = calculate_edge_weight(edge.txId1, edge.txId2, value_map)
        net.add_edge(
            edge.txId1,
            edge.txId2,
            label=format_btc_weight(weight),
            title=f"weight: {weight:.8f} BTC",
            value=max(weight, 0.00000001),
            width=1 + min(weight * 3, 8),
        )

    with NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        tmp_path = Path(tmp.name)

    net.save_graph(str(tmp_path))
    html = tmp_path.read_text(encoding="utf-8")
    tmp_path.unlink(missing_ok=True)
    return html, graph_note


def calculate_edge_weight(
    source: str,
    target: str,
    value_map: dict[str, float],
) -> float:
    """Возвращает вес ребра в BTC."""
    target_value = float(value_map.get(target, 0.0) or 0.0)
    if target_value > 0:
        return target_value
    return float(value_map.get(source, 0.0) or 0.0)


def format_btc_weight(weight: float) -> str:
    """Форматирует вес ребра для подписи."""
    if weight >= 1:
        return f"{weight:.2f} BTC"
    if weight >= 0.001:
        return f"{weight:.4f} BTC"
    if weight > 0:
        return f"{weight:.8f} BTC"
    return "0 BTC"


def build_graph_gif(
    features: pd.DataFrame,
    edges: pd.DataFrame,
    cumulative: bool,
) -> bytes | None:
    """Строит GIF изменения графа во времени."""
    frames_data = build_gif_frames_data(features, cumulative)
    if len(frames_data) < 2:
        return None

    all_features = frames_data[-1][1] if cumulative else features
    all_features = limit_gif_nodes(all_features)
    full_edges = select_gif_edges(all_features, edges)
    full_graph = build_frame_graph(all_features, full_edges)
    if full_graph.number_of_nodes() == 0:
        return None

    positions = nx.spring_layout(full_graph, seed=42, k=0.35, iterations=60)
    images = []
    for label, frame_features in frames_data:
        frame_features = limit_gif_nodes(frame_features)
        frame_edges = select_gif_edges(frame_features, edges)
        image = draw_graph_frame(frame_features, frame_edges, positions, label)
        images.append(image)

    if not images:
        return None

    output = BytesIO()
    images[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=800,
        loop=0,
    )
    return output.getvalue()


def build_gif_frames_data(
    features: pd.DataFrame,
    cumulative: bool,
) -> list[tuple[str, pd.DataFrame]]:
    """Формирует кадры GIF."""
    ordered = features.sort_values(["datetime", "tx_hash"]).copy()
    if "block_height" in ordered.columns and ordered["block_height"].nunique() > 1:
        values = sorted(ordered["block_height"].dropna().unique())
        axis = "block_height"
        label_prefix = "Блок"
    elif "time_step" in ordered.columns and ordered["time_step"].nunique() > 1:
        values = sorted(ordered["time_step"].dropna().unique())
        axis = "time_step"
        label_prefix = "time_step"
    else:
        ordered["frame_id"] = range(1, len(ordered) + 1)
        chunk_size = max(len(ordered) // MAX_GIF_FRAMES, 1)
        ordered["frame_id"] = ((ordered["frame_id"] - 1) // chunk_size) + 1
        values = sorted(ordered["frame_id"].dropna().unique())
        axis = "frame_id"
        label_prefix = "Кадр"

    values = reduce_frame_values(values)
    frames = []
    for value in values:
        if cumulative:
            frame = ordered[ordered[axis] <= value]
        else:
            frame = ordered[ordered[axis] == value]
        if frame.empty:
            continue
        frames.append((f"{label_prefix}: {int(value)}", frame))
    return frames


def reduce_frame_values(values: list[object]) -> list[object]:
    """Ограничивает число кадров GIF."""
    if len(values) <= MAX_GIF_FRAMES:
        return values

    step = max(len(values) // MAX_GIF_FRAMES, 1)
    reduced = values[::step]
    if reduced[-1] != values[-1]:
        reduced.append(values[-1])
    return reduced[:MAX_GIF_FRAMES]


def limit_gif_nodes(features: pd.DataFrame) -> pd.DataFrame:
    """Ограничивает число узлов GIF."""
    if len(features) <= MAX_GIF_NODES:
        return features
    return features.nlargest(MAX_GIF_NODES, "risk_score")


def select_gif_edges(features: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Выбирает ребра для GIF без внешних звездчатых соседей."""
    tx_ids = set(features["tx_hash"].astype(str))
    if not edges.empty:
        inner_edges = edges[
            edges["txId1"].isin(tx_ids) & edges["txId2"].isin(tx_ids)
        ]
        if not inner_edges.empty:
            return inner_edges.head(MAX_GRAPH_EDGES)
    return build_demo_edges(features)


def build_frame_graph(features: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    """Строит граф кадра GIF."""
    graph = nx.DiGraph()
    graph.add_nodes_from(features["tx_hash"].astype(str))
    if not edges.empty:
        graph.add_edges_from(edges[["txId1", "txId2"]].itertuples(index=False))
    return graph


def draw_graph_frame(
    features: pd.DataFrame,
    edges: pd.DataFrame,
    positions: dict[str, tuple[float, float]],
    label: str,
) -> object:
    """Рисует один кадр GIF."""
    from PIL import Image, ImageDraw

    graph = build_frame_graph(features, edges)
    node_ids = list(graph.nodes())
    score_map = features.set_index("tx_hash")["risk_score"].to_dict()
    value_map = features.set_index("tx_hash")["total_output_value"].to_dict()
    canvas_size = (1100, 720)
    image = Image.new("RGB", canvas_size, "white")
    draw = ImageDraw.Draw(image)
    title_font = load_gif_font(18)
    text_font = load_gif_font(14)
    node_font = load_gif_font(10)
    pixel_positions = project_positions(positions, node_ids, canvas_size)

    draw.text((24, 18), label, fill="#222222", font=title_font)
    draw.text(
        (24, 40),
        f"Узлов: {graph.number_of_nodes()}  Ребер: {graph.number_of_edges()}",
        fill="#555555",
        font=text_font,
    )

    for source, target in graph.edges():
        if source not in pixel_positions or target not in pixel_positions:
            continue
        x1, y1 = pixel_positions[source]
        x2, y2 = pixel_positions[target]
        draw.line((x1, y1, x2, y2), fill="#b0b0b0", width=1)

    labels = build_frame_labels(features)
    for node_id in node_ids:
        x, y = pixel_positions.get(node_id, (canvas_size[0] // 2, canvas_size[1] // 2))
        score = float(score_map.get(node_id, 0.0))
        value = float(value_map.get(node_id, 0.0) or 0.0)
        radius = int(4 + min(value * 1.5, 10) + score * 8)
        color = risk_color(score)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline="#333333",
            width=1,
        )
        if node_id in labels:
            draw.text(
                (x + radius + 2, y - radius),
                labels[node_id],
                fill="#333333",
                font=node_font,
            )

    return image


def load_gif_font(size: int) -> object:
    """Загружает шрифт с поддержкой кириллицы для GIF."""
    from PIL import ImageFont

    font_paths = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for font_path in font_paths:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def project_positions(
    positions: dict[str, tuple[float, float]],
    node_ids: list[str],
    canvas_size: tuple[int, int],
) -> dict[str, tuple[int, int]]:
    """Проецирует координаты layout на область изображения."""
    if not node_ids:
        return {}

    width, height = canvas_size
    padding_x = 80
    padding_y = 90
    xs = [float(positions.get(node, (0.0, 0.0))[0]) for node in node_ids]
    ys = [float(positions.get(node, (0.0, 0.0))[1]) for node in node_ids]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0e-9)
    span_y = max(max_y - min_y, 1.0e-9)

    projected = {}
    for node in node_ids:
        x, y = positions.get(node, (0.0, 0.0))
        px = padding_x + (float(x) - min_x) / span_x * (width - 2 * padding_x)
        py = padding_y + (float(y) - min_y) / span_y * (height - 2 * padding_y)
        projected[node] = (int(px), int(py))
    return projected


def build_frame_labels(features: pd.DataFrame) -> dict[str, str]:
    """Формирует подписи узлов кадра."""
    if len(features) <= 35:
        label_features = features
    else:
        label_features = features.nlargest(35, "risk_score")
    return {
        str(row.tx_hash): short_hash(str(row.tx_hash))
        for row in label_features.itertuples(index=False)
    }


def select_graph_edges(
    features: pd.DataFrame,
    edges: pd.DataFrame,
    tx_ids: set[str],
    allow_external_neighbors: bool = True,
) -> pd.DataFrame:
    """Выбирает ребра для графа с сохранением внешних соседей."""
    if edges.empty:
        return pd.DataFrame(columns=["txId1", "txId2"])

    inner_edges = edges[
        edges["txId1"].isin(tx_ids) & edges["txId2"].isin(tx_ids)
    ]
    if not inner_edges.empty:
        return inner_edges.head(MAX_GRAPH_EDGES)

    if not allow_external_neighbors:
        return pd.DataFrame(columns=["txId1", "txId2"])

    neighbor_edges = edges[
        edges["txId1"].isin(tx_ids) | edges["txId2"].isin(tx_ids)
    ]
    if neighbor_edges.empty:
        return neighbor_edges

    known_risks = features.set_index("tx_hash")["risk_score"]
    ranked = neighbor_edges.copy()
    ranked["rank_score"] = ranked["txId1"].map(known_risks).fillna(
        ranked["txId2"].map(known_risks),
    )
    return (
        ranked.sort_values("rank_score", ascending=False)
        .drop(columns=["rank_score"])
        .head(MAX_GRAPH_EDGES)
    )


def build_demo_edges(features: pd.DataFrame) -> pd.DataFrame:
    """Строит демонстрационные связи, если edge list пуст для периода."""
    if len(features) < 2:
        return pd.DataFrame(columns=["txId1", "txId2"])

    sort_columns = [
        column
        for column in ["block_height", "timestamp", "time_step", "tx_hash"]
        if column in features.columns
    ]
    ordered = features.sort_values(sort_columns)["tx_hash"].astype(str).head(80)
    return pd.DataFrame(
        {
            "txId1": ordered.iloc[:-1].to_list(),
            "txId2": ordered.iloc[1:].to_list(),
        }
    )


def risk_color(score: float) -> str:
    """Возвращает цвет узла по риску."""
    if score >= HIGH_RISK_THRESHOLD:
        return "#d62728"
    if score >= MEDIUM_RISK_THRESHOLD:
        return "#ffbf00"
    return "#2ca02c"


def short_hash(value: str) -> str:
    """Сокращает tx_id для подписи."""
    if len(value) <= 12:
        return value
    return f"{value[:6]}...{value[-6:]}"


def render_transaction_check(
    features: pd.DataFrame,
    edges: pd.DataFrame,
    inputs: pd.DataFrame,
    outputs: pd.DataFrame,
) -> None:
    """Показывает проверку конкретной транзакции."""
    st.subheader("Проверка транзакции")
    example_tx_id = get_example_tx_id(features)
    if example_tx_id:
        st.caption(f"Пример: `{example_tx_id}`")

    tx_id = st.text_input("Введите tx_id").strip()
    if not tx_id:
        st.info("Введите идентификатор транзакции для анализа.")
        return

    match = features[features["tx_hash"] == tx_id]
    if match.empty:
        st.warning("Транзакция не найдена в текущей витрине.")
        return

    tx = match.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("risk_score", f"{tx['risk_score']:.3f}")
    col2.metric("Уровень риска", tx["risk_label"])
    col3.metric("Сумма перевода BTC", f"{tx.get('total_output_value', 0):.8f}")
    col4.metric("Входы / выходы", f"{tx.get('input_count', 0)} / {tx.get('output_count', 0)}")

    st.caption(build_risk_explanation(tx))

    col_graph, col_tables = st.columns([1.35, 1])
    with col_graph:
        local_features = local_graph_features(tx_id, features, edges, depth=1)
        html, graph_note = build_pyvis_html(local_features, edges, height_px=350)
        if html:
            if graph_note:
                st.caption(graph_note)
            components.html(html, height=365, scrolling=False)
        else:
            st.info("Для транзакции нет локальных ребер.")

    with col_tables:
        st.caption("Входы")
        st.dataframe(
            filter_tx_rows(inputs, {tx_id}).head(20),
            use_container_width=True,
            hide_index=True,
            height=145,
        )
        st.caption("Выходы")
        st.dataframe(
            filter_tx_rows(outputs, {tx_id}).head(20),
            use_container_width=True,
            hide_index=True,
            height=145,
        )
        st.caption(
            "Паттерны: "
            + (", ".join(detect_transaction_patterns(tx)) or "не найдены")
        )


def get_example_tx_id(features: pd.DataFrame) -> str | None:
    """Возвращает пример tx_id для ручной проверки."""
    if features.empty or "tx_hash" not in features.columns:
        return None
    if "risk_score" in features.columns:
        return str(features.sort_values("risk_score", ascending=False).iloc[0]["tx_hash"])
    return str(features.iloc[0]["tx_hash"])


def local_graph_features(
    tx_id: str,
    features: pd.DataFrame,
    edges: pd.DataFrame,
    depth: int = 1,
) -> pd.DataFrame:
    """Возвращает признаки узлов ego-network."""
    graph = build_graph(edges)
    if tx_id not in graph:
        return features[features["tx_hash"] == tx_id]

    nodes = {tx_id}
    frontier = {tx_id}
    for _ in range(depth):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(graph.predecessors(node))
            next_frontier.update(graph.successors(node))
        nodes.update(next_frontier)
        frontier = next_frontier

    return features[features["tx_hash"].isin(nodes)]


def build_risk_explanation(tx: pd.Series) -> str:
    """Формирует короткое объяснение риска."""
    reasons = []
    if tx.get("output_count", 0) >= 10:
        reasons.append("имеет большое количество выходов")
    if tx.get("input_count", 0) >= 10:
        reasons.append("имеет большое количество входов")
    if tx.get("has_high_fee", 0) == 1:
        reasons.append("имеет аномально высокую комиссию")
    if tx.get("total_degree", 0) >= 5:
        reasons.append("находится в плотном транзакционном окружении")
    if tx.get("has_small_outputs", 0) == 1:
        reasons.append("содержит мелкие выходы")

    if not reasons:
        reasons.append("имеет умеренные структурные признаки риска")

    reason_text = ", ".join(reasons)
    return (
        "Транзакция получила текущий risk_score, так как "
        f"{reason_text}. Оценка является индикатором риска, а не доказательством."
    )


def detect_transaction_patterns(tx: pd.Series) -> list[str]:
    """Определяет простые suspicious-паттерны для транзакции."""
    patterns = []
    if tx.get("input_count", 0) >= 10 and tx.get("output_count", 0) >= 10:
        patterns.append("много входов -> много выходов")
    if tx.get("total_degree", 0) >= 5:
        patterns.append("плотное транзакционное окружение")
    if tx.get("output_count", 0) == 2 and tx.get("has_small_outputs", 0) == 1:
        patterns.append("peeling chain-like pattern")
    if tx.get("input_count", 0) >= 5 and tx.get("output_count", 0) >= 5:
        patterns.append("mixing-like pattern")
    return patterns


def render_patterns(features: pd.DataFrame, address_risks: pd.DataFrame) -> None:
    """Показывает карточки suspicious-паттернов."""
    st.subheader("Подозрительные паттерны")
    input_count = safe_column(features, "input_count")
    output_count = safe_column(features, "output_count")
    total_degree = safe_column(features, "total_degree")
    component_size = safe_column(features, "weak_component_size")
    small_outputs = safe_column(features, "has_small_outputs")
    address_tx_count = safe_column(address_risks, "tx_count")
    address_inputs = safe_column(address_risks, "input_tx_count")
    address_outputs = safe_column(address_risks, "output_tx_count")

    pattern_rows = [
        (
            "Много входов -> много выходов",
            int(((input_count >= 10) & (output_count >= 10)).sum()),
        ),
        (
            "Цепочка быстрых переводов",
            int((total_degree >= 4).sum()),
        ),
        (
            "Адрес с резким всплеском активности",
            int((address_tx_count >= 5).sum()),
        ),
        (
            "Плотная группа адресов",
            int((component_size >= 10).sum()),
        ),
        (
            "Адрес-посредник",
            int(((address_inputs > 0) & (address_outputs > 0)).sum()),
        ),
        (
            "Peeling chain-like pattern",
            int(((output_count == 2) & (small_outputs == 1)).sum()),
        ),
        (
            "Mixing-like pattern",
            int(((input_count >= 5) & (output_count >= 5)).sum()),
        ),
    ]

    cols = st.columns(3)
    for index, (title, count) in enumerate(pattern_rows):
        with cols[index % 3]:
            st.metric(title, count)

    st.caption(
        "Паттерны являются эвристиками мониторинга и не доказывают незаконную активность."
    )


def render_model_page() -> None:
    """Показывает описание модели и ограничений."""
    st.subheader("О модели")
    st.markdown(
        """
        Прототип использует rule-based risk_score от 0 до 1. Оценка строится по
        локальным, графовым и адресным признакам: количество входов и выходов,
        объем перевода, комиссия, степень узла, PageRank, размер компоненты,
        активность соседей и бинарные suspicious-флаги.

        Уровни риска: низкий до 0.4, средний от 0.4, высокий от 0.7.
        В MVP оценка является объяснимой эвристикой. На следующем этапе ее можно
        заменить ML-моделью, обученной на размеченном Elliptic-like датасете.

        Ограничения: система оценивает структурный риск и не утверждает факт
        принадлежности транзакции к незаконной активности. Для промышленного AML
        нужны внешние источники меток, санкционные списки и валидация качества.
        """
    )


def render_empty_state() -> None:
    """Показывает инструкцию, если данных еще нет."""
    st.warning("Готовая витрина транзакций не найдена.")
    st.code(
        "cd ..\\parser\n"
        "python scripts\\collect_blocks.py --max-blocks 20\n"
        "python scripts\\build_dataset.py --time-window-hours 24",
        language="powershell",
    )


def safe_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Возвращает числовую колонку или пустой Series."""
    if df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(0)


def filter_tx_rows(df: pd.DataFrame, tx_ids: set[str]) -> pd.DataFrame:
    """Фильтрует таблицу по tx_hash с защитой от пустых данных."""
    if df.empty or "tx_hash" not in df.columns:
        return pd.DataFrame()
    return df[df["tx_hash"].isin(tx_ids)]


def main() -> None:
    """Запускает Streamlit-приложение."""
    configure_compact_layout()
    st.title("Система мониторинга подозрительных Bitcoin-транзакций")
    st.caption("Динамический граф, risk_score, проверка tx_id")

    data = load_site_data()
    features = data["features"]
    if features.empty:
        render_empty_state()
        return

    filtered_features = filter_features(features)
    tx_ids = set(filtered_features["tx_hash"].astype(str))
    inputs = filter_tx_rows(data["inputs"], tx_ids)
    outputs = filter_tx_rows(data["outputs"], tx_ids)
    address_risks = build_address_risks(filtered_features, inputs, outputs)

    tabs = st.tabs(
        [
            "Обзор",
            "Динамический граф",
            "Проверка транзакции",
        ]
    )

    with tabs[0]:
        render_dashboard(filtered_features, data["edges"], address_risks)
    with tabs[1]:
        render_transaction_graph(filtered_features, data["edges"])
    with tabs[2]:
        render_transaction_check(
            filtered_features,
            data["edges"],
            inputs,
            outputs,
        )


if __name__ == "__main__":
    main()
