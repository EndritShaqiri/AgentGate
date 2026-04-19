from __future__ import annotations

import base64
import html
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from firewall_proxy.runtime_state import is_global_firewall_enabled, set_global_firewall_enabled


PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "firewall_logs.db"
ASSET_DIR = PROJECT_ROOT / "assets"
SLASHID_LOGO_DARK_PATH = ASSET_DIR / "slashid-logo-dark.jfif"

DECISION_COLORS = {
    "ALLOW": "#5df4c7",
    "WARN": "#ffd66b",
    "DENY": "#ff7a9d",
    "BYPASS": "#78d8ff",
}


@st.cache_data(ttl=2)
def load_logs() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()

    with sqlite3.connect(DB_PATH) as connection:
        frame = pd.read_sql_query(
            """
            SELECT
                id,
                created_at,
                agent_id,
                user_input,
                decision,
                trigger_layer,
                scope_score,
                prompt_injection_score,
                raw_scores,
                request_payload,
                response_payload
            FROM firewall_logs
            ORDER BY id DESC
            """,
            connection,
        )

    return frame


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-main: #070b12;
            --bg-deep: #0b1120;
            --bg-elevated: #101827;
            --bg-panel: rgba(13, 20, 34, 0.86);
            --bg-panel-strong: rgba(16, 24, 39, 0.96);
            --border-soft: rgba(124, 150, 186, 0.18);
            --text-main: #f7fbff;
            --text-soft: rgba(209, 221, 240, 0.76);
            --accent: #6fb6ff;
            --accent-bright: #83f5ff;
            --allow: #5df4c7;
            --warn: #ffd66b;
            --deny: #ff7a9d;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 14% 0%, rgba(111, 245, 255, 0.16), transparent 22%),
                radial-gradient(circle at 88% 12%, rgba(110, 182, 255, 0.17), transparent 24%),
                radial-gradient(circle at 50% 100%, rgba(93, 244, 199, 0.08), transparent 24%),
                linear-gradient(180deg, #05070b 0%, var(--bg-main) 44%, var(--bg-deep) 100%);
            color: var(--text-main);
        }

        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(8, 12, 19, 0.98), rgba(11, 17, 32, 0.98));
            border-right: 1px solid var(--border-soft);
        }

        .block-container {
            max-width: 1320px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, p, label, span, div {
            color: var(--text-main);
        }

        .hero-card,
        .metric-card,
        .panel-card {
            background:
                linear-gradient(180deg, rgba(16, 24, 39, 0.96), rgba(10, 17, 30, 0.92));
            border: 1px solid var(--border-soft);
            border-radius: 24px;
            box-shadow: 0 28px 72px rgba(0, 0, 0, 0.34);
            backdrop-filter: blur(18px);
        }

        .hero-card {
            padding: 1.55rem 1.7rem;
            margin-bottom: 1rem;
            position: relative;
            overflow: hidden;
        }

        .hero-card::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(130deg, rgba(111, 245, 255, 0.08), transparent 28%),
                linear-gradient(320deg, rgba(111, 182, 255, 0.08), transparent 24%);
            pointer-events: none;
        }

        .hero-card::after {
            content: "";
            position: absolute;
            inset: auto -70px -85px auto;
            width: 280px;
            height: 280px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(111, 245, 255, 0.18), transparent 66%);
        }

        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.74rem;
            color: var(--accent-bright);
            margin-bottom: 0.55rem;
            font-weight: 700;
        }

        .hero-title {
            font-size: 2.35rem;
            line-height: 1.05;
            font-weight: 780;
            margin: 0;
        }

        .hero-subtitle {
            color: var(--text-soft);
            font-size: 1rem;
            margin-top: 0.75rem;
            max-width: 760px;
        }

        .source-pill {
            display: inline-block;
            margin-top: 1rem;
            padding: 0.5rem 0.8rem;
            border-radius: 999px;
            background: rgba(111, 182, 255, 0.08);
            border: 1px solid rgba(111, 182, 255, 0.18);
            color: var(--text-soft);
            font-size: 0.85rem;
        }

        .brand-row {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
            position: relative;
            z-index: 1;
        }

        .brand-mark {
            width: 62px;
            height: 62px;
            border-radius: 18px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(180deg, rgba(111, 245, 255, 0.12), rgba(111, 182, 255, 0.08));
            border: 1px solid rgba(111, 245, 255, 0.16);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
        }

        .brand-mark img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 14px;
        }

        .brand-copy {
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
        }

        .brand-name {
            font-size: 1.02rem;
            font-weight: 760;
            letter-spacing: 0.01em;
        }

        .brand-tag {
            color: var(--text-soft);
            font-size: 0.82rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            margin-left: 0.75rem;
            padding: 0.48rem 0.82rem;
            border-radius: 999px;
            background: rgba(111, 245, 255, 0.08);
            border: 1px solid rgba(111, 245, 255, 0.15);
            font-size: 0.83rem;
            color: var(--text-main);
        }

        .metric-card {
            padding: 1rem 1.1rem;
            min-height: 132px;
            position: relative;
            overflow: hidden;
        }

        .metric-card::after {
            content: "";
            position: absolute;
            inset: auto auto -35px -20px;
            width: 120px;
            height: 120px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(111, 245, 255, 0.1), transparent 70%);
            pointer-events: none;
        }

        .metric-label {
            color: var(--text-soft);
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }

        .metric-value {
            font-size: 2rem;
            line-height: 1.05;
            font-weight: 760;
            margin: 0.45rem 0 0.35rem 0;
        }

        .metric-detail {
            color: var(--text-soft);
            font-size: 0.94rem;
        }

        .panel-card {
            padding: 1.15rem 1.2rem;
            margin-top: 0.5rem;
            background:
                linear-gradient(180deg, rgba(14, 21, 35, 0.95), rgba(10, 16, 28, 0.95));
        }

        .section-title {
            font-size: 1.06rem;
            font-weight: 720;
            margin-bottom: 0.9rem;
        }

        .badge-row {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: 0.65rem;
        }

        .decision-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.38rem 0.78rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            border: 1px solid rgba(255, 255, 255, 0.14);
            background: rgba(111, 182, 255, 0.08);
        }

        .score-card {
            padding: 0.9rem 1rem;
            border-radius: 18px;
            background: rgba(18, 28, 46, 0.72);
            border: 1px solid rgba(124, 150, 186, 0.16);
            margin-bottom: 0.75rem;
        }

        .score-label {
            color: var(--text-soft);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            font-weight: 700;
            margin-bottom: 0.42rem;
        }

        .score-value {
            font-size: 1.4rem;
            font-weight: 750;
            margin-bottom: 0.6rem;
        }

        .score-bar {
            height: 10px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.1);
            overflow: hidden;
        }

        .score-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(111, 182, 255, 0.86), rgba(111, 245, 255, 0.96));
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .summary-item {
            padding: 0.9rem 1rem;
            border-radius: 18px;
            background: rgba(18, 28, 46, 0.72);
            border: 1px solid rgba(124, 150, 186, 0.16);
        }

        .summary-key {
            color: var(--text-soft);
            font-size: 0.79rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
            font-weight: 700;
        }

        .summary-value {
            font-size: 0.97rem;
            line-height: 1.5;
            word-break: break-word;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border-soft);
            border-radius: 20px;
            overflow: hidden;
            background: rgba(11, 17, 30, 0.92);
        }

        div[data-testid="stMetric"] {
            background: transparent;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--border-soft);
            border-radius: 18px;
            overflow: hidden;
            background: rgba(11, 17, 30, 0.9);
        }

        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div {
            background: rgba(18, 28, 46, 0.82);
            border-color: rgba(124, 150, 186, 0.16);
        }

        [data-testid="stSidebar"] .sidebar-panel {
            padding: 1rem 1rem 1.1rem 1rem;
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(16, 24, 39, 0.96), rgba(10, 17, 30, 0.92));
            border: 1px solid rgba(124, 150, 186, 0.16);
            margin-bottom: 1rem;
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.24);
        }

        [data-testid="stSidebar"] .sidebar-title {
            font-size: 0.95rem;
            font-weight: 750;
            margin-bottom: 0.3rem;
        }

        [data-testid="stSidebar"] .sidebar-logo {
            width: 118px;
            border-radius: 16px;
            margin-bottom: 0.8rem;
            display: block;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.24);
        }

        [data-testid="stSidebar"] .sidebar-copy {
            color: var(--text-soft);
            font-size: 0.85rem;
            line-height: 1.5;
            margin-bottom: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_json_loads(raw_value: Any) -> dict[str, Any]:
    if isinstance(raw_value, dict):
        return raw_value
    if not raw_value:
        return {}
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return {"raw": raw_value}


def compute_top_trigger_layer(frame: pd.DataFrame) -> str:
    triggered = frame.loc[frame["trigger_layer"] != "NONE", "trigger_layer"]
    if triggered.empty:
        return "NONE"
    return str(triggered.mode().iloc[0])


def filter_logs(
    frame: pd.DataFrame,
    query: str,
    agent_filter: str,
    decision_filter: str,
) -> pd.DataFrame:
    filtered = frame.copy()

    if agent_filter != "All":
        filtered = filtered.loc[filtered["agent_id"] == agent_filter]

    if decision_filter != "All":
        filtered = filtered.loc[filtered["decision"] == decision_filter]

    if not query:
        return filtered

    mask = (
        filtered["user_input"].fillna("").str.contains(query, case=False, regex=False)
        | filtered["decision"].fillna("").str.contains(query, case=False, regex=False)
        | filtered["trigger_layer"].fillna("").str.contains(query, case=False, regex=False)
        | filtered["agent_id"].fillna("").str.contains(query, case=False, regex=False)
    )
    return filtered.loc[mask].copy()


def clamp_score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def metric_card(label: str, value: str, detail: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{accent};">{value}</div>
            <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_card(label: str, value: Any, accent: str) -> None:
    numeric = clamp_score(value)
    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-label">{label}</div>
            <div class="score-value" style="color:{accent};">{numeric:.4f}</div>
            <div class="score-bar">
                <div class="score-fill" style="width:{numeric * 100:.1f}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def summary_item(label: str, value: Any) -> str:
    rendered = "—" if value in (None, "", []) else str(value)
    safe_label = html.escape(label)
    safe_value = html.escape(rendered)
    return (
        '<div class="summary-item">'
        f'<div class="summary-key">{safe_label}</div>'
        f'<div class="summary-value">{safe_value}</div>'
        "</div>"
    )


def image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def decision_badge(decision: str) -> str:
    color = DECISION_COLORS.get(decision, "#8ab3ff")
    return (
        f'<span class="decision-badge" style="color:{color};">'
        f'<span style="width:8px;height:8px;border-radius:50%;background:{color};display:inline-block;"></span>'
        f"{decision}"
        "</span>"
    )


st.set_page_config(page_title="AI Firewall Dashboard", layout="wide")
inject_styles()
slashid_logo_src = image_data_uri(SLASHID_LOGO_DARK_PATH)

initial_firewall_enabled = is_global_firewall_enabled()

with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-panel">
            <img class="sidebar-logo" src="{slashid_logo_src}" alt="SlashID" />
            <div class="sidebar-title">SlashID AI Firewall</div>
            <div class="sidebar-copy">
                Toggle global protection live. When off, requests still go through the proxy but bypass
                firewall enforcement and forward upstream untouched.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    firewall_enabled = st.toggle(
        "Protect Requests Through Proxy",
        value=initial_firewall_enabled,
        help="Turn SlashID AI Firewall enforcement on or off globally.",
    )
    if firewall_enabled != initial_firewall_enabled:
        set_global_firewall_enabled(firewall_enabled)
        st.rerun()

    st.caption(
        "Protection is active." if firewall_enabled else "Protection is bypassed globally."
    )

st.markdown(
    f"""
    <div class="hero-card">
        <div class="brand-row">
            <div class="brand-mark">
                <img src="{slashid_logo_src}" alt="SlashID" />
            </div>
            <div class="brand-copy">
                <div class="brand-name">SlashID AI Firewall</div>
                <div class="brand-tag">Identity Protection Operations</div>
            </div>
        </div>
        <div class="hero-kicker">Operational Intelligence</div>
        <h1 class="hero-title">AI Firewall Dashboard</h1>
        <div class="hero-subtitle">
            Monitor identity-protection decisions, review request risk, and control live enforcement
            from a security-first dashboard inspired by SlashID’s product style.
        </div>
        <div>
            <span class="source-pill">SQLite source: {DB_PATH}</span>
            <span class="status-pill">
                <span style="width:10px;height:10px;border-radius:50%;background:{'#5df4c7' if firewall_enabled else '#78d8ff'};display:inline-block;"></span>
                {"Protection ON" if firewall_enabled else "Protection OFF"}
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

logs = load_logs()
if logs.empty:
    st.info("No firewall logs found yet. Start the FastAPI app and send a few requests first.")
    st.stop()

logs["created_at_raw"] = pd.to_datetime(logs["created_at"], utc=True, errors="coerce")
logs["created_at"] = logs["created_at_raw"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")

agent_options = ["All", *sorted(logs["agent_id"].dropna().astype(str).unique().tolist())]
decision_options = ["All", "ALLOW", "BYPASS", "WARN", "DENY"]

control_col_1, control_col_2, control_col_3 = st.columns([1.25, 1, 1])
with control_col_1:
    search_query = st.text_input(
        "Search logs",
        placeholder="Search by prompt, decision, layer, or agent",
    )
with control_col_2:
    agent_filter = st.selectbox("Agent", agent_options, index=0)
with control_col_3:
    decision_filter = st.selectbox("Decision", decision_options, index=0)

filtered_logs = filter_logs(logs, search_query, agent_filter, decision_filter)

if filtered_logs.empty:
    st.warning("No logs match the current filters.")
    st.stop()

total_requests = int(len(filtered_logs))
blocked_count = int(filtered_logs["decision"].isin(["WARN", "DENY"]).sum())
allow_count = int((filtered_logs["decision"] == "ALLOW").sum())
bypass_count = int((filtered_logs["decision"] == "BYPASS").sum())
block_rate = (blocked_count / total_requests) * 100 if total_requests else 0.0
top_layer = compute_top_trigger_layer(filtered_logs)
avg_scope = float(filtered_logs["scope_score"].fillna(0).mean())
avg_injection = float(filtered_logs["prompt_injection_score"].fillna(0).mean())

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
with metric_col_1:
    metric_card("Requests In View", f"{total_requests:,}", "Filtered event volume", "#f6f8ff")
with metric_col_2:
    metric_card("Allowed", f"{allow_count:,}", "Requests that passed upstream", DECISION_COLORS["ALLOW"])
with metric_col_3:
    metric_card(
        "Protection Mix",
        f"{block_rate:.1f}%",
        f"WARN or DENY; {bypass_count:,} bypassed",
        DECISION_COLORS["WARN"],
    )
with metric_col_4:
    metric_card("Top Trigger Layer", top_layer, "Most frequent non-pass layer", "#8ab3ff")

analytics_col_1, analytics_col_2 = st.columns([1.2, 0.8])
with analytics_col_1:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Decision Flow</div>', unsafe_allow_html=True)
    decision_counts = (
        filtered_logs["decision"]
        .value_counts()
        .reindex(["ALLOW", "BYPASS", "WARN", "DENY"], fill_value=0)
        .rename_axis("decision")
        .reset_index(name="count")
        .set_index("decision")
    )
    st.bar_chart(decision_counts, color="#8ab3ff", use_container_width=True)
    st.markdown(
        '<div class="badge-row">'
        + decision_badge("ALLOW")
        + decision_badge("BYPASS")
        + decision_badge("WARN")
        + decision_badge("DENY")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with analytics_col_2:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Signal Snapshot</div>', unsafe_allow_html=True)
    score_card("Average Scope Score", avg_scope, "#8ab3ff")
    score_card("Average Injection Score", avg_injection, DECISION_COLORS["WARN"])
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Event Log</div>', unsafe_allow_html=True)

table_columns = [
    "id",
    "created_at",
    "agent_id",
    "decision",
    "trigger_layer",
    "scope_score",
    "prompt_injection_score",
    "user_input",
]

table_event = st.dataframe(
    filtered_logs[table_columns],
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "scope_score": st.column_config.NumberColumn(format="%.4f"),
        "prompt_injection_score": st.column_config.NumberColumn(format="%.4f"),
        "user_input": st.column_config.TextColumn(width="large"),
    },
)
st.markdown("</div>", unsafe_allow_html=True)

selected_row_index = None
selected_rows = getattr(getattr(table_event, "selection", None), "rows", [])
if selected_rows:
    selected_row_index = selected_rows[0]
elif not filtered_logs.empty:
    selected_row_index = 0

st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Deep Dive</div>', unsafe_allow_html=True)

if selected_row_index is None:
    st.info("Select a row from the table to inspect the raw scope and classifier scores.")
    st.stop()

selected_row = filtered_logs.iloc[selected_row_index]
raw_scores = safe_json_loads(selected_row["raw_scores"])
request_payload = safe_json_loads(selected_row["request_payload"])
response_payload = safe_json_loads(selected_row["response_payload"])

scope_scores = raw_scores.get("scope", {})
injection_scores = raw_scores.get("prompt_injection", {})

overview_col, scope_col, injection_col = st.columns([0.9, 1, 1])
with overview_col:
    st.markdown(
        '<div class="summary-grid">'
        + summary_item("Request ID", int(selected_row["id"]))
        + summary_item("Agent", selected_row["agent_id"])
        + summary_item("Decision", selected_row["decision"])
        + summary_item("Trigger Layer", selected_row["trigger_layer"])
        + summary_item("Created", selected_row["created_at"])
        + summary_item("Prompt", selected_row["user_input"])
        + "</div>",
        unsafe_allow_html=True,
    )

with scope_col:
    st.markdown("**Scope Check**")
    score_card("Combined Scope Score", scope_scores.get("scope_score", 0.0), "#8ab3ff")
    score_card("Description Similarity", scope_scores.get("description_similarity", 0.0), "#77b0ff")
    score_card("Allowed Max Similarity", scope_scores.get("allowed_max_similarity", 0.0), DECISION_COLORS["ALLOW"])
    score_card("Denied Max Similarity", scope_scores.get("denied_max_similarity", 0.0), DECISION_COLORS["DENY"])
    st.caption("Top allowed example")
    st.code(scope_scores.get("top_allowed_example", ""), language="text")
    st.caption("Top denied example")
    st.code(scope_scores.get("top_denied_example", ""), language="text")

with injection_col:
    st.markdown("**Prompt Injection Check**")
    score_card(
        "Malicious Probability",
        injection_scores.get("malicious_probability", 0.0),
        DECISION_COLORS["DENY"],
    )
    score_card(
        "Benign Probability",
        injection_scores.get("benign_probability", 0.0),
        DECISION_COLORS["ALLOW"],
    )
    st.caption("Classifier context")
    st.code(injection_scores.get("recent_context", ""), language="text")

st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Raw Scores JSON", expanded=False):
    st.json(raw_scores)

with st.expander("Request Payload", expanded=False):
    st.json(request_payload)

with st.expander("Response Payload", expanded=False):
    st.json(response_payload)
