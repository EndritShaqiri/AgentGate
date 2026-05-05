from __future__ import annotations

import atexit
import base64
import html
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from firewall_proxy.config import load_config
from firewall_proxy.policy import (
    compute_policy_source_hash,
    ensure_policy_tables,
    format_policy_json,
    generate_policy_document,
    load_active_policy_document,
    load_latest_policy_document,
    load_recent_pbac_decisions,
    save_active_policy_document,
    validate_policy_document,
)
from firewall_proxy.runtime_config_store import (
    clear_runtime_agent_setup,
    format_tool_registry,
    load_runtime_agent_setup,
    parse_tool_registry,
    split_examples,
    upsert_runtime_agent_setup,
)
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
            SELECT *
            FROM firewall_logs
            ORDER BY id DESC
            """,
            connection,
        )

    defaults = {
        "pg2_main": 0.0,
        "scope_main": frame.get("scope_score", 0.0),
        "pii_main": 0.0,
        "doc_pg2_max": 0.0,
        "doc_scope_min": 1.0,
        "doc_flagged_ratio": 0.0,
        "lg4_unsafe": 0,
        "lg4_code_abuse": 0,
        "final_risk": 0.0,
        "attachment_summary": "{}",
        "decision_reasons": "[]",
        "chunk_summaries": "[]",
        "model_versions": "{}",
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default

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

        .section-kicker {
            margin: 1.4rem 0 0.45rem 0;
            color: var(--accent-bright);
            font-size: 0.74rem;
            font-weight: 780;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }

        .section-heading {
            font-size: 1.42rem;
            line-height: 1.15;
            font-weight: 780;
            margin: 0 0 0.35rem 0;
        }

        .section-copy {
            color: var(--text-soft);
            font-size: 0.94rem;
            line-height: 1.55;
            margin-bottom: 0.85rem;
            max-width: 880px;
        }

        .flow-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.8rem 0 1rem 0;
        }

        .flow-step {
            padding: 0.82rem 0.95rem;
            border-radius: 16px;
            background: rgba(18, 28, 46, 0.68);
            border: 1px solid rgba(124, 150, 186, 0.16);
        }

        .flow-step-key {
            color: var(--accent-bright);
            font-size: 0.72rem;
            font-weight: 780;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 0.26rem;
        }

        .flow-step-value {
            font-size: 0.93rem;
            color: var(--text-main);
            line-height: 1.4;
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


def compact_tool_json(raw_value: Any) -> str:
    if not raw_value:
        return ""
    try:
        loaded = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
    except json.JSONDecodeError:
        return str(raw_value)

    if isinstance(loaded, list):
        names: list[str] = []
        for item in loaded:
            if isinstance(item, dict):
                name = item.get("name")
                usage = item.get("usage")
                if name and usage:
                    names.append(f"{name} ({usage})")
                elif name:
                    names.append(str(name))
            elif item not in (None, ""):
                names.append(str(item))
        return ", ".join(names)

    if isinstance(loaded, dict):
        return ", ".join(f"{key}: {value}" for key, value in loaded.items())

    return str(loaded)


def generate_and_store_pbac_policy_draft(setup: Any) -> Any:
    config = load_config()
    outcome = generate_policy_document(setup, config.policy_generation)
    st.session_state["pbac_policy_draft"] = format_policy_json(outcome.policy)
    st.session_state["pbac_policy_draft_hash"] = outcome.policy["source_hash"]
    st.session_state["pbac_policy_editing"] = False
    st.session_state["pbac_policy_collapsed"] = False
    st.session_state["pbac_policy_generation_status"] = {
        "mode": outcome.mode,
        "used_llm": outcome.used_llm,
        "fallback_used": outcome.fallback_used,
        "error": outcome.error,
    }
    st.session_state.pop("pbac_policy_editor_text", None)
    return outcome


def render_section_header(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-kicker">{html.escape(kicker)}</div>
        <div class="section-heading">{html.escape(title)}</div>
        <div class="section-copy">{html.escape(copy)}</div>
        """,
        unsafe_allow_html=True,
    )


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


def load_dashboard_runtime_setup_values() -> dict[str, Any]:
    setup = load_runtime_agent_setup(DB_PATH)
    if setup is not None:
        return {
            "configured": True,
            "source": "Dashboard SQLite runtime setup",
            "agent_id": setup.agent_id,
            "description": setup.description,
            "allowed_examples": "\n".join(setup.allowed_examples),
            "denied_examples": "\n".join(setup.denied_examples),
            "tool_registry": format_tool_registry(setup.tool_registry),
            "tool_count": len(setup.tool_registry),
            "use_local_mock": setup.use_local_mock,
            "base_url": setup.base_url or "",
            "timeout_seconds": setup.timeout_seconds,
            "default_model": setup.default_model,
            "updated_at": setup.updated_at,
        }

    fallback = load_config()
    return {
        "configured": False,
        "source": "Not configured yet",
        "agent_id": "",
        "description": "",
        "allowed_examples": "",
        "denied_examples": "",
        "tool_registry": "",
        "tool_count": 0,
        "use_local_mock": fallback.upstream.use_local_mock,
        "base_url": fallback.upstream.base_url or "",
        "timeout_seconds": fallback.upstream.timeout_seconds,
        "default_model": fallback.upstream.default_model,
        "updated_at": "This local session starts empty.",
    }


def render_runtime_agent_setup_editor() -> None:
    values = load_dashboard_runtime_setup_values()
    mode_label = "Local mock" if values["use_local_mock"] else "Remote upstream"
    base_url_label = values["base_url"] or "Internal /mock routes"

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Runtime Agent Setup</div>', unsafe_allow_html=True)
    st.caption(
        "Fill this once per local demo run. The dashboard saves one SQLite runtime row containing "
        "both the scope profile and the upstream target, and FastAPI reloads that same row before requests."
    )

    if not values["configured"]:
        st.warning(
            "No runtime setup is active yet. Save this form before sending protected requests."
        )

    status_col_1, status_col_2 = st.columns([0.58, 0.42])
    with status_col_1:
        st.markdown(
            '<div class="summary-grid">'
            + summary_item("Agent ID", values["agent_id"] or "Not set")
            + summary_item("Forwarding Mode", mode_label)
            + summary_item("Upstream Base URL", base_url_label)
            + summary_item("Default Model", values["default_model"])
            + summary_item("Registered Tools", values["tool_count"])
            + "</div>",
            unsafe_allow_html=True,
        )
    with status_col_2:
        st.markdown(
            '<div class="summary-grid">'
            + summary_item("Settings Source", values["source"])
            + summary_item("Updated", values["updated_at"])
            + summary_item("Timeout", f'{float(values["timeout_seconds"]):.1f}s')
            + "</div>",
            unsafe_allow_html=True,
        )
        st.info(
            "If local mock is ON, allowed requests intentionally return a mock reply. "
            "Turn it OFF and enter your real upstream URL to get real chatbot answers."
        )

    with st.form("runtime_agent_setup_form", clear_on_submit=False):
        scope_col, upstream_col = st.columns([0.54, 0.46])
        with scope_col:
            agent_id = st.text_input(
                "Agent ID",
                value=str(values["agent_id"]),
                placeholder="star",
                help="Requests can select this with `x-agent-id`, top-level `agent_id`, or metadata.",
            )
            description = st.text_area(
                "Description",
                value=str(values["description"]),
                height=150,
                placeholder="Describe what this protected agent is allowed to do.",
            )
            allowed_raw = st.text_area(
                "Allowed examples",
                value=str(values["allowed_examples"]),
                height=150,
                placeholder="One legitimate user request per line.",
            )
            denied_raw = st.text_area(
                "Denied examples",
                value=str(values["denied_examples"]),
                height=150,
                placeholder="One denied, unsafe, or out-of-scope request per line.",
            )
            tool_registry_raw = st.text_area(
                "Allowed tool names",
                value=str(values["tool_registry"]),
                height=125,
                placeholder=(
                    "search_massachusetts_law\n"
                    "summarize_uploaded_pdf\n"
                    "send_email\n"
                    "tool_A | external_action | Send summaries to the configured recipient"
                ),
                help=(
                    "Exact names are enough when they describe the function. "
                    "For generic names like tool_A, add: name | category | short purpose | risk."
                ),
            )
            st.warning(
                "If the tool name clearly describes its function, like `send_email`, no description is needed. "
                "If it is generic, like `tool_A`, add one short sentence explaining what it does.",
                icon="⚠️",
            )

        with upstream_col:
            use_local_mock = st.toggle(
                "Use local mock upstream",
                value=bool(values["use_local_mock"]),
                help="When enabled, allowed requests go to the built-in /mock endpoints.",
            )
            base_url = st.text_input(
                "Remote upstream base URL",
                value=str(values["base_url"]),
                placeholder="https://api.openai.com/v1 or http://127.0.0.1:1234/v1",
                help="Required when local mock mode is off. Do not point this at the dashboard or firewall.",
            )
            timeout_seconds = st.number_input(
                "Timeout seconds",
                min_value=1.0,
                max_value=300.0,
                value=float(values["timeout_seconds"]),
                step=1.0,
            )
            default_model = st.text_input(
                "Default model",
                value=str(values["default_model"]),
                placeholder="gpt-4.1-mini, local-demo-model, or mock-guarded-model",
            )
            st.caption(
                "Forwarded requests keep the original path, query string, body, and auth headers."
            )

        save_col, clear_col, hint_col = st.columns([0.18, 0.18, 0.64])
        with save_col:
            save_clicked = st.form_submit_button("Save Setup", type="primary", use_container_width=True)
        with clear_col:
            clear_clicked = st.form_submit_button("Clear Setup", use_container_width=True)
        with hint_col:
            st.caption(
                "This runtime setup is intentionally ephemeral: normal FastAPI or dashboard shutdown clears it."
            )

    if save_clicked:
        try:
            saved = upsert_runtime_agent_setup(
                DB_PATH,
                agent_id=agent_id,
                description=description,
                allowed_examples=split_examples(allowed_raw),
                denied_examples=split_examples(denied_raw),
                tool_registry=parse_tool_registry(tool_registry_raw),
                use_local_mock=use_local_mock,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                default_model=default_model,
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            mode = "local mock" if saved.use_local_mock else "remote upstream"
            try:
                outcome = generate_and_store_pbac_policy_draft(saved)
            except Exception as exc:
                st.error(f"Saved setup, but PBAC policy generation failed: {exc}")
                return
            st.success(
                f"Saved `{saved.agent_id}` with {mode}. Review and accept the generated PBAC policy below before protected requests continue."
            )
            if outcome.used_llm:
                st.caption("PBAC draft generated by the configured LLM policy compiler and validated locally.")
            elif outcome.fallback_used:
                st.warning(f"LLM policy generation failed; deterministic fallback draft was generated. {outcome.error}")

    if clear_clicked:
        clear_runtime_agent_setup(DB_PATH)
        for key in (
            "pbac_policy_draft",
            "pbac_policy_draft_hash",
            "pbac_policy_editing",
            "pbac_policy_collapsed",
            "pbac_policy_generation_status",
        ):
            st.session_state.pop(key, None)
        st.warning("Cleared the runtime setup. Protected requests will pause until you save a new one.")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_pbac_policy_review() -> None:
    ensure_policy_tables(DB_PATH)
    setup = load_runtime_agent_setup(DB_PATH)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">PBAC Policy Review</div>', unsafe_allow_html=True)

    if setup is None:
        st.info("Save a runtime agent setup to generate a PBAC policy draft.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    source_hash = compute_policy_source_hash(setup)
    active_policy = load_active_policy_document(DB_PATH, setup.agent_id, source_hash)
    latest_policy = load_latest_policy_document(DB_PATH, setup.agent_id)

    status_col, action_col = st.columns([0.62, 0.38])
    with status_col:
        if active_policy is not None:
            policy_id, _policy = active_policy
            st.success(f"Accepted PBAC policy #{policy_id} is active for `{setup.agent_id}`.")
            st.caption("Runtime PBAC uses this policy as a binary structural gate before L0-L4 scoring.")
        elif latest_policy and latest_policy.get("source_hash") != source_hash:
            st.warning("The saved setup changed after the last accepted policy. Generate and accept a fresh policy.")
        else:
            st.warning("No accepted PBAC policy is active for the current setup. Protected requests with registered tools are denied until acceptance.")

        st.markdown(
            '<div class="summary-grid">'
            + summary_item("Agent", setup.agent_id)
            + summary_item("Source Hash", source_hash[:16])
            + summary_item("Registered Tools", len(setup.tool_registry))
            + summary_item("Active Policy", f"#{active_policy[0]}" if active_policy else "Not accepted")
            + "</div>",
            unsafe_allow_html=True,
        )

    with action_col:
        if st.button("Generate LLM Policy Draft", type="primary", use_container_width=True):
            try:
                generate_and_store_pbac_policy_draft(setup)
            except Exception as exc:
                st.error(f"Policy generation failed: {exc}")
                return
            st.rerun()
        if active_policy is not None and st.button("Load Active Policy", use_container_width=True):
            st.session_state["pbac_policy_draft"] = format_policy_json(active_policy[1])
            st.session_state["pbac_policy_draft_hash"] = source_hash
            st.session_state["pbac_policy_editing"] = False
            st.session_state["pbac_policy_collapsed"] = False
            st.session_state.pop("pbac_policy_editor_text", None)
            st.rerun()

    draft_text = st.session_state.get("pbac_policy_draft")
    draft_hash = st.session_state.get("pbac_policy_draft_hash")
    if not draft_text or draft_hash != source_hash:
        if active_policy is not None:
            draft_text = format_policy_json(active_policy[1])
            st.session_state["pbac_policy_collapsed"] = True
        else:
            try:
                outcome = generate_and_store_pbac_policy_draft(setup)
            except Exception as exc:
                st.error(f"Policy generation failed: {exc}")
                st.markdown("</div>", unsafe_allow_html=True)
                return
            draft_text = format_policy_json(outcome.policy)
            st.session_state["pbac_policy_collapsed"] = False
        st.session_state["pbac_policy_draft"] = draft_text
        st.session_state["pbac_policy_draft_hash"] = source_hash
        st.session_state["pbac_policy_editing"] = False
        st.session_state.pop("pbac_policy_editor_text", None)

    editing = bool(st.session_state.get("pbac_policy_editing", False))
    st.caption(
        "Review the generated policy. `ACCEPT` persists it as the active PBAC policy; `EDIT` lets you adjust the JSON before saving."
    )
    generation_status = st.session_state.get("pbac_policy_generation_status")
    if isinstance(generation_status, dict):
        if generation_status.get("used_llm"):
            st.caption("Draft source: LLM policy compiler, locally normalized and validated.")
        elif generation_status.get("fallback_used"):
            st.warning(
                "Draft source: deterministic fallback after LLM generation failed. "
                f"{generation_status.get('error') or ''}"
            )

    policy_expanded = editing or not bool(st.session_state.get("pbac_policy_collapsed", False))
    with st.expander("Policy JSON", expanded=policy_expanded):
        if editing:
            edited_text = st.text_area(
                "PBAC policy JSON",
                value=str(draft_text),
                height=460,
                key="pbac_policy_editor_text",
            )
            button_col_1, button_col_2 = st.columns([0.22, 0.78])
            with button_col_1:
                accept_clicked = st.button("ACCEPT", type="primary", use_container_width=True, key="accept_edited_pbac")
            with button_col_2:
                if st.button("Cancel Edit", use_container_width=True):
                    st.session_state["pbac_policy_editing"] = False
                    st.session_state["pbac_policy_collapsed"] = False
                    st.session_state.pop("pbac_policy_editor_text", None)
                    st.rerun()

            if accept_clicked:
                _accept_policy_text(edited_text, setup)
        else:
            try:
                st.json(json.loads(str(draft_text)))
            except json.JSONDecodeError:
                st.code(str(draft_text), language="json")

            button_col_1, button_col_2, button_col_3 = st.columns([0.18, 0.18, 0.64])
            with button_col_1:
                if st.button("ACCEPT", type="primary", use_container_width=True, key="accept_generated_pbac"):
                    _accept_policy_text(str(draft_text), setup)
            with button_col_2:
                if st.button("EDIT", use_container_width=True, key="edit_generated_pbac"):
                    st.session_state["pbac_policy_editing"] = True
                    st.session_state["pbac_policy_collapsed"] = False
                    st.rerun()
            with button_col_3:
                st.caption("PBAC policies are stored separately from firewall scores and never feed risk fusion.")

    st.markdown("</div>", unsafe_allow_html=True)


def _accept_policy_text(raw_text: str, setup: Any) -> None:
    try:
        policy = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        st.error(f"Policy JSON is invalid: {exc}")
        return
    if not isinstance(policy, dict):
        st.error("Policy must be a JSON object.")
        return

    validation_errors = validate_policy_document(policy, setup)
    if validation_errors:
        st.error("Policy validation failed.")
        for error in validation_errors:
            st.write(f"- {error}")
        return

    try:
        policy_id = save_active_policy_document(DB_PATH, policy, setup)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.session_state["pbac_policy_draft"] = format_policy_json(policy)
    st.session_state["pbac_policy_draft_hash"] = compute_policy_source_hash(setup)
    st.session_state["pbac_policy_editing"] = False
    st.session_state["pbac_policy_collapsed"] = True
    st.session_state.pop("pbac_policy_editor_text", None)
    st.success(f"Accepted PBAC policy #{policy_id}. Protected requests can continue through the PBAC plane.")
    st.rerun()


def render_enforcement_flow() -> None:
    st.markdown(
        """
        <div class="flow-strip">
            <div class="flow-step">
                <div class="flow-step-key">Plane 1</div>
                <div class="flow-step-value">PBAC structural policy: active policy, inferred tool intent, exact tool names.</div>
            </div>
            <div class="flow-step">
                <div class="flow-step-key">Plane 2</div>
                <div class="flow-step-value">L0-L4 content firewall: scope, prompt injection, PII, attachments, multimodal/tool misuse.</div>
            </div>
            <div class="flow-step">
                <div class="flow-step-key">Gateway</div>
                <div class="flow-step-value">Tool execution goes through /agentgate/tools/execute for a second PBAC check.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pbac_operations() -> None:
    render_section_header(
        "Structural Plane",
        "PBAC Detection And Prevention",
        "Binary policy decisions for requested or inferred tool use. These records are separate from L0-L4 scores.",
    )

    pbac_rows = load_recent_pbac_decisions(DB_PATH, max_rows=40)
    if not pbac_rows:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.info("No PBAC decisions recorded yet. Save and accept a policy, then send protected traffic.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    pbac_frame = pd.DataFrame(pbac_rows)
    pbac_frame["created_at_raw"] = pd.to_datetime(pbac_frame["created_at"], utc=True, errors="coerce")
    pbac_frame["created_at"] = pbac_frame["created_at_raw"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    for column in ("requested_tools", "denied_tools", "required_tools"):
        if column in pbac_frame.columns:
            pbac_frame[column] = pbac_frame[column].apply(compact_tool_json)
        else:
            pbac_frame[column] = ""

    pbac_total = int(len(pbac_frame))
    pbac_allowed = int((pbac_frame["decision"] == "ALLOW").sum())
    pbac_denied = int((pbac_frame["decision"] == "DENY").sum())
    pbac_deny_rate = (pbac_denied / pbac_total) * 100 if pbac_total else 0.0
    top_pbac_trigger = (
        str(pbac_frame.loc[pbac_frame["trigger"] != "PBAC_ALLOW", "trigger"].mode().iloc[0])
        if not pbac_frame.loc[pbac_frame["trigger"] != "PBAC_ALLOW", "trigger"].empty
        else "PBAC_ALLOW"
    )

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    with metric_col_1:
        metric_card("PBAC Events", f"{pbac_total:,}", "Recent structural decisions", "#f6f8ff")
    with metric_col_2:
        metric_card("PBAC Allowed", f"{pbac_allowed:,}", "Passed to content firewall", DECISION_COLORS["ALLOW"])
    with metric_col_3:
        metric_card("PBAC Denied", f"{pbac_denied:,}", f"{pbac_deny_rate:.1f}% structural blocks", DECISION_COLORS["DENY"])
    with metric_col_4:
        metric_card("Top PBAC Trigger", top_pbac_trigger, "Most common policy outcome", "#8ab3ff")

    pbac_chart_col, pbac_table_col = st.columns([0.42, 0.58])
    with pbac_chart_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">PBAC Outcomes</div>', unsafe_allow_html=True)
        pbac_counts = (
            pbac_frame["decision"]
            .value_counts()
            .reindex(["ALLOW", "DENY"], fill_value=0)
            .rename_axis("decision")
            .reset_index(name="count")
            .set_index("decision")
        )
        st.bar_chart(pbac_counts, color="#8ab3ff", use_container_width=True)
        st.markdown(
            '<div class="badge-row">'
            + decision_badge("ALLOW")
            + decision_badge("DENY")
            + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with pbac_table_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recent Structural Decisions</div>', unsafe_allow_html=True)
        pbac_columns = [
            "id",
            "created_at",
            "agent_id",
            "policy_id",
            "decision",
            "trigger",
            "reason",
            "requested_tools",
            "denied_tools",
            "required_tools",
        ]
        st.dataframe(
            pbac_frame[pbac_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "reason": st.column_config.TextColumn(width="large"),
                "requested_tools": st.column_config.TextColumn(width="medium"),
                "denied_tools": st.column_config.TextColumn(width="medium"),
                "required_tools": st.column_config.TextColumn(width="medium"),
            },
        )
        st.markdown("</div>", unsafe_allow_html=True)


st.set_page_config(page_title="AI Firewall Dashboard", layout="wide")


@st.cache_resource
def register_runtime_cleanup(db_path_value: str) -> bool:
    atexit.register(clear_runtime_agent_setup, Path(db_path_value))
    return True


register_runtime_cleanup(str(DB_PATH))
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
            from a security-first dashboard.
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

render_runtime_agent_setup_editor()
render_pbac_policy_review()
render_enforcement_flow()
render_pbac_operations()

render_section_header(
    "Content Plane",
    "L0-L4 Detection And Prevention",
    "Layered content decisions for protected requests after PBAC has allowed the structural policy check.",
)

logs = load_logs()
if logs.empty:
    st.info("No firewall logs found yet. Start the FastAPI app and send a few requests first.")
    st.stop()

logs["created_at_raw"] = pd.to_datetime(logs["created_at"], utc=True, errors="coerce")
logs["created_at"] = logs["created_at_raw"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")

agent_options = ["All", *sorted(logs["agent_id"].dropna().astype(str).unique().tolist())]
decision_options = ["All", "ALLOW", "BYPASS", "WARN", "DENY"]

st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">L0-L4 Filters</div>', unsafe_allow_html=True)
control_col_1, control_col_2, control_col_3 = st.columns([1.25, 1, 1])
with control_col_1:
    search_query = st.text_input(
        "Search content logs",
        placeholder="Search by prompt, decision, layer, or agent",
    )
with control_col_2:
    agent_filter = st.selectbox("Content Agent", agent_options, index=0)
with control_col_3:
    decision_filter = st.selectbox("Content Decision", decision_options, index=0)
st.markdown("</div>", unsafe_allow_html=True)

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
avg_injection = float(filtered_logs["pg2_main"].fillna(filtered_logs["prompt_injection_score"]).fillna(0).mean())
avg_risk = float(filtered_logs["final_risk"].fillna(0).mean())

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
with metric_col_1:
    metric_card("L0-L4 Events", f"{total_requests:,}", "Filtered content decisions", "#f6f8ff")
with metric_col_2:
    metric_card("Content Allowed", f"{allow_count:,}", "Forwarded upstream", DECISION_COLORS["ALLOW"])
with metric_col_3:
    metric_card(
        "Content Blocks",
        f"{block_rate:.1f}%",
        f"WARN or DENY; {bypass_count:,} bypassed",
        DECISION_COLORS["WARN"],
    )
with metric_col_4:
    metric_card("Top L0-L4 Trigger", top_layer, "Most frequent non-pass layer", "#8ab3ff")

analytics_col_1, analytics_col_2 = st.columns([1.2, 0.8])
with analytics_col_1:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">L0-L4 Outcome Flow</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-title">Layer Signal Snapshot</div>', unsafe_allow_html=True)
    score_card("Average Scope Score", avg_scope, "#8ab3ff")
    score_card("Average PG2 Score", avg_injection, DECISION_COLORS["WARN"])
    score_card("Average Final Risk", avg_risk, DECISION_COLORS["DENY"])
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">L0-L4 Event Log</div>', unsafe_allow_html=True)

table_columns = [
    "id",
    "created_at",
    "agent_id",
    "decision",
    "trigger_layer",
    "scope_main",
    "pg2_main",
    "pii_main",
    "final_risk",
    "user_input",
]

table_event = st.dataframe(
    filtered_logs[table_columns],
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "scope_main": st.column_config.NumberColumn(format="%.4f"),
        "pg2_main": st.column_config.NumberColumn(format="%.4f"),
        "pii_main": st.column_config.NumberColumn(format="%.4f"),
        "final_risk": st.column_config.NumberColumn(format="%.4f"),
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
st.markdown('<div class="section-title">L0-L4 Deep Dive</div>', unsafe_allow_html=True)

if selected_row_index is None:
    st.info("Select a row from the table to inspect the raw scope and classifier scores.")
    st.stop()

selected_row = filtered_logs.iloc[selected_row_index]
raw_scores = safe_json_loads(selected_row["raw_scores"])
request_payload = safe_json_loads(selected_row["request_payload"])
response_payload = safe_json_loads(selected_row["response_payload"])

scope_scores = raw_scores.get("scope", {})
injection_scores = raw_scores.get("prompt_injection", {})
pii_scores = raw_scores.get("pii", {})
attachment_scores = raw_scores.get("attachments", {})
llama_guard_scores = raw_scores.get("llama_guard", {})
risk_scores = raw_scores.get("risk", {})
decision_reasons = safe_json_loads(selected_row.get("decision_reasons", "[]"))
chunk_summaries = safe_json_loads(selected_row.get("chunk_summaries", "[]"))
model_versions = safe_json_loads(selected_row.get("model_versions", "{}"))

overview_col, layer_col, scope_col, injection_col = st.columns([0.9, 1, 1, 1])
with overview_col:
    st.markdown(
        '<div class="summary-grid">'
        + summary_item("Request ID", int(selected_row["id"]))
        + summary_item("Agent", selected_row["agent_id"])
        + summary_item("Decision", selected_row["decision"])
        + summary_item("Trigger Layer", selected_row["trigger_layer"])
        + summary_item("Created", selected_row["created_at"])
        + summary_item("Final Risk", f'{clamp_score(selected_row.get("final_risk", 0.0)):.4f}')
        + summary_item("Prompt", selected_row["user_input"])
        + "</div>",
        unsafe_allow_html=True,
    )

with layer_col:
    st.markdown("**Layer Results**")
    score_card("L0 Scope Main", risk_scores.get("scope_main", selected_row.get("scope_main", 0.0)), "#8ab3ff")
    score_card("L1 PG2 Main", risk_scores.get("pg2_main", selected_row.get("pg2_main", 0.0)), DECISION_COLORS["WARN"])
    score_card("L2 PII Main", risk_scores.get("pii_main", selected_row.get("pii_main", 0.0)), "#f6f8ff")
    score_card("Final Risk", risk_scores.get("final_risk", selected_row.get("final_risk", 0.0)), DECISION_COLORS["DENY"])
    st.caption("Layers executed")
    st.code(", ".join(risk_scores.get("layers_executed", [])), language="text")

with scope_col:
    st.markdown("**Scope Check**")
    score_card("Combined Scope Score", scope_scores.get("scope_score", 0.0), "#8ab3ff")
    score_card("Raw Scope Formula", scope_scores.get("raw_scope_score", 0.0), "#77b0ff")
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
    score_card("Attachment PG2 Max", attachment_scores.get("doc_pg2_max", 0.0), DECISION_COLORS["WARN"])
    score_card("Attachment Scope Min", attachment_scores.get("doc_scope_min", 1.0), "#8ab3ff")
    st.caption("Prompt Guard context")
    st.code(injection_scores.get("recent_context", ""), language="text")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Decision Reasons And L3/L4 Findings</div>', unsafe_allow_html=True)
reason_values = decision_reasons if isinstance(decision_reasons, list) else risk_scores.get("reasons", [])
if reason_values:
    for reason in reason_values:
        st.write(f"- {reason}")
else:
    st.caption("No warning or deny reason recorded.")

detail_col_1, detail_col_2, detail_col_3 = st.columns(3)
with detail_col_1:
    st.markdown("**PII Summary**")
    st.json(pii_scores or {"severity": selected_row.get("pii_main", 0.0)})
with detail_col_2:
    st.markdown("**Attachment Summary**")
    st.json(attachment_scores.get("summary", safe_json_loads(selected_row.get("attachment_summary", "{}"))))
with detail_col_3:
    st.markdown("**Llama Guard 4**")
    st.json(llama_guard_scores or {"lg4_unsafe": selected_row.get("lg4_unsafe", 0)})
st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Chunk-Level Findings", expanded=False):
    chunks = chunk_summaries if isinstance(chunk_summaries, list) and chunk_summaries else attachment_scores.get("chunks", [])
    if chunks:
        st.dataframe(pd.DataFrame(chunks), use_container_width=True, hide_index=True)
    else:
        st.caption("No text attachment chunks were logged for this request.")

with st.expander("Model Versions", expanded=False):
    st.json(model_versions or raw_scores.get("model_versions", {}))

with st.expander("Raw Scores JSON", expanded=False):
    st.json(raw_scores)

with st.expander("Request Payload", expanded=False):
    st.json(request_payload)

with st.expander("Response Payload", expanded=False):
    st.json(response_payload)
