"""CSS styling for the Streamlit ACWR application."""

from __future__ import annotations

import streamlit as st

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* Base */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: #F0F4FA !important;
}
[data-testid="stMainBlockContainer"] {
    padding-top: 2rem !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] button, [data-testid="stToolbar"] a,
[data-testid="stToolbar"] span, [data-testid="stToolbar"] p {
    color: #0F172A !important;
}
[data-testid="stToolbar"] svg { fill: #0F172A !important; stroke: #0F172A !important; }
[data-testid="stSidebarCollapsedControl"] {
    background: #003D78 !important;
    border-radius: 0 6px 6px 0 !important;
}
[data-testid="stSidebarCollapsedControl"] button { color: #FEBE10 !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #002D5A 0%, #00529F 100%) !important;
    border-right: none !important;
    box-shadow: 3px 0 20px rgba(0,0,0,0.18) !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
    margin: 0.25rem 1rem 0.5rem !important;
}


/* Sidebar radio — styled as navigation list */
[data-testid="stSidebar"] [data-testid="stRadio"] {
    padding: 0.25rem 0.75rem 1rem !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div:last-child {
    display: flex !important;
    flex-direction: column !important;
    gap: 3px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    padding: 0.65rem 1rem !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: background 0.15s, color 0.15s !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: rgba(255,255,255,0.72) !important;
    border-left: 3px solid transparent !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: rgba(254,190,16,0.16) !important;
    color: #FEBE10 !important;
    font-weight: 700 !important;
    border-left-color: #FEBE10 !important;
}
/* Hide radio circle indicators */
[data-testid="stSidebar"] [data-baseweb="radio"] { display: none !important; }

/* Cards */
.rm-card {
    background: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #E2EBF6;
    box-shadow: 0 1px 8px rgba(0,60,140,0.07);
    padding: 1.25rem 1.5rem;
    height: 100%;
}

/* Stat cards */
.stat-card {
    background: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #E2EBF6;
    border-top: 4px solid;
    box-shadow: 0 2px 10px rgba(0,60,140,0.07);
    padding: 1.25rem 1.5rem;
}
.planner-stat-card {
    min-height: 122px;
}
.stat-val { font-size: 2.4rem; font-weight: 800; line-height: 1; margin-bottom: 6px; }
.stat-lbl { font-size: 0.82rem; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; }

/* Page header */
.page-header {
    margin-bottom: 2rem;
    padding: 2rem 1rem 1.5rem;
    text-align: center;
    border-bottom: 1px solid #E2EBF6;
}
.page-header-accent {
    display: inline-block;
    width: 48px;
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(90deg, #FEBE10, #00529F);
    margin-bottom: 1rem;
}
.page-title {
    font-size: 2.4rem;
    font-weight: 900;
    color: #0F172A;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.6rem;
}
.page-title span { color: #00529F; }
.page-sub {
    font-size: 1rem;
    color: #64748B;
    font-weight: 400;
    line-height: 1.6;
}

/* Section label */
.section-label {
    font-size: 0.82rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #00529F;
    margin-bottom: 0.75rem;
    margin-top: 0.25rem;
}

/* Zone pill */
.zone-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    border: 1px solid;
    white-space: nowrap;
}
.zone-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
    display: inline-block;
}

/* Player grid card */
.player-card {
    background: #FFFFFF;
    border: 1px solid #E2EBF6;
    border-left: 4px solid #E2EBF6;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    box-shadow: 0 1px 5px rgba(0,60,140,0.06);
    transition: box-shadow 0.2s, border-left-color 0.2s;
}
.player-card:hover {
    box-shadow: 0 4px 16px rgba(0,60,140,0.12);
}
.player-card.danger { border-left-color: #EE324E !important; }
.player-card.caution { border-left-color: #F59E0B !important; }
.player-card.optimal { border-left-color: #10B981 !important; }
.card-id {
    font-size: 0.9rem; font-weight: 800; color: #0F172A;
    font-family: 'Courier New', monospace; letter-spacing: 0.5px;
}
.card-pos {
    font-size: 0.75rem; color: #00529F; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.6px; margin-top: 3px; margin-bottom: 10px;
}
.card-rule { border: none; border-top: 1px solid #EEF3FA; margin: 0 0 10px; }
.metric-row {
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;
}
.metric-lbl { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.3px; }
.metric-rhs { display: flex; align-items: center; gap: 6px; }
.metric-val { font-size: 0.95rem; font-weight: 800; font-family: 'Courier New', monospace; }
.metric-badge {
    font-size: 0.68rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.4px;
    padding: 2px 7px; border-radius: 6px; border: 1px solid;
}

/* Planner workspace */
.planner-legend-pill {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    min-height: 54px;
    border: 1px solid;
    border-radius: 12px;
    padding: 0.65rem 0.85rem;
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 0.3px;
    text-align: center;
}
.planner-legend-dot {
    width: 9px;
    height: 9px;
    border-radius: 999px;
    flex-shrink: 0;
}
.planner-hint {
    background: rgba(0,82,159,0.06);
    border: 1px solid rgba(0,82,159,0.12);
    border-radius: 10px;
    padding: 0.8rem 0.95rem;
    color: #334D6E;
    font-size: 0.86rem;
    line-height: 1.45;
}
.planner-side-copy {
    color: #334D6E;
    font-size: 0.9rem;
    line-height: 1.6;
}
.planner-event-card {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
}
.planner-event-title {
    color: #0F172A;
    font-size: 0.98rem;
    font-weight: 800;
    line-height: 1.3;
}
.planner-event-meta,
.planner-event-location {
    color: #64748B;
    font-size: 0.82rem;
    line-height: 1.45;
}
.planner-event-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.1rem;
}
.session-chip {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 28px;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    border: 1px solid;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}
.planner-results-header {
    margin-top: 0.25rem;
    padding-top: 1.5rem;
}
.planner-status-banner {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    min-height: 48px;
    border: 1px solid;
    border-radius: 10px;
    font-size: 0.85rem;
    font-weight: 800;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}


/* Planner sidebar — forecast window box */
.forecast-window-box {
    background: linear-gradient(135deg, #002D5A 0%, #00529F 100%);
    border-radius: 12px;
    padding: 1rem 1.15rem 0.9rem;
    color: #FFFFFF;
}
.fw-label {
    font-size: 0.68rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: rgba(255,255,255,0.55);
    margin-bottom: 0.3rem;
}
.fw-dates {
    font-size: 1.05rem;
    font-weight: 800;
    color: #FEBE10;
    line-height: 1.25;
    margin-bottom: 0.25rem;
}
.fw-sub {
    font-size: 0.73rem;
    color: rgba(255,255,255,0.50);
    font-weight: 400;
}

/* Planner sidebar — session type legend rows */
.sidebar-session-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 3px 0;
}
.sidebar-session-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.sidebar-session-code {
    font-size: 0.77rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    min-width: 38px;
}
.sidebar-session-name {
    font-size: 0.80rem;
    color: #475569;
    font-weight: 500;
}

/* Planned sessions — panel and single-line rows */
.session-list-row {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 0;
    margin: 0;
    line-height: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.session-list-row p { margin: 0 !important; }
.slr-date {
    font-size: 0.80rem;
    font-weight: 700;
    color: #0F172A;
    flex-shrink: 0;
}
.slr-types {
    font-size: 0.80rem;
    color: #00529F;
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
}
/* Session count inline with heading */
.slr-count {
    font-size: 0.75rem;
    font-weight: 500;
    color: #64748B;
    text-transform: none;
    letter-spacing: 0;
}
/* Remove top padding inside the session panel bordered container */
[data-testid="stVerticalBlockBorderWrapper"]:has(.session-list-row) > [data-testid="stVerticalBlock"] {
    padding-top: 0 !important;
    gap: 0 !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.session-list-row) [data-testid="stHorizontalBlock"]:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
/* Force row alignment — strip all wrapper margins inside session rows */
[data-testid="stHorizontalBlock"]:has(.session-list-row) {
    align-items: center !important;
    gap: 4px !important;
    padding: 3px 0 !important;
    margin: 0 !important;
    border-bottom: 1px solid #EEF3FA;
}
[data-testid="stHorizontalBlock"]:has(.session-list-row) > [data-testid="stColumn"] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    min-width: 0 !important;
}
[data-testid="stHorizontalBlock"]:has(.session-list-row) [data-testid="stElementContainer"],
[data-testid="stHorizontalBlock"]:has(.session-list-row) [data-testid="stElementContainer"] > div,
[data-testid="stHorizontalBlock"]:has(.session-list-row) [data-testid="stMarkdownContainer"],
[data-testid="stHorizontalBlock"]:has(.session-list-row) [data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
}
[data-testid="stHorizontalBlock"]:has(.session-list-row) .stButton {
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
}
[data-testid="stHorizontalBlock"]:has(.session-list-row) .stButton > button {
    padding: 0 !important;
    min-height: 26px !important;
    height: 26px !important;
    width: 100% !important;
    font-size: 0.85rem !important;
    line-height: 1 !important;
    border-radius: 5px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Alert banner */
.rm-alert {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 14px 18px;
    background: rgba(238,50,78,0.05);
    border: 1px solid rgba(238,50,78,0.25);
    border-left: 4px solid #EE324E;
    border-radius: 10px;
    color: #B91C3C; font-size: 0.95rem; font-weight: 500;
    margin-bottom: 1.5rem;
    line-height: 1.6;
}

/* Summary table */
.rm-table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
.rm-table th {
    padding: 12px 16px;
    background: #F0F4FA;
    border-bottom: 2px solid #C5D8EE;
    font-size: 0.75rem; font-weight: 800; color: #00529F;
    text-transform: uppercase; letter-spacing: 1px;
    text-align: left; white-space: nowrap;
}
.rm-table td { padding: 11px 16px; border-bottom: 1px solid #EEF3FA; vertical-align: middle; }
.rm-table tr:last-child td { border-bottom: none; }
.rm-table tr:hover td { background: rgba(254,190,16,0.04); }
.td-pid { font-weight: 700; color: #0F172A; font-family: 'Courier New', monospace; font-size: 0.92rem; }
.td-pos { color: #00529F; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px; }

/* Buttons */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.15s !important;
    font-size: 0.875rem !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #FEBE10 0%, #FFD84A 100%) !important;
    color: #0F172A !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 12px rgba(254,190,16,0.40) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 20px rgba(254,190,16,0.55) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"] {
    background: #FFFFFF !important;
    color: #00529F !important;
    border: 1.5px solid #C5D8EE !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #FEBE10 !important;
    background: rgba(254,190,16,0.06) !important;
}

/* Select box */
[data-testid="stSelectbox"] label {
    font-size: 0.82rem !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: #00529F !important;
}
[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border-color: #D1E2F4 !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    color: #00529F !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] div[class*="singleValue"],
[data-baseweb="select"] div[class*="placeholder"] {
    color: #00529F !important;
}

/* Data editor */
[data-testid="stDataEditor"] {
    border: 1px solid #D1DCE8 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 6px rgba(0,60,140,0.06) !important;
    background: #EAEEF4 !important;
}
[data-testid="stDataEditor"] [data-testid="glideDataEditorContainer"],
[data-testid="stDataEditor"] canvas,
[data-testid="stDataEditor"] > div {
    background: #EAEEF4 !important;
}

/* HR */
hr { border-color: #E2EBF6 !important; margin: 1.5rem 0 !important; }

/* Spinner */
[data-testid="stSpinner"] p { color: #00529F !important; }
</style>
"""


def inject_styles() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
