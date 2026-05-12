"""
Real Madrid ACWR Monitor — Streamlit app.
Run with: streamlit run app.py
"""

import base64
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

from real_madrid_acwr.acwr import classify_acwr_zone, compute_acwr, compute_acwr_with_forecast
from real_madrid_acwr.config import DATA_DIR, STATIC_DIR
from real_madrid_acwr.modeling.artifacts import (
    TARGETS,
    ArtifactLoadError,
    load_model_artifacts,
)

LOGO_PATH = STATIC_DIR / "img" / "Real-Madrid-CF-v2002.svg"

st.set_page_config(
    page_title="Real Madrid · ACWR Monitor",
    page_icon=str(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
SEASON_START  = pd.Timestamp("2024-07-15")
# Ordered to match has_* columns in model_data.parquet; changing order breaks feature alignment
SESSION_TYPES = ["G", "TAC", "BP", "TEC", "MATCH"]
PAGES         = ["Dashboard", "Plan Sessions", "Forecast Results"]

# Colors map to Real Madrid's official palette: navy (#00529F), gold (#FEBE10), red (#EE324E)
TARGET_META = {
    "total_distance": {"label": "Total Distance",     "unit": "m",      "color": "#00529F", "fill": "rgba(0,82,159,0.12)"},
    "acc_total":      {"label": "Accelerations",      "unit": "efforts", "color": "#FEBE10", "fill": "rgba(254,190,16,0.12)"},
    "vel_total":      {"label": "High-Speed Running", "unit": "m",      "color": "#EE324E", "fill": "rgba(238,50,78,0.12)"},
}
SESSION_COLORS = {
    "G": "#00529F", "TAC": "#FEBE10", "BP": "#8B5CF6", "TEC": "#10B981", "MATCH": "#EE324E",
}
SESSION_LABELS = {
    "G": "Game / SSG", "TAC": "Tactical", "BP": "Set Pieces", "TEC": "Technical", "MATCH": "Official Match",
}
ZONE_COLORS = {
    "undertraining": "#64748B", "optimal": "#10B981",
    "caution": "#F59E0B", "danger": "#EE324E", "unknown": "#94A3B8",
}
ZONE_LABELS = {
    "undertraining": "Under", "optimal": "Optimal",
    "caution": "Caution", "danger": "Danger", "unknown": "—",
}
# Caution/danger bands use slightly higher opacity (0.22) than others to draw the eye to risk zones
ZONE_BANDS = [
    {"y0": 0,   "y1": 0.8, "color": "rgba(100,116,139,0.18)"},
    {"y0": 0.8, "y1": 1.3, "color": "rgba(16,185,129,0.18)"},
    {"y0": 1.3, "y1": 1.5, "color": "rgba(245,158,11,0.22)"},
    {"y0": 1.5, "y1": 2.5, "color": "rgba(238,50,78,0.22)"},
]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
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
""", unsafe_allow_html=True)


# ── Resolve pending navigation BEFORE sidebar renders ────────────────────────
# Must run before the radio widget renders: Streamlit raises StreamlitAPIException
# if you set a widget's session_state key after that widget has already rendered
# in the same script run. We stage the target page in _pending_nav, then apply
# it here at the very top so the radio picks up the correct value on re-render.
if "_pending_nav" in st.session_state:
    st.session_state.nav_page = st.session_state.pop("_pending_nav")
elif "nav_page" not in st.session_state:
    st.session_state.nav_page = PAGES[0]

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return load_model_artifacts()


def get_models_or_stop():
    try:
        return load_models()
    except ArtifactLoadError as exc:
        st.error("Model artifacts are missing or invalid. Run `python train_models.py` to regenerate them.")
        st.exception(exc)
        st.stop()


@st.cache_resource
def load_player_data():
    df = pd.read_parquet(DATA_DIR / "processed" / "model_data.parquet")
    df["date"] = SEASON_START + pd.to_timedelta(df["days_since_start"], unit="D")

    all_pids = sorted(df["player_id"].unique().tolist())
    player_data = {}

    for pid, grp in df.groupby("player_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        dr = pd.date_range(grp["date"].min(), grp["date"].max(), freq="D")
        # Left-merge creates a complete daily calendar grid; fillna(0) ensures rest days carry
        # zero load rather than NaN — EWMA accumulates over every calendar day, so gaps must be 0
        merged = pd.DataFrame({"date": dr}).merge(
            grp[["date", "total_distance", "acc_total", "vel_total", "has_MATCH"]],
            on="date", how="left",
        ).fillna({"total_distance": 0.0, "acc_total": 0.0, "vel_total": 0.0, "has_MATCH": 0})

        pos_label = "Unknown"
        for col in ["pos_central_back", "pos_central_midfielder", "pos_forward",
                    "pos_full_back", "pos_winger"]:
            if int(grp.iloc[-1][col]) == 1:
                pos_label = col.replace("pos_", "").replace("_", " ").title()
                break

        match_rows = grp[grp["has_MATCH"] == 1]
        player_data[pid] = {
            "grid":          merged,
            "profile":       grp.iloc[-1],
            "position":      pos_label,
            "last_active":   grp["date"].max(),
            "last_match":    match_rows["date"].max() if len(match_rows) > 0 else None,
            "n_active_days": len(grp),
        }

    current_acwr = {}
    for pid, data in player_data.items():
        current_acwr[pid] = {}
        for m in TARGETS:
            loads = data["grid"][m].values.astype(float)
            adf   = compute_acwr(loads)
            valid = adf["acwr"].dropna()
            val   = float(valid.iloc[-1]) if len(valid) > 0 else None
            current_acwr[pid][m] = {
                "value": round(val, 3) if val is not None else None,
                "zone":  classify_acwr_zone(val) if val is not None else "unknown",
            }

    return player_data, all_pids, current_acwr


# ── Inference ─────────────────────────────────────────────────────────────────
def build_forecast(plan_days):
    models = get_models_or_stop()
    player_data, all_pids, _ = load_player_data()

    HIST_SHOW = 60
    results   = {}

    for pid in all_pids:
        pdata   = player_data[pid]
        profile = pdata["profile"]
        grid    = pdata["grid"]

        player_feats = {
            "height":                  float(profile["height"]),
            "weight":                  float(profile["weight"]),
            "age":                     float(profile["age"]),
            "pos_central_back":        int(profile["pos_central_back"]),
            "pos_central_midfielder":  int(profile["pos_central_midfielder"]),
            "pos_forward":             int(profile["pos_forward"]),
            "pos_full_back":           int(profile["pos_full_back"]),
            "pos_winger":              int(profile["pos_winger"]),
        }
        # All 28 players must appear as pid_ columns even for rest-day rows; model expects a fixed 45-feature vector
        pid_feats = {f"pid_{p}": int(p == pid) for p in all_pids}

        last_dss    = int(profile["days_since_start"])
        last_active = pdata["last_active"]
        last_match  = pdata["last_match"]

        # Track offsets relative to the forecast window start rather than absolute dates
        # so days_since_last_activity/match stay valid across the 15-day roll-forward
        prev_active_d = 0
        prev_match_d  = (
            -int((last_active - last_match).days)
            if last_match is not None and not pd.isna(last_match)
            else -21
        )

        forecast_loads = {m: [] for m in TARGETS}

        for d, day in enumerate(plan_days, start=1):
            dsla = min(d - prev_active_d, 21)  # capped at 21 to match training distribution; extrapolating beyond degrades predictions
            dslm = min(d - prev_match_d, 21)

            if day["is_rest"]:
                for m in TARGETS:
                    forecast_loads[m].append(0.0)
                continue

            sess = {
                "has_G":           int(day.get("G",     False)),
                "has_TAC":         int(day.get("TAC",   False)),
                "has_BP":          int(day.get("BP",    False)),
                "has_TEC":         int(day.get("TEC",   False)),
                "has_MATCH":       int(day.get("MATCH", False)),
                "n_session_types": sum(int(day.get(t, False)) for t in SESSION_TYPES),
            }
            day_feats = {
                "days_since_start":        last_dss + d,
                "days_since_last_activity": float(dsla),
                "days_since_last_match":    float(dslm),
            }

            for target, art in models.items():
                fc   = art.feature_cols
                feat = {c: 0 for c in fc}
                feat.update(player_feats)
                feat.update(sess)
                feat.update(day_feats)
                feat.update(pid_feats)
                X   = pd.DataFrame([feat])[fc]
                raw = float(art.model.predict(xgb.DMatrix(X))[0])
                if art.transform["type"] == "log1p":
                    raw = float(np.expm1(raw))
                forecast_loads[target].append(max(0.0, raw))

            prev_active_d = d
            if day.get("MATCH"):
                prev_match_d = d

        pr = {"position": pdata["position"], "n_active_days": pdata["n_active_days"]}

        for m in TARGETS:
            hist_loads = grid[m].values.astype(float)
            fore_loads = np.array(forecast_loads[m])
            full       = compute_acwr_with_forecast(hist_loads, fore_loads)
            n_hist     = len(hist_loads)
            chunk      = full.iloc[max(0, n_hist - HIST_SHOW):]

            hist_c = chunk[~chunk["is_forecast"]]
            fore_c = chunk[chunk["is_forecast"]]

            hist_start = last_active - pd.Timedelta(days=len(hist_c) - 1)

            def to_dates(base, n):
                return [(base + pd.Timedelta(days=i)).strftime("%d %b") for i in range(n)]

            def clean(vals):
                return [
                    None if (v is None or (isinstance(v, float) and np.isnan(v)))
                    else round(float(v), 3) for v in vals
                ]

            valid_fore = fore_c["acwr"].dropna()
            day15_val  = float(valid_fore.iloc[-1]) if len(valid_fore) > 0 else None

            pr[m] = {
                "hist_dates": to_dates(hist_start, len(hist_c)),
                "hist_acwr":  clean(hist_c["acwr"].values),
                "fore_dates": to_dates(last_active + pd.Timedelta(days=1), len(fore_c)),
                "fore_acwr":  clean(fore_c["acwr"].values),
                "day15_acwr": round(day15_val, 3) if day15_val is not None else None,
                "day15_zone": classify_acwr_zone(day15_val) if day15_val is not None else "unknown",
            }

        results[str(pid)] = pr

    return results


# ── Chart ─────────────────────────────────────────────────────────────────────
def build_acwr_chart(mdata: dict, meta: dict) -> go.Figure:
    hist_x = mdata["hist_dates"]
    fore_x = mdata["fore_dates"]
    hist_y = mdata["hist_acwr"]
    fore_y = mdata["fore_acwr"]

    fig = go.Figure()

    for band in ZONE_BANDS:
        fig.add_hrect(y0=band["y0"], y1=band["y1"],
                      fillcolor=band["color"], layer="below", line_width=0)

    for y_val in [0.8, 1.3, 1.5]:
        fig.add_hline(y=y_val, line_dash="dot",
                      line_color="rgba(0,82,159,0.20)", line_width=1)

    if fore_x:
        fig.add_vrect(x0=fore_x[0], x1=fore_x[-1],
                      fillcolor="rgba(254,190,16,0.05)", layer="below", line_width=0)
        fig.add_annotation(
            x=fore_x[0], y=2.1, text="Forecast Window",
            showarrow=False, font=dict(size=10, color="#B8920A"),
            xanchor="left", yanchor="top",
        )

    # Extend historical trace by one forecast point so the two lines visually join at the boundary
    join_x = hist_x + ([fore_x[0]] if fore_x else [])
    join_y = hist_y + ([fore_y[0]] if fore_y else [])
    fig.add_trace(go.Scatter(
        x=join_x, y=join_y,
        mode="lines", name="Historical",
        line=dict(color="rgba(0,82,159,0.40)", width=2),
        connectgaps=False,
        hovertemplate="<b>%{x}</b><br>ACWR: %{y:.3f}<extra>Historical</extra>",
    ))

    fig.add_trace(go.Scatter(
        x=fore_x, y=fore_y,
        mode="lines+markers", name="Forecast",
        line=dict(color=meta["color"], width=2.5),
        marker=dict(size=5, color=meta["color"],
                    line=dict(color="#FFFFFF", width=1.5)),
        fill="tozeroy", fillcolor=meta["fill"],
        connectgaps=False,
        hovertemplate="<b>%{x}</b><br>ACWR: %{y:.3f}<extra>Forecast</extra>",
    ))

    fig.update_layout(
        height=400,
        plot_bgcolor="#FAFCFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, system-ui", color="#334D6E", size=12),
        margin=dict(l=8, r=8, t=32, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#DCE8F5",
                        font=dict(color="#0F172A", size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11)),
        xaxis=dict(tickfont=dict(size=10, color="#64748B"),
                   gridcolor="rgba(0,82,159,0.06)",
                   showline=False, tickangle=40, nticks=22),
        yaxis=dict(range=[0, 2.3],
                   tickfont=dict(size=10, color="#64748B"),
                   gridcolor="rgba(0,82,159,0.06)",
                   title=dict(text="ACWR", font=dict(size=11, color="#64748B")),
                   zeroline=False),
    )
    return fig


# ── Page: Dashboard ───────────────────────────────────────────────────────────
def page_dashboard():
    player_data, all_pids, current_acwr = load_player_data()
    get_models_or_stop()
    last_date = player_data[all_pids[0]]["last_active"].strftime("%d %B %Y")

    st.markdown(f"""
    <div class="page-header">
        <div class="page-header-accent"></div>
        <div class="page-title">Squad <span>ACWR</span> Dashboard</div>
        <div class="page-sub">
            Data through <strong style="color:#334D6E">{last_date}</strong>
            &nbsp;&middot;&nbsp; {len(all_pids)} players tracked
            &nbsp;&middot;&nbsp; 3 load metrics
        </div>
    </div>""", unsafe_allow_html=True)

    # Stat cards
    n_danger  = sum(1 for p in all_pids for m in TARGETS if current_acwr[p][m]["zone"] == "danger")
    n_caution = sum(1 for p in all_pids for m in TARGETS if current_acwr[p][m]["zone"] == "caution")
    n_optimal = sum(1 for p in all_pids for m in TARGETS if current_acwr[p][m]["zone"] == "optimal")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class="stat-card" style="border-top-color:#EE324E">
        <div class="stat-val" style="color:#EE324E">{n_danger}</div>
        <div class="stat-lbl">Danger Flags</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="stat-card" style="border-top-color:#F59E0B">
        <div class="stat-val" style="color:#F59E0B">{n_caution}</div>
        <div class="stat-lbl">Caution Flags</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="stat-card" style="border-top-color:#10B981">
        <div class="stat-val" style="color:#10B981">{n_optimal}</div>
        <div class="stat-lbl">Optimal Flags</div>
    </div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="stat-card" style="border-top-color:#00529F">
        <div class="stat-val" style="color:#00529F">{len(all_pids)}</div>
        <div class="stat-lbl">Players Tracked</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

    # Zone legend
    st.markdown('<div class="section-label" style="font-size:1rem;letter-spacing:0.5px">Risk Zones</div>', unsafe_allow_html=True)
    lc = st.columns(4)
    for i, (zone, rng) in enumerate([
        ("undertraining", "ACWR < 0.8"),
        ("optimal",       "ACWR 0.8 – 1.3"),
        ("caution",       "ACWR 1.3 – 1.5"),
        ("danger",        "ACWR ≥ 1.5"),
    ]):
        col = ZONE_COLORS[zone]
        lbl = ZONE_LABELS[zone]
        lc[i].markdown(f"""
        <div class="zone-pill" style="color:{col};border-color:{col};background:{col}15">
            <span class="zone-dot" style="background:{col}"></span>
            {lbl} &nbsp; <span style="font-weight:400;opacity:0.8">{rng}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label" style="font-size:1rem;letter-spacing:0.5px">Player Status — Current ACWR</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    for i, pid in enumerate(all_pids):
        pdata     = player_data[pid]
        acwr      = current_acwr[pid]

        worst_zone = "optimal"
        zone_order = ["danger", "caution", "undertraining", "optimal", "unknown"]
        for m in TARGETS:
            z = acwr[m]["zone"]
            if zone_order.index(z) < zone_order.index(worst_zone):
                worst_zone = z

        rows_html = ""
        for m in TARGETS:
            ma    = acwr[m]
            col   = ZONE_COLORS[ma["zone"]]
            val_s = f"{ma['value']:.2f}" if ma["value"] is not None else "—"
            rows_html += f"""
            <div class="metric-row">
                <span class="metric-lbl" style="color:{TARGET_META[m]['color']}">{TARGET_META[m]['label']}</span>
                <div class="metric-rhs">
                    <span class="metric-val" style="color:{col}">{val_s}</span>
                    <span class="metric-badge" style="color:{col};border-color:{col};background:{col}18">
                        {ZONE_LABELS[ma['zone']]}
                    </span>
                </div>
            </div>"""

        with cols[i % 4]:
            st.markdown(f"""
            <div class="player-card {worst_zone}">
                <div class="card-id">{pid}</div>
                <div class="card-pos">{pdata['position']}</div>
                <hr class="card-rule">
                {rows_html}
            </div>""", unsafe_allow_html=True)


# ── Page: Session Planner ─────────────────────────────────────────────────────
def page_planner():
    player_data, all_pids, _ = load_player_data()
    last_active = player_data[all_pids[0]]["last_active"]
    plan_dates  = [(last_active + pd.Timedelta(days=i + 1)).strftime("%a %d %b") for i in range(15)]

    st.markdown("""
    <div class="page-header">
        <div class="page-title">Session <span>Planner</span></div>
        <div class="page-sub">
            Select session types for each of the next 15 days.
            Leaving all unchecked marks the day as REST.
        </div>
    </div>""", unsafe_allow_html=True)

    # Session type legend
    st.markdown('<div class="section-label" style="font-size:1rem;letter-spacing:0.5px">Session Types</div>', unsafe_allow_html=True)
    lc = st.columns(6)
    for i, (t, lbl) in enumerate({**SESSION_LABELS, "REST": "Rest Day"}.items()):
        col = SESSION_COLORS.get(t, "#64748B")
        lc[i % 6].markdown(f"""
        <div style="border:1px solid {col};border-radius:20px;padding:5px 12px;
                    font-size:0.7rem;font-weight:700;color:{col};
                    text-transform:uppercase;letter-spacing:0.4px;
                    text-align:center;margin-bottom:10px;background:{col}0D">
            {t} &middot; {lbl}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    init = pd.DataFrame({
        "Day":   [f"Day {i+1:02d}" for i in range(15)],
        "Date":  plan_dates,
        "G":     [False] * 15,
        "TAC":   [False] * 15,
        "BP":    [False] * 15,
        "TEC":   [False] * 15,
        "MATCH": [False] * 15,
    })

    edited = st.data_editor(
        init,
        column_config={
            "Day":   st.column_config.TextColumn("Day",   disabled=True, width="small"),
            "Date":  st.column_config.TextColumn("Date",  disabled=True, width="medium"),
            "G":     st.column_config.CheckboxColumn("G",     help="Game / Small-Sided Game"),
            "TAC":   st.column_config.CheckboxColumn("TAC",   help="Tactical session"),
            "BP":    st.column_config.CheckboxColumn("BP",    help="Set pieces"),
            "TEC":   st.column_config.CheckboxColumn("TEC",   help="Technical drills"),
            "MATCH": st.column_config.CheckboxColumn("MATCH", help="Official match"),
        },
        hide_index=True,
        use_container_width=True,
        height=576,
        key="plan_editor",
    )

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    _, btn_col = st.columns([4, 1])
    if btn_col.button("Run Forecast →", type="primary", use_container_width=True):
        plan_days = []
        for _, row in edited.iterrows():
            selected = [t for t in SESSION_TYPES if row.get(t, False)]
            plan_days.append({
                "is_rest": len(selected) == 0,
                **{t: (t in selected) for t in SESSION_TYPES},
            })

        with st.spinner("Computing 15-day ACWR forecasts for all 28 players…"):
            results = build_forecast(plan_days)

        if results:
            st.session_state.forecast   = results
            st.session_state.plan_days  = plan_days
            st.session_state.plan_dates = plan_dates
            st.session_state._pending_nav = PAGES[2]  # stage nav change; resolved at script top before radio renders
            st.rerun()
        else:
            st.error("Forecast failed — check models are trained (`python train_models.py`).")


# ── Page: Forecast Results ────────────────────────────────────────────────────
def page_results():
    if "forecast" not in st.session_state:
        st.markdown("""
        <div class="page-header">
            <div class="page-title">Forecast <span>Results</span></div>
            <div class="page-sub">No forecast computed yet.</div>
        </div>""", unsafe_allow_html=True)
        st.info("Go to **Plan Sessions** and click **Run Forecast →** to generate results.")
        return

    player_data, all_pids, _ = load_player_data()
    forecast   = st.session_state.forecast
    plan_days  = st.session_state.get("plan_days", [])
    plan_dates = st.session_state.get("plan_dates", [])

    st.markdown(f"""
    <div class="page-header">
        <div class="page-title">Forecast <span>Results</span></div>
        <div class="page-sub">
            15-day ACWR projection &nbsp;&middot;&nbsp;
            {len(all_pids)} players &nbsp;&middot;&nbsp;
            3 load metrics
        </div>
    </div>""", unsafe_allow_html=True)

    # Planned sessions — weekly calendar
    st.markdown('<div class="section-label" style="font-size:1rem;letter-spacing:0.5px">Planned Sessions</div>', unsafe_allow_html=True)

    DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_index = {d: i for i, d in enumerate(DOW_ORDER)}
    first_dow = dow_index.get(plan_dates[0][:3], 0)

    # Pad start with empty slots so day 1 lands on its weekday column
    slots = [None] * first_dow + list(range(len(plan_days)))
    while len(slots) % 7:
        slots.append(None)
    weeks = [slots[i:i + 7] for i in range(0, len(slots), 7)]

    header_cells = "".join(
        f'<th style="width:14.28%;padding:6px 0;font-size:0.65rem;font-weight:800;'
        f'text-transform:uppercase;letter-spacing:1.2px;color:#00529F;'
        f'text-align:center;border-bottom:2px solid #E2EBF6">{d}</th>'
        for d in DOW_ORDER
    )

    rows_html = ""
    for week in weeks:
        row = ""
        for idx in week:
            if idx is None:
                row += '<td style="padding:4px;background:#F8FAFD;border:1px solid #EEF3FA"></td>'
            else:
                day      = plan_days[idx]
                date_str = plan_dates[idx]
                day_num  = date_str[4:6]
                mon_str  = date_str[7:]
                types    = [t for t in SESSION_TYPES if day.get(t)]
                is_rest  = len(types) == 0
                is_match = "MATCH" in types

                bg = "#FFF8F8" if is_match else ("#F8FAFD" if is_rest else "#FFFFFF")
                border_top = "3px solid #EE324E" if is_match else ("3px solid #E2EBF6" if is_rest else "3px solid #FEBE10")

                badges = ""
                if is_rest:
                    badges = '<span style="font-size:0.62rem;color:#94A3B8;font-weight:600">REST</span>'
                else:
                    for t in types:
                        c = SESSION_COLORS[t]
                        badges += (
                            f'<span style="display:inline-block;margin:1px 2px;'
                            f'padding:2px 6px;border-radius:4px;font-size:0.6rem;'
                            f'font-weight:800;background:{c}22;color:{c};'
                            f'border:1px solid {c}66">{t}</span>'
                        )

                row += f"""
                <td style="padding:4px;border:1px solid #EEF3FA;vertical-align:top">
                  <div style="background:{bg};border-radius:7px;padding:7px 8px;
                              border-top:{border_top};min-height:72px">
                    <div style="font-size:0.7rem;font-weight:700;color:#334D6E;
                                margin-bottom:4px;line-height:1">
                      {day_num}
                      <span style="font-size:0.6rem;font-weight:500;color:#94A3B8">{mon_str}</span>
                    </div>
                    <div style="display:flex;flex-wrap:wrap;gap:2px">{badges}</div>
                  </div>
                </td>"""
        rows_html += f"<tr>{row}</tr>"

    st.markdown(f"""
    <div style="border:2px solid #3D4A5C;border-radius:10px;overflow:hidden;
                box-shadow:0 1px 6px rgba(0,60,140,0.06);margin-bottom:1.5rem">
        <table style="width:100%;border-collapse:collapse;background:#F0F4FA">
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>""", unsafe_allow_html=True)

    # Danger alert
    danger_entries = []
    for p in all_pids:
        bad_metrics = [
            TARGET_META[m]["label"]
            for m in TARGETS
            if forecast[str(p)][m]["day15_zone"] == "danger"
        ]
        if bad_metrics:
            danger_entries.append((str(p), bad_metrics))

    if danger_entries:
        rows = "".join(
            f'<div style="margin-top:4px">&#x2022; Player <strong>{p}</strong> — '
            f'{", ".join(ms)}</div>'
            for p, ms in danger_entries
        )
        st.markdown(f"""
        <div class="rm-alert">
            <div>
                <strong>Injury Risk Alert:</strong> {len(danger_entries)} player(s) projected
                in DANGER zone by Day 15:
                {rows}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Player selector + back button
    c1, c2 = st.columns([3, 1])
    with c1:
        pid_opts      = {f"Player {p} · {player_data[p]['position']}": str(p) for p in all_pids}
        sel_pid_label = st.selectbox("PLAYER", list(pid_opts.keys()), label_visibility="visible")
        sel_pid       = pid_opts[sel_pid_label]
    with c2:
        st.markdown("<div style='margin-top:1.65rem'></div>", unsafe_allow_html=True)
        if st.button("← Adjust Plan", type="secondary", use_container_width=True):
            st.session_state._pending_nav = PAGES[1]  # stage nav change; resolved at script top before radio renders
            st.rerun()

    # Three charts stacked
    for metric in TARGETS:
        meta = TARGET_META[metric]
        st.markdown(
            f'<div class="section-label" style="color:{meta["color"]};margin-top:1rem">'
            f'{meta["label"]} &nbsp;·&nbsp; ACWR (unitless)</div>',
            unsafe_allow_html=True,
        )
        fig = build_acwr_chart(forecast[sel_pid][metric], meta)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Day-15 summary table
    st.markdown("---")
    st.markdown('<div class="section-label">Day-15 Summary — All Players</div>', unsafe_allow_html=True)

    ZONE_ORDER   = ["danger", "caution", "undertraining", "optimal", "unknown"]
    header_cols  = "".join(
        f'<th style="color:{TARGET_META[m]["color"]}">{TARGET_META[m]["label"]}</th>'
        for m in TARGETS
    )
    rows_html = ""
    for pid in [str(p) for p in all_pids]:
        pdata = player_data[int(pid)]
        cells = f'<td class="td-pid">Player {pid}</td><td class="td-pos">{pdata["position"]}</td>'
        worst = "optimal"
        for m in TARGETS:
            v    = forecast[pid][m]["day15_acwr"]
            zone = forecast[pid][m]["day15_zone"]
            col  = ZONE_COLORS[zone]
            lbl  = ZONE_LABELS[zone]
            if ZONE_ORDER.index(zone) < ZONE_ORDER.index(worst):
                worst = zone
            val_s = f"{v:.2f}" if v is not None else "—"
            cells += f"""
            <td>
                <span style="color:{col};font-weight:700;font-family:'Courier New',monospace">{val_s}</span>
                <span style="font-size:0.56rem;font-weight:800;text-transform:uppercase;
                             padding:2px 6px;border-radius:5px;border:1px solid {col};
                             color:{col};background:{col}18;margin-left:5px">{lbl}</span>
            </td>"""
        status_icons = {"danger": "HIGH RISK", "caution": "CAUTION", "optimal": "OK", "undertraining": "LOW"}
        icol  = ZONE_COLORS[worst]
        itext = status_icons.get(worst, "—")
        cells += f'<td><span style="color:{icol};font-size:0.65rem;font-weight:800;letter-spacing:0.8px">{itext}</span></td>'
        rows_html += f"<tr>{cells}</tr>"

    st.markdown(f"""
    <div style="overflow-x:auto;border:1px solid #E2EBF6;border-radius:10px;
                box-shadow:0 1px 6px rgba(0,60,140,0.06)">
        <table class="rm-table">
            <thead><tr><th>Player</th><th>Position</th>{header_cols}<th>Status</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    _logo_b64 = ""
    if LOGO_PATH.exists():
        _logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()

    st.markdown(f"""
    <div style="padding:1.75rem 1.25rem 1.1rem;text-align:center">
        {'<img src="data:image/svg+xml;base64,' + _logo_b64 + '" style="width:112px;height:112px;display:block;margin:0 auto 14px">' if _logo_b64 else ''}
        <div style="font-size:1.13rem;font-weight:900;letter-spacing:3px;
                    color:#FEBE10;text-transform:uppercase;line-height:1.2">Real Madrid C.F.</div>
        <div style="font-size:0.96rem;font-weight:400;letter-spacing:2px;
                    color:rgba(255,255,255,0.55);text-transform:uppercase;margin-top:5px">
            ACWR Monitor
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        PAGES,
        key="nav_page",
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Season info footer
    st.markdown("""
    <div style="padding:0.5rem 1rem 1rem;font-size:0.65rem;color:rgba(255,255,255,0.4);line-height:1.8">
        <div style="font-weight:700;color:rgba(255,255,255,0.6);
                    text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">
            Season 2024/25
        </div>
        <div>28 Players &nbsp;&middot;&nbsp; 3 Metrics</div>
        <div>EWMA · α<sub>acute</sub>=0.25 · α<sub>chronic</sub>≈0.07</div>
    </div>""", unsafe_allow_html=True)

    # Developed by footer
    _team_logo_path = STATIC_DIR / "img" / "trAIn_labs.png"
    _team_b64 = ""
    if _team_logo_path.exists():
        _team_b64 = base64.b64encode(_team_logo_path.read_bytes()).decode()

    st.markdown("---")
    st.markdown(f"""
    <div style="padding:0.5rem 1rem 1.5rem;text-align:center">
        <div style="font-size:0.68rem;font-weight:600;letter-spacing:1.5px;
                    color:rgba(255,255,255,0.35);text-transform:uppercase;margin-bottom:10px">
            Developed by
        </div>
        {'<div style="display:inline-block;background:#FFFFFF;border-radius:8px;padding:6px 14px"><img src="data:image/png;base64,' + _team_b64 + '" style="width:110px;display:block"></div>' if _team_b64 else '<span style="color:rgba(255,255,255,0.5);font-weight:700">trAIn Labs</span>'}
    </div>""", unsafe_allow_html=True)


# ── Route ─────────────────────────────────────────────────────────────────────
if page == PAGES[0]:
    page_dashboard()
elif page == PAGES[1]:
    page_planner()
elif page == PAGES[2]:
    page_results()
