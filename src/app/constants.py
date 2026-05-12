"""Constants for the Streamlit ACWR application."""

from __future__ import annotations

import pandas as pd

SEASON_START = pd.Timestamp("2024-07-15")
# Ordered to match has_* columns in model_data.parquet; changing order breaks feature alignment.
SESSION_TYPES = ["G", "TAC", "BP", "TEC", "MATCH"]
PAGES = ["Dashboard", "Plan Sessions", "Forecast Results"]
TARGETS = ("total_distance", "acc_total", "vel_total")

# Colors map to Real Madrid's official palette: navy (#00529F), gold (#FEBE10), red (#EE324E).
TARGET_META = {
    "total_distance": {
        "label": "Total Distance",
        "unit": "m",
        "color": "#00529F",
        "fill": "rgba(0,82,159,0.12)",
    },
    "acc_total": {
        "label": "Accelerations",
        "unit": "efforts",
        "color": "#FEBE10",
        "fill": "rgba(254,190,16,0.12)",
    },
    "vel_total": {
        "label": "High-Speed Running",
        "unit": "m",
        "color": "#EE324E",
        "fill": "rgba(238,50,78,0.12)",
    },
}
SESSION_COLORS = {
    "G": "#00529F",
    "TAC": "#FEBE10",
    "BP": "#8B5CF6",
    "TEC": "#10B981",
    "MATCH": "#EE324E",
}
SESSION_LABELS = {
    "G": "Game / SSG",
    "TAC": "Tactical",
    "BP": "Set Pieces",
    "TEC": "Technical",
    "MATCH": "Official Match",
}
ZONE_COLORS = {
    "undertraining": "#64748B",
    "optimal": "#10B981",
    "caution": "#F59E0B",
    "danger": "#EE324E",
    "unknown": "#94A3B8",
}
ZONE_LABELS = {
    "undertraining": "Under",
    "optimal": "Optimal",
    "caution": "Caution",
    "danger": "Danger",
    "unknown": "—",
}
# Caution/danger bands use slightly higher opacity (0.22) than others to draw the eye to risk zones.
ZONE_BANDS = [
    {"y0": 0, "y1": 0.8, "color": "rgba(100,116,139,0.18)"},
    {"y0": 0.8, "y1": 1.3, "color": "rgba(16,185,129,0.18)"},
    {"y0": 1.3, "y1": 1.5, "color": "rgba(245,158,11,0.22)"},
    {"y0": 1.5, "y1": 2.5, "color": "rgba(238,50,78,0.22)"},
]
