"""Cached data and model loaders for the Streamlit application."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.constants import SEASON_START, TARGETS
from real_madrid_acwr.acwr import classify_acwr_zone, compute_acwr
from real_madrid_acwr.config import DATA_DIR


@st.cache_resource
def load_models():
    from real_madrid_acwr.modeling.artifacts import load_model_artifacts

    return load_model_artifacts()


def get_models_or_stop():
    try:
        return load_models()
    except ModuleNotFoundError as exc:
        st.error(
            "A required Python package is missing. Activate the project environment with "
            "`source .venv/bin/activate` and rerun `streamlit run main.py`."
        )
        st.exception(exc)
        st.stop()
    except Exception as exc:
        st.error("Model artifacts are missing or invalid. Run `python train_models.py` to regenerate them.")
        st.exception(exc)
        st.stop()


@st.cache_resource
def load_player_data():
    df = pd.read_parquet(DATA_DIR / "processed" / "model_data.parquet")
    df["date"] = SEASON_START + pd.to_timedelta(df["days_since_start"], unit="D")

    all_pids = sorted(df["player_id"].unique().tolist())
    player_data: dict[Any, dict[str, Any]] = {}

    for pid, grp in df.groupby("player_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        start_date = pd.Timestamp(grp["date"].min())
        end_date = pd.Timestamp(grp["date"].max())
        dr = pd.date_range(start_date, end_date, freq="D")
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

    current_acwr: dict[Any, dict[str, dict[str, Any]]] = {}
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
