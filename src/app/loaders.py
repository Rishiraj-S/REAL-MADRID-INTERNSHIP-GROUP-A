"""Cached data and model loaders for the Streamlit application."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.constants import TARGETS
from real_madrid_acwr.acwr import classify_acwr_zone, compute_acwr
from real_madrid_acwr.config import DAILY_PARQUET


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
    daily_df = pd.read_parquet(DAILY_PARQUET)
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    all_pids = sorted(daily_df["player_id"].unique().tolist())
    player_data: dict[Any, dict[str, Any]] = {}

    for pid, grp in daily_df.groupby("player_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        start_date = pd.Timestamp(grp["date"].min())
        end_date   = pd.Timestamp(grp["date"].max())
        dr = pd.date_range(start_date, end_date, freq="D")

        # Complete daily calendar grid; fillna(0) ensures rest days carry zero load
        has_cols = [c for c in grp.columns if c.startswith("has_")]
        merge_cols = ["date", "total_distance", "accelerations", "sprint_distance"] + has_cols
        merged = pd.DataFrame({"date": dr}).merge(
            grp[merge_cols],
            on="date", how="left",
        ).fillna({c: 0.0 for c in merge_cols if c != "date"})

        pos_label = str(grp.iloc[-1]["position"]) if "position" in grp.columns else "Unknown"

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

    return player_data, all_pids, current_acwr, daily_df
