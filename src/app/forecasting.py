"""Forecast construction for the Streamlit ACWR application."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.constants import SESSION_TYPES, TARGETS
from app.loaders import get_models_or_stop, load_player_data
from real_madrid_acwr.acwr import classify_acwr_zone, compute_acwr_with_forecast
from real_madrid_acwr.modeling.train import _add_features

HIST_DAYS = 90  # calendar days of history to seed EWMA (chronic_load needs >= 28 days)
HIST_SHOW = 60  # days of history shown in the chart


def _build_plan_frame(
    plan_days: list[dict],
    all_pids: list,
    player_data: dict,
    last_active: pd.Timestamp,
    target: str,
) -> pd.DataFrame:
    """Convert coach plan to a DataFrame ready for recursive forecasting."""
    has_cols = ["has_G", "has_TAC", "has_BP", "has_TEC", "has_MATCH"]
    rows = []
    for d, day in enumerate(plan_days, start=1):
        date = last_active + pd.Timedelta(days=d)
        is_rest = day.get("is_rest", False)
        n_types = 0 if is_rest else sum(int(day.get(t, False)) for t in SESSION_TYPES)
        for pid in all_pids:
            profile = player_data[pid]["profile"]
            row: dict[str, object] = {
                "player_id":       pid,
                "date":            date,
                "n_periods":       0 if is_rest else 1,
                "n_exercise_types": n_types,
                "height":          float(profile["height"]),
                "weight":          float(profile["weight"]),
                "age":             float(profile["age"]),
                "position":        str(profile["position"]) if "position" in profile.index else "Unknown",
            }
            for metric in TARGETS:
                row[metric] = np.nan if metric == target else 0.0
            for h in has_cols:
                key = h[len("has_"):]
                row[h] = 0 if is_rest else int(day.get(key, False))
            rows.append(row)
    return pd.DataFrame(rows)


def _get_history(daily_df: pd.DataFrame, target: str, last_active: pd.Timestamp) -> pd.DataFrame:
    """Return the last HIST_DAYS of history for all players, keeping only columns needed."""
    cutoff = last_active - pd.Timedelta(days=HIST_DAYS - 1)
    hist = daily_df[daily_df["date"] >= cutoff].copy()
    keep = ["player_id", "date", target, "n_periods", "n_exercise_types",
            "height", "weight", "age", "position"] + \
           [c for c in daily_df.columns if c.startswith("has_")]
    hist = hist[[c for c in keep if c in hist.columns]].reset_index(drop=True)
    return hist


def _recursive_forecast(
    history: pd.DataFrame,
    plan: pd.DataFrame,
    target: str,
    bundle,
) -> pd.DataFrame:
    """Predict `target` for each date in `plan` using the recursive one-step-ahead approach."""
    model        = bundle.model
    scaler       = bundle.scaler
    feature_cols = bundle.feature_cols

    plan_work = plan.copy()

    for current_date in sorted(plan_work["date"].unique()):
        combined   = pd.concat([history, plan_work], ignore_index=True)
        featurized = _add_features(combined, target)

        day_rows = featurized[featurized["date"] == current_date].copy()
        missing_feature_cols = [col for col in feature_cols if col not in day_rows.columns]
        for col in missing_feature_cols:
            day_rows[col] = 0.0

        X_raw    = day_rows[feature_cols].values
        X_scaled = scaler.transform(X_raw)
        pred     = np.clip(np.expm1(model.predict(X_scaled)), 0, None)

        # Rest days always produce 0 load
        pred[day_rows["n_periods"].values == 0] = 0.0

        for pid, val in zip(day_rows["player_id"].values, pred, strict=False):
            mask = (plan_work["date"] == current_date) & (plan_work["player_id"] == pid)
            plan_work.loc[mask, target] = val

    return plan_work


def build_forecast(plan_days: list[dict]) -> dict:
    bundles = get_models_or_stop()
    player_data, all_pids, _, daily_df = load_player_data()

    last_active: pd.Timestamp = max(player_data[pid]["last_active"] for pid in all_pids)

    results: dict = {}
    for pid in all_pids:
        results[str(pid)] = {
            "position":      player_data[pid]["position"],
            "n_active_days": player_data[pid]["n_active_days"],
        }

    for target, bundle in bundles.items():
        history = _get_history(daily_df, target, last_active)

        plan_frame = _build_plan_frame(plan_days, all_pids, player_data, last_active, target)

        forecast_result = _recursive_forecast(history, plan_frame, target, bundle)

        for pid in all_pids:
            pdata     = player_data[pid]
            hist_grid = pdata["grid"]

            # Collect predicted daily loads in plan order
            pid_fore = forecast_result[forecast_result["player_id"] == pid].sort_values("date")
            fore_loads = pid_fore[target].fillna(0.0).values.astype(float)

            hist_loads = hist_grid[target].values.astype(float)
            full       = compute_acwr_with_forecast(hist_loads, fore_loads)
            n_hist     = len(hist_loads)
            chunk      = full.iloc[max(0, n_hist - HIST_SHOW):]

            hist_c = chunk[~chunk["is_forecast"]]
            fore_c = chunk[chunk["is_forecast"]]

            hist_start = pdata["last_active"] - pd.Timedelta(days=len(hist_c) - 1)

            def to_dates(base: pd.Timestamp, n: int) -> list[str]:
                return [(base + pd.Timedelta(days=i)).strftime("%d %b") for i in range(n)]

            def clean(vals) -> list:
                return [
                    None if (v is None or (isinstance(v, float) and np.isnan(v)))
                    else round(float(v), 3) for v in vals
                ]

            valid_fore = fore_c["acwr"].dropna()
            day15_val  = float(valid_fore.iloc[-1]) if len(valid_fore) > 0 else None

            results[str(pid)][target] = {
                "hist_dates": to_dates(hist_start, len(hist_c)),
                "hist_acwr":  clean(hist_c["acwr"].values),
                "fore_dates": to_dates(last_active + pd.Timedelta(days=1), len(fore_c)),
                "fore_acwr":  clean(fore_c["acwr"].values),
                "day15_acwr": round(day15_val, 3) if day15_val is not None else None,
                "day15_zone": classify_acwr_zone(day15_val) if day15_val is not None else "unknown",
            }

    return results
