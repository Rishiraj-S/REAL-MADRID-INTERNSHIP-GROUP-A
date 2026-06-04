"""Forecast construction for the Streamlit ACWR application."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.constants import SESSION_TYPES, TARGETS
from app.loaders import get_models_or_stop, load_player_data
from real_madrid_acwr.acwr import classify_acwr_zone, compute_acwr_with_forecast
from real_madrid_acwr.modeling.datapipeline import add_features, encode_dow

HIST_SHOW = 60  # days of history shown in the chart


def _build_plan_frame(
    plan_days: list[dict],
    all_pids: list,
    player_data: dict,
    last_active: pd.Timestamp,
    target: str,
) -> pd.DataFrame:
    """Convert coach plan to a DataFrame ready for model inference."""
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


def _direct_forecast(
    plan: pd.DataFrame,
    target: str,
    bundle,
) -> pd.DataFrame:
    """Predict `target` for all plan rows in a single pass (no recursion)."""
    model        = bundle.model
    scaler       = bundle.scaler
    feature_cols = bundle.feature_cols

    featurized = encode_dow(add_features(plan.copy()))

    for col in feature_cols:
        if col not in featurized.columns:
            featurized[col] = 0.0

    X_raw    = featurized[feature_cols].values
    X_scaled = scaler.transform(X_raw)
    pred     = np.clip(np.expm1(model.predict(X_scaled)), 0, None)

    pred[featurized["n_periods"].values == 0] = 0.0

    result        = plan.copy()
    result[target] = pred
    return result


def build_forecast(plan_days: list[dict]) -> dict:
    bundles = get_models_or_stop()
    player_data, all_pids, _, _ = load_player_data()

    last_active: pd.Timestamp = max(player_data[pid]["last_active"] for pid in all_pids)

    results: dict = {}
    for pid in all_pids:
        results[str(pid)] = {
            "position":      player_data[pid]["position"],
            "n_active_days": player_data[pid]["n_active_days"],
        }

    for target, bundle in bundles.items():
        plan_frame = _build_plan_frame(plan_days, all_pids, player_data, last_active, target)

        forecast_result = _direct_forecast(plan_frame, target, bundle)

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

            # Activities per hist day from grid has_* columns
            has_cols = sorted([c for c in hist_grid.columns if c.startswith("has_")])
            act_lookup = hist_grid.set_index("date")[has_cols] if has_cols else None
            hist_activities: list[list[str]] = []
            for i in range(len(hist_c)):
                dt = hist_start + pd.Timedelta(days=i)
                if act_lookup is not None and dt in act_lookup.index:
                    row = act_lookup.loc[dt]
                    hist_activities.append([h[4:] for h in has_cols if row[h] > 0])
                else:
                    hist_activities.append([])

            # Activities per forecast day from plan_days
            fore_activities: list[list[str]] = [
                [] if plan_days[i].get("is_rest", False)
                else [s for s in SESSION_TYPES if plan_days[i].get(s, False)]
                for i in range(min(len(fore_c), len(plan_days)))
            ]

            valid_fore = fore_c["acwr"].dropna()
            day15_val  = float(valid_fore.iloc[-1]) if len(valid_fore) > 0 else None

            results[str(pid)][target] = {
                "hist_dates":      to_dates(hist_start, len(hist_c)),
                "hist_acwr":       clean(hist_c["acwr"].values),
                "hist_load":       clean(hist_c["load"].values),
                "hist_activities": hist_activities,
                "fore_dates":      to_dates(last_active + pd.Timedelta(days=1), len(fore_c)),
                "fore_acwr":       clean(fore_c["acwr"].values),
                "fore_load":       clean(fore_c["load"].values),
                "fore_activities": fore_activities,
                "day15_acwr":      round(day15_val, 3) if day15_val is not None else None,
                "day15_zone":      classify_acwr_zone(day15_val) if day15_val is not None else "unknown",
            }

    return results
