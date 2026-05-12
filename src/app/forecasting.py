"""Forecast construction for the Streamlit ACWR application."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.constants import SESSION_TYPES, TARGETS
from app.loaders import get_models_or_stop, load_player_data
from real_madrid_acwr.acwr import classify_acwr_zone, compute_acwr_with_forecast

FeatureValue = int | float


def build_forecast(plan_days):
    import xgboost as xgb

    models = get_models_or_stop()
    player_data, all_pids, _ = load_player_data()

    HIST_SHOW = 60
    results   = {}

    for pid in all_pids:
        pdata   = player_data[pid]
        profile = pdata["profile"]
        grid    = pdata["grid"]

        player_feats: dict[str, FeatureValue] = {
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
        pid_feats: dict[str, FeatureValue] = {f"pid_{p}": int(p == pid) for p in all_pids}

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

        forecast_loads: dict[str, list[float]] = {m: [] for m in TARGETS}

        for d, day in enumerate(plan_days, start=1):
            dsla = min(d - prev_active_d, 21)  # capped at 21 to match training distribution; extrapolating beyond degrades predictions
            dslm = min(d - prev_match_d, 21)

            if day["is_rest"]:
                for m in TARGETS:
                    forecast_loads[m].append(0.0)
                continue

            sess: dict[str, FeatureValue] = {
                "has_G":           int(day.get("G",     False)),
                "has_TAC":         int(day.get("TAC",   False)),
                "has_BP":          int(day.get("BP",    False)),
                "has_TEC":         int(day.get("TEC",   False)),
                "has_MATCH":       int(day.get("MATCH", False)),
                "n_session_types": sum(int(day.get(t, False)) for t in SESSION_TYPES),
            }
            day_feats: dict[str, FeatureValue] = {
                "days_since_start":        last_dss + d,
                "days_since_last_activity": float(dsla),
                "days_since_last_match":    float(dslm),
            }

            for target, art in models.items():
                fc   = art.feature_cols
                feat: dict[str, FeatureValue] = {c: 0 for c in fc}
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
