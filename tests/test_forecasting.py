"""Tests for Streamlit forecast assembly helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from app.forecasting import _build_plan_frame, _direct_forecast


class _IdentityScaler:
    def transform(self, values: np.ndarray) -> np.ndarray:
        return values


class _ZeroModel:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.zeros(len(values))


def test_build_plan_frame_includes_all_target_columns() -> None:
    """Planned rows should include every load target expected by trained bundles."""
    profile = pd.Series({"height": 180, "weight": 75, "age": 24, "position": "Forward"})
    plan_frame = _build_plan_frame(
        plan_days=[{"is_rest": False, "G": True, "TAC": False, "BP": False, "TEC": False, "MATCH": False}],
        all_pids=[7],
        player_data={7: {"profile": profile}},
        last_active=pd.Timestamp("2025-01-01"),
        target="accelerations",
    )

    assert {"total_distance", "accelerations", "sprint_distance"}.issubset(plan_frame.columns)
    assert pd.isna(plan_frame.loc[0, "accelerations"])
    assert plan_frame.loc[0, "total_distance"] == 0.0
    assert plan_frame.loc[0, "sprint_distance"] == 0.0


def test_direct_forecast_handles_missing_feature_columns() -> None:
    """Forecasting should tolerate bundles that expect columns absent from the plan frame."""
    plan = pd.DataFrame(
        {
            "player_id": [7],
            "date": pd.to_datetime(["2025-01-03"]),
            "accelerations": [np.nan],
            "sprint_distance": [0.0],
            "n_periods": [1],
            "n_exercise_types": [1],
            "height": [180.0],
            "weight": [75.0],
            "age": [24.0],
            "position": ["Forward"],
            "has_G": [1],
            "has_TAC": [0],
            "has_BP": [0],
            "has_TEC": [0],
            "has_MATCH": [0],
        }
    )
    bundle = SimpleNamespace(
        model=_ZeroModel(),
        scaler=_IdentityScaler(),
        feature_cols=[
            "sprint_distance",
            "n_periods",
            "n_exercise_types",
            "height",
            "weight",
            "age",
            "has_BP",
            "has_G",
            "has_MATCH",
            "has_TAC",
            "has_TEC",
            "day_of_week",
            "microcycle_load_sum",
            "microcycle_load_std_dev",
            "monotony",
            "strain",
            "acute_load",
            "chronic_load",
            "training_stress_balance",
            "acwr",
            "load_lag_1",
            "load_lag_3",
            "load_lag_5",
            "load_lag_7",
            "load_lag_14",
            "load_ma_3",
            "load_ma_7",
            "load_ma_14",
        ],
    )

    result = _direct_forecast(plan, "accelerations", bundle)

    assert result.loc[0, "accelerations"] == 0.0
