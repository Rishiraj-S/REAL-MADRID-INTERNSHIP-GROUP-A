"""Regression tests for planner page helpers."""

from __future__ import annotations

from datetime import time

import pandas as pd

from app.pages import _coerce_calendar_boundary, _selection_bounds
from app.planning import build_planning_window


def test_coerce_calendar_boundary_keeps_date_only_values_on_same_day() -> None:
    """Date-only calendar payloads should not shift to the previous day because of timezone parsing."""
    parsed = _coerce_calendar_boundary("2025-07-01", time(hour=10, minute=0))

    assert parsed == pd.Timestamp("2025-07-01 10:00:00").to_pydatetime()


def test_selection_bounds_uses_clicked_day_for_all_day_cells() -> None:
    """Clicking a day cell should default the event to that exact calendar day."""
    last_active: pd.Timestamp = pd.Timestamp("2025-06-26")
    plan_dates = build_planning_window(last_active, horizon_days=15)

    start_dt, end_dt = _selection_bounds(
        {
            "allDay": True,
            "dateStr": "2025-07-01",
            "date": "2025-07-01T00:00:00.000Z",
        },
        plan_dates,
    )

    assert start_dt == pd.Timestamp("2025-07-01 10:00:00").to_pydatetime()
    assert end_dt == pd.Timestamp("2025-07-01 11:30:00").to_pydatetime()


def test_selection_bounds_converts_utc_all_day_payloads_to_madrid_date() -> None:
    """UTC callback timestamps should map back to the intended Madrid calendar day."""
    last_active: pd.Timestamp = pd.Timestamp("2025-06-26")
    plan_dates = build_planning_window(last_active, horizon_days=15)

    start_dt, end_dt = _selection_bounds(
        {
            "allDay": True,
            "date": "2025-06-30T22:00:00.000Z",
        },
        plan_dates,
    )

    assert start_dt == pd.Timestamp("2025-07-01 10:00:00").to_pydatetime()
    assert end_dt == pd.Timestamp("2025-07-01 11:30:00").to_pydatetime()


