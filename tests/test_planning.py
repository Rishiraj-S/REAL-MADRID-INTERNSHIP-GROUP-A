"""Tests for the interactive planner helper layer."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from app.planning import (
    build_event_record,
    build_plan_days_from_events,
    build_planning_window,
    plan_signature,
    summarise_plan,
)


def test_build_plan_days_from_events_merges_multiple_sessions_on_same_day() -> None:
    """Multiple planned events on one date should union their session types for forecasting."""
    last_active = pd.Timestamp("2025-01-01")
    plan_dates = build_planning_window(last_active, horizon_days=3)
    events = [
        build_event_record(
            event_id="evt-1",
            start_dt=datetime(2025, 1, 2, 10, 0),
            end_dt=datetime(2025, 1, 2, 11, 30),
            session_types=["G"],
            location="Valdebebas",
            notes="",
        ),
        build_event_record(
            event_id="evt-2",
            start_dt=datetime(2025, 1, 2, 15, 0),
            end_dt=datetime(2025, 1, 2, 16, 0),
            session_types=["TEC", "MATCH"],
            location="Santiago Bernabéu",
            notes="",
        ),
    ]

    plan_days = build_plan_days_from_events(events, plan_dates)

    assert plan_days[0] == {
        "is_rest": False,
        "G": True,
        "TAC": False,
        "BP": False,
        "TEC": True,
        "MATCH": True,
    }
    assert plan_days[1]["is_rest"] is True
    assert plan_days[2]["is_rest"] is True


def test_summarise_plan_counts_active_rest_and_match_days() -> None:
    """Planner summary cards should reflect the derived daily plan state."""
    plan_days = [
        {"is_rest": False, "G": True, "TAC": False, "BP": False, "TEC": False, "MATCH": False},
        {"is_rest": False, "G": False, "TAC": False, "BP": False, "TEC": False, "MATCH": True},
        {"is_rest": True, "G": False, "TAC": False, "BP": False, "TEC": False, "MATCH": False},
    ]

    summary = summarise_plan(plan_days)

    assert summary == {
        "total_days": 3,
        "active_days": 2,
        "rest_days": 1,
        "match_days": 1,
    }


def test_build_plan_days_ignores_events_outside_forecast_window() -> None:
    """Only events inside the active 15-day horizon should affect the forecast plan."""
    last_active = pd.Timestamp("2025-01-10")
    plan_dates = build_planning_window(last_active, horizon_days=2)
    events = [
        build_event_record(
            event_id="before-window",
            start_dt=datetime(2025, 1, 10, 9, 0),
            end_dt=datetime(2025, 1, 10, 10, 0),
            session_types=["MATCH"],
            location="Away",
            notes="",
        ),
        build_event_record(
            event_id="in-window",
            start_dt=datetime(2025, 1, 11, 11, 0),
            end_dt=datetime(2025, 1, 11, 12, 0),
            session_types=["TAC"],
            location="Valdebebas",
            notes="",
        ),
        build_event_record(
            event_id="after-window",
            start_dt=datetime(2025, 1, 20, 11, 0),
            end_dt=datetime(2025, 1, 20, 12, 0),
            session_types=["G"],
            location="Gym",
            notes="",
        ),
    ]

    plan_days = build_plan_days_from_events(events, plan_dates)

    assert plan_days == [
        {"is_rest": False, "G": False, "TAC": True, "BP": False, "TEC": False, "MATCH": False},
        {"is_rest": True, "G": False, "TAC": False, "BP": False, "TEC": False, "MATCH": False},
    ]


def test_plan_signature_is_stable_for_equivalent_plans() -> None:
    """The stale-forecast check should ignore ordering noise and compare the daily plan only."""
    plan_days_a = [
        {"is_rest": False, "G": True, "TAC": False, "BP": False, "TEC": False, "MATCH": False},
        {"is_rest": True, "G": False, "TAC": False, "BP": False, "TEC": False, "MATCH": False},
    ]
    plan_days_b = [
        {"MATCH": False, "TEC": False, "BP": False, "TAC": False, "G": True, "is_rest": False},
        {"MATCH": False, "TEC": False, "BP": False, "TAC": False, "G": False, "is_rest": True},
    ]

    assert plan_signature(plan_days_a) == plan_signature(plan_days_b)

