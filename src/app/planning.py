"""Planning helpers for the Streamlit scheduling workspace."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime, time, timedelta

import pandas as pd

from app.constants import SESSION_LABELS, SESSION_TYPES

PLANNING_HORIZON_DAYS = 15
DEFAULT_EVENT_DURATION_MINUTES = 90
DEFAULT_EVENT_START_TIME = time(hour=10, minute=0)
LOCATION_OPTIONS = [
    "Valdebebas",
    "Santiago Bernabéu",
    "Away",
    "Gym",
    "Recovery Centre",
    "Travel",
    "Other",
]


EventDict = dict[str, object]
PlanDay = dict[str, bool]


def _parse_datetime_value(value: object) -> datetime:
    """Parse a datetime-like value into a naive Python datetime."""
    parsed = pd.Timestamp(str(value)).to_pydatetime()
    if parsed.tzinfo is not None:
        return parsed.replace(tzinfo=None)
    return parsed


def _mapping_session_types(event: Mapping[str, object]) -> list[str]:
    """Return normalized session types from a generic event mapping."""
    raw_types = event.get("session_types", [])
    if isinstance(raw_types, list):
        return normalise_event_types(raw_types)
    return []


def build_planning_window(last_active: pd.Timestamp, horizon_days: int = PLANNING_HORIZON_DAYS) -> list[pd.Timestamp]:
    """Return the forecast window dates starting the day after the latest observed load."""
    start = last_active.normalize() + pd.Timedelta(days=1)
    return [start + pd.Timedelta(days=offset) for offset in range(horizon_days)]


def build_plan_date_labels(plan_dates: Iterable[pd.Timestamp]) -> list[str]:
    """Format planner dates exactly as the legacy results page expects."""
    return [dt.strftime("%a %d %b") for dt in plan_dates]


def default_event_bounds(plan_dates: list[pd.Timestamp]) -> tuple[datetime, datetime]:
    """Provide a stable default start/end time for the event creation dialog."""
    first_day = plan_dates[0].date()
    start_dt = datetime.combine(first_day, DEFAULT_EVENT_START_TIME)
    end_dt = start_dt + timedelta(minutes=DEFAULT_EVENT_DURATION_MINUTES)
    return start_dt, end_dt


def combine_date_and_time(day: date, clock: time) -> datetime:
    """Combine separate date/time inputs into a naive local datetime."""
    return datetime.combine(day, clock)


def clamp_date_to_window(day: date, plan_dates: list[pd.Timestamp]) -> date:
    """Clamp an arbitrary date into the supported 15-day planning window."""
    lower = plan_dates[0].date()
    upper = plan_dates[-1].date()
    return min(max(day, lower), upper)


def normalise_event_types(raw_types: Iterable[object]) -> list[str]:
    """Return session types in model-compatible order."""
    selected = {str(value) for value in raw_types if str(value) in SESSION_TYPES}
    return [session_type for session_type in SESSION_TYPES if session_type in selected]


def compose_event_title(session_types: Iterable[object], location: str) -> str:
    """Build a concise event title for the calendar surface."""
    ordered_types = normalise_event_types(session_types)
    label = " / ".join(ordered_types) if ordered_types else "Session"
    if location and location != "Other":
        return f"{label} · {location}"
    return label


def build_event_record(
    *,
    event_id: str,
    start_dt: datetime,
    end_dt: datetime,
    session_types: Iterable[object],
    location: str,
    notes: str,
) -> EventDict:
    """Create the persisted planner event structure used across the page."""
    ordered_types = normalise_event_types(session_types)
    clean_location = location.strip()
    return {
        "id": event_id,
        "title": compose_event_title(ordered_types, clean_location),
        "start": start_dt.isoformat(timespec="minutes"),
        "end": end_dt.isoformat(timespec="minutes"),
        "allDay": False,
        "session_types": ordered_types,
        "location": clean_location,
        "notes": notes.strip(),
    }


def event_start_datetime(event: Mapping[str, object]) -> datetime:
    """Parse an event start timestamp from persisted state or calendar payloads."""
    return _parse_datetime_value(event["start"])


def event_end_datetime(event: Mapping[str, object]) -> datetime:
    """Parse an event end timestamp and add a default duration if the source omits it."""
    end_value = event.get("end")
    if end_value:
        return _parse_datetime_value(end_value)
    return event_start_datetime(event) + timedelta(minutes=DEFAULT_EVENT_DURATION_MINUTES)


def sort_events(events: Iterable[Mapping[str, object]]) -> list[EventDict]:
    """Return planner events in display order without mutating the source list."""
    return sorted((dict(event) for event in events), key=lambda item: (str(item["start"]), str(item["id"])))


def build_calendar_events(events: Iterable[Mapping[str, object]]) -> list[EventDict]:
    """Project planner state into FullCalendar-compatible event objects."""
    calendar_events: list[EventDict] = []
    for event in sort_events(events):
        event_types = _mapping_session_types(event)
        calendar_events.append(
            {
                "id": str(event["id"]),
                "title": compose_event_title(event_types, str(event.get("location", ""))),
                "start": str(event["start"]),
                "end": str(event.get("end") or ""),
                "allDay": bool(event.get("allDay", False)),
                "extendedProps": {
                    "session_types": event_types,
                    "location": str(event.get("location", "")),
                    "notes": str(event.get("notes", "")),
                },
            }
        )
    return calendar_events


def build_plan_days_from_events(
    events: Iterable[Mapping[str, object]],
    plan_dates: list[pd.Timestamp],
) -> list[PlanDay]:
    """Collapse detailed events into the daily session flags expected by forecasting."""
    event_dates: dict[date, set[str]] = {}
    for event in events:
        start_day = event_start_datetime(event).date()
        selected_types = _mapping_session_types(event)
        if not selected_types:
            continue
        event_dates.setdefault(start_day, set()).update(selected_types)

    plan_days: list[PlanDay] = []
    for plan_date in plan_dates:
        day_types = event_dates.get(plan_date.date(), set())
        plan_days.append(
            {
                "is_rest": len(day_types) == 0,
                **{session_type: session_type in day_types for session_type in SESSION_TYPES},
            }
        )
    return plan_days


def summarise_plan(plan_days: Iterable[Mapping[str, bool]]) -> dict[str, int]:
    """Build lightweight KPIs for the planner header."""
    days = list(plan_days)
    active_days = sum(1 for day in days if not day.get("is_rest", False))
    return {
        "total_days": len(days),
        "active_days": active_days,
        "rest_days": sum(1 for day in days if day.get("is_rest", False)),
        "match_days": sum(1 for day in days if day.get("MATCH", False)),
    }


def plan_signature(plan_days: Iterable[Mapping[str, bool]]) -> str:
    """Generate a stable signature so the UI can detect stale forecast output."""
    payload = [
        {session_type: bool(day.get(session_type, False)) for session_type in [*SESSION_TYPES, "is_rest"]}
        for day in plan_days
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def describe_event(event: Mapping[str, object]) -> str:
    """Return a readable summary for secondary planner surfaces."""
    start_dt = event_start_datetime(event)
    types = _mapping_session_types(event)
    type_labels = [SESSION_LABELS.get(session_type, session_type) for session_type in types]
    return (
        f"{start_dt.strftime('%a %d %b')}"
        f" · {', '.join(type_labels) if type_labels else 'No session types'}"
    )




