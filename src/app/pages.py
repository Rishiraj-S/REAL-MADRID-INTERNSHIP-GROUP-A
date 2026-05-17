"""Streamlit page renderers for the ACWR monitor."""

from __future__ import annotations

import base64
import importlib.util
import json
from datetime import date, datetime, time
from typing import cast
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from app.charts import build_acwr_chart
from app.constants import (
    PAGES,
    SESSION_COLORS,
    SESSION_LABELS,
    SESSION_TYPES,
    TARGET_META,
    TARGETS,
    ZONE_COLORS,
    ZONE_LABELS,
)
from app.forecasting import build_forecast
from app.loaders import get_models_or_stop, load_player_data
from app.planning import (
    LOCATION_OPTIONS,
    build_calendar_events,
    build_event_record,
    build_plan_date_labels,
    build_plan_days_from_events,
    build_planning_window,
    combine_date_and_time,
    default_event_bounds,
    describe_event,
    event_end_datetime,
    event_start_datetime,
    normalise_event_types,
    plan_signature,
    sort_events,
    summarise_plan,
)
from real_madrid_acwr.config import STATIC_DIR

APP_TIMEZONE = ZoneInfo("Europe/Madrid")


def _render_standard_header(title: str, subtitle: str) -> None:
    """Render a consistent top-of-page header."""
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-header-accent"></div>
            <div class="page-title">{title}</div>
            <div class="page-sub">{subtitle}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def page_dashboard() -> None:
    """Render the squad-level ACWR status dashboard."""
    player_data, all_pids, current_acwr = load_player_data()
    get_models_or_stop()
    last_date = player_data[all_pids[0]]["last_active"].strftime("%d %B %Y")

    st.markdown(f"""
    <div class="page-header">
        <div class="page-header-accent"></div>
        <div class="page-title">Squad <span>ACWR</span> Dashboard</div>
        <div class="page-sub">
            Data through <strong style="color:#334D6E">{last_date}</strong>
            &nbsp;&middot;&nbsp; {len(all_pids)} players tracked
            &nbsp;&middot;&nbsp; 3 load metrics
        </div>
    </div>""", unsafe_allow_html=True)

    n_danger = sum(1 for pid in all_pids for metric in TARGETS if current_acwr[pid][metric]["zone"] == "danger")
    n_caution = sum(1 for pid in all_pids for metric in TARGETS if current_acwr[pid][metric]["zone"] == "caution")
    n_optimal = sum(1 for pid in all_pids for metric in TARGETS if current_acwr[pid][metric]["zone"] == "optimal")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class="stat-card" style="border-top-color:#EE324E">
        <div class="stat-val" style="color:#EE324E">{n_danger}</div>
        <div class="stat-lbl">Danger Flags</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="stat-card" style="border-top-color:#F59E0B">
        <div class="stat-val" style="color:#F59E0B">{n_caution}</div>
        <div class="stat-lbl">Caution Flags</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="stat-card" style="border-top-color:#10B981">
        <div class="stat-val" style="color:#10B981">{n_optimal}</div>
        <div class="stat-lbl">Optimal Flags</div>
    </div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="stat-card" style="border-top-color:#00529F">
        <div class="stat-val" style="color:#00529F">{len(all_pids)}</div>
        <div class="stat-lbl">Players Tracked</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label" style="font-size:1rem;letter-spacing:0.5px">Risk Zones</div>',
        unsafe_allow_html=True,
    )

    legend_cols = st.columns(4)
    for index, (zone, rng) in enumerate([
        ("undertraining", "ACWR < 0.8"),
        ("optimal", "ACWR 0.8 – 1.3"),
        ("caution", "ACWR 1.3 – 1.5"),
        ("danger", "ACWR ≥ 1.5"),
    ]):
        color = ZONE_COLORS[zone]
        label = ZONE_LABELS[zone]
        legend_cols[index].markdown(f"""
        <div class="zone-pill" style="color:{color};border-color:{color};background:{color}15">
            <span class="zone-dot" style="background:{color}"></span>
            {label} &nbsp; <span style="font-weight:400;opacity:0.8">{rng}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div class="section-label" style="font-size:1rem;letter-spacing:0.5px">Player Status — Current ACWR</div>',
        unsafe_allow_html=True,
    )

    zone_order = ["danger", "caution", "undertraining", "optimal", "unknown"]
    player_cols = st.columns(4)
    for index, pid in enumerate(all_pids):
        pdata = player_data[pid]
        acwr = current_acwr[pid]
        worst_zone = min((acwr[metric]["zone"] for metric in TARGETS), key=zone_order.index)

        rows_html = ""
        for metric in TARGETS:
            metric_acwr = acwr[metric]
            color = ZONE_COLORS[metric_acwr["zone"]]
            value_string = f"{metric_acwr['value']:.2f}" if metric_acwr["value"] is not None else "—"
            rows_html += f"""
            <div class="metric-row">
                <span class="metric-lbl" style="color:{TARGET_META[metric]['color']}">{TARGET_META[metric]['label']}</span>
                <div class="metric-rhs">
                    <span class="metric-val" style="color:{color}">{value_string}</span>
                    <span class="metric-badge" style="color:{color};border-color:{color};background:{color}18">
                        {ZONE_LABELS[metric_acwr['zone']]}
                    </span>
                </div>
            </div>"""

        with player_cols[index % 4]:
            st.markdown(f"""
            <div class="player-card {worst_zone}">
                <div class="card-id">{pid}</div>
                <div class="card-pos">{pdata['position']}</div>
                <hr class="card-rule">
                {rows_html}
            </div>""", unsafe_allow_html=True)


def _legacy_plan_days_to_events(plan_days: list[dict[str, bool]], plan_dates: list[pd.Timestamp]) -> list[dict[str, object]]:
    """Lift the legacy day-grid state into the richer event model once per session."""
    migrated_events: list[dict[str, object]] = []
    for index, day in enumerate(plan_days):
        session_types = [session_type for session_type in SESSION_TYPES if day.get(session_type, False)]
        if not session_types:
            continue
        migrated_events.append(
            build_event_record(
                event_id=f"legacy-{index}",
                start_dt=datetime.combine(plan_dates[index].date(), time(hour=10, minute=0)),
                end_dt=datetime.combine(plan_dates[index].date(), time(hour=11, minute=30)),
                session_types=session_types,
                location="Valdebebas",
                notes="Migrated from the previous day-based planning grid.",
            )
        )
    return migrated_events


def _initialise_planner_state(plan_dates: list[pd.Timestamp]) -> None:
    """Bootstrap the planner-specific session state."""
    if "plan_events" not in st.session_state:
        legacy_plan_days = st.session_state.get("plan_days", [])
        if isinstance(legacy_plan_days, list):
            st.session_state.plan_events = _legacy_plan_days_to_events(legacy_plan_days, plan_dates)
        else:
            st.session_state.plan_events = []
    st.session_state.setdefault("planner_dialog_request", None)
    st.session_state.setdefault("_calendar_action_token", None)


def _event_session_types(event: dict[str, object]) -> list[str]:
    """Return normalized session types from an event record."""
    raw_types = event.get("session_types", [])
    if isinstance(raw_types, list):
        return normalise_event_types(raw_types)
    return []


def _coerce_timestamp(value: object) -> datetime:
    """Parse a calendar payload timestamp into a naive Python datetime."""
    parsed = pd.Timestamp(str(value)).to_pydatetime()
    if not isinstance(parsed, datetime):
        raise ValueError(f"Unsupported datetime payload: {value!r}")
    if parsed.tzinfo is not None:
        return parsed.astimezone(APP_TIMEZONE).replace(tzinfo=None)
    return parsed


def _coerce_date_input(value: object, fallback: date) -> date:
    """Normalize Streamlit date input output into a concrete single date."""
    if isinstance(value, date):
        return value
    return fallback


def _coerce_time_input(value: object, fallback: time) -> time:
    """Normalize Streamlit time input output into a concrete time."""
    if isinstance(value, time):
        return value
    return fallback


def _coerce_calendar_boundary(value: object, fallback_time: time) -> datetime:
    """Parse a FullCalendar boundary value without shifting date-only payloads across timezones."""
    if isinstance(value, str):
        stripped = value.strip()
        if len(stripped) == 10 and stripped[4] == "-" and stripped[7] == "-":
            return datetime.combine(date.fromisoformat(stripped), fallback_time)
    return _coerce_timestamp(value)


def _badge_html(session_types: list[str]) -> str:
    """Render HTML badges for session types using the application palette."""
    return "".join(
        (
            f'<span class="session-chip" style="border-color:{SESSION_COLORS[session_type]};'
            f'background:{SESSION_COLORS[session_type]}14;color:{SESSION_COLORS[session_type]}">'
            f'{session_type}</span>'
        )
        for session_type in session_types
    )


def _event_lookup(events: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Index the in-memory event list by identifier for fast retrieval."""
    return {str(event["id"]): event for event in events}


def _upsert_event(updated_event: dict[str, object]) -> None:
    """Insert or replace an event inside the persisted planner state."""
    remaining_events = [
        event for event in st.session_state.plan_events if str(event["id"]) != str(updated_event["id"])
    ]
    remaining_events.append(updated_event)
    st.session_state.plan_events = sort_events(remaining_events)


def _remove_event(event_id: str) -> None:
    """Delete an event by identifier from the planner state."""
    st.session_state.plan_events = [
        event for event in st.session_state.plan_events if str(event["id"]) != event_id
    ]


def _normalise_callback_payload(payload: object) -> dict[str, object]:
    """Return a safe mapping for calendar callback payloads."""
    return payload if isinstance(payload, dict) else {}


def _extract_calendar_callback(calendar_state: object) -> tuple[str | None, dict[str, object]]:
    """Read the latest callback emitted by the calendar component."""
    if not isinstance(calendar_state, dict):
        return None, {}

    callback_name = calendar_state.get("callback")
    if isinstance(callback_name, str):
        return callback_name, _normalise_callback_payload(calendar_state.get(callback_name))

    for candidate in ("dateClick", "select", "eventClick", "eventChange"):
        if candidate in calendar_state:
            return candidate, _normalise_callback_payload(calendar_state.get(candidate))

    return None, {}


def _calendar_action_token(callback_name: str | None, payload: dict[str, object]) -> str | None:
    """Build a stable token so each calendar callback is handled once per rerun cycle."""
    if callback_name is None:
        return None
    return json.dumps({"callback": callback_name, "payload": payload}, sort_keys=True, default=str)


def _extract_event_id(payload: dict[str, object]) -> str | None:
    """Extract an event id from the supported callback payload shapes."""
    event_payload = payload.get("event")
    if isinstance(event_payload, dict) and event_payload.get("id") is not None:
        return str(event_payload["id"])
    if payload.get("id") is not None:
        return str(payload["id"])
    return None


def _calendar_payload_to_event(
    payload: dict[str, object],
    existing_event: dict[str, object] | None,
) -> dict[str, object] | None:
    """Translate a drag/drop callback payload back into the persisted event schema."""
    event_payload = payload.get("event") if isinstance(payload.get("event"), dict) else payload
    if not isinstance(event_payload, dict) or event_payload.get("id") is None or event_payload.get("start") is None:
        return None

    extended_props = event_payload.get("extendedProps")
    if not isinstance(extended_props, dict):
        extended_props = {}

    fallback_event = existing_event or {}
    start_dt = _coerce_timestamp(event_payload["start"])
    end_value = event_payload.get("end") or fallback_event.get("end")
    end_dt = _coerce_timestamp(end_value) if end_value is not None else start_dt + pd.Timedelta(minutes=90).to_pytimedelta()

    fallback_types = fallback_event.get("session_types", [])
    session_types = normalise_event_types(extended_props.get("session_types", fallback_types))
    location = str(extended_props.get("location", fallback_event.get("location", "Other")))
    notes = str(extended_props.get("notes", fallback_event.get("notes", "")))
    return build_event_record(
        event_id=str(event_payload["id"]),
        start_dt=start_dt,
        end_dt=end_dt,
        session_types=session_types,
        location=location,
        notes=notes,
    )


def _selection_bounds(payload: dict[str, object], plan_dates: list[pd.Timestamp]) -> tuple[datetime, datetime]:
    """Convert calendar selection payloads into dialog defaults."""
    default_start, default_end = default_event_bounds(plan_dates)
    is_all_day = bool(payload.get("allDay", False))
    start_value = (
        payload.get("startStr")
        or payload.get("dateStr")
        or payload.get("start")
        or payload.get("date")
    )
    end_value = payload.get("endStr") or payload.get("end")
    if start_value is None:
        return default_start, default_end

    start_dt = _coerce_calendar_boundary(start_value, default_start.time())
    if is_all_day:
        start_dt = datetime.combine(start_dt.date(), default_start.time())

    if end_value is not None and not is_all_day:
        end_dt = _coerce_calendar_boundary(end_value, default_end.time())
    else:
        end_dt = start_dt + pd.Timedelta(minutes=90).to_pytimedelta()

    start_dt = start_dt.replace(second=0, microsecond=0)
    end_dt = end_dt.replace(second=0, microsecond=0)
    if end_dt <= start_dt:
        end_dt = start_dt + pd.Timedelta(minutes=90).to_pytimedelta()
    return start_dt, end_dt


def _open_event_dialog(request: dict[str, object]) -> None:
    """Store the next dialog request and trigger an immediate rerender."""
    st.session_state.planner_dialog_request = request
    st.rerun()


def _render_session_legend() -> None:
    """Show the session palette used across the planner surfaces."""
    st.markdown(
        '<div class="section-label" style="font-size:1rem;letter-spacing:0.5px">Session Types</div>',
        unsafe_allow_html=True,
    )
    legend_cols = st.columns(len(SESSION_TYPES))
    for index, session_type in enumerate(SESSION_TYPES):
        legend_cols[index].markdown(
            f"""
            <div class="planner-legend-pill" style="border-color:{SESSION_COLORS[session_type]};color:{SESSION_COLORS[session_type]};background:{SESSION_COLORS[session_type]}12">
                <span class="planner-legend-dot" style="background:{SESSION_COLORS[session_type]}"></span>
                {session_type} · {SESSION_LABELS[session_type]}
            </div>""",
            unsafe_allow_html=True,
        )


def _render_planner_metrics(summary: dict[str, int], event_count: int) -> None:
    """Render top-line planner KPIs."""
    metrics = [
        ("Planned Events", event_count, "#00529F"),
        ("Active Days", summary["active_days"], "#FEBE10"),
        ("Match Days", summary["match_days"], "#EE324E"),
        ("Rest Days", summary["rest_days"], "#64748B"),
    ]
    metric_cols = st.columns(4)
    for column, (label, value, color) in zip(metric_cols, metrics, strict=False):
        column.markdown(
            f"""
            <div class="stat-card planner-stat-card" style="border-top-color:{color}">
                <div class="stat-val" style="color:{color}">{value}</div>
                <div class="stat-lbl">{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )


def _build_calendar_payload(plan_dates: list[pd.Timestamp], plan_events: list[dict[str, object]]) -> list[dict[str, object]]:
    """Combine editable events with a background highlight for the active forecast window."""
    highlight_end = (plan_dates[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    planning_window_event = {
        "id": "planning-window-highlight",
        "title": "Forecast window",
        "start": plan_dates[0].strftime("%Y-%m-%d"),
        "end": highlight_end,
        "allDay": True,
        "display": "background",
        "editable": False,
        "overlap": True,
        "classNames": ["planning-window-highlight"],
    }
    return [planning_window_event, *build_calendar_events(plan_events)]


def _render_calendar(plan_dates: list[pd.Timestamp], plan_events: list[dict[str, object]]) -> object:
    """Render the interactive FullCalendar instance."""
    if importlib.util.find_spec("streamlit_calendar") is None:
        st.error(
            "The interactive calendar dependency is missing. Run `python3 -m pip install -e .` "
            "to install the updated project dependencies."
        )
        return None

    from streamlit_calendar import calendar as calendar_fn

    calendar_options = {
        "editable": True,
        "selectable": True,
        "eventStartEditable": True,
        "eventDurationEditable": True,
        "firstDay": 1,
        "initialView": "dayGridMonth",
        "headerToolbar": {
            "left": "today prev,next",
            "center": "title",
            "right": "dayGridMonth,timeGridWeek",
        },
        "initialDate": plan_dates[0].strftime("%Y-%m-%d"),
        "height": 760,
        "nowIndicator": True,
        "dayMaxEvents": 3,
        "slotMinTime": "06:00:00",
        "slotMaxTime": "23:00:00",
        "allDaySlot": False,
        "selectMirror": True,
        "eventTimeFormat": {"hour": "2-digit", "minute": "2-digit", "meridiem": False},
    }
    custom_css = """
    .fc {
        font-family: Inter, system-ui, sans-serif;
        background: #FFFFFF;
        border-radius: 14px;
        padding: 0.5rem;
    }
    .fc .fc-toolbar-title {
        color: #0F172A;
        font-size: 1.05rem;
        font-weight: 800;
    }
    .fc .fc-button {
        background: #FFFFFF;
        border: 1px solid #D7E4F1;
        color: #00529F;
        box-shadow: none;
    }
    .fc .fc-button-primary:not(:disabled).fc-button-active,
    .fc .fc-button-primary:not(:disabled):active {
        background: #00529F;
        border-color: #00529F;
        color: #FFFFFF;
    }
    .fc .fc-daygrid-event,
    .fc .fc-timegrid-event {
        border: none;
        border-radius: 8px;
        padding: 2px 4px;
        background: linear-gradient(135deg, #00529F 0%, #3A78B6 100%);
    }
    .fc .fc-col-header-cell-cushion,
    .fc .fc-daygrid-day-number {
        color: #334D6E;
        font-weight: 700;
    }
    .fc .fc-highlight {
        background: rgba(254, 190, 16, 0.18);
    }
    .fc .planning-window-highlight {
        background: rgba(254, 190, 16, 0.14) !important;
    }
    """
    return calendar_fn(
        events=_build_calendar_payload(plan_dates, plan_events),
        options=calendar_options,
        custom_css=custom_css,
        callbacks=["dateClick", "select", "eventClick", "eventChange"],
        key="session_planner_calendar",
    )


def _handle_calendar_interactions(plan_dates: list[pd.Timestamp]) -> None:
    """Handle create/edit/drag actions emitted by the interactive calendar."""
    calendar_state = _render_calendar(plan_dates, sort_events(st.session_state.plan_events))
    callback_name, payload = _extract_calendar_callback(calendar_state)
    action_token = _calendar_action_token(callback_name, payload)
    if action_token is None or action_token == st.session_state.get("_calendar_action_token"):
        return

    st.session_state._calendar_action_token = action_token
    events_by_id = _event_lookup(sort_events(st.session_state.plan_events))

    if callback_name in {"dateClick", "select"}:
        start_dt, end_dt = _selection_bounds(payload, plan_dates)
        _open_event_dialog(
            {
                "mode": "create",
                "start": start_dt.isoformat(timespec="minutes"),
                "end": end_dt.isoformat(timespec="minutes"),
            }
        )

    if callback_name == "eventClick":
        event_id = _extract_event_id(payload)
        if event_id is not None:
            _open_event_dialog({"mode": "edit", "event_id": event_id})
        return

    if callback_name == "eventChange":
        event_id = _extract_event_id(payload)
        existing_event = events_by_id.get(event_id or "")
        if existing_event is None:
            return

        updated_event = _calendar_payload_to_event(payload, existing_event)
        if updated_event is not None:
            _upsert_event(updated_event)
            st.rerun()


def _render_schedule_sidebar(
    plan_dates: list[pd.Timestamp],
    plan_events: list[dict[str, object]],
    last_active: pd.Timestamp,
) -> None:
    """Render helpful planner context, event list, and secondary event actions."""
    st.markdown('<div class="section-label">Planning Summary</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="planner-side-copy">
                <strong>Latest historical data</strong><br>
                {last_active.strftime('%d %b %Y')}<br><br>
                <strong>Forecast window</strong><br>
                {plan_dates[0].strftime('%d %b %Y')} → {plan_dates[-1].strftime('%d %b %Y')}<br><br>
                The highlighted calendar days define the active 15-day forecast horizon.
                Events outside that highlight remain visible for planning, but they do not affect the current forecast run.
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label" style="margin-top:1rem">Upcoming Sessions</div>', unsafe_allow_html=True)
    if not plan_events:
        with st.container(border=True):
            st.info("No sessions are planned yet. Add an event to start building the next 15-day schedule.")
        return

    upcoming_events = plan_events[:3]
    if len(plan_events) > 3:
        st.caption(f"Showing the next 3 planned sessions out of {len(plan_events)} total events.")

    for event in upcoming_events:
        event_id = str(event["id"])
        event_types = _event_session_types(event)
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="planner-event-card">
                    <div class="planner-event-title">{event['title']}</div>
                    <div class="planner-event-meta">{describe_event(event)}</div>
                    <div class="planner-event-location">📍 {event.get('location', 'Other')}</div>
                    <div class="planner-event-badges">{_badge_html(event_types)}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if event.get("notes"):
                st.caption(str(event["notes"]))
            edit_col, delete_col = st.columns(2)
            if edit_col.button("Edit", key=f"edit_event_{event_id}", use_container_width=True):
                _open_event_dialog({"mode": "edit", "event_id": event_id})
            if delete_col.button("Delete", key=f"delete_event_{event_id}", use_container_width=True):
                _remove_event(event_id)
                st.rerun()


def _run_forecast(plan_dates: list[pd.Timestamp]) -> None:
    """Build and persist a fresh forecast for the current event plan."""
    plan_events = sort_events(st.session_state.plan_events)
    plan_days = build_plan_days_from_events(plan_events, plan_dates)

    with st.spinner("Computing 15-day ACWR forecasts for all 28 players…"):
        results = build_forecast(plan_days)

    if results:
        st.session_state.forecast = results
        st.session_state.plan_days = plan_days
        st.session_state.plan_dates = build_plan_date_labels(plan_dates)
        st.session_state.forecast_plan_signature = plan_signature(plan_days)
    else:
        st.error("Forecast failed — check models are trained (`python train_models.py`).")


def _render_forecast_results(
    *,
    player_data: dict[int, dict[str, object]],
    all_pids: list[int],
    forecast: dict[str, dict[str, dict[str, object]]],
    forecast_is_stale: bool,
) -> None:
    """Render the results portion below the planner once a forecast exists."""
    st.markdown("---")
    st.markdown(
        f"""
        <div class="page-header planner-results-header">
            <div class="page-title">Forecast <span>Results</span></div>
            <div class="page-sub">
                15-day ACWR projection &nbsp;&middot;&nbsp;
                {len(all_pids)} players &nbsp;&middot;&nbsp; 3 load metrics
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    if forecast_is_stale:
        st.warning(
            "The planning calendar has changed since the last forecast run. The results below are preserved for reference; run the forecast again to refresh them."
        )

    danger_entries: list[tuple[str, list[str]]] = []
    for pid in all_pids:
        bad_metrics = [
            TARGET_META[metric]["label"]
            for metric in TARGETS
            if forecast[str(pid)][metric]["day15_zone"] == "danger"
        ]
        if bad_metrics:
            danger_entries.append((str(pid), bad_metrics))

    if danger_entries:
        rows = "".join(
            f'<div style="margin-top:4px">&#x2022; Player <strong>{player}</strong> — {", ".join(metrics)}</div>'
            for player, metrics in danger_entries
        )
        st.markdown(
            f"""
            <div class="rm-alert">
                <div>
                    <strong>Injury Risk Alert:</strong> {len(danger_entries)} player(s) projected
                    in DANGER zone by Day 15:
                    {rows}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    selector_col, status_col = st.columns([3, 2])
    with selector_col:
        pid_options = {f"Player {pid} · {player_data[pid]['position']}": str(pid) for pid in all_pids}
        selected_label = st.selectbox("Player", list(pid_options.keys()), key="forecast_player_selector")
        selected_pid = pid_options[selected_label]
    with status_col:
        st.markdown("<div style='margin-top:1.85rem'></div>", unsafe_allow_html=True)
        status_text = "Plan updated after last run" if forecast_is_stale else "Forecast is up to date"
        status_color = "#F59E0B" if forecast_is_stale else "#10B981"
        st.markdown(
            f'<div class="planner-status-banner" style="border-color:{status_color};color:{status_color};background:{status_color}12">{status_text}</div>',
            unsafe_allow_html=True,
        )

    for metric in TARGETS:
        meta = TARGET_META[metric]
        st.markdown(
            f'<div class="section-label" style="color:{meta["color"]};margin-top:1rem">{meta["label"]} &nbsp;·&nbsp; ACWR (unitless)</div>',
            unsafe_allow_html=True,
        )
        metric_forecast = forecast[selected_pid][metric]
        fig = build_acwr_chart(metric_forecast, meta)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    st.markdown('<div class="section-label">Day-15 Summary — All Players</div>', unsafe_allow_html=True)

    zone_order = ["danger", "caution", "undertraining", "optimal", "unknown"]
    header_cols = "".join(
        f'<th style="color:{TARGET_META[metric]["color"]}">{TARGET_META[metric]["label"]}</th>'
        for metric in TARGETS
    )
    rows_html = ""
    for pid_str in [str(player_id) for player_id in all_pids]:
        pdata = player_data[int(pid_str)]
        cells = f'<td class="td-pid">Player {pid_str}</td><td class="td-pos">{pdata["position"]}</td>'
        worst_zone = "optimal"
        for metric in TARGETS:
            value = forecast[pid_str][metric]["day15_acwr"]
            zone = cast(str, forecast[pid_str][metric]["day15_zone"])
            color = ZONE_COLORS[zone]
            label = ZONE_LABELS[zone]
            if zone_order.index(zone) < zone_order.index(worst_zone):
                worst_zone = zone
            value_string = f"{value:.2f}" if isinstance(value, (int, float)) else "—"
            cells += f"""
            <td>
                <span style="color:{color};font-weight:700;font-family:'Courier New',monospace">{value_string}</span>
                <span style="font-size:0.56rem;font-weight:800;text-transform:uppercase;
                             padding:2px 6px;border-radius:5px;border:1px solid {color};
                             color:{color};background:{color}18;margin-left:5px">{label}</span>
            </td>"""
        status_icons = {"danger": "HIGH RISK", "caution": "CAUTION", "optimal": "OK", "undertraining": "LOW"}
        status_color = ZONE_COLORS[worst_zone]
        status_label = status_icons.get(worst_zone, "—")
        cells += f'<td><span style="color:{status_color};font-size:0.65rem;font-weight:800;letter-spacing:0.8px">{status_label}</span></td>'
        rows_html += f"<tr>{cells}</tr>"

    st.markdown(
        f"""
        <div style="overflow-x:auto;border:1px solid #E2EBF6;border-radius:10px;
                    box-shadow:0 1px 6px rgba(0,60,140,0.06)">
            <table class="rm-table">
                <thead><tr><th>Player</th><th>Position</th>{header_cols}<th>Status</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>""",
        unsafe_allow_html=True,
    )


@st.dialog("Session Event", width="large")
def _event_editor_dialog(plan_dates: list[pd.Timestamp]) -> None:
    """Modal dialog used to create or edit planning events."""
    request = st.session_state.get("planner_dialog_request")
    if not isinstance(request, dict):
        return

    events_by_id = _event_lookup(sort_events(st.session_state.plan_events))
    current_event = events_by_id.get(str(request.get("event_id", "")))
    if current_event is None:
        default_start, default_end = default_event_bounds(plan_dates)
        start_dt = _coerce_timestamp(request.get("start", default_start.isoformat()))
        end_dt = _coerce_timestamp(request.get("end", default_end.isoformat()))
        event_id = str(uuid4())
        selected_types: list[str] = []
        location = LOCATION_OPTIONS[0]
        notes = ""
        dialog_title = "Create a new session"
    else:
        start_dt = event_start_datetime(current_event)
        end_dt = event_end_datetime(current_event)
        event_id = str(current_event["id"])
        selected_types = _event_session_types(current_event)
        location = str(current_event.get("location", LOCATION_OPTIONS[0]))
        notes = str(current_event.get("notes", ""))
        dialog_title = "Edit planned session"

    safe_start_date = start_dt.date() if isinstance(start_dt, datetime) else plan_dates[0].date()

    st.markdown(f"**{dialog_title}**")
    selected_date = st.date_input(
        "Date",
        value=safe_start_date,
        format="YYYY-MM-DD",
    )
    time_col1, time_col2 = st.columns(2)
    with time_col1:
        start_time_input = st.time_input("Start time", value=start_dt.time(), step=900)
    with time_col2:
        end_time_input = st.time_input("End time", value=end_dt.time(), step=900)

    selected_date_value = _coerce_date_input(selected_date, safe_start_date)
    start_time_value = _coerce_time_input(start_time_input, start_dt.time())
    end_time_value = _coerce_time_input(end_time_input, end_dt.time())

    session_types = st.multiselect(
        "Session types",
        SESSION_TYPES,
        default=selected_types,
        format_func=lambda session_type: f"{session_type} · {SESSION_LABELS[session_type]}",
    )
    location_value = location if location in LOCATION_OPTIONS else "Other"
    selected_location = st.selectbox(
        "Location",
        LOCATION_OPTIONS,
        index=LOCATION_OPTIONS.index(location_value),
    )
    notes_value = st.text_area(
        "Notes",
        value=notes,
        placeholder="Optional coaching context, travel information, or drill notes.",
        height=120,
    )

    selected_location_value = selected_location or LOCATION_OPTIONS[0]
    notes_text = notes_value or ""

    save_col, cancel_col = st.columns(2)
    if save_col.button("Save Event", type="primary", use_container_width=True):
        if not session_types:
            st.error("Select at least one session type before saving the event.")
            st.stop()

        start_value = combine_date_and_time(selected_date_value, start_time_value)
        end_value = combine_date_and_time(selected_date_value, end_time_value)
        if end_value <= start_value:
            st.error("The event end time must be later than the start time.")
            st.stop()

        _upsert_event(
            build_event_record(
                event_id=event_id,
                start_dt=start_value,
                end_dt=end_value,
                session_types=session_types,
                location=selected_location_value,
                notes=notes_text,
            )
        )
        st.session_state.planner_dialog_request = None
        st.rerun()

    if cancel_col.button("Cancel", use_container_width=True):
        st.session_state.planner_dialog_request = None
        st.rerun()

    if current_event is not None and st.button("Delete Event", type="secondary", use_container_width=True):
        _remove_event(str(current_event["id"]))
        st.session_state.planner_dialog_request = None
        st.rerun()


def page_planner() -> None:
    """Render the unified planning and forecasting workspace."""
    player_data, all_pids, _ = load_player_data()
    get_models_or_stop()
    last_active = player_data[all_pids[0]]["last_active"]
    plan_dates = build_planning_window(last_active)

    _initialise_planner_state(plan_dates)
    plan_events = sort_events(st.session_state.plan_events)
    current_plan_days = build_plan_days_from_events(plan_events, plan_dates)
    plan_summary = summarise_plan(current_plan_days)
    forecast_is_stale = (
        "forecast" in st.session_state
        and st.session_state.get("forecast_plan_signature") != plan_signature(current_plan_days)
    )

    _render_standard_header(
        "Planning & <span>Forecast</span>",
        "Schedule the next 15 days in an interactive calendar, then run a squad-wide ACWR forecast without leaving the page.",
    )
    st.caption(
        "Forecast window is anchored to the latest available squad data on "
        f"{last_active.strftime('%d %b %Y')}. Highlighted calendar days mark the active 15-day forecast horizon; events outside the highlight are stored but ignored by the current forecast run."
    )
    _render_session_legend()
    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    _render_planner_metrics(plan_summary, len(plan_events))

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    action_cols = st.columns([1, 1, 1.3, 1.7])
    if action_cols[0].button("New Event", type="secondary", use_container_width=True):
        default_start, default_end = default_event_bounds(plan_dates)
        _open_event_dialog(
            {
                "mode": "create",
                "start": default_start.isoformat(timespec="minutes"),
                "end": default_end.isoformat(timespec="minutes"),
            }
        )
    if action_cols[1].button("Clear Plan", type="secondary", use_container_width=True):
        st.session_state.plan_events = []
        st.rerun()
    if action_cols[2].button("Run Forecast", type="primary", use_container_width=True):
        _run_forecast(plan_dates)
        st.rerun()
    action_cols[3].markdown(
        '<div class="planner-hint">Tip: click a date, drag across the weekly view, or move existing events directly on the calendar.</div>',
        unsafe_allow_html=True,
    )

    calendar_col, schedule_col = st.columns([2.25, 1], gap="large")
    with calendar_col:
        st.markdown('<div class="section-label">Interactive Session Calendar</div>', unsafe_allow_html=True)
        _handle_calendar_interactions(plan_dates)
    with schedule_col:
        _render_schedule_sidebar(plan_dates, sort_events(st.session_state.plan_events), last_active)

    if st.session_state.get("planner_dialog_request") is not None:
        _event_editor_dialog(plan_dates)

    if "forecast" not in st.session_state:
        st.markdown("---")
        st.info("Build the upcoming schedule above, then click **Run Forecast** to generate the integrated ACWR results view.")
        return

    _render_forecast_results(
        player_data=player_data,
        all_pids=all_pids,
        forecast=cast(dict[str, dict[str, dict[str, object]]], st.session_state.forecast),
        forecast_is_stale=forecast_is_stale,
    )


def page_results() -> None:
    """Compatibility wrapper for any legacy imports targeting the old results page."""
    page_planner()


def render_sidebar(logo_path):
    with st.sidebar:
        _logo_b64 = ""
        if logo_path.exists():
            _logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()

        st.markdown(f"""
        <div style="padding:1.75rem 1.25rem 1.1rem;text-align:center">
            {'<img src="data:image/svg+xml;base64,' + _logo_b64 + '" style="width:112px;height:112px;display:block;margin:0 auto 14px">' if _logo_b64 else ''}
            <div style="font-size:1.13rem;font-weight:900;letter-spacing:3px;
                        color:#FEBE10;text-transform:uppercase;line-height:1.2">Real Madrid C.F.</div>
            <div style="font-size:0.96rem;font-weight:400;letter-spacing:2px;
                        color:rgba(255,255,255,0.55);text-transform:uppercase;margin-top:5px">
                ACWR Monitor
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        page = st.radio("Navigation", PAGES, key="nav_page", label_visibility="collapsed")
        st.markdown("---")

        st.markdown("""
        <div style="padding:0.5rem 1rem 1rem;font-size:0.65rem;color:rgba(255,255,255,0.4);line-height:1.8">
            <div style="font-weight:700;color:rgba(255,255,255,0.6);
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">
                Season 2024/25
            </div>
            <div>28 Players &nbsp;&middot;&nbsp; 3 Metrics</div>
            <div>EWMA · α<sub>acute</sub>=0.25 · α<sub>chronic</sub>≈0.07</div>
        </div>""", unsafe_allow_html=True)

        team_logo_path = STATIC_DIR / "img" / "trAIn_labs.png"
        team_b64 = ""
        if team_logo_path.exists():
            team_b64 = base64.b64encode(team_logo_path.read_bytes()).decode()

        st.markdown("---")
        st.markdown(f"""
        <div style="padding:0.5rem 1rem 1.5rem;text-align:center">
            <div style="font-size:0.68rem;font-weight:600;letter-spacing:1.5px;
                        color:rgba(255,255,255,0.35);text-transform:uppercase;margin-bottom:10px">
                Developed by
            </div>
            {'<div style="display:inline-block;background:#FFFFFF;border-radius:8px;padding:6px 14px"><img src="data:image/png;base64,' + team_b64 + '" style="width:110px;display:block"></div>' if team_b64 else '<span style="color:rgba(255,255,255,0.5);font-weight:700">trAIn Labs</span>'}
        </div>""", unsafe_allow_html=True)

        return page
