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
    SESSION_TYPES,
    TARGET_META,
    TARGETS,
    ZONE_COLORS,
)
from app.forecasting import build_forecast, build_player_forecast
from app.i18n import (
    fmt_date_full,
    fmt_date_long,
    fmt_date_medium,
    fmt_date_short,
    fmt_dow_date,
    t,
    t_pos,
)
from app.loaders import get_models_or_stop, load_player_data
from app.planning import (
    build_event_record,
    build_plan_date_labels,
    build_plan_days_from_events,
    build_planning_window,
    combine_date_and_time,
    default_event_bounds,
    event_end_datetime,
    event_start_datetime,
    normalise_event_types,
    plan_signature,
    sort_events,
    summarise_plan,
)
from real_madrid_acwr.config import STATIC_DIR

APP_TIMEZONE = ZoneInfo("Europe/Madrid")


def get_session_labels() -> dict[str, str]:
    """Return session type display labels for the active UI language."""
    return {stype: t(f"session_type_{stype}") for stype in SESSION_TYPES}


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
    """Render the ACWR methodology dashboard."""
    player_data, all_pids, current_acwr, _ = load_player_data()
    get_models_or_stop()
    last_date = fmt_date_full(player_data[all_pids[0]]["last_active"])

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="padding:2.2rem 0 1.6rem;border-bottom:1px solid #E2EBF6;margin-bottom:2rem;text-align:center">
        <div style="display:inline-block;width:44px;height:4px;border-radius:2px;
                    background:linear-gradient(90deg,#FEBE10,#00529F);margin-bottom:1rem"></div>
        <div style="font-size:2.2rem;font-weight:900;color:#00529F;letter-spacing:-1px;line-height:1.1;margin-bottom:0.5rem">
            Squad ACWR Dashboard
        </div>
        <div style="font-size:0.95rem;color:#64748B;font-weight:400">
            Data through <strong style="color:#334D6E">{last_date}</strong>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Season / squad / method info bar ─────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:2rem">
        <div style="flex:1;min-width:160px;background:linear-gradient(135deg,#00529F,#0369a1);
                    border-radius:12px;padding:1.1rem 1.3rem;color:#fff">
            <div style="font-size:0.65rem;font-weight:800;text-transform:uppercase;letter-spacing:1.2px;
                        opacity:0.65;margin-bottom:5px">{t("info_season")}</div>
            <div style="font-size:1.15rem;font-weight:800">2024 &ndash; 25</div>
        </div>
        <div style="flex:1;min-width:160px;background:linear-gradient(135deg,#00529F,#0369a1);
                    border-radius:12px;padding:1.1rem 1.3rem;color:#fff">
            <div style="font-size:0.65rem;font-weight:800;text-transform:uppercase;letter-spacing:1.2px;
                        opacity:0.65;margin-bottom:5px">{t("info_squad")}</div>
            <div style="font-size:1.15rem;font-weight:800">{t("info_players_metrics").format(n=len(all_pids))}</div>
        </div>
        <div style="flex:2;min-width:260px;background:linear-gradient(135deg,#00529F,#0369a1);
                    border-radius:12px;padding:1.1rem 1.3rem;color:#fff">
            <div style="font-size:0.65rem;font-weight:800;text-transform:uppercase;letter-spacing:1.2px;
                        opacity:0.65;margin-bottom:5px">{t("info_smoothing")}</div>
            <div style="font-size:1.05rem;font-weight:800">
                EWMA &nbsp;&middot;&nbsp;
                &alpha;<sub>acute</sub> = 0.25 &nbsp;&middot;&nbsp;
                &alpha;<sub>chronic</sub> &asymp; 0.07
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── ACWR formula ─────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-label" style="margin-bottom:0.9rem">{t("section_what_is_acwr")}</div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1.6], gap="large")
    with left_col:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #D7E4F1;border-radius:12px;
                    padding:1.5rem 1.6rem;height:100%">
            <div style="font-size:0.88rem;color:#475569;line-height:1.75;margin-bottom:1.2rem">
                {t("acwr_intro")}
            </div>
            <div style="display:flex;align-items:center;justify-content:center;gap:1.4rem;
                        padding:1.1rem;background:#F0F4FA;border-radius:10px">
                <span style="font-size:1.25rem;font-weight:900;color:#00529F">ACWR =</span>
                <div style="display:inline-flex;flex-direction:column;align-items:center">
                    <span style="font-size:1rem;font-weight:700;color:#00529F;padding-bottom:5px">{t("acwr_acute_label")}</span>
                    <div style="width:100%;height:2.5px;background:#00529F;border-radius:2px"></div>
                    <span style="font-size:1rem;font-weight:700;color:#00529F;padding-top:5px">{t("acwr_chronic_label")}</span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with right_col:
        for title, days, alpha, desc in [
            (t("acwr_acute_label"), t("acwr_acute_days"), "&alpha; = 0.25",
             t("acwr_acute_desc")),
            (t("acwr_chronic_label"), t("acwr_chronic_days"), "&alpha; &asymp; 0.07",
             t("acwr_chronic_desc")),
            (t("acwr_ewma_label"), "", "",
             t("acwr_ewma_desc")),
        ]:
            badge = f'<span style="font-family:Courier New,monospace;font-size:0.72rem;font-weight:700;color:#00529F;background:#EEF3FF;padding:1px 7px;border-radius:4px;margin-left:6px">{days}</span>' if days else ""
            alpha_badge = f'<span style="font-family:Courier New,monospace;font-size:0.72rem;font-weight:700;color:#475569;background:#F1F5F9;padding:1px 7px;border-radius:4px;margin-left:4px">{alpha}</span>' if alpha else ""
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid #D7E4F1;border-radius:10px;
                        padding:0.85rem 1.1rem;margin-bottom:0.7rem;display:flex;gap:0.9rem;align-items:flex-start">
                <div style="width:6px;min-width:6px;height:6px;border-radius:50%;background:#00529F;margin-top:7px"></div>
                <div>
                    <div style="font-size:0.8rem;font-weight:800;color:#00529F;text-transform:uppercase;
                                letter-spacing:0.5px;margin-bottom:3px">
                        {title}{badge}{alpha_badge}
                    </div>
                    <div style="font-size:0.83rem;color:#475569;line-height:1.55">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Risk zones ───────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-label" style="margin-top:1.8rem;margin-bottom:0.9rem">{t("section_risk_zones")}</div>', unsafe_allow_html=True)
    zone_info = [
        ("undertraining", t("zone_undertraining_range"), t("zone_undertraining_label"),
         t("zone_undertraining_desc")),
        ("optimal",       t("zone_optimal_range"),       t("zone_optimal_label"),
         t("zone_optimal_desc")),
        ("caution",       t("zone_caution_range"),       t("zone_caution_label"),
         t("zone_caution_desc")),
        ("danger",        t("zone_danger_range"),        t("zone_danger_label"),
         t("zone_danger_desc")),
    ]
    zone_cols = st.columns(4)
    for col, (zone, rng, label, desc) in zip(zone_cols, zone_info, strict=False):
        color = ZONE_COLORS[zone]
        col.markdown(f"""
        <div style="border:1px solid {color}28;border-top:4px solid {color};border-radius:12px;
                    padding:1.1rem 1.1rem 1.2rem;background:#FFFFFF;height:100%;
                    box-shadow:0 2px 8px {color}10">
            <div style="display:flex;align-items:center;gap:7px;margin-bottom:7px">
                <span style="width:9px;height:9px;border-radius:50%;background:{color};flex-shrink:0"></span>
                <span style="font-weight:800;color:{color};font-size:0.9rem">{label}</span>
            </div>
            <div style="font-family:Courier New,monospace;font-size:0.76rem;font-weight:700;
                        color:{color};margin-bottom:10px;padding:2px 9px;background:{color}14;
                        border-radius:5px;display:inline-block">{rng}</div>
            <div style="font-size:0.8rem;color:#64748B;line-height:1.6">{desc}</div>
        </div>""", unsafe_allow_html=True)

    # ── Current player status ─────────────────────────────────────────────────
    st.markdown(f'<div class="section-label" style="margin-top:2rem;margin-bottom:0.9rem">{t("section_player_status")}</div>', unsafe_allow_html=True)

    # KPI summary
    n_danger    = sum(1 for pid in all_pids for m in TARGETS if current_acwr[pid][m]["zone"] == "danger")
    n_caution   = sum(1 for pid in all_pids for m in TARGETS if current_acwr[pid][m]["zone"] == "caution")
    n_optimal   = sum(1 for pid in all_pids for m in TARGETS if current_acwr[pid][m]["zone"] == "optimal")
    n_under     = sum(1 for pid in all_pids for m in TARGETS if current_acwr[pid][m]["zone"] == "undertraining")

    k1, k2, k3, k4 = st.columns(4)
    for col, label, val, color in [
        (k1, t("kpi_high_risk_flags"), n_danger,  "#EE324E"),
        (k2, t("kpi_caution_flags"),   n_caution, "#F59E0B"),
        (k3, t("kpi_optimal_flags"),   n_optimal, "#10B981"),
        (k4, t("kpi_undertraining"),   n_under,   "#64748B"),
    ]:
        col.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2EBF6;border-top:4px solid {color};
                    border-radius:12px;padding:1.1rem 1.3rem;box-shadow:0 2px 8px rgba(0,60,140,0.06);
                    margin-bottom:1rem">
            <div style="font-size:2rem;font-weight:900;color:{color};line-height:1;margin-bottom:5px">{val}</div>
            <div style="font-size:0.72rem;font-weight:800;text-transform:uppercase;letter-spacing:0.8px;
                        color:#64748B">{label}</div>
        </div>""", unsafe_allow_html=True)

    # Player table — sorted by worst zone then position
    zone_order   = ["danger", "caution", "undertraining", "optimal", "unknown"]
    pos_order    = {"Full Back": 0, "Central Back": 1, "Central Midfielder": 2, "Winger": 3, "Forward": 4, "Unknown": 99}

    def _worst(pid: int) -> str:
        return min((current_acwr[pid][m]["zone"] for m in TARGETS), key=zone_order.index)

    sorted_pids = sorted(all_pids, key=lambda pid: (zone_order.index(_worst(pid)), pos_order.get(player_data[pid]["position"], 99)))

    header = (
        '<thead><tr style="background:#F0F4FA;border-bottom:2px solid #D7E4F1">'
        f'<th style="padding:9px 14px;text-align:left;font-size:0.68rem;font-weight:800;color:#00529F;text-transform:uppercase;letter-spacing:0.8px;white-space:nowrap">{t("table_player")}</th>'
        f'<th style="padding:9px 14px;text-align:left;font-size:0.68rem;font-weight:800;color:#00529F;text-transform:uppercase;letter-spacing:0.8px">{t("table_position")}</th>'
        + "".join(
            f'<th style="padding:9px 14px;text-align:center;font-size:0.68rem;font-weight:800;color:#00529F;text-transform:uppercase;letter-spacing:0.8px">{t(f"target_{m}")}</th>'
            for m in TARGETS
        )
        + f'<th style="padding:9px 14px;text-align:center;font-size:0.68rem;font-weight:800;color:#00529F;text-transform:uppercase;letter-spacing:0.8px">{t("table_status")}</th>'
        + '</tr></thead>'
    )

    body = ""
    for pid in sorted_pids:
        pos     = t_pos(player_data[pid]["position"])
        worst   = _worst(pid)
        wcolor  = ZONE_COLORS[worst]
        status_labels = {"danger": t("status_high_risk"), "caution": t("status_caution"), "optimal": t("status_ok"), "undertraining": t("status_low"), "unknown": "—"}
        status  = status_labels.get(worst, "—")

        cells = ""
        for m in TARGETS:
            z     = current_acwr[pid][m]["zone"]
            val   = current_acwr[pid][m]["value"]
            zc    = ZONE_COLORS[z]
            vs    = f"{val:.2f}" if val is not None else "—"
            cells += (
                f'<td style="padding:9px 14px;text-align:center">'
                f'<span style="font-family:Courier New,monospace;font-weight:700;font-size:0.88rem;color:{zc}">{vs}</span>'
                f'<span style="display:block;font-size:0.6rem;font-weight:800;text-transform:uppercase;'
                f'color:{zc};opacity:0.8;letter-spacing:0.4px">{t(f"zone_{z}")}</span>'
                f'</td>'
            )

        body += (
            f'<tr style="border-bottom:1px solid #EEF3FA">'
            f'<td style="padding:9px 14px;font-family:Courier New,monospace;font-weight:700;color:#0F172A;font-size:0.88rem;white-space:nowrap">{pid}</td>'
            f'<td style="padding:9px 14px;font-size:0.8rem;font-weight:600;color:#00529F;text-transform:uppercase;letter-spacing:0.4px;white-space:nowrap">{pos}</td>'
            f'{cells}'
            f'<td style="padding:9px 14px;text-align:center">'
            f'<span style="font-size:0.65rem;font-weight:800;letter-spacing:0.5px;text-transform:uppercase;'
            f'padding:3px 9px;border-radius:5px;border:1px solid {wcolor};color:{wcolor};background:{wcolor}12">{status}</span>'
            f'</td>'
            f'</tr>'
        )

    table_html = (
        '<div style="overflow-x:auto;border:1px solid #D7E4F1;border-radius:12px;'
        'box-shadow:0 2px 8px rgba(0,60,140,0.06);background:#FFFFFF;margin-bottom:1rem">'
        f'<table style="width:100%;border-collapse:collapse;font-size:0.85rem">'
        f'{header}<tbody>{body}</tbody>'
        '</table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


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
    """Extract the parent event id from a callback payload. Strips the |stype suffix from per-type blocks."""
    event_payload = payload.get("event")
    if isinstance(event_payload, dict) and event_payload.get("id") is not None:
        return str(event_payload["id"]).split("|")[0]
    if payload.get("id") is not None:
        return str(payload["id"]).split("|")[0]
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
        f'<div class="section-label" style="font-size:1rem;letter-spacing:0.5px">{t("section_session_types")}</div>',
        unsafe_allow_html=True,
    )
    legend_cols = st.columns(len(SESSION_TYPES))
    for index, session_type in enumerate(SESSION_TYPES):
        legend_cols[index].markdown(
            f"""
            <div class="planner-legend-pill" style="border-color:{SESSION_COLORS[session_type]};color:{SESSION_COLORS[session_type]};background:{SESSION_COLORS[session_type]}12">
                <span class="planner-legend-dot" style="background:{SESSION_COLORS[session_type]}"></span>
                {session_type} · {get_session_labels()[session_type]}
            </div>""",
            unsafe_allow_html=True,
        )


def _render_planner_metrics(summary: dict[str, int], event_count: int) -> None:
    """Render top-line planner KPIs."""
    metrics = [
        (t("kpi_sessions_planned"), event_count, "#00529F"),
        (t("kpi_training_days"), summary["active_days"], "#FEBE10"),
        (t("kpi_match_days"), summary["match_days"], "#EE324E"),
        (t("kpi_rest_days"), summary["rest_days"], "#64748B"),
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
    """Combine one block per session type with a background highlight for the forecast window."""
    highlight_end = (plan_dates[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    planning_window_event = {
        "id": "planning-window-highlight",
        "title": t("section_forecast_window"),
        "start": plan_dates[0].strftime("%Y-%m-%d"),
        "end": highlight_end,
        "allDay": True,
        "display": "background",
        "editable": False,
        "overlap": True,
        "classNames": ["planning-window-highlight"],
    }
    expanded: list[dict[str, object]] = []
    for event in sort_events(plan_events):
        parent_id = str(event["id"])
        event_types = _event_session_types(event)
        if not event_types:
            expanded.append({
                "id": parent_id,
                "title": t("calendar_session_fallback"),
                "start": str(event["start"]),
                "end": str(event.get("end") or ""),
                "allDay": False,
                "editable": False,
            })
        else:
            for stype in event_types:
                expanded.append({
                    "id": f"{parent_id}|{stype}",
                    "title": get_session_labels()[stype],
                    "start": str(event["start"]),
                    "end": str(event.get("end") or ""),
                    "allDay": False,
                    "backgroundColor": SESSION_COLORS[stype],
                    "borderColor": SESSION_COLORS[stype],
                    "editable": False,
                })
    return [planning_window_event, *expanded]


def _render_calendar(plan_dates: list[pd.Timestamp], plan_events: list[dict[str, object]]) -> object:
    """Render the interactive FullCalendar instance."""
    if importlib.util.find_spec("streamlit_calendar") is None:
        st.error(
            "The interactive calendar dependency is missing. Run `python3 -m pip install -e .` "
            "to install the updated project dependencies."
        )
        return None

    from streamlit_calendar import calendar as calendar_fn

    lang = st.session_state.get("lang", "ENG")
    calendar_options = {
        "editable": True,
        "selectable": True,
        "eventStartEditable": True,
        "eventDurationEditable": True,
        "firstDay": 1,
        "initialView": "dayGridMonth",
        "locale": "es" if lang == "ESP" else "en",
        "headerToolbar": {
            "left": "today prev,next",
            "center": "title",
            "right": "",
        },
        "initialDate": plan_dates[0].strftime("%Y-%m-%d"),
        "validRange": {
            "start": plan_dates[0].strftime("%Y-%m-%d"),
            "end": (plan_dates[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        },
        "height": 760,
        "nowIndicator": True,
        "dayMaxEvents": 3,
        "displayEventTime": False,
        "allDaySlot": False,
        "selectMirror": True,
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
    .fc .fc-day-disabled {
        background: rgba(220, 230, 242, 0.35) !important;
    }
    .fc .fc-day-disabled .fc-daygrid-day-number {
        color: #94A3B8;
        opacity: 0.7;
    }
    """
    return calendar_fn(
        events=_build_calendar_payload(plan_dates, plan_events),
        options=calendar_options,
        custom_css=custom_css,
        callbacks=["dateClick", "select", "eventClick", "eventChange"],
        key=f"session_planner_calendar_{lang}",
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
    """Render the planning sidebar: window info, session legend, and event list."""
    st.markdown(
        f"""
        <div class="forecast-window-box">
            <div class="fw-label">{t("section_forecast_window")}</div>
            <div class="fw-dates">{fmt_date_short(plan_dates[0])} &rarr; {fmt_date_medium(plan_dates[-1])}</div>
            <div class="fw-sub">{t("fw_15days")} &nbsp;·&nbsp; {t("fw_model_data_to")} {fmt_date_medium(last_active)}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    if not plan_events:
        st.markdown(
            f'<div class="section-label" style="margin-top:1.1rem">{t("section_planned_sessions")}</div>',
            unsafe_allow_html=True,
        )
        st.info(t("no_sessions_tip"), icon="📅")
        return

    n = len(plan_events)
    st.markdown(
        f'<div class="section-label" style="margin-top:1.1rem;margin-bottom:0.3rem">'
        f'{t("section_planned_sessions")} <span class="slr-count">({n})</span></div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        for event in plan_events:
            event_id = str(event["id"])
            event_types = _event_session_types(event)
            start_dt = event_start_datetime(event)
            date_str = fmt_dow_date(start_dt)
            types_str = " · ".join(event_types) if event_types else "—"
            text_col, edit_col, del_col = st.columns([5, 0.55, 0.55], gap="small")
            text_col.markdown(
                f'<div class="session-list-row">'
                f'<span class="slr-date">{date_str}:</span> '
                f'<span class="slr-types">{types_str}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
            if edit_col.button("✏️", key=f"edit_event_{event_id}", use_container_width=True):
                _open_event_dialog({"mode": "edit", "event_id": event_id})
            if del_col.button("🗑️", key=f"delete_event_{event_id}", use_container_width=True):
                _remove_event(event_id)
                st.rerun()


def _run_forecast(plan_dates: list[pd.Timestamp]) -> None:
    """Build and persist a fresh forecast for the current event plan."""
    plan_events = sort_events(st.session_state.plan_events)
    plan_days = build_plan_days_from_events(plan_events, plan_dates)

    with st.spinner(t("spinner_forecast")):
        results = build_forecast(plan_days)

    if results:
        st.session_state.forecast = results
        st.session_state.plan_days = plan_days
        st.session_state.plan_dates = build_plan_date_labels(plan_dates)
        st.session_state.forecast_plan_signature = plan_signature(plan_days)
    else:
        st.error(t("err_forecast_failed"))


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
            <div class="page-title">{t("forecast_results_title")}</div>
            <div class="page-sub">
                {t("forecast_results_sub")} &nbsp;&middot;&nbsp;
                {t("forecast_header_meta").format(n=len(all_pids))}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    if forecast_is_stale:
        st.info(t("forecast_stale_warning"), icon="⚠️")

    _ = st.session_state.get("plan_days", [])  # unused here; accessed per-player below

    danger_entries: list[tuple[str, list[tuple[str, str | None, float | None, float | None]]]] = []
    for pid in all_pids:
        bad_metrics = []
        for metric in TARGETS:
            mdata        = cast(dict, forecast[str(pid)][metric])
            fore_dates   = cast(list, mdata["fore_dates"])
            fore_acwr    = cast(list, mdata["fore_acwr"])
            first_danger_idx: int | None = next(
                (i for i, (_, v) in enumerate(zip(fore_dates, fore_acwr, strict=False)) if v is not None and v >= 1.5),
                None,
            )
            if first_danger_idx is None:
                continue
            first_danger_date: str | None = fore_dates[first_danger_idx]
            first_danger_acwr: float | None = fore_acwr[first_danger_idx]
            bad_metrics.append((t(f"target_{metric}"), first_danger_date, first_danger_acwr, mdata.get("day15_acwr")))
        if bad_metrics:
            danger_entries.append((str(pid), bad_metrics))

    if danger_entries:
        import streamlit.components.v1 as components

        player_prefix = t("player_prefix")

        # Split: still in danger at day 15 vs briefly entered then recovered
        critical = [(p, m) for p, m in danger_entries if any(isinstance(a, float) and a >= 1.5 for _, _, _, a in m)]
        recovered = [(p, m) for p, m in danger_entries if not any(isinstance(a, float) and a >= 1.5 for _, _, _, a in m)]

        # Sort critical by worst first_acwr desc
        critical.sort(key=lambda e: -max((a or 0.0) for _, _, a, _ in e[1]))

        def _table_rows_html(entries: list) -> str:
            rows = ""
            for player, metrics in entries:
                pos = t_pos(str(player_data[int(player)]["position"]))
                for i, (name, date, first_acwr, day15_acwr) in enumerate(metrics):
                    d15_color = "#EE324E" if (isinstance(day15_acwr, float) and day15_acwr >= 1.5) \
                        else "#F59E0B" if (isinstance(day15_acwr, float) and day15_acwr >= 1.3) \
                        else "#10B981"
                    player_cell = (
                        f'<td rowspan="{len(metrics)}" style="vertical-align:middle;border-right:1px solid #FEE2E2;padding:10px 14px;white-space:nowrap">'
                        f'<div style="font-weight:800;font-family:Courier New,monospace;color:#0F172A;font-size:0.9rem">{player_prefix} {player}</div>'
                        f'<div style="font-size:0.72rem;color:#00529F;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-top:2px">{pos}</div>'
                        f'</td>'
                    ) if i == 0 else ""
                    fa = f"{first_acwr:.2f}" if isinstance(first_acwr, float) else "—"
                    d15 = f"{day15_acwr:.2f}" if isinstance(day15_acwr, float) else "—"
                    rows += (
                        f'<tr style="border-bottom:1px solid #FEE2E2">'
                        f'{player_cell}'
                        f'<td style="padding:9px 14px;font-weight:700;color:#0F172A;font-size:0.84rem">{name}</td>'
                        f'<td style="padding:9px 14px;white-space:nowrap">'
                        f'<span style="font-weight:800;color:#EE324E">{date or "—"}</span>'
                        f'<span style="font-family:Courier New,monospace;font-size:0.84rem;color:#EE324E;margin-left:5px">({fa})</span>'
                        f'</td>'
                        f'<td style="padding:9px 14px;text-align:center">'
                        f'<span style="font-family:Courier New,monospace;font-weight:800;font-size:0.9rem;color:{d15_color}">{d15}</span>'
                        f'</td>'
                        f'</tr>'
                    )
            return rows

        thead = (
            '<thead><tr style="background:#FFF5F5;border-bottom:2px solid #FECACA">'
            f'<th style="padding:8px 14px;text-align:left;font-size:0.68rem;font-weight:800;color:#B91C3C;text-transform:uppercase;letter-spacing:0.8px;white-space:nowrap">{t("table_player")}</th>'
            f'<th style="padding:8px 14px;text-align:left;font-size:0.68rem;font-weight:800;color:#B91C3C;text-transform:uppercase;letter-spacing:0.8px">{t("table_load_metric")}</th>'
            f'<th style="padding:8px 14px;text-align:left;font-size:0.68rem;font-weight:800;color:#B91C3C;text-transform:uppercase;letter-spacing:0.8px;white-space:nowrap">{t("table_enters_danger")}</th>'
            f'<th style="padding:8px 14px;text-align:center;font-size:0.68rem;font-weight:800;color:#B91C3C;text-transform:uppercase;letter-spacing:0.8px;white-space:nowrap">{t("table_day15_acwr")}</th>'
            '</tr></thead>'
        )

        recovered_names = ", ".join(
            f"{player_prefix} {p}" for p, _ in recovered
        )

        html_parts = [
            '<div style="font-family:Inter,system-ui,sans-serif;border:1px solid #FECACA;border-left:4px solid #EE324E;'
            'border-radius:10px;overflow:hidden;margin-bottom:4px;box-shadow:0 1px 6px rgba(238,50,78,0.08)">',
            # Header
            '<div style="background:#FFF5F5;padding:11px 16px;border-bottom:1px solid #FECACA;display:flex;align-items:center;gap:10px;flex-wrap:wrap">',
            '<span style="font-size:0.95rem">&#9888;&#65039;</span>',
            f'<span style="font-weight:800;color:#B91C3C;font-size:0.88rem;text-transform:uppercase;letter-spacing:0.6px">{t("injury_risk_alert")}</span>',
            f'<span style="color:#64748B;font-size:0.8rem">— {len(critical)} player{"s" if len(critical)!=1 else ""} {t("alert_still_in_danger")}</span>',
            '</div>',
        ]

        if critical:
            html_parts += [
                '<table style="width:100%;border-collapse:collapse;font-size:0.84rem;background:#fff">',
                thead,
                f'<tbody>{_table_rows_html(critical)}</tbody>',
                '</table>',
            ]

        if recovered:
            html_parts += [
                f'<div style="background:#FFF5F5;padding:8px 16px;border-top:1px solid #FECACA;font-size:0.78rem;color:#64748B">'
                f'<span style="font-weight:700;color:#B91C3C">{t("alert_recovered_note")}</span> {recovered_names}'
                f'</div>',
            ]

        html_parts.append('</div>')

        total_h = 56 + (len(critical) * 3) * 40 + (60 if recovered else 0)
        components.html("".join(html_parts), height=min(total_h, 520), scrolling=True)

    selector_col, status_col = st.columns([3, 2])
    player_prefix = t("player_prefix")
    with selector_col:
        pid_options = {f"{player_prefix} {pid} · {t_pos(str(player_data[pid]['position']))}": str(pid) for pid in all_pids}
        selected_label = st.selectbox(t("table_player"), list(pid_options.keys()), key="forecast_player_selector")
        selected_pid = pid_options[selected_label]
    with status_col:
        st.markdown("<div style='margin-top:1.85rem'></div>", unsafe_allow_html=True)
        status_text = t("status_stale") if forecast_is_stale else t("status_fresh")
        status_color = "#F59E0B" if forecast_is_stale else "#10B981"
        st.markdown(
            f'<div class="planner-status-banner" style="border-color:{status_color};color:{status_color};background:{status_color}12">{status_text}</div>',
            unsafe_allow_html=True,
        )

    show_load = st.checkbox(t("label_show_load"), key="forecast_show_load", value=False)

    for metric in TARGETS:
        meta = TARGET_META[metric]
        st.markdown(
            f'<div class="section-label" style="color:{meta["color"]};margin-top:1rem">{t(f"target_{metric}")} &nbsp;·&nbsp; {t("target_acwr_unit")}</div>',
            unsafe_allow_html=True,
        )
        metric_forecast = forecast[selected_pid][metric]
        meta_translated = {**meta, "label": t(f"target_{metric}")}
        fig = build_acwr_chart(metric_forecast, meta_translated, show_load=show_load)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    st.markdown(f'<div class="section-label">{t("section_day15_summary")}</div>', unsafe_allow_html=True)

    zone_order = ["danger", "caution", "undertraining", "optimal", "unknown"]
    header_cols = "".join(
        f'<th style="color:{TARGET_META[metric]["color"]}">{t(f"target_{metric}")}</th>'
        for metric in TARGETS
    )
    rows_html = ""
    for pid_str in [str(player_id) for player_id in all_pids]:
        pdata = player_data[int(pid_str)]
        cells = f'<td class="td-pid">{t("player_prefix")} {pid_str}</td><td class="td-pos">{t_pos(str(pdata["position"]))}</td>'
        worst_zone = "optimal"
        for metric in TARGETS:
            value = forecast[pid_str][metric]["day15_acwr"]
            zone = cast(str, forecast[pid_str][metric]["day15_zone"])
            color = ZONE_COLORS[zone]
            label = t(f"zone_{zone}")
            if zone_order.index(zone) < zone_order.index(worst_zone):
                worst_zone = zone
            value_string = f"{value:.2f}" if isinstance(value, int | float) else "—"
            cells += f"""
            <td>
                <span style="color:{color};font-weight:700;font-family:'Courier New',monospace">{value_string}</span>
                <span style="font-size:0.56rem;font-weight:800;text-transform:uppercase;
                             padding:2px 6px;border-radius:5px;border:1px solid {color};
                             color:{color};background:{color}18;margin-left:5px">{label}</span>
            </td>"""
        status_icons = {
            "danger": t("status_high_risk"),
            "caution": t("status_caution"),
            "optimal": t("status_ok"),
            "undertraining": t("status_low"),
        }
        status_color = ZONE_COLORS[worst_zone]
        status_label = status_icons.get(worst_zone, "—")
        cells += f'<td><span style="color:{status_color};font-size:0.65rem;font-weight:800;letter-spacing:0.8px">{status_label}</span></td>'
        rows_html += f"<tr>{cells}</tr>"

    st.markdown(
        f"""
        <div style="overflow-x:auto;border:1px solid #E2EBF6;border-radius:10px;
                    box-shadow:0 1px 6px rgba(0,60,140,0.06)">
            <table class="rm-table">
                <thead><tr><th>{t("table_player")}</th><th>{t("table_position")}</th>{header_cols}<th>{t("table_status")}</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>""",
        unsafe_allow_html=True,
    )


@st.dialog("Session Event", width="small")
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
        # Stable event_id across reruns — uuid4() called once, then reused from session state
        if "_dialog_event_id" not in st.session_state:
            st.session_state._dialog_event_id = str(uuid4())
        event_id = st.session_state._dialog_event_id
        selected_types: list[str] = []
        notes = ""
        dialog_title = t("dialog_create_title")
    else:
        start_dt = event_start_datetime(current_event)
        end_dt = event_end_datetime(current_event)
        event_id = str(current_event["id"])
        selected_types = _event_session_types(current_event)
        notes = str(current_event.get("notes", ""))
        dialog_title = t("dialog_edit_title")

    safe_start_date = start_dt.date() if isinstance(start_dt, datetime) else plan_dates[0].date()
    selected_date_value = safe_start_date
    start_time_value = start_dt.time()
    end_time_value = end_dt.time()

    st.markdown(f"**{dialog_title}**")
    st.markdown(f"**{t('dialog_date_label')}** {fmt_date_long(safe_start_date)}")

    st.markdown(f"**{t('dialog_training_types')}**")
    session_labels = get_session_labels()
    session_types = [
        stype for stype in SESSION_TYPES
        if st.checkbox(
            f"{stype} · {session_labels[stype]}",
            value=stype in selected_types,
            key=f"dialog_stype_{stype}_{event_id}",
        )
    ]

    notes_value = st.text_area(
        t("dialog_notes"),
        value=notes,
        placeholder=t("dialog_notes_placeholder"),
        height=120,
    )
    notes_text = notes_value or ""

    save_col, cancel_col = st.columns(2)
    if save_col.button(t("dialog_save"), type="primary", use_container_width=True):
        if not session_types:
            st.error(t("dialog_err_no_types"))
            st.stop()

        start_value = combine_date_and_time(selected_date_value, start_time_value)
        end_value = combine_date_and_time(selected_date_value, end_time_value)
        if end_value <= start_value:
            st.error(t("dialog_err_time"))
            st.stop()

        _upsert_event(
            build_event_record(
                event_id=event_id,
                start_dt=start_value,
                end_dt=end_value,
                session_types=session_types,
                location="",
                notes=notes_text,
            )
        )
        st.session_state.planner_dialog_request = None
        st.session_state.pop("_dialog_event_id", None)
        st.rerun()

    if cancel_col.button(t("dialog_cancel"), use_container_width=True):
        st.session_state.planner_dialog_request = None
        st.session_state.pop("_dialog_event_id", None)
        st.rerun()

    if current_event is not None and st.button(t("dialog_delete"), type="secondary", use_container_width=True):
        _remove_event(str(current_event["id"]))
        st.session_state.planner_dialog_request = None
        st.session_state.pop("_dialog_event_id", None)
        st.rerun()


def page_planner() -> None:
    """Render the unified planning and forecasting workspace."""
    player_data, all_pids, _, _daily = load_player_data()
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

    _render_standard_header(t("planner_title"), t("planner_sub"))
    st.caption(
        t("planner_caption").format(
            start=fmt_date_short(plan_dates[0]),
            end=fmt_date_medium(plan_dates[-1]),
        )
    )
    _render_planner_metrics(plan_summary, len(plan_events))
    st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)
    _render_session_legend()

    calendar_col, schedule_col = st.columns([2.25, 1], gap="large")
    with calendar_col:
        st.markdown(f'<div class="section-label">{t("section_session_calendar")}</div>', unsafe_allow_html=True)
        _handle_calendar_interactions(plan_dates)
        st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)
        action_cols = st.columns([1, 1.3])
        if action_cols[0].button(t("btn_clear_plan"), type="secondary", use_container_width=True):
            st.session_state.plan_events = []
            st.rerun()
        if action_cols[1].button(t("btn_run_forecast"), type="primary", use_container_width=True):
            _run_forecast(plan_dates)
            st.rerun()
    with schedule_col:
        st.markdown("<div style='margin-top:2.1rem'></div>", unsafe_allow_html=True)
        _render_schedule_sidebar(plan_dates, sort_events(st.session_state.plan_events), last_active)

    if st.session_state.get("planner_dialog_request") is not None:
        _event_editor_dialog(plan_dates)

    if "forecast" not in st.session_state:
        st.markdown("---")
        st.info(t("info_no_forecast"), icon="📊")
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


# =============================================================================
# Player Customization page
# =============================================================================

def page_player_customization() -> None:
    """Customise squad plan for an individual at-risk player and compare forecasts."""
    player_data, all_pids, current_acwr, _ = load_player_data()
    get_models_or_stop()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="padding:2rem 0 1.4rem;border-bottom:1px solid #E2EBF6;margin-bottom:1.8rem;text-align:center">
        <div style="display:inline-block;width:44px;height:4px;border-radius:2px;
                    background:linear-gradient(90deg,#FEBE10,#00529F);margin-bottom:1rem"></div>
        <div style="font-size:2.1rem;font-weight:900;color:#00529F;letter-spacing:-1px;margin-bottom:0.4rem">
            {t("pc_title")}
        </div>
        <div style="font-size:0.95rem;color:#64748B">
            {t("pc_subtitle")}
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Guard: squad forecast must exist ──────────────────────────────────────
    squad_forecast  = cast(dict, st.session_state.get("forecast"))
    squad_plan_days = st.session_state.get("plan_days", [])

    if not squad_forecast or not squad_plan_days:
        st.info(t("pc_no_forecast"), icon="📅")
        return

    player_prefix = t("player_prefix")
    zone_order    = ["danger", "caution", "undertraining", "optimal", "unknown"]

    # ── Identify at-risk players from squad forecast ──────────────────────────
    def _first_danger(pid: int) -> str | None:
        for m in TARGETS:
            mdata = squad_forecast.get(str(pid), {}).get(m, {})
            for d, v in zip(mdata.get("fore_dates", []), mdata.get("fore_acwr", []), strict=False):
                if v is not None and v >= 1.5:
                    return d
        return None

    at_risk_pids = [pid for pid in all_pids if _first_danger(pid) is not None]

    if not at_risk_pids:
        st.success(t("pc_no_danger"), icon="✅")
        return

    # At-risk summary chips
    chips = " ".join(
        f'<span style="display:inline-block;padding:3px 10px;margin:2px;border-radius:5px;'
        f'background:#EE324E14;color:#EE324E;font-weight:700;font-size:0.78rem;'
        f'border:1px solid #EE324E30">{player_prefix} {pid}</span>'
        for pid in at_risk_pids
    )
    st.markdown(
        f'<div style="margin-bottom:1.4rem"><span style="font-size:0.75rem;font-weight:800;'
        f'text-transform:uppercase;letter-spacing:0.8px;color:#EE324E;margin-right:8px">{t("pc_at_risk")}</span>'
        f'{chips}</div>',
        unsafe_allow_html=True,
    )

    # ── Player selector (default to first at-risk player) ────────────────────
    pid_options = {
        f"{player_prefix} {pid} · {t_pos(str(player_data[pid]['position']))}": pid
        for pid in at_risk_pids
    }
    all_pid_options = {
        f"{player_prefix} {pid} · {t_pos(str(player_data[pid]['position']))}": pid
        for pid in all_pids
    }
    show_all = st.checkbox(t("pc_show_all"), key="pc_show_all", value=False)
    options  = all_pid_options if show_all else pid_options

    selected_label = st.selectbox(t("pc_select_player"), list(options.keys()), key="pc_player_selector")
    selected_pid   = options[selected_label]
    pdata          = player_data[selected_pid]
    last_active    = pdata["last_active"]
    pid_str        = str(selected_pid)

    # ── Player profile + squad forecast status ────────────────────────────────
    worst_zone = min(
        (current_acwr[selected_pid][m]["zone"] for m in TARGETS),
        key=zone_order.index,
    )
    wcolor = ZONE_COLORS[worst_zone]

    prof_col, status_col = st.columns([1, 2.2], gap="large")
    with prof_col:
        first_danger_date = _first_danger(selected_pid) or "—"
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #D7E4F1;border-left:5px solid {wcolor};
                    border-radius:12px;padding:1.3rem 1.4rem;box-shadow:0 2px 8px rgba(0,60,140,0.06)">
            <div style="font-family:Courier New,monospace;font-size:1.1rem;font-weight:800;
                        color:#0F172A;margin-bottom:3px">{player_prefix} {selected_pid}</div>
            <div style="font-size:0.75rem;font-weight:800;color:#00529F;text-transform:uppercase;
                        letter-spacing:0.6px;margin-bottom:1rem">{t_pos(str(pdata["position"]))}</div>
            <div style="font-size:0.82rem;color:#475569;display:flex;flex-direction:column;gap:5px">
                <div><span style="font-weight:700;color:#334D6E">{t("label_last_active")}:</span> {fmt_date_medium(last_active)}</div>
                <div><span style="font-weight:700;color:#334D6E">{t("label_active_days")}:</span> {pdata["n_active_days"]}</div>
                <div><span style="font-weight:700;color:#EE324E">{t("label_enters_danger")}:</span>
                    <span style="color:#EE324E;font-weight:800">{first_danger_date}</span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with status_col:
        st.markdown(f'<div style="font-size:0.72rem;font-weight:800;text-transform:uppercase;letter-spacing:0.8px;color:#00529F;margin-bottom:8px">{t("pc_squad_forecast")}</div>', unsafe_allow_html=True)
        metric_cols = st.columns(3)
        for col, metric in zip(metric_cols, TARGETS, strict=False):
            sq = squad_forecast.get(pid_str, {}).get(metric, {})
            d15  = sq.get("day15_acwr")
            zone = sq.get("day15_zone", "unknown")
            zc   = ZONE_COLORS[zone]
            vs   = f"{d15:.2f}" if isinstance(d15, float) else "—"
            col.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid {zc}28;border-top:4px solid {zc};
                        border-radius:10px;padding:0.9rem 1rem;text-align:center">
                <div style="font-size:0.68rem;font-weight:800;text-transform:uppercase;
                            letter-spacing:0.6px;color:#64748B;margin-bottom:6px">{t(f"target_{metric}")}</div>
                <div style="font-family:Courier New,monospace;font-size:1.5rem;font-weight:900;
                            color:{zc};line-height:1">{vs}</div>
                <div style="font-size:0.65rem;font-weight:800;text-transform:uppercase;
                            color:{zc};margin-top:5px;padding:2px 7px;background:{zc}14;
                            border-radius:4px;display:inline-block">{t(f"zone_{zone}")}</div>
            </div>""", unsafe_allow_html=True)

    # ── Custom plan builder (pre-populated from squad plan) ───────────────────
    st.markdown("<div style='margin-top:1.8rem'></div>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-label">{t("pc_custom_plan_title")}</div>', unsafe_allow_html=True)
    st.caption(t("pc_custom_plan_caption"))

    session_labels = get_session_labels()
    custom_plan_days: list[dict] = []

    # Seed widget defaults from squad plan on first view of this player
    seed_key = f"pc_seeded_{selected_pid}"
    if seed_key not in st.session_state:
        for d, day in enumerate(squad_plan_days):
            st.session_state[f"pc_rest_{selected_pid}_{d}"]  = day.get("is_rest", True)
            st.session_state[f"pc_types_{selected_pid}_{d}"] = [s for s in SESSION_TYPES if day.get(s, False)]
        st.session_state[seed_key] = True

    for d in range(min(15, len(squad_plan_days))):
        squad_day = squad_plan_days[d]
        day_date  = last_active + pd.Timedelta(days=d + 1)

        col_label, col_rest, col_types = st.columns([1.5, 0.9, 4], gap="small")

        with col_label:
            squad_types = [s for s in SESSION_TYPES if squad_day.get(s, False)]
            squad_tag   = (
                " ".join(f'<span style="font-size:0.65rem;font-weight:700;color:#00529F;'
                         f'background:#EEF3FF;padding:0px 5px;border-radius:3px">{s}</span>'
                         for s in squad_types)
                if not squad_day.get("is_rest") and squad_types
                else f'<span style="font-size:0.65rem;color:#94A3B8">{t("pc_rest")}</span>'
            )
            st.markdown(
                f'<div style="padding-top:0.55rem">'
                f'<div style="font-size:0.85rem;font-weight:700;color:#334D6E">'
                f'{t("pc_day_label")} {d+1} <span style="font-weight:400;color:#94A3B8;font-size:0.77rem">'
                f'{day_date.strftime("%a %d %b")}</span></div>'
                f'<div style="margin-top:2px">{squad_tag}</div></div>',
                unsafe_allow_html=True,
            )

        with col_rest:
            is_rest = st.checkbox(t("pc_rest"), key=f"pc_rest_{selected_pid}_{d}")

        with col_types:
            if not is_rest:
                selected_types = st.multiselect(
                    "",
                    options=SESSION_TYPES,
                    format_func=lambda s: f"{s} · {session_labels[s]}",
                    key=f"pc_types_{selected_pid}_{d}",
                    label_visibility="collapsed",
                )
            else:
                selected_types = []

        custom_plan_days.append({
            "is_rest": is_rest,
            **{s: s in selected_types for s in SESSION_TYPES},
        })

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    btn_run, btn_reset, btn_clear = st.columns([2, 1.2, 1])

    if btn_run.button(t("pc_btn_run"), type="primary", use_container_width=True, key="pc_run"):
        with st.spinner(t("pc_spinner")):
            st.session_state[f"pc_forecast_{selected_pid}"] = build_player_forecast(custom_plan_days, selected_pid)

    if btn_reset.button(t("pc_btn_reset"), type="secondary", use_container_width=True, key="pc_reset"):
        st.session_state.pop(seed_key, None)
        st.session_state.pop(f"pc_forecast_{selected_pid}", None)
        st.rerun()

    if btn_clear.button(t("pc_btn_clear"), type="secondary", use_container_width=True, key="pc_clear"):
        st.session_state.pop(f"pc_forecast_{selected_pid}", None)
        st.rerun()

    # ── Comparison: Squad Plan vs Custom Plan ─────────────────────────────────
    custom_forecast = st.session_state.get(f"pc_forecast_{selected_pid}")
    if custom_forecast is None:
        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
        st.info(t("pc_info_no_forecast"), icon="📊")
        return

    st.markdown("---")
    st.markdown(f'<div class="section-label" style="margin-top:0.5rem">{t("pc_comparison_title")}</div>', unsafe_allow_html=True)

    # Day-15 comparison summary
    cmp_cols = st.columns(3)
    for col, metric in zip(cmp_cols, TARGETS, strict=False):
        sq_d15   = squad_forecast.get(pid_str, {}).get(metric, {}).get("day15_acwr")
        cu_d15   = custom_forecast.get(pid_str, {}).get(metric, {}).get("day15_acwr")
        cu_zone  = custom_forecast.get(pid_str, {}).get(metric, {}).get("day15_zone", "unknown")
        zc       = ZONE_COLORS[cu_zone]
        sq_str   = f"{sq_d15:.2f}" if isinstance(sq_d15, float) else "—"
        cu_str   = f"{cu_d15:.2f}" if isinstance(cu_d15, float) else "—"
        delta    = (cu_d15 - sq_d15) if isinstance(cu_d15, float) and isinstance(sq_d15, float) else None
        arrow    = ("▼ " + f"{abs(delta):.2f}") if delta is not None and delta < 0 else \
                   ("▲ " + f"{abs(delta):.2f}") if delta is not None and delta > 0 else "—"
        arrow_color = "#10B981" if delta is not None and delta < 0 else "#EE324E" if delta is not None and delta > 0 else "#94A3B8"
        col.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid {zc}28;border-top:4px solid {zc};
                    border-radius:10px;padding:1rem;text-align:center;margin-bottom:1rem">
            <div style="font-size:0.68rem;font-weight:800;text-transform:uppercase;
                        letter-spacing:0.6px;color:#64748B;margin-bottom:8px">{t(f"target_{metric}")}</div>
            <div style="display:flex;justify-content:space-around;align-items:center;margin-bottom:8px">
                <div>
                    <div style="font-size:0.6rem;color:#94A3B8;font-weight:600;margin-bottom:2px">{t("label_squad")}</div>
                    <div style="font-family:Courier New,monospace;font-size:1.1rem;font-weight:800;color:#64748B">{sq_str}</div>
                </div>
                <div style="font-size:1.2rem;color:#D7E4F1">→</div>
                <div>
                    <div style="font-size:0.6rem;color:{zc};font-weight:600;margin-bottom:2px">{t("label_custom")}</div>
                    <div style="font-family:Courier New,monospace;font-size:1.1rem;font-weight:800;color:{zc}">{cu_str}</div>
                </div>
            </div>
            <div style="font-size:0.75rem;font-weight:800;color:{arrow_color}">{arrow}</div>
        </div>""", unsafe_allow_html=True)

    # Per-metric charts with both squad + custom forecast overlaid
    show_load = st.checkbox(t("label_show_load_pc"), key="pc_show_load", value=False)

    for metric in TARGETS:
        meta = TARGET_META[metric]
        st.markdown(
            f'<div class="section-label" style="color:{meta["color"]};margin-top:0.8rem">'
            f'{t(f"target_{metric}")} &nbsp;·&nbsp; {t("target_acwr_unit")}</div>',
            unsafe_allow_html=True,
        )

        sq_mdata = squad_forecast.get(pid_str, {}).get(metric, {})
        cu_mdata = custom_forecast.get(pid_str, {}).get(metric, {})
        meta_translated = {**meta, "label": t(f"target_{metric}")}

        import plotly.graph_objects as go  # noqa: PLC0415


        # Build chart from custom forecast (history shared, two forecast lines)
        fig = build_acwr_chart(cu_mdata, meta_translated, show_load=show_load)

        # Add squad forecast line as reference
        sq_fore_x = sq_mdata.get("fore_dates", [])
        sq_fore_y = sq_mdata.get("fore_acwr", [])
        if sq_fore_x and sq_fore_y:
            fig.add_trace(go.Scatter(
                x=sq_fore_x, y=sq_fore_y,
                mode="lines",
                name=t("pc_squad_plan_trace"),
                line=dict(color="rgba(100,116,139,0.5)", width=2, dash="dash"),
                connectgaps=False,
                hoverinfo="skip",
            ))

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _player_effective_plan(pid: int, squad_plan: list[dict]) -> list[dict]:
    """Return the effective plan for a player: custom if the coach modified it, otherwise squad."""
    if f"pc_seeded_{pid}" not in st.session_state:
        return squad_plan
    return [
        {
            "is_rest": st.session_state.get(f"pc_rest_{pid}_{d}", True),
            **{
                s: s in st.session_state.get(f"pc_types_{pid}_{d}", [])
                for s in SESSION_TYPES
            },
        }
        for d in range(len(squad_plan))
    ]


def _cell_text(day: dict) -> str:
    if day.get("is_rest", True):
        return "REST"
    types = [s for s in SESSION_TYPES if day.get(s, False)]
    return " · ".join(types) if types else "REST"


def page_export_plan() -> None:
    """Render the training plan export page."""
    import base64

    import streamlit.components.v1 as components

    player_data, all_pids, _, _ = load_player_data()
    get_models_or_stop()

    # ── Header row ────────────────────────────────────────────────────────────
    hdr_col, btn_col = st.columns([3, 1])
    with hdr_col:
        st.markdown(f"""
        <div style="padding:1.8rem 0 1.2rem;border-bottom:1px solid #E2EBF6;margin-bottom:1.6rem">
            <div style="display:inline-block;width:40px;height:4px;border-radius:2px;
                        background:linear-gradient(90deg,#FEBE10,#00529F);margin-bottom:0.8rem"></div>
            <div style="font-size:2rem;font-weight:900;color:#00529F;letter-spacing:-1px;margin-bottom:0.3rem">
                {t("ep_title")}
            </div>
            <div style="font-size:0.9rem;color:#64748B">
                {t("ep_subtitle")}
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Guard ─────────────────────────────────────────────────────────────────
    squad_plan: list[dict] = st.session_state.get("plan_days", [])
    plan_dates_labels: list[str] = st.session_state.get("plan_dates", [])

    if not squad_plan:
        st.info(t("ep_no_plan"), icon="📅")
        return

    # ── Build column headers (date labels) ────────────────────────────────────
    last_active = max(player_data[pid]["last_active"] for pid in all_pids)
    if plan_dates_labels:
        col_headers = plan_dates_labels
    else:
        col_headers = [
            (last_active + pd.Timedelta(days=d + 1)).strftime("%a %d %b")
            for d in range(len(squad_plan))
        ]

    # ── Sort players by position ───────────────────────────────────────────────
    pos_order = {"Full Back": 0, "Central Back": 1, "Central Midfielder": 2, "Winger": 3, "Forward": 4, "Unknown": 99}
    sorted_pids = sorted(all_pids, key=lambda pid: pos_order.get(player_data[pid]["position"], 99))

    # ── Read logo ─────────────────────────────────────────────────────────────
    logo_b64 = ""
    try:
        logo_b64 = base64.b64encode(STATIC_DIR.joinpath("img", "Real-Madrid-CF-v2002.svg").read_bytes()).decode()
    except Exception:
        pass

    # ── Build HTML table rows ─────────────────────────────────────────────────
    SESSION_COLORS_MAP = SESSION_COLORS  # already imported

    def _cell_html(day: dict) -> str:
        if day.get("is_rest", True):
            return f'<span style="color:#94A3B8;font-size:0.72rem;font-weight:600">{t("ep_rest_cell")}</span>'
        types = [s for s in SESSION_TYPES if day.get(s, False)]
        if not types:
            return f'<span style="color:#94A3B8;font-size:0.72rem;font-weight:600">{t("ep_rest_cell")}</span>'
        badges = "".join(
            f'<span style="display:inline-block;padding:1px 5px;margin:1px;border-radius:3px;'
            f'font-size:0.65rem;font-weight:800;letter-spacing:0.3px;'
            f'background:{SESSION_COLORS_MAP.get(s,"#64748B")}22;'
            f'color:{SESSION_COLORS_MAP.get(s,"#64748B")};'
            f'border:1px solid {SESSION_COLORS_MAP.get(s,"#64748B")}44">{s}</span>'
            for s in types
        )
        return badges

    header_cells = "".join(
        f'<th style="padding:6px 8px;white-space:nowrap;font-size:0.68rem;font-weight:800;'
        f'text-transform:uppercase;letter-spacing:0.5px;color:#00529F;background:#F0F4FA;'
        f'border:1px solid #D7E4F1;text-align:center">{h}</th>'
        for h in col_headers
    )

    body_rows = ""
    for pid in sorted_pids:
        pos  = t_pos(str(player_data[pid]["position"]))
        plan = _player_effective_plan(pid, squad_plan)
        customised = f"pc_seeded_{pid}" in st.session_state
        badge = (
            f'<span style="font-size:0.58rem;font-weight:800;background:#FEBE1022;color:#B8920A;'
            f'border:1px solid #FEBE1044;border-radius:3px;padding:0px 4px;margin-left:4px">{t("ep_custom_badge")}</span>'
            if customised else ""
        )
        cells = "".join(
            f'<td style="padding:6px 8px;border:1px solid #E2EBF6;text-align:center;'
            f'min-width:70px">{_cell_html(d)}</td>'
            for d in plan
        )
        body_rows += (
            f'<tr>'
            f'<td style="padding:6px 10px;border:1px solid #D7E4F1;white-space:nowrap;'
            f'background:#F8FAFD;font-family:Courier New,monospace;font-size:0.82rem;font-weight:700;'
            f'color:#0F172A;position:sticky;left:0;z-index:1">'
            f'{pid}{badge}'
            f'<div style="font-size:0.62rem;font-weight:700;color:#00529F;text-transform:uppercase;'
            f'letter-spacing:0.4px;margin-top:1px">{pos}</div>'
            f'</td>'
            f'{cells}'
            f'</tr>'
        )

    logo_tag = (
        f'<img src="data:image/svg+xml;base64,{logo_b64}" '
        f'style="width:64px;height:64px;display:block;margin-bottom:8px" />'
        if logo_b64 else ""
    )

    n_custom = sum(1 for pid in all_pids if f"pc_seeded_{pid}" in st.session_state)
    note = (
        f'<p style="font-size:0.75rem;color:#64748B;margin:0 0 12px 0">'
        f'{n_custom} player{"s have" if n_custom != 1 else " has"} a customised plan '
        f'(marked <span style="background:#FEBE1022;color:#B8920A;border:1px solid #FEBE1044;'
        f'border-radius:3px;padding:0 4px;font-weight:800;font-size:0.68rem">{t("ep_custom_badge")}</span>).'
        f'</p>'
        if n_custom else ""
    )

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Inter, system-ui, sans-serif; background: #fff; color: #0F172A; padding: 0; }}
  .toolbar {{
    padding: 10px 16px;
    background: #F0F4FA;
    border-bottom: 1px solid #D7E4F1;
    display: flex;
    justify-content: flex-end;
    gap: 8px;
  }}
  .btn-print {{
    background: linear-gradient(135deg,#FEBE10,#FFD84A);
    color: #0F172A;
    font-weight: 800;
    font-size: 0.85rem;
    border: none;
    border-radius: 7px;
    padding: 8px 20px;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(254,190,16,0.4);
    letter-spacing: 0.3px;
  }}
  .btn-print:hover {{ box-shadow: 0 4px 16px rgba(254,190,16,0.55); }}
  .content {{ padding: 20px 24px; }}
  .doc-header {{ display:flex; align-items:center; gap:18px; margin-bottom:20px; padding-bottom:14px; border-bottom:2px solid #00529F; }}
  .doc-title {{ font-size:1.5rem; font-weight:900; color:#00529F; letter-spacing:-0.5px; }}
  .doc-sub {{ font-size:0.8rem; color:#64748B; margin-top:3px; }}
  .table-wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.8rem; }}
  thead th:first-child {{
    position: sticky; left: 0; z-index: 2; background: #E8EFF8;
  }}
  .legend {{ margin-top: 14px; font-size: 0.72rem; color: #64748B; }}
  @page {{ size: A3 landscape; margin: 14mm 12mm; }}
  @media print {{
    .toolbar {{ display: none !important; }}
    body {{ padding: 0; }}
    .content {{ padding: 10px 14px; }}
    table {{ font-size: 0.68rem; }}
    thead th {{ background: #E8EFF8 !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    td span {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  }}
</style>
</head>
<body>
<div class="toolbar">
  <button class="btn-print" onclick="window.print()">{t("ep_btn_export")}</button>
</div>
<div class="content">
  <div class="doc-header">
    {logo_tag}
    <div>
      <div class="doc-title">{t("ep_pdf_title")}</div>
      <div class="doc-sub">{t("ep_pdf_subtitle")} &nbsp;·&nbsp; {t("ep_pdf_schedule").format(n=len(squad_plan), p=len(all_pids))}</div>
    </div>
  </div>
  {note}
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th style="padding:6px 10px;font-size:0.68rem;font-weight:800;text-transform:uppercase;
                     letter-spacing:0.5px;color:#00529F;background:#E8EFF8;border:1px solid #D7E4F1;
                     white-space:nowrap;position:sticky;left:0;z-index:2">{t("ep_col_player")}</th>
          {header_cells}
        </tr>
      </thead>
      <tbody>{body_rows}</tbody>
    </table>
  </div>
  <div class="legend" style="margin-top:14px;display:flex;gap:14px;flex-wrap:wrap">
    {''.join(f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{SESSION_COLORS_MAP.get(s,"#64748B")};margin-right:4px;vertical-align:middle"></span><strong>{s}</strong> &nbsp;{get_session_labels().get(s,"")}</span>' for s in SESSION_TYPES)}
    <span><span style="color:#94A3B8">{t("ep_rest_cell")}</span> {t("ep_legend_rest")}</span>
  </div>
</div>
</body>
</html>"""

    # ── Show in Streamlit ──────────────────────────────────────────────────────
    # Calculate height: header ~80px + toolbar ~44px + n_players * 42px + legend ~50px
    table_height = min(80 + 44 + len(all_pids) * 42 + 80, 820)
    components.html(full_html, height=table_height, scrolling=True)


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
                {t("sidebar_app_name")}
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        page = st.radio(
            t("sidebar_navigation"),
            PAGES,
            key="nav_page",
            label_visibility="collapsed",
            format_func=lambda p: (
                t("nav_dashboard") if p == PAGES[0]
                else t("nav_planner") if p == PAGES[1]
                else t("nav_player_customization") if p == PAGES[2]
                else t("nav_export_plan")
            ),
        )
        st.markdown("---")


        team_logo_path = STATIC_DIR / "img" / "trAIn_labs.png"
        team_b64 = ""
        if team_logo_path.exists():
            team_b64 = base64.b64encode(team_logo_path.read_bytes()).decode()

        st.markdown("---")
        st.markdown(f"""
        <div style="padding:0.5rem 1rem 1.5rem;text-align:center">
            <div style="font-size:0.68rem;font-weight:600;letter-spacing:1.5px;
                        color:rgba(255,255,255,0.35);text-transform:uppercase;margin-bottom:10px">
                {t("sidebar_developed_by")}
            </div>
            {'<div style="display:inline-block;background:#FFFFFF;border-radius:8px;padding:6px 14px"><img src="data:image/png;base64,' + team_b64 + '" style="width:110px;display:block"></div>' if team_b64 else '<span style="color:rgba(255,255,255,0.5);font-weight:700">trAIn Labs</span>'}
        </div>""", unsafe_allow_html=True)

        def _on_lang_change() -> None:
            # Preserve the active page across language changes.
            # format_func switching can cause Streamlit to reset the nav radio;
            # writing to _pending_nav ensures main.py restores the correct page.
            st.session_state["_pending_nav"] = st.session_state.get("nav_page", PAGES[0])

        st.radio(
            t("sidebar_language"),
            ["ENG", "ESP"],
            key="lang",
            horizontal=True,
            label_visibility="collapsed",
            on_change=_on_lang_change,
        )

        return page
