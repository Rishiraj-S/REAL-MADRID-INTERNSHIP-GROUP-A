"""Streamlit page renderers for the ACWR monitor."""

from __future__ import annotations

import base64

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
from real_madrid_acwr.config import STATIC_DIR


def page_dashboard():
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

    # Stat cards
    n_danger  = sum(1 for p in all_pids for m in TARGETS if current_acwr[p][m]["zone"] == "danger")
    n_caution = sum(1 for p in all_pids for m in TARGETS if current_acwr[p][m]["zone"] == "caution")
    n_optimal = sum(1 for p in all_pids for m in TARGETS if current_acwr[p][m]["zone"] == "optimal")

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

    # Zone legend
    st.markdown('<div class="section-label" style="font-size:1rem;letter-spacing:0.5px">Risk Zones</div>', unsafe_allow_html=True)
    lc = st.columns(4)
    for i, (zone, rng) in enumerate([
        ("undertraining", "ACWR < 0.8"),
        ("optimal",       "ACWR 0.8 – 1.3"),
        ("caution",       "ACWR 1.3 – 1.5"),
        ("danger",        "ACWR ≥ 1.5"),
    ]):
        col = ZONE_COLORS[zone]
        lbl = ZONE_LABELS[zone]
        lc[i].markdown(f"""
        <div class="zone-pill" style="color:{col};border-color:{col};background:{col}15">
            <span class="zone-dot" style="background:{col}"></span>
            {lbl} &nbsp; <span style="font-weight:400;opacity:0.8">{rng}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label" style="font-size:1rem;letter-spacing:0.5px">Player Status — Current ACWR</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    for i, pid in enumerate(all_pids):
        pdata     = player_data[pid]
        acwr      = current_acwr[pid]

        worst_zone = "optimal"
        zone_order = ["danger", "caution", "undertraining", "optimal", "unknown"]
        for m in TARGETS:
            z = acwr[m]["zone"]
            if zone_order.index(z) < zone_order.index(worst_zone):
                worst_zone = z

        rows_html = ""
        for m in TARGETS:
            ma    = acwr[m]
            col   = ZONE_COLORS[ma["zone"]]
            val_s = f"{ma['value']:.2f}" if ma["value"] is not None else "—"
            rows_html += f"""
            <div class="metric-row">
                <span class="metric-lbl" style="color:{TARGET_META[m]['color']}">{TARGET_META[m]['label']}</span>
                <div class="metric-rhs">
                    <span class="metric-val" style="color:{col}">{val_s}</span>
                    <span class="metric-badge" style="color:{col};border-color:{col};background:{col}18">
                        {ZONE_LABELS[ma['zone']]}
                    </span>
                </div>
            </div>"""

        with cols[i % 4]:
            st.markdown(f"""
            <div class="player-card {worst_zone}">
                <div class="card-id">{pid}</div>
                <div class="card-pos">{pdata['position']}</div>
                <hr class="card-rule">
                {rows_html}
            </div>""", unsafe_allow_html=True)


# ── Page: Session Planner ─────────────────────────────────────────────────────
def page_planner():
    player_data, all_pids, _ = load_player_data()
    last_active = player_data[all_pids[0]]["last_active"]
    plan_dates  = [(last_active + pd.Timedelta(days=i + 1)).strftime("%a %d %b") for i in range(15)]

    st.markdown("""
    <div class="page-header">
        <div class="page-title">Session <span>Planner</span></div>
        <div class="page-sub">
            Select session types for each of the next 15 days.
            Leaving all unchecked marks the day as REST.
        </div>
    </div>""", unsafe_allow_html=True)

    # Session type legend
    st.markdown('<div class="section-label" style="font-size:1rem;letter-spacing:0.5px">Session Types</div>', unsafe_allow_html=True)
    lc = st.columns(6)
    for i, (t, lbl) in enumerate({**SESSION_LABELS, "REST": "Rest Day"}.items()):
        col = SESSION_COLORS.get(t, "#64748B")
        lc[i % 6].markdown(f"""
        <div style="border:1px solid {col};border-radius:20px;padding:5px 12px;
                    font-size:0.7rem;font-weight:700;color:{col};
                    text-transform:uppercase;letter-spacing:0.4px;
                    text-align:center;margin-bottom:10px;background:{col}0D">
            {t} &middot; {lbl}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    init = pd.DataFrame({
        "Day":   [f"Day {i+1:02d}" for i in range(15)],
        "Date":  plan_dates,
        "G":     [False] * 15,
        "TAC":   [False] * 15,
        "BP":    [False] * 15,
        "TEC":   [False] * 15,
        "MATCH": [False] * 15,
    })

    edited = st.data_editor(
        init,
        column_config={
            "Day":   st.column_config.TextColumn("Day",   disabled=True, width="small"),
            "Date":  st.column_config.TextColumn("Date",  disabled=True, width="medium"),
            "G":     st.column_config.CheckboxColumn("G",     help="Game / Small-Sided Game"),
            "TAC":   st.column_config.CheckboxColumn("TAC",   help="Tactical session"),
            "BP":    st.column_config.CheckboxColumn("BP",    help="Set pieces"),
            "TEC":   st.column_config.CheckboxColumn("TEC",   help="Technical drills"),
            "MATCH": st.column_config.CheckboxColumn("MATCH", help="Official match"),
        },
        hide_index=True,
        use_container_width=True,
        height=576,
        key="plan_editor",
    )

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    _, btn_col = st.columns([4, 1])
    if btn_col.button("Run Forecast →", type="primary", use_container_width=True):
        plan_days = []
        for _, row in edited.iterrows():
            selected = [t for t in SESSION_TYPES if row.get(t, False)]
            plan_days.append({
                "is_rest": len(selected) == 0,
                **{t: (t in selected) for t in SESSION_TYPES},
            })

        with st.spinner("Computing 15-day ACWR forecasts for all 28 players…"):
            results = build_forecast(plan_days)

        if results:
            st.session_state.forecast   = results
            st.session_state.plan_days  = plan_days
            st.session_state.plan_dates = plan_dates
            st.session_state._pending_nav = PAGES[2]  # stage nav change; resolved at script top before radio renders
            st.rerun()
        else:
            st.error("Forecast failed — check models are trained (`python train_models.py`).")


# ── Page: Forecast Results ────────────────────────────────────────────────────
def page_results():
    if "forecast" not in st.session_state:
        st.markdown("""
        <div class="page-header">
            <div class="page-title">Forecast <span>Results</span></div>
            <div class="page-sub">No forecast computed yet.</div>
        </div>""", unsafe_allow_html=True)
        st.info("Go to **Plan Sessions** and click **Run Forecast →** to generate results.")
        return

    player_data, all_pids, _ = load_player_data()
    forecast   = st.session_state.forecast
    plan_days  = st.session_state.get("plan_days", [])
    plan_dates = st.session_state.get("plan_dates", [])

    st.markdown(f"""
    <div class="page-header">
        <div class="page-title">Forecast <span>Results</span></div>
        <div class="page-sub">
            15-day ACWR projection &nbsp;&middot;&nbsp;
            {len(all_pids)} players &nbsp;&middot;&nbsp;
            3 load metrics
        </div>
    </div>""", unsafe_allow_html=True)

    # Planned sessions — weekly calendar
    st.markdown('<div class="section-label" style="font-size:1rem;letter-spacing:0.5px">Planned Sessions</div>', unsafe_allow_html=True)

    DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_index = {d: i for i, d in enumerate(DOW_ORDER)}
    first_dow = dow_index.get(plan_dates[0][:3], 0)

    # Pad start with empty slots so day 1 lands on its weekday column
    slots = [None] * first_dow + list(range(len(plan_days)))
    while len(slots) % 7:
        slots.append(None)
    weeks = [slots[i:i + 7] for i in range(0, len(slots), 7)]

    header_cells = "".join(
        f'<th style="width:14.28%;padding:6px 0;font-size:0.65rem;font-weight:800;'
        f'text-transform:uppercase;letter-spacing:1.2px;color:#00529F;'
        f'text-align:center;border-bottom:2px solid #E2EBF6">{d}</th>'
        for d in DOW_ORDER
    )

    rows_html = ""
    for week in weeks:
        row = ""
        for idx in week:
            if idx is None:
                row += '<td style="padding:4px;background:#F8FAFD;border:1px solid #EEF3FA"></td>'
            else:
                day      = plan_days[idx]
                date_str = plan_dates[idx]
                day_num  = date_str[4:6]
                mon_str  = date_str[7:]
                types    = [t for t in SESSION_TYPES if day.get(t)]
                is_rest  = len(types) == 0
                is_match = "MATCH" in types

                bg = "#FFF8F8" if is_match else ("#F8FAFD" if is_rest else "#FFFFFF")
                border_top = "3px solid #EE324E" if is_match else ("3px solid #E2EBF6" if is_rest else "3px solid #FEBE10")

                badges = ""
                if is_rest:
                    badges = '<span style="font-size:0.62rem;color:#94A3B8;font-weight:600">REST</span>'
                else:
                    for t in types:
                        c = SESSION_COLORS[t]
                        badges += (
                            f'<span style="display:inline-block;margin:1px 2px;'
                            f'padding:2px 6px;border-radius:4px;font-size:0.6rem;'
                            f'font-weight:800;background:{c}22;color:{c};'
                            f'border:1px solid {c}66">{t}</span>'
                        )

                row += f"""
                <td style="padding:4px;border:1px solid #EEF3FA;vertical-align:top">
                  <div style="background:{bg};border-radius:7px;padding:7px 8px;
                              border-top:{border_top};min-height:72px">
                    <div style="font-size:0.7rem;font-weight:700;color:#334D6E;
                                margin-bottom:4px;line-height:1">
                      {day_num}
                      <span style="font-size:0.6rem;font-weight:500;color:#94A3B8">{mon_str}</span>
                    </div>
                    <div style="display:flex;flex-wrap:wrap;gap:2px">{badges}</div>
                  </div>
                </td>"""
        rows_html += f"<tr>{row}</tr>"

    st.markdown(f"""
    <div style="border:2px solid #3D4A5C;border-radius:10px;overflow:hidden;
                box-shadow:0 1px 6px rgba(0,60,140,0.06);margin-bottom:1.5rem">
        <table style="width:100%;border-collapse:collapse;background:#F0F4FA">
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>""", unsafe_allow_html=True)

    # Danger alert
    danger_entries = []
    for p in all_pids:
        bad_metrics = [
            TARGET_META[m]["label"]
            for m in TARGETS
            if forecast[str(p)][m]["day15_zone"] == "danger"
        ]
        if bad_metrics:
            danger_entries.append((str(p), bad_metrics))

    if danger_entries:
        rows = "".join(
            f'<div style="margin-top:4px">&#x2022; Player <strong>{p}</strong> — '
            f'{", ".join(ms)}</div>'
            for p, ms in danger_entries
        )
        st.markdown(f"""
        <div class="rm-alert">
            <div>
                <strong>Injury Risk Alert:</strong> {len(danger_entries)} player(s) projected
                in DANGER zone by Day 15:
                {rows}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Player selector + back button
    c1, c2 = st.columns([3, 1])
    with c1:
        pid_opts      = {f"Player {p} · {player_data[p]['position']}": str(p) for p in all_pids}
        sel_pid_label = st.selectbox("PLAYER", list(pid_opts.keys()), label_visibility="visible")
        sel_pid       = pid_opts[sel_pid_label]
    with c2:
        st.markdown("<div style='margin-top:1.65rem'></div>", unsafe_allow_html=True)
        if st.button("← Adjust Plan", type="secondary", use_container_width=True):
            st.session_state._pending_nav = PAGES[1]  # stage nav change; resolved at script top before radio renders
            st.rerun()

    # Three charts stacked
    for metric in TARGETS:
        meta = TARGET_META[metric]
        st.markdown(
            f'<div class="section-label" style="color:{meta["color"]};margin-top:1rem">'
            f'{meta["label"]} &nbsp;·&nbsp; ACWR (unitless)</div>',
            unsafe_allow_html=True,
        )
        fig = build_acwr_chart(forecast[sel_pid][metric], meta)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Day-15 summary table
    st.markdown("---")
    st.markdown('<div class="section-label">Day-15 Summary — All Players</div>', unsafe_allow_html=True)

    ZONE_ORDER   = ["danger", "caution", "undertraining", "optimal", "unknown"]
    header_cols  = "".join(
        f'<th style="color:{TARGET_META[m]["color"]}">{TARGET_META[m]["label"]}</th>'
        for m in TARGETS
    )
    rows_html = ""
    for pid in [str(p) for p in all_pids]:
        pdata = player_data[int(pid)]
        cells = f'<td class="td-pid">Player {pid}</td><td class="td-pos">{pdata["position"]}</td>'
        worst = "optimal"
        for m in TARGETS:
            v    = forecast[pid][m]["day15_acwr"]
            zone = forecast[pid][m]["day15_zone"]
            col  = ZONE_COLORS[zone]
            lbl  = ZONE_LABELS[zone]
            if ZONE_ORDER.index(zone) < ZONE_ORDER.index(worst):
                worst = zone
            val_s = f"{v:.2f}" if v is not None else "—"
            cells += f"""
            <td>
                <span style="color:{col};font-weight:700;font-family:'Courier New',monospace">{val_s}</span>
                <span style="font-size:0.56rem;font-weight:800;text-transform:uppercase;
                             padding:2px 6px;border-radius:5px;border:1px solid {col};
                             color:{col};background:{col}18;margin-left:5px">{lbl}</span>
            </td>"""
        status_icons = {"danger": "HIGH RISK", "caution": "CAUTION", "optimal": "OK", "undertraining": "LOW"}
        icol  = ZONE_COLORS[worst]
        itext = status_icons.get(worst, "—")
        cells += f'<td><span style="color:{icol};font-size:0.65rem;font-weight:800;letter-spacing:0.8px">{itext}</span></td>'
        rows_html += f"<tr>{cells}</tr>"

    st.markdown(f"""
    <div style="overflow-x:auto;border:1px solid #E2EBF6;border-radius:10px;
                box-shadow:0 1px 6px rgba(0,60,140,0.06)">
        <table class="rm-table">
            <thead><tr><th>Player</th><th>Position</th>{header_cols}<th>Status</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>""", unsafe_allow_html=True)


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

        page = st.radio(
            "Navigation",
            PAGES,
            key="nav_page",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Season info footer
        st.markdown("""
        <div style="padding:0.5rem 1rem 1rem;font-size:0.65rem;color:rgba(255,255,255,0.4);line-height:1.8">
            <div style="font-weight:700;color:rgba(255,255,255,0.6);
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">
                Season 2024/25
            </div>
            <div>28 Players &nbsp;&middot;&nbsp; 3 Metrics</div>
            <div>EWMA · α<sub>acute</sub>=0.25 · α<sub>chronic</sub>≈0.07</div>
        </div>""", unsafe_allow_html=True)

        # Developed by footer
        _team_logo_path = STATIC_DIR / "img" / "trAIn_labs.png"
        _team_b64 = ""
        if _team_logo_path.exists():
            _team_b64 = base64.b64encode(_team_logo_path.read_bytes()).decode()

        st.markdown("---")
        st.markdown(f"""
        <div style="padding:0.5rem 1rem 1.5rem;text-align:center">
            <div style="font-size:0.68rem;font-weight:600;letter-spacing:1.5px;
                        color:rgba(255,255,255,0.35);text-transform:uppercase;margin-bottom:10px">
                Developed by
            </div>
            {'<div style="display:inline-block;background:#FFFFFF;border-radius:8px;padding:6px 14px"><img src="data:image/png;base64,' + _team_b64 + '" style="width:110px;display:block"></div>' if _team_b64 else '<span style="color:rgba(255,255,255,0.5);font-weight:700">trAIn Labs</span>'}
        </div>""", unsafe_allow_html=True)

        return page
