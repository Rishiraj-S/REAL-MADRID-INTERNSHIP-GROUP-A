"""Constants for the Streamlit ACWR application."""

from __future__ import annotations

import pandas as pd

SEASON_START = pd.Timestamp("2024-07-15")
# Ordered to match has_* columns in model_data.parquet; changing order breaks feature alignment.
SESSION_TYPES = ["G", "TAC", "BP", "TEC", "MATCH"]
PAGES = ["Dashboard", "Planning & Forecast"]
TARGETS = ("total_distance", "acc_total", "vel_total")

# Colors map to Real Madrid's official palette: navy (#00529F), gold (#FEBE10), red (#EE324E).
TARGET_META = {
    "total_distance": {
        "label": "Total Distance",
        "unit": "m",
        "color": "#00529F",
        "fill": "rgba(0,82,159,0.12)",
    },
    "acc_total": {
        "label": "Accelerations",
        "unit": "efforts",
        "color": "#FEBE10",
        "fill": "rgba(254,190,16,0.12)",
    },
    "vel_total": {
        "label": "High-Speed Running",
        "unit": "m",
        "color": "#EE324E",
        "fill": "rgba(238,50,78,0.12)",
    },
}
SESSION_COLORS = {
    "G": "#00529F",
    "TAC": "#FEBE10",
    "BP": "#8B5CF6",
    "TEC": "#10B981",
    "MATCH": "#EE324E",
}
SESSION_LABELS = {
    "G": "Game / SSG",
    "TAC": "Tactical",
    "BP": "Set Pieces",
    "TEC": "Technical",
    "MATCH": "Official Match",
}
ZONE_COLORS = {
    "undertraining": "#64748B",
    "optimal": "#10B981",
    "caution": "#F59E0B",
    "danger": "#EE324E",
    "unknown": "#94A3B8",
}
ZONE_LABELS = {
    "undertraining": "Under",
    "optimal": "Optimal",
    "caution": "Caution",
    "danger": "Danger",
    "unknown": "—",
}
# Caution/danger bands use slightly higher opacity (0.22) than others to draw the eye to risk zones.
ZONE_BANDS = [
    {"y0": 0, "y1": 0.8, "color": "rgba(100,116,139,0.18)"},
    {"y0": 0.8, "y1": 1.3, "color": "rgba(16,185,129,0.18)"},
    {"y0": 1.3, "y1": 1.5, "color": "rgba(245,158,11,0.22)"},
    {"y0": 1.5, "y1": 2.5, "color": "rgba(238,50,78,0.22)"},
]

TRANSLATIONS: dict[str, dict[str, str]] = {
    "ENG": {
        # Navigation
        "nav_dashboard": "Dashboard",
        "nav_planner": "Planning & Forecast",
        # Dashboard
        "dashboard_title": "Squad <span>ACWR</span> Dashboard",
        "dashboard_sub": "Data through <strong style='color:#334D6E'>{date}</strong>&nbsp;&middot;&nbsp; {n} players tracked&nbsp;&middot;&nbsp; 3 load metrics",
        "kpi_danger_flags": "Danger Flags",
        "kpi_caution_flags": "Caution Flags",
        "kpi_optimal_flags": "Optimal Flags",
        "kpi_players_tracked": "Players Tracked",
        "section_risk_zones": "Risk Zones",
        "section_player_status": "Player Status — Current ACWR",
        # Zones
        "zone_undertraining": "Under",
        "zone_optimal": "Optimal",
        "zone_caution": "Caution",
        "zone_danger": "Danger",
        "zone_unknown": "—",
        # Targets
        "target_total_distance": "Total Distance",
        "target_acc_total": "Accelerations",
        "target_vel_total": "High-Speed Running",
        "target_acwr_unit": "ACWR (unitless)",
        # Planner page
        "planner_title": "Planning & <span>Forecast</span>",
        "planner_sub": "Schedule the next 15 days in an interactive calendar, then run a squad-wide ACWR forecast without leaving the page.",
        "planner_caption": "Click any highlighted date to plan a training session. Schedule runs {start} – {end} (15 days from the latest available squad data). Hit Run Forecast to see projected ACWR for all 28 players.",
        "kpi_sessions_planned": "Sessions Planned",
        "kpi_training_days": "Training Days",
        "kpi_match_days": "Match Days",
        "kpi_rest_days": "Rest Days",
        "section_session_types": "Session Types",
        "section_session_calendar": "Session Calendar",
        "section_forecast_window": "Forecast Window",
        "section_planned_sessions": "Planned Sessions",
        "fw_15days": "15 days",
        "fw_model_data_to": "model data to",
        "no_sessions_tip": "Tap any highlighted date on the calendar to add a training session.",
        "btn_clear_plan": "Clear Plan",
        "btn_run_forecast": "Run Forecast",
        # Forecast results
        "forecast_results_title": "Forecast <span>Results</span>",
        "forecast_results_sub": "15-day ACWR projection",
        "forecast_header_meta": "{n} players &nbsp;&middot;&nbsp; 3 load metrics",
        "forecast_stale_warning": "Your plan has changed since the last run. The results below are preserved for reference — click **Run Forecast** to refresh.",
        "injury_risk_alert": "Injury Risk Alert",
        "injury_risk_msg": "{n} player(s) projected in DANGER zone by Day 15:",
        "status_stale": "Plan updated after last run",
        "status_fresh": "Forecast is up to date",
        "status_high_risk": "HIGH RISK",
        "status_caution": "CAUTION",
        "status_ok": "OK",
        "status_low": "LOW",
        "table_player": "Player",
        "table_position": "Position",
        "table_status": "Status",
        "section_day15_summary": "Day-15 Summary — All Players",
        "player_prefix": "Player",
        # Dialog
        "dialog_create_title": "Create a new session",
        "dialog_edit_title": "Edit planned session",
        "dialog_date_label": "Date:",
        "dialog_training_types": "Training types",
        "dialog_notes": "Notes",
        "dialog_notes_placeholder": "Optional coaching context, travel information, or drill notes.",
        "dialog_save": "Save Event",
        "dialog_cancel": "Cancel",
        "dialog_delete": "Delete Event",
        "dialog_err_no_types": "Select at least one session type before saving the event.",
        "dialog_err_time": "The event end time must be later than the start time.",
        # Spinner / errors
        "spinner_forecast": "Computing 15-day ACWR forecasts for all 28 players…",
        "info_no_forecast": "Add sessions to the calendar above, then click **Run Forecast** to see projected ACWR for all 28 players over the next 15 days.",
        "err_forecast_failed": "Forecast failed — check models are trained (`python train_models.py`).",
        # Sidebar
        "sidebar_season": "Season 2024/25",
        "sidebar_players_metrics": "28 Players &nbsp;&middot;&nbsp; 3 Metrics",
        "sidebar_developed_by": "Developed by",
        # Player positions
        "pos_central_back": "Central Back",
        "pos_central_midfielder": "Central Midfielder",
        "pos_forward": "Forward",
        "pos_full_back": "Full Back",
        "pos_winger": "Winger",
        "pos_unknown": "Unknown",
        # Calendar fallback / sidebar app name
        "calendar_session_fallback": "Session",
        "sidebar_app_name": "ACWR Monitor",
        # Session type labels
        "session_type_G": "Game / SSG",
        "session_type_TAC": "Tactical",
        "session_type_BP": "Set Pieces",
        "session_type_TEC": "Technical",
        "session_type_MATCH": "Official Match",
    },
    "ESP": {
        # Navigation
        "nav_dashboard": "Panel",
        "nav_planner": "Planificación y Pronóstico",
        # Dashboard
        "dashboard_title": "Panel <span>ACWR</span> del Equipo",
        "dashboard_sub": "Datos hasta <strong style='color:#334D6E'>{date}</strong>&nbsp;&middot;&nbsp; {n} jugadores seguidos&nbsp;&middot;&nbsp; 3 métricas de carga",
        "kpi_danger_flags": "Alertas de Peligro",
        "kpi_caution_flags": "Alertas de Precaución",
        "kpi_optimal_flags": "Alertas Óptimas",
        "kpi_players_tracked": "Jugadores Seguidos",
        "section_risk_zones": "Zonas de Riesgo",
        "section_player_status": "Estado del Jugador — ACWR Actual",
        # Zones
        "zone_undertraining": "Bajo",
        "zone_optimal": "Óptimo",
        "zone_caution": "Precaución",
        "zone_danger": "Peligro",
        "zone_unknown": "—",
        # Targets
        "target_total_distance": "Distancia Total",
        "target_acc_total": "Aceleraciones",
        "target_vel_total": "Carrera de Alta Velocidad",
        "target_acwr_unit": "ACWR (adimensional)",
        # Planner page
        "planner_title": "Planificación y <span>Pronóstico</span>",
        "planner_sub": "Planifica los próximos 15 días en un calendario interactivo y ejecuta un pronóstico ACWR del equipo sin salir de la página.",
        "planner_caption": "Haz clic en cualquier fecha resaltada para planificar una sesión. El calendario abarca {start} – {end} (15 días desde los datos más recientes del equipo). Pulsa Ejecutar Pronóstico para ver el ACWR proyectado para los 28 jugadores.",
        "kpi_sessions_planned": "Sesiones Planificadas",
        "kpi_training_days": "Días de Entrenamiento",
        "kpi_match_days": "Días de Partido",
        "kpi_rest_days": "Días de Descanso",
        "section_session_types": "Tipos de Sesión",
        "section_session_calendar": "Calendario de Sesiones",
        "section_forecast_window": "Ventana de Pronóstico",
        "section_planned_sessions": "Sesiones Planificadas",
        "fw_15days": "15 días",
        "fw_model_data_to": "datos del modelo hasta",
        "no_sessions_tip": "Toca cualquier fecha resaltada en el calendario para añadir una sesión de entrenamiento.",
        "btn_clear_plan": "Borrar Plan",
        "btn_run_forecast": "Ejecutar Pronóstico",
        # Forecast results
        "forecast_results_title": "Pronóstico <span>Resultados</span>",
        "forecast_results_sub": "Proyección ACWR de 15 días",
        "forecast_header_meta": "{n} jugadores &nbsp;&middot;&nbsp; 3 métricas de carga",
        "forecast_stale_warning": "Tu plan ha cambiado desde la última ejecución. Los resultados a continuación se conservan como referencia — haz clic en **Ejecutar Pronóstico** para actualizar.",
        "injury_risk_alert": "Alerta de Riesgo de Lesión",
        "injury_risk_msg": "{n} jugador(es) proyectado(s) en zona de PELIGRO para el Día 15:",
        "status_stale": "Plan actualizado después de la última ejecución",
        "status_fresh": "Pronóstico actualizado",
        "status_high_risk": "ALTO RIESGO",
        "status_caution": "PRECAUCIÓN",
        "status_ok": "OK",
        "status_low": "BAJO",
        "table_player": "Jugador",
        "table_position": "Posición",
        "table_status": "Estado",
        "section_day15_summary": "Resumen del Día 15 — Todos los Jugadores",
        "player_prefix": "Jugador",
        # Dialog
        "dialog_create_title": "Crear una nueva sesión",
        "dialog_edit_title": "Editar sesión planificada",
        "dialog_date_label": "Fecha:",
        "dialog_training_types": "Tipos de entrenamiento",
        "dialog_notes": "Notas",
        "dialog_notes_placeholder": "Contexto del entrenador, información de viaje o notas de ejercicios.",
        "dialog_save": "Guardar Evento",
        "dialog_cancel": "Cancelar",
        "dialog_delete": "Eliminar Evento",
        "dialog_err_no_types": "Selecciona al menos un tipo de sesión antes de guardar el evento.",
        "dialog_err_time": "La hora de fin del evento debe ser posterior a la hora de inicio.",
        # Spinner / errors
        "spinner_forecast": "Calculando pronósticos ACWR de 15 días para los 28 jugadores…",
        "info_no_forecast": "Añade sesiones al calendario arriba, luego haz clic en **Ejecutar Pronóstico** para ver el ACWR proyectado para los 28 jugadores durante los próximos 15 días.",
        "err_forecast_failed": "Error en el pronóstico — verifica que los modelos estén entrenados (`python train_models.py`).",
        # Sidebar
        "sidebar_season": "Temporada 2024/25",
        "sidebar_players_metrics": "28 Jugadores &nbsp;&middot;&nbsp; 3 Métricas",
        "sidebar_developed_by": "Desarrollado por",
        # Player positions
        "pos_central_back": "Defensa Central",
        "pos_central_midfielder": "Centrocampista",
        "pos_forward": "Delantero",
        "pos_full_back": "Lateral",
        "pos_winger": "Extremo",
        "pos_unknown": "Desconocido",
        # Calendar fallback / sidebar app name
        "calendar_session_fallback": "Sesión",
        "sidebar_app_name": "Monitor ACWR",
        # Session type labels
        "session_type_G": "Juego / PRP",
        "session_type_TAC": "Táctica",
        "session_type_BP": "Balón Parado",
        "session_type_TEC": "Técnica",
        "session_type_MATCH": "Partido Oficial",
    },
}
