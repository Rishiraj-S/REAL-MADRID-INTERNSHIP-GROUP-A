"""Full ENG / ESP translation table for the ACWR Monitor application."""

from __future__ import annotations

TRANSLATIONS: dict[str, dict[str, str]] = {
    "ENG": {
        # ── Navigation ──────────────────────────────────────────────────────
        "nav_dashboard":             "Dashboard",
        "nav_planner":               "Planning & Forecast",
        "nav_player_customization":  "Player Customization",
        "nav_export_plan":           "Export Plan",

        # ── Dashboard header ────────────────────────────────────────────────
        "dashboard_title":       "Squad <span>ACWR</span> Dashboard",
        "dashboard_sub":         "Data through <strong style='color:#334D6E'>{date}</strong>&nbsp;&middot;&nbsp; {n} players tracked&nbsp;&middot;&nbsp; 3 load metrics",
        "dashboard_data_through": "Data through",

        # ── Info bar ────────────────────────────────────────────────────────
        "info_season":          "Season",
        "info_squad":           "Squad",
        "info_players_metrics": "{n} Players &nbsp;&middot;&nbsp; 3 Metrics",
        "info_smoothing":       "Smoothing Method",

        # ── ACWR explanation ────────────────────────────────────────────────
        "section_what_is_acwr": "What is ACWR?",
        "acwr_intro":           "The <strong style='color:#00529F'>Acute:Chronic Workload Ratio</strong> compares a player's recent training load against their longer term baseline. A ratio far above 1.0 signals a sudden spike in load, a known injury risk indicator in sport science.",
        "acwr_acute_label":     "Acute Load",
        "acwr_chronic_label":   "Chronic Load",
        "acwr_ewma_label":      "EWMA",
        "acwr_acute_days":      "7 days",
        "acwr_chronic_days":    "28 days",
        "acwr_acute_desc":      "Short term fatigue. Reflects how hard the player has trained recently. High sensitivity to recent session spikes.",
        "acwr_chronic_desc":    "Long term fitness baseline. Represents the player's accumulated conditioning over the past month.",
        "acwr_ewma_desc":       "Exponentially Weighted Moving Average. Gives higher weight to recent sessions and decays smoothly on rest days, more responsive than a simple rolling average.",

        # ── Risk zones section ──────────────────────────────────────────────
        "section_risk_zones":          "Risk Zones",
        "zone_undertraining_label":    "Undertraining",
        "zone_optimal_label":          "Optimal",
        "zone_caution_label":          "Caution",
        "zone_danger_label":           "High Risk",
        "zone_undertraining_range":    "ACWR < 0.8",
        "zone_optimal_range":          "0.8 to 1.3",
        "zone_caution_range":          "1.3 to 1.5",
        "zone_danger_range":           "ACWR ≥ 1.5",
        "zone_undertraining_desc":     "Recent load well below baseline. Fitness may be declining; player not adequately prepared for match demands.",
        "zone_optimal_desc":           "Acute and chronic load are balanced. Sweet spot for performance and injury prevention.",
        "zone_caution_desc":           "Load noticeably above baseline. Injury risk is elevated. Monitor closely and consider reducing intensity.",
        "zone_danger_desc":            "Acute load far exceeds baseline. Significantly increased soft tissue injury risk. Immediate load reduction recommended.",

        # ── Zone names (short, used in badges) ─────────────────────────────
        "zone_undertraining": "Under",
        "zone_optimal":       "Optimal",
        "zone_caution":       "Caution",
        "zone_danger":        "Danger",
        "zone_unknown":       "—",

        # ── KPI cards ───────────────────────────────────────────────────────
        "kpi_danger_flags":      "Danger Flags",
        "kpi_caution_flags":     "Caution Flags",
        "kpi_optimal_flags":     "Optimal Flags",
        "kpi_players_tracked":   "Players Tracked",
        "kpi_high_risk_flags":   "High Risk Flags",
        "kpi_undertraining":     "Undertraining",
        "kpi_sessions_planned":  "Sessions Planned",
        "kpi_training_days":     "Training Days",
        "kpi_match_days":        "Match Days",
        "kpi_rest_days":         "Rest Days",

        # ── Player status section ───────────────────────────────────────────
        "section_player_status":  "Player Status — Current ACWR",
        "player_prefix":          "Player",
        "label_last_active":      "Last active",
        "label_active_days":      "Active days",
        "label_enters_danger":    "Enters danger",
        "status_high_risk":       "HIGH RISK",
        "status_caution":         "CAUTION",
        "status_ok":              "OK",
        "status_low":             "LOW",

        # ── Targets ─────────────────────────────────────────────────────────
        "target_total_distance":  "Total Distance",
        "target_accelerations":   "Accelerations",
        "target_sprint_distance": "Sprint Distance",
        "target_acwr_unit":       "ACWR (unitless)",

        # ── Planner page ────────────────────────────────────────────────────
        "planner_title":   "Planning & <span>Forecast</span>",
        "planner_sub":     "Schedule the next 15 days in an interactive calendar, then run a squad-wide ACWR forecast without leaving the page.",
        "planner_caption": "Click any highlighted date to plan a training session. Schedule runs {start} – {end} (15 days from the latest available squad data). Hit Run Forecast to see projected ACWR for all 28 players.",
        "section_session_types":    "Session Types",
        "section_session_calendar": "Session Calendar",
        "section_forecast_window":  "Forecast Window",
        "section_planned_sessions": "Planned Sessions",
        "fw_15days":         "15 days",
        "fw_model_data_to":  "model data to",
        "no_sessions_tip":   "Tap any highlighted date on the calendar to add a training session.",
        "btn_clear_plan":    "Clear Plan",
        "btn_run_forecast":  "Run Forecast",
        "calendar_session_fallback": "Session",

        # ── Forecast results ────────────────────────────────────────────────
        "forecast_results_title":   "Forecast <span>Results</span>",
        "forecast_results_sub":     "15-day ACWR projection",
        "forecast_header_meta":     "{n} players &nbsp;&middot;&nbsp; 3 load metrics",
        "forecast_stale_warning":   "Your plan has changed since the last run. The results below are preserved for reference — click **Run Forecast** to refresh.",
        "section_day15_summary":    "Day-15 Summary — All Players",
        "label_day15":              "Day 15",
        "label_show_load":          "Show load",
        "label_squad":              "SQUAD",
        "label_custom":             "CUSTOM",

        # ── Injury Risk Alert ────────────────────────────────────────────────
        "injury_risk_alert":         "Injury Risk Alert",
        "injury_risk_msg":           "{n} player(s) projected in DANGER zone by Day 15:",
        "alert_metric_flags":        "{n} metric flag{s} across {p} player{ps}",
        "alert_still_in_danger":     "still in danger",
        "alert_recovered":           "recovered by day 15",
        "alert_recovered_note":      "Briefly entered danger (recovered by Day 15):",
        "table_player":              "Player",
        "table_position":            "Position",
        "table_status":              "Status",
        "table_load_metric":         "Load Metric",
        "table_enters_danger":       "Enters Danger",
        "table_day15_acwr":          "Day 15 ACWR",

        # ── Dialog ──────────────────────────────────────────────────────────
        "dialog_create_title":      "Create a new session",
        "dialog_edit_title":        "Edit planned session",
        "dialog_date_label":        "Date:",
        "dialog_training_types":    "Training types",
        "dialog_notes":             "Notes",
        "dialog_notes_placeholder": "Optional coaching context, travel information, or drill notes.",
        "dialog_save":              "Save Event",
        "dialog_cancel":            "Cancel",
        "dialog_delete":            "Delete Event",
        "dialog_err_no_types":      "Select at least one session type before saving the event.",
        "dialog_err_time":          "The event end time must be later than the start time.",

        # ── Spinner / errors ─────────────────────────────────────────────────
        "spinner_forecast":   "Computing 15-day ACWR forecasts for all 28 players…",
        "info_no_forecast":   "Add sessions to the calendar above, then click **Run Forecast** to see projected ACWR for all 28 players over the next 15 days.",
        "err_forecast_failed": "Forecast failed — check models are trained (`python train_models.py`).",

        # ── Sidebar ──────────────────────────────────────────────────────────
        "sidebar_app_name":      "ACWR Monitor",
        "sidebar_season":        "Season 2024/25",
        "sidebar_players_metrics": "28 Players &nbsp;&middot;&nbsp; 3 Metrics",
        "sidebar_developed_by":  "Developed by",
        "sidebar_language":      "Language",
        "sidebar_navigation":    "Navigation",

        # ── Session type labels ──────────────────────────────────────────────
        "session_type_G":     "Game / SSG",
        "session_type_TAC":   "Tactical",
        "session_type_BP":    "Set Pieces",
        "session_type_TEC":   "Technical",
        "session_type_MATCH": "Official Match",

        # ── Player positions ─────────────────────────────────────────────────
        "pos_central_back":        "Central Back",
        "pos_central_midfielder":  "Central Midfielder",
        "pos_forward":             "Forward",
        "pos_full_back":           "Full Back",
        "pos_winger":              "Winger",
        "pos_unknown":             "Unknown",

        # ── Player Customization ─────────────────────────────────────────────
        "pc_title":              "Player Customization",
        "pc_subtitle":           "Modify the squad plan for an individual player and compare the custom ACWR forecast against the original.",
        "pc_no_forecast":        "No squad forecast found. Go to **Planning & Forecast**, build a session plan and click **Run Forecast** first — then come back here to customise individual players.",
        "pc_no_danger":          "No players are projected to enter the danger zone under the current squad plan. No customization needed.",
        "pc_at_risk":            "At Risk",
        "pc_show_all":           "Show all players (not just at-risk)",
        "pc_select_player":      "Select Player to Customise",
        "pc_current_status":     "Current ACWR Status",
        "pc_squad_forecast":     "Squad Plan — Day 15 Forecast",
        "pc_custom_plan_title":  "Custom Plan — Modify From Squad Plan",
        "pc_custom_plan_caption": "Each day is pre-loaded from the squad plan. Adjust rest days or session types for this player only.",
        "pc_day_label":          "Day",
        "pc_rest":               "Rest",
        "pc_btn_run":            "Run Custom Forecast",
        "pc_btn_reset":          "Reset to Squad Plan",
        "pc_btn_clear":          "Clear",
        "pc_spinner":            "Computing custom forecast…",
        "pc_info_no_forecast":   "Adjust the plan above and click **Run Custom Forecast** to compare with the squad plan.",
        "pc_comparison_title":   "Comparison — Squad Plan vs Custom Plan",
        "pc_squad_plan_trace":   "Squad Plan",
        "label_show_load_pc":    "Show load",

        # ── Export Plan ──────────────────────────────────────────────────────
        "ep_title":           "Export Plan",
        "ep_subtitle":        "Full squad training plan. Click Export PDF to save or print.",
        "ep_no_plan":         "No squad plan found. Go to **Planning & Forecast**, build a plan and click **Run Forecast** first.",
        "ep_btn_export":      "⬇ Export PDF",
        "ep_pdf_title":       "Training Plan",
        "ep_pdf_subtitle":    "Real Madrid C.F. &nbsp;&middot;&nbsp; Season 2024/25",
        "ep_pdf_schedule":    "{n}-Day Schedule &nbsp;&middot;&nbsp; {p} Players",
        "ep_legend_rest":     "= No session scheduled",
        "ep_custom_note":     "{n} player{s} a customised plan (marked",
        "ep_col_player":      "Player",
        "ep_custom_badge":    "CUSTOM",
        "ep_rest_cell":       "REST",

        # ── Status / fresh-stale ─────────────────────────────────────────────
        "status_stale": "Plan updated after last run",
        "status_fresh": "Forecast is up to date",
    },

    "ESP": {
        # ── Navigation ──────────────────────────────────────────────────────
        "nav_dashboard":             "Panel",
        "nav_planner":               "Planificación y Pronóstico",
        "nav_player_customization":  "Personalización del Jugador",
        "nav_export_plan":           "Exportar Plan",

        # ── Dashboard header ────────────────────────────────────────────────
        "dashboard_title":       "Panel <span>ACWR</span> del Equipo",
        "dashboard_sub":         "Datos hasta <strong style='color:#334D6E'>{date}</strong>&nbsp;&middot;&nbsp; {n} jugadores seguidos&nbsp;&middot;&nbsp; 3 métricas de carga",
        "dashboard_data_through": "Datos hasta",

        # ── Info bar ────────────────────────────────────────────────────────
        "info_season":          "Temporada",
        "info_squad":           "Plantilla",
        "info_players_metrics": "{n} Jugadores &nbsp;&middot;&nbsp; 3 Métricas",
        "info_smoothing":       "Método de Suavizado",

        # ── ACWR explanation ────────────────────────────────────────────────
        "section_what_is_acwr": "¿Qué es el ACWR?",
        "acwr_intro":           "La <strong style='color:#00529F'>Relación Aguda:Crónica de Carga de Trabajo</strong> compara la carga de entrenamiento reciente de un jugador con su referencia a largo plazo. Una ratio muy superior a 1.0 señala un pico repentino de carga, indicador de riesgo de lesión en la ciencia deportiva.",
        "acwr_acute_label":     "Carga Aguda",
        "acwr_chronic_label":   "Carga Crónica",
        "acwr_ewma_label":      "EWMA",
        "acwr_acute_days":      "7 días",
        "acwr_chronic_days":    "28 días",
        "acwr_acute_desc":      "Fatiga a corto plazo. Refleja la intensidad del entrenamiento reciente del jugador. Alta sensibilidad a los picos de sesión recientes.",
        "acwr_chronic_desc":    "Línea base de condición física a largo plazo. Representa el acondicionamiento acumulado del jugador durante el último mes.",
        "acwr_ewma_desc":       "Media Móvil Exponencialmente Ponderada. Da mayor peso a las sesiones recientes y decae suavemente en días de descanso, más sensible que una media móvil simple.",

        # ── Risk zones section ──────────────────────────────────────────────
        "section_risk_zones":          "Zonas de Riesgo",
        "zone_undertraining_label":    "Subcarga",
        "zone_optimal_label":          "Óptimo",
        "zone_caution_label":          "Precaución",
        "zone_danger_label":           "Alto Riesgo",
        "zone_undertraining_range":    "ACWR < 0.8",
        "zone_optimal_range":          "0.8 a 1.3",
        "zone_caution_range":          "1.3 a 1.5",
        "zone_danger_range":           "ACWR ≥ 1.5",
        "zone_undertraining_desc":     "Carga reciente muy por debajo de la línea base. La condición física puede estar disminuyendo; el jugador no está adecuadamente preparado para las exigencias del partido.",
        "zone_optimal_desc":           "La carga aguda y crónica están equilibradas. Punto óptimo para el rendimiento y la prevención de lesiones.",
        "zone_caution_desc":           "Carga notablemente superior a la línea base. El riesgo de lesión es elevado. Supervisar de cerca y considerar reducir la intensidad.",
        "zone_danger_desc":            "La carga aguda supera ampliamente la línea base. Riesgo significativamente mayor de lesión de tejido blando. Se recomienda reducción inmediata de la carga.",

        # ── Zone names (short, used in badges) ─────────────────────────────
        "zone_undertraining": "Bajo",
        "zone_optimal":       "Óptimo",
        "zone_caution":       "Precaución",
        "zone_danger":        "Peligro",
        "zone_unknown":       "—",

        # ── KPI cards ───────────────────────────────────────────────────────
        "kpi_danger_flags":      "Alertas de Peligro",
        "kpi_caution_flags":     "Alertas de Precaución",
        "kpi_optimal_flags":     "Alertas Óptimas",
        "kpi_players_tracked":   "Jugadores Seguidos",
        "kpi_high_risk_flags":   "Alertas de Alto Riesgo",
        "kpi_undertraining":     "Subcarga",
        "kpi_sessions_planned":  "Sesiones Planificadas",
        "kpi_training_days":     "Días de Entrenamiento",
        "kpi_match_days":        "Días de Partido",
        "kpi_rest_days":         "Días de Descanso",

        # ── Player status section ───────────────────────────────────────────
        "section_player_status":  "Estado del Jugador — ACWR Actual",
        "player_prefix":          "Jugador",
        "label_last_active":      "Último activo",
        "label_active_days":      "Días activos",
        "label_enters_danger":    "Entra en peligro",
        "status_high_risk":       "ALTO RIESGO",
        "status_caution":         "PRECAUCIÓN",
        "status_ok":              "OK",
        "status_low":             "BAJO",

        # ── Targets ─────────────────────────────────────────────────────────
        "target_total_distance":  "Distancia Total",
        "target_accelerations":   "Aceleraciones",
        "target_sprint_distance": "Distancia de Sprint",
        "target_acwr_unit":       "ACWR (adimensional)",

        # ── Planner page ────────────────────────────────────────────────────
        "planner_title":   "Planificación y <span>Pronóstico</span>",
        "planner_sub":     "Planifica los próximos 15 días en un calendario interactivo y ejecuta un pronóstico ACWR del equipo sin salir de la página.",
        "planner_caption": "Haz clic en cualquier fecha resaltada para planificar una sesión. El calendario abarca {start} – {end} (15 días desde los datos más recientes del equipo). Pulsa Ejecutar Pronóstico para ver el ACWR proyectado para los 28 jugadores.",
        "section_session_types":    "Tipos de Sesión",
        "section_session_calendar": "Calendario de Sesiones",
        "section_forecast_window":  "Ventana de Pronóstico",
        "section_planned_sessions": "Sesiones Planificadas",
        "fw_15days":         "15 días",
        "fw_model_data_to":  "datos del modelo hasta",
        "no_sessions_tip":   "Toca cualquier fecha resaltada en el calendario para añadir una sesión de entrenamiento.",
        "btn_clear_plan":    "Borrar Plan",
        "btn_run_forecast":  "Ejecutar Pronóstico",
        "calendar_session_fallback": "Sesión",

        # ── Forecast results ────────────────────────────────────────────────
        "forecast_results_title":   "Pronóstico <span>Resultados</span>",
        "forecast_results_sub":     "Proyección ACWR de 15 días",
        "forecast_header_meta":     "{n} jugadores &nbsp;&middot;&nbsp; 3 métricas de carga",
        "forecast_stale_warning":   "Tu plan ha cambiado desde la última ejecución. Los resultados a continuación se conservan como referencia — haz clic en **Ejecutar Pronóstico** para actualizar.",
        "section_day15_summary":    "Resumen del Día 15 — Todos los Jugadores",
        "label_day15":              "Día 15",
        "label_show_load":          "Mostrar carga",
        "label_squad":              "EQUIPO",
        "label_custom":             "PERSONALIZADO",

        # ── Injury Risk Alert ────────────────────────────────────────────────
        "injury_risk_alert":         "Alerta de Riesgo de Lesión",
        "injury_risk_msg":           "{n} jugador(es) proyectado(s) en zona de PELIGRO para el Día 15:",
        "alert_metric_flags":        "{n} alerta{s} de métrica en {p} jugador{ps}",
        "alert_still_in_danger":     "aún en peligro",
        "alert_recovered":           "recuperado en el día 15",
        "alert_recovered_note":      "Entró brevemente en peligro (recuperado en el Día 15):",
        "table_player":              "Jugador",
        "table_position":            "Posición",
        "table_status":              "Estado",
        "table_load_metric":         "Métrica de Carga",
        "table_enters_danger":       "Entra en Peligro",
        "table_day15_acwr":          "ACWR Día 15",

        # ── Dialog ──────────────────────────────────────────────────────────
        "dialog_create_title":      "Crear una nueva sesión",
        "dialog_edit_title":        "Editar sesión planificada",
        "dialog_date_label":        "Fecha:",
        "dialog_training_types":    "Tipos de entrenamiento",
        "dialog_notes":             "Notas",
        "dialog_notes_placeholder": "Contexto del entrenador, información de viaje o notas de ejercicios.",
        "dialog_save":              "Guardar Evento",
        "dialog_cancel":            "Cancelar",
        "dialog_delete":            "Eliminar Evento",
        "dialog_err_no_types":      "Selecciona al menos un tipo de sesión antes de guardar el evento.",
        "dialog_err_time":          "La hora de fin del evento debe ser posterior a la hora de inicio.",

        # ── Spinner / errors ─────────────────────────────────────────────────
        "spinner_forecast":    "Calculando pronósticos ACWR de 15 días para los 28 jugadores…",
        "info_no_forecast":    "Añade sesiones al calendario arriba, luego haz clic en **Ejecutar Pronóstico** para ver el ACWR proyectado para los 28 jugadores durante los próximos 15 días.",
        "err_forecast_failed": "Error en el pronóstico — verifica que los modelos estén entrenados (`python train_models.py`).",

        # ── Sidebar ──────────────────────────────────────────────────────────
        "sidebar_app_name":        "Monitor ACWR",
        "sidebar_season":          "Temporada 2024/25",
        "sidebar_players_metrics": "28 Jugadores &nbsp;&middot;&nbsp; 3 Métricas",
        "sidebar_developed_by":    "Desarrollado por",
        "sidebar_language":        "Idioma",
        "sidebar_navigation":      "Navegación",

        # ── Session type labels ──────────────────────────────────────────────
        "session_type_G":     "Juego / PRP",
        "session_type_TAC":   "Táctica",
        "session_type_BP":    "Balón Parado",
        "session_type_TEC":   "Técnica",
        "session_type_MATCH": "Partido Oficial",

        # ── Player positions ─────────────────────────────────────────────────
        "pos_central_back":       "Defensa Central",
        "pos_central_midfielder": "Centrocampista",
        "pos_forward":            "Delantero",
        "pos_full_back":          "Lateral",
        "pos_winger":             "Extremo",
        "pos_unknown":            "Desconocido",

        # ── Player Customization ─────────────────────────────────────────────
        "pc_title":              "Personalización del Jugador",
        "pc_subtitle":           "Modifica el plan del equipo para un jugador individual y compara el pronóstico ACWR personalizado con el original.",
        "pc_no_forecast":        "No se encontró pronóstico del equipo. Ve a **Planificación y Pronóstico**, crea un plan y haz clic en **Ejecutar Pronóstico** primero.",
        "pc_no_danger":          "Ningún jugador está proyectado para entrar en zona de peligro con el plan actual. No se necesita personalización.",
        "pc_at_risk":            "En Riesgo",
        "pc_show_all":           "Mostrar todos los jugadores (no solo los de riesgo)",
        "pc_select_player":      "Seleccionar Jugador a Personalizar",
        "pc_current_status":     "Estado ACWR Actual",
        "pc_squad_forecast":     "Plan del Equipo — Pronóstico Día 15",
        "pc_custom_plan_title":  "Plan Personalizado — Modificar desde el Plan del Equipo",
        "pc_custom_plan_caption": "Cada día está precargado desde el plan del equipo. Ajusta los días de descanso o los tipos de sesión solo para este jugador.",
        "pc_day_label":          "Día",
        "pc_rest":               "Descanso",
        "pc_btn_run":            "Ejecutar Pronóstico Personalizado",
        "pc_btn_reset":          "Restablecer al Plan del Equipo",
        "pc_btn_clear":          "Limpiar",
        "pc_spinner":            "Calculando pronóstico personalizado…",
        "pc_info_no_forecast":   "Ajusta el plan y haz clic en **Ejecutar Pronóstico Personalizado** para comparar con el plan del equipo.",
        "pc_comparison_title":   "Comparación — Plan del Equipo vs Plan Personalizado",
        "pc_squad_plan_trace":   "Plan del Equipo",
        "label_show_load_pc":    "Mostrar carga",

        # ── Export Plan ──────────────────────────────────────────────────────
        "ep_title":           "Exportar Plan",
        "ep_subtitle":        "Plan de entrenamiento completo del equipo. Haz clic en Exportar PDF para guardar o imprimir.",
        "ep_no_plan":         "No se encontró plan del equipo. Ve a **Planificación y Pronóstico**, crea un plan y haz clic en **Ejecutar Pronóstico** primero.",
        "ep_btn_export":      "⬇ Exportar PDF",
        "ep_pdf_title":       "Plan de Entrenamiento",
        "ep_pdf_subtitle":    "Real Madrid C.F. &nbsp;&middot;&nbsp; Temporada 2024/25",
        "ep_pdf_schedule":    "Programa de {n} días &nbsp;&middot;&nbsp; {p} Jugadores",
        "ep_legend_rest":     "= Sin sesión programada",
        "ep_custom_note":     "{n} jugador{s} tiene un plan personalizado (marcado",
        "ep_col_player":      "Jugador",
        "ep_custom_badge":    "PERSONALIZADO",
        "ep_rest_cell":       "DESCANSO",

        # ── Status / fresh-stale ─────────────────────────────────────────────
        "status_stale": "Plan actualizado después de la última ejecución",
        "status_fresh": "Pronóstico actualizado",
    },
}
