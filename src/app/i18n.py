"""Internationalisation helpers for the ACWR monitor."""

from __future__ import annotations

from datetime import date as _date, datetime as _datetime
from typing import Union

import streamlit as st

from app.constants import TRANSLATIONS

_DateLike = Union[_date, _datetime]

_ES_MONTH_ABBR = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                   "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
_ES_MONTH_FULL = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
                   "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
_ES_DOW_ABBR = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
_ES_DOW_FULL = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


def t(key: str) -> str:
    """Return the translation for key in the active UI language."""
    lang = st.session_state.get("lang", "ENG")
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS["ENG"])
    return lang_dict.get(key, TRANSLATIONS["ENG"].get(key, key))


def _is_esp() -> bool:
    return st.session_state.get("lang") == "ESP"


def fmt_date_short(dt: _DateLike) -> str:
    """'26 May' — abbreviated month, no year."""
    if _is_esp():
        return f"{dt.day:02d} {_ES_MONTH_ABBR[dt.month - 1]}"
    return dt.strftime("%d %b")


def fmt_date_medium(dt: _DateLike) -> str:
    """'26 May 2025' — abbreviated month with year."""
    if _is_esp():
        return f"{dt.day:02d} {_ES_MONTH_ABBR[dt.month - 1]} {dt.year}"
    return dt.strftime("%d %b %Y")


def fmt_date_full(dt: _DateLike) -> str:
    """'26 May 2025' / '26 de mayo de 2025' — full month name."""
    if _is_esp():
        return f"{dt.day} de {_ES_MONTH_FULL[dt.month - 1]} de {dt.year}"
    return dt.strftime("%d %B %Y")


def fmt_date_long(dt: _DateLike) -> str:
    """'Monday, 26 May 2025' / 'Lunes, 26 de mayo de 2025' — full weekday and month."""
    if _is_esp():
        return f"{_ES_DOW_FULL[dt.weekday()]}, {dt.day} de {_ES_MONTH_FULL[dt.month - 1]} de {dt.year}"
    return dt.strftime("%A, %d %B %Y")


def t_pos(position: str) -> str:
    """Translate a player position label (e.g. 'Central Back' → 'Defensa Central')."""
    key = f"pos_{position.lower().replace(' ', '_')}"
    return t(key)


def fmt_dow_date(dt: _DateLike) -> str:
    """'Mon 26 May' / 'Lun 26 May' — abbreviated weekday + day + abbreviated month."""
    if _is_esp():
        return f"{_ES_DOW_ABBR[dt.weekday()]} {dt.day:02d} {_ES_MONTH_ABBR[dt.month - 1]}"
    return dt.strftime("%a %d %b")
