"""
Real Madrid ACWR Monitor — Streamlit app.
Run with: streamlit run main.py
"""

# ruff: noqa: E402

import sys
import warnings
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from app.constants import PAGES
from app.pages import page_dashboard, page_planner, render_sidebar
from app.styles import inject_styles
from real_madrid_acwr.config import STATIC_DIR

LOGO_PATH = STATIC_DIR / "img" / "Real-Madrid-CF-v2002.svg"

st.set_page_config(
    page_title="Real Madrid · ACWR Monitor",
    page_icon=str(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

warnings.filterwarnings("ignore")
inject_styles()

# Must run before the radio widget renders: Streamlit raises StreamlitAPIException
# if you set a widget's session_state key after that widget has already rendered
# in the same script run. We stage the target page in _pending_nav, then apply
# it here at the very top so the radio picks up the correct value on re-render.
if "_pending_nav" in st.session_state:
    st.session_state.nav_page = st.session_state.pop("_pending_nav")
elif "nav_page" not in st.session_state:
    st.session_state.nav_page = PAGES[0]

page = render_sidebar(LOGO_PATH)

if page == PAGES[0]:
    page_dashboard()
else:
    page_planner()
