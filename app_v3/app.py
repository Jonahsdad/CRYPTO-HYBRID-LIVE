# streamlit_app.py (or app_v3/app.py)

from __future__ import annotations
import os
import importlib
import streamlit as st

BRAND = "HYBRID INTELLIGENCE SYSTEMS"

# ✅ MUST be the first Streamlit call:
st.set_page_config(
    page_title=f"{BRAND} — Command",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# (optional) CSS helper — safe to call after set_page_config
def inject_css(path: str = "assets/style.css"):
    try:
        with open(path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

inject_css()

# Import AFTER set_page_config; ensure this module does NOT call st.* at import time
from pages._registry import ARENAS  # noqa: E402

# Sidebar
st.sidebar.title(BRAND)
st.sidebar.caption("Forecast OS • powered by LIPE")
st.sidebar.markdown("---")

# Persist arena selection across reruns
arena_keys = [a.key for a in ARENAS]
if "arena_selected" not in st.session_state:
    st.session_state.arena_selected = arena_keys[0]

selected = st.sidebar.selectbox(
    "Choose Your Arena",
    arena_keys,
    index=arena_keys.index(st.session_state.arena_selected),
    key="arena_selected",
)

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"### **{BRAND}**  | Global Forecast OS")
with col2:
    st.metric("Engine", st.session_state.get("engine_status", "Idle"))

# Load & render selected arena page
arena = next(a for a in ARENAS if a.key == selected)
try:
    mod = importlib.import_module(arena.module)
    show = getattr(mod, "show", None)
    if callable(show):
        show()  # delegate rendering
    else:
        st.error(f"`{arena.module}` is missing a `show()` function.")
except Exception as e:
    st.error(f"Failed to load `{arena.module}`")
    st.exception(e)
