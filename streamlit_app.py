# streamlit_app.py
from __future__ import annotations
import importlib
import streamlit as st

BRAND = "HYBRID INTELLIGENCE SYSTEMS"

# ✅ MUST be the first Streamlit call in the whole app:
st.set_page_config(page_title=f"{BRAND} — Command", page_icon="⚡", layout="wide")

# optional css
def inject_css(path="assets/style.css"):
    try:
        with open(path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass
inject_css()

# import after set_page_config
from pages._registry import ARENAS  # noqa: E402

# sidebar
st.sidebar.title(BRAND)
st.sidebar.caption("Forecast OS • powered by LIPE")
st.sidebar.markdown("---")

# persist arena selection
keys = [a.key for a in ARENAS]
sel = st.sidebar.selectbox("Choose Your Arena", keys, index=0, key="arena_selected")

# header
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown(f"### **{BRAND}** | Global Forecast OS")
with c2:
    st.metric("Engine", "Idle")

# render selected arena
arena = next(a for a in ARENAS if a.key == sel)
try:
    mod = importlib.import_module(arena.module)
    show = getattr(mod, "show", None)
    if callable(show):
        show()
    else:
        st.error(f"`{arena.module}` is missing a `show()` function.")
except Exception as e:
    st.error(f"Failed to load `{arena.module}`")
    st.exception(e)
