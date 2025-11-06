# app_v3/app.py
import importlib
import streamlit as st
from pages._registry import ARENAS

st.set_page_config(page_title="PunchLogic Command", layout="wide")

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### **PunchLogic Command**  | Global Forecast OS")
with col2:
    st.metric("Engine", "Idle")

# Choose Your Arena (left panel / top control)
arena_names = [a.key for a in ARENAS]
selected = st.selectbox("Choose Your Arena", arena_names, index=0)

# Load and run the selected arena's Overview.show()
arena = next(a for a in ARENAS if a.key == selected)

try:
    mod = importlib.import_module(arena.module)
    if hasattr(mod, "show"):
        mod.show()
    else:
        st.error(f"`{arena.module}` is missing a `show()` function.")
except Exception as e:
    st.error(f"Failed to load `{arena.module}`")
    st.exception(e)

# Footer tip
st.caption("Tip: Navigation is instant. No full reloads — state stays in session.")
