import streamlit as st
from importlib import import_module

st.set_page_config(page_title="PunchLogic | Command", layout="wide")

# -------- GLOBAL CONSTANTS ----------
ARENAS = [
    "Home", "Crypto", "Sports", "Lottery", "Stocks",
    "Options", "Real Estate", "Commodities", "Forex", "RWA"
]

# -------- SESSION DEFAULTS ----------
st.session_state.setdefault("arena", "Home")
st.session_state.setdefault("engine_status", "idle")

# -------- HEADER CARD ----------
with st.container(border=True):
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        st.subheader("PunchLogic")
        st.caption("Command")
    with c2:
        st.caption("Global")
        st.write("Forecast OS")
    with c3:
        dot = "🟢" if st.session_state["engine_status"] != "idle" else "🟡"
        st.caption(f"Engine: {st.session_state['engine_status']}")
        st.write(dot)

# -------- LEFT PANEL ----------
with st.sidebar:
    st.header("Control Panel")
    cA, cB = st.columns(2)
    if cA.button("▶ Run / Refresh"):
        st.session_state["engine_status"] = "running"
        st.toast("Run triggered")
    if cB.button("⟳ Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.toast("Cache cleared")

    st.divider()
    st.caption("Quick Jump")
    for a in ARENAS:
        if st.button(a, key=f"jump_{a}"):
            st.session_state["arena"] = a

# -------- ARENA SELECTOR ----------
choice = st.selectbox(
    "Choose Your Arena",
    ARENAS,
    index=ARENAS.index(st.session_state["arena"])
)
if choice != st.session_state["arena"]:
    st.session_state["arena"] = choice

# -------- HOME GRID ----------
def arena_grid():
    cols = st.columns(3)
    for i, a in enumerate(ARENAS[1:10]):  # skip Home
        with cols[i % 3]:
            if st.button(a, use_container_width=True, key=f"grid_{a}"):
                st.session_state["arena"] = a

# -------- CENTRAL ROUTER ----------
def render_current_arena():
    arena = st.session_state["arena"]
    if arena == "Home":
        from app_v3.pages._registry import HOME_RENDER
        HOME_RENDER()
        return

    from app_v3.pages._registry import MODULE_MAP
    try:
        mod_path = MODULE_MAP[arena]
        mod = import_module(mod_path)
        mod.render()
    except Exception as ex:
        st.error(f"Failed to load page for {arena}: {ex}")

render_current_arena()
