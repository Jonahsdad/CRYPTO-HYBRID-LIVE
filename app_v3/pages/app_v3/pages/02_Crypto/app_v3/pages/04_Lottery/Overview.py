import streamlit as st

def show():
    st.title("🎲 Lottery — Overview")
    st.caption("HYBRID INTELLIGENCE SYSTEMS | Global Forecast OS")
st.session_state["arena_name"] = "Lottery"

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Games", "Pick 3 / Pick 4")
    with col2: st.metric("Window", "Midday / Evening")
    with col3: st.metric("Mode", "Forecast")

    st.markdown("### Filters")
    c1, c2, c3 = st.columns(3)
    with c1: _ = st.selectbox("Game", ["Pick 3", "Pick 4"])
    with c2: _ = st.selectbox("Session", ["Midday", "Evening"])
    with c3: _ = st.selectbox("Lookback", ["30 draws", "60 draws", "120 draws"])

    st.markdown("### Actions")
    a1, a2, a3 = st.columns(3)
    a1.button("Run Pattern Scan")
    a2.button("Generate Forecast")
    a3.button("Export Set")

    st.markdown("---")
    st.success("Forecast numbers and diagnostics will appear here.")
