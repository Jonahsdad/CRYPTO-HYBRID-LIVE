import streamlit as st

def show():
    st.title("⛏️ Commodities — Overview")
    st.caption("HYBRID INTELLIGENCE SYSTEMS | Global Forecast OS")
st.session_state["arena_name"] = "Commodities"

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Universe", "Gold / Oil / Agri")
    with col2: st.metric("Window", "Daily")
    with col3: st.metric("Signal", "Trend")

    st.markdown("### Filters")
    c1, c2, c3 = st.columns(3)
    with c1: _ = st.selectbox("Asset", ["Gold","Silver","WTI","Brent","NatGas","Corn","Wheat"])
    with c2: _ = st.selectbox("Timeframe", ["1h","4h","1d","1w"])
    with c3: _ = st.selectbox("Model", ["Trend","Mean-Revert","Breakout"])

    st.markdown("### Actions")
    a1, a2, a3 = st.columns(3)
    a1.button("Run Scan")
    a2.button("Show Levels")
    a3.button("Export Chart Pack")

    st.markdown("---")
    st.success("Signals & charts will appear here after data wiring.")
