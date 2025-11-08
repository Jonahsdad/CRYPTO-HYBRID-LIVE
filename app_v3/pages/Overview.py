import streamlit as st

def show():
    st.title("HYBRID INTELLIGENCE SYSTEMS")
    st.caption("Forecast OS • powered by LIPE")

    arena = st.session_state.get("arena_name", "Unknown Arena")
    st.subheader(f"Arena: {arena}")

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Active Forecasts", "0")
    with col2: st.metric("Confidence", "—")
    with col3: st.metric("Next Update", "⟳")

    st.markdown("---")
    st.markdown("### Actions")
    c1, c2, c3 = st.columns(3)
    c1.button("Run Forecast")
    c2.button("View History")
    c3.button("Sync Data")

    st.markdown("---")
    st.info("All forecasts processed through LIPE Core Intelligence.")
