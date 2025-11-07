import streamlit as st

def show():
    st.title("🏠 HYBRID INTELLIGENCE SYSTEMS")
    st.subheader("Global Forecast OS")
    st.info("Choose an arena from the selector above to enter a live workspace.")

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Installed Arenas", "9")
    with col2: st.metric("Engine", "Idle")
    with col3: st.metric("Mode", "UI Shell Ready")

    st.markdown("---")
    st.markdown("**Quick start**")
    st.markdown("1) Pick an arena → 2) Use the left panel → 3) Run a scan / forecast")
