import streamlit as st
from ui.widgets import StatCard, PrimaryButton, Loader, Badge

def view(theme):
    st.subheader("Home")
    Badge("Status: UI Shell Ready", "success")
    StatCard("Arenas Installed", "3", "Crypto, Sports, Lottery")
    StatCard("Engine", "Live", "Status light shows action name + ms")

    col1,col2,col3 = st.columns(3)
    with col1: PrimaryButton("Quick Tour", key="home_tour", run=lambda: Loader("Showing tour…"))
    with col2: PrimaryButton("Run Demo Forecast", key="home_demo", run=lambda: Loader("Simulating…"))
    with col3: PrimaryButton("Open Logs", key="home_logs", run=lambda: Loader("Opening…"))

    st.divider()
    if st.button("Force Clear Cache"): st.cache_data.clear(); st.success("Cache cleared. Press Rerun.")
