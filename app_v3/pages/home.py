import streamlit as st
from ui.widgets import StatCard, PrimaryButton, Loader, Badge

def view(theme):
    st.subheader("Home")
    Badge("Status: UI Shell Ready", "success")
    st.write("This is the Phase-1 polished shell. All actions give feedback; data hookups come in Phase-2.")

    StatCard("Arenas Installed", "3", "Crypto, Sports, Lottery")
    StatCard("Engine State", "Idle", "Phase-1 uses UI stubs (no live data yet)")

    col1, col2, col3 = st.columns(3)
    with col1:
        PrimaryButton("Quick Tour", key="home_tour", run=lambda: Loader("Showing tour…"))
    with col2:
        PrimaryButton("Run Demo Forecast", key="home_demo", run=lambda: Loader("Simulating…"))
    with col3:
        PrimaryButton("Open Logs", key="home_logs", run=lambda: Loader("Opening…"))
