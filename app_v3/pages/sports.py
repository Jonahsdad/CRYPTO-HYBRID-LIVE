import streamlit as st
from ui.widgets import PrimaryButton, Loader, StatCard, Badge

def view(theme):
    st.subheader("Sports Arena")
    Badge("Stub Mode: UI only — Phase-2 will wire real schedules/odds", "warning")

    colA, colB, colC = st.columns(3)
    with colA:
        PrimaryButton("Fetch Games", key="s_fetch", run=lambda: (Loader("Fetching games…"), StatCard("Games Today", "12")))
    with colB:
        PrimaryButton("Best Bets (beta)", key="s_bets", run=lambda: (Loader("Analyzing…"), StatCard("Edge Found", "+7.2%")))
    with colC:
        PrimaryButton("Trend Compare", key="s_trend", run=lambda: (Loader("Comparing…"), StatCard("Correlations", "0.63")))
