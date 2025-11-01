import streamlit as st
from ui.widgets import PrimaryButton, Loader, StatCard, Badge

def _stub_result(title, value, help=""):
    StatCard(title, value, help)

def view(theme):
    st.subheader("Crypto Arena")
    Badge("Stub Mode: UI only — Phase-2 will wire real feeds", "warning")

    colA, colB, colC = st.columns(3)
    with colA:
        PrimaryButton("Refresh Prices", key="c_refresh", run=lambda: (_stub_result("Last Refresh", "OK"), Loader("Refreshing…")))
    with colB:
        PrimaryButton("Run Forecast", key="c_forecast", run=lambda: (_stub_result("Forecast Sentiment", "BULLISH"), Loader("Computing…")))
    with colC:
        PrimaryButton("View History", key="c_history", run=lambda: (_stub_result("Records Loaded", "30 days"), Loader("Loading…")))
