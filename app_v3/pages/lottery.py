import streamlit as st
from ui.widgets import PrimaryButton, Loader, StatCard, Badge

def view(theme):
    st.subheader("Lottery Arena")
    Badge("Stub Mode: UI only — Phase-2 will wire IL/IN draws", "warning")

    colA, colB, colC = st.columns(3)
    with colA:
        PrimaryButton("Load Draws", key="l_load", run=lambda: (Loader("Loading draws…"), StatCard("Draws Loaded", "90d")))
    with colB:
        PrimaryButton("Pattern Scan", key="l_scan", run=lambda: (Loader("Scanning…"), StatCard("Hot Pattern", "Echo-13")))
    with colC:
        PrimaryButton("Anomaly Map", key="l_anom", run=lambda: (Loader("Mapping…"), StatCard("Anomalies", "3")))
