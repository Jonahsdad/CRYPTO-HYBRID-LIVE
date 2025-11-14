from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

def show():
    st.subheader("🔥 Crypto Flagship")
    st.caption("Demo panel — proves routing works. You can wire the real API next.")

    # Controls
    c1, c2 = st.columns([2,1])
    with c1:
        symbol = st.text_input("Symbol", value="BTCUSDT")
    with c2:
        horizon = st.slider("Horizon (days)", 1, 30, value=5)

    # Fake series so the page renders even without backend
    n = 200
    xs = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="H")
    base = np.cumsum(np.random.randn(n)) + 100
    yhat = base + np.linspace(0, 2, n)
    q10  = yhat - 2.5
    q90  = yhat + 2.5

    # Chart
    fig = go.Figure()
    fig.add_scatter(x=xs, y=base, name="Actual", mode="lines")
    fig.add_scatter(x=xs, y=q90,  name="q90", mode="lines", line=dict(width=0.1), showlegend=False)
    fig.add_scatter(x=xs, y=q10,  name="q10", mode="lines", fill="tonexty",
                    line=dict(width=0.1), fillcolor="rgba(124,92,255,0.20)", showlegend=False)
    fig.add_scatter(x=xs, y=yhat, name="Forecast", mode="lines", line=dict(dash="dash", width=2))
    fig.update_layout(margin=dict(l=30,r=10,t=10,b=30), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # KPIs (placeholder)
    k1, k2, k3 = st.columns(3)
    k1.metric("Regime", "Expansion")
    k2.metric("Entropy", "0.34")
    k3.metric("Edge", "3.1%")

    st.info("✅ Routing works. Next: replace the demo series with your backend call.")
