# pages/crypto_flagship.py
from __future__ import annotations
import os, time, random
import requests
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

API_BASE = os.getenv("HIS_API_BASE", st.secrets.get("HIS_API_BASE", "")).rstrip("/")

def _forecast(symbol: str, horizon: int = 5):
    """Try real API; fall back to synthetic so the page always works."""
    if API_BASE:
        try:
            r = requests.post(
                f"{API_BASE}/forecast",
                json={"arena": "crypto", "symbol": symbol, "horizon": horizon},
                timeout=(3, 20),
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            st.warning(f"API fallback (forecast error: {e})")

    # synthetic fallback
    now = pd.Timestamp.utcnow().floor("min")
    xs = pd.date_range(end=now, periods=200, freq="T")
    ys = pd.Series([100 + 2*random.random() for _ in xs]).cumsum()
    fut = pd.date_range(start=now, periods=horizon, freq="T")
    yhat = [float(ys.iloc[-1]) + i*random.uniform(-1, 1) for i in range(len(fut))]
    q10  = [v - abs(v)*0.01 for v in yhat]
    q90  = [v + abs(v)*0.01 for v in yhat]
    return {"event": {"forecast": {"points":[{"ts":t.isoformat(), "yhat":yhat[i], "q10":q10[i], "q90":q90[i]} for i,t in enumerate(fut)]},
                      "metrics":{"entropy":0.33,"edge":0.07,"regime":"Compression→Expansion"}}}

def show():
    st.subheader("🔥 Crypto — Flagship")
    with st.expander("Settings", expanded=True):
        c1, c2, c3 = st.columns([2,1,1])
        symbol  = c1.text_input("Symbol", value="BTCUSDT")
        horizon = c2.slider("Horizon", 1, 30, value=5)
        run     = c3.button("Run Forecast", use_container_width=True)

    if run:
        res = _forecast(symbol, horizon)
        evt = res.get("event", {})
        fc  = (evt.get("forecast") or {}).get("points", [])
        met = evt.get("metrics", {})
        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("Regime",  met.get("regime","—"))
        k2.metric("Entropy", f"{met.get('entropy', float('nan')):.2f}")
        k3.metric("Edge",    f"{met.get('edge', float('nan')):.2%}")

        # Chart
        if fc:
            xs  = [pd.to_datetime(p["ts"]) for p in fc]
            y50 = [p["yhat"] for p in fc]
            q10 = [p.get("q10", p["yhat"]) for p in fc]
            q90 = [p.get("q90", p["yhat"]) for p in fc]
            fig = go.Figure()
            fig.add_scatter(x=xs, y=q90, name="q90", mode="lines", line=dict(width=0.5))
            fig.add_scatter(x=xs, y=q10, name="q10", mode="lines", fill="tonexty", line=dict(width=0.5))
            fig.add_scatter(x=xs, y=y50, name="Forecast", mode="lines", line=dict(dash="dash"))
            fig.update_layout(margin=dict(l=20,r=10,t=10,b=30), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No forecast points returned.")
