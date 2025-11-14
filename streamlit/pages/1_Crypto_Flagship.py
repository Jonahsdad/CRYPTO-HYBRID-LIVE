# streamlit/pages/1_Crypto_Flagship.py
from __future__ import annotations
import requests, pandas as pd, plotly.graph_objs as go
import streamlit as st

def _api() -> str:
    base = st.session_state.get("api_base") or ""
    if not base:
        st.error("Set API Base URL in the sidebar and click Connect.")
        st.stop()
    if not st.session_state.get("connected"):
        st.warning("Not connected. Click Connect in the sidebar.")
    return base

st.set_page_config(page_title="Crypto Flagship — HIS", page_icon="🟣", layout="wide")

st.subheader("🟣 Crypto Flagship")
st.caption("Bands • Entropy • Regime • Share links")

symbol = st.text_input("Symbol", value="BTCUSDT")
horizon = st.slider("Horizon (days)", 1, 30, 5)

c1, c2 = st.columns([1, 1])
with c1:
    go_btn = st.button("Run Forecast", type="primary")
with c2:
    share_btn = st.button("Create Share Link", disabled=True)

ph_chart = st.empty()
k1, k2, k3, k4 = st.columns(4)
ph_msg = st.empty()

if go_btn:
    try:
        r = requests.post(f"{_api()}/v1/forecast",
                          json={"arena": "crypto", "symbol": symbol, "horizon": horizon},
                          timeout=20)
        r.raise_for_status()
        evt = r.json()
        st.session_state["last_event"] = evt
        ph_msg.success("Forecast ready")
        # plot
        ser = pd.DataFrame(evt["series"])
        fc = pd.DataFrame(evt["forecast"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ser["ts"], y=ser["close"], mode="lines", name="Actual"))
        fig.add_trace(go.Scatter(x=fc["ts"], y=fc["q90"], mode="lines", name="q90", line=dict(width=0.1)))
        fig.add_trace(go.Scatter(x=fc["ts"], y=fc["q10"], mode="lines", name="q10",
                                 fill="tonexty", line=dict(width=0.1)))
        fig.add_trace(go.Scatter(x=fc["ts"], y=fc["yhat"], mode="lines", name="Forecast", line=dict(dash="dash")))
        fig.update_layout(margin=dict(l=20,r=10,t=10,b=30), hovermode="x unified")
        ph_chart.plotly_chart(fig, use_container_width=True)

        m = evt["metrics"]
        k1.metric("Regime", m["regime"])
        k2.metric("Entropy", m["entropy"])
        k3.metric("Edge", m["edge"])
        k4.metric("RP", m["rp"])
        share_btn = st.button("Create Share Link", key="share2", type="secondary")
        if share_btn:
            sr = requests.post(f"{_api()}/v1/share/create", json={"ttl_minutes": 120}, timeout=10)
            sr.raise_for_status()
            u = sr.json()["url"]
            st.success(f"Share created: `{u}`")
            st.write(f"Open: `{_api()}{u}`")

    except Exception as e:
        ph_msg.error(f"Failed: {e}")
