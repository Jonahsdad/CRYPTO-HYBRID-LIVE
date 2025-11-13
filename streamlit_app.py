# ============================================================
# FILE: streamlit_app.py
# HIS — Streamlit | Global Forecast OS (Crypto Flagship v1)
# ============================================================

from __future__ import annotations
import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.graph_objs as go

# API bindings
from lib.api import (
    api_login, api_symbols_crypto, api_timeseries_crypto,
    api_predict_crypto, api_signals_current,
    api_backtest, api_strategy_backtest, api_checkout
)

st.set_page_config(
    page_title="HIS — Powered by LIPE",
    page_icon="⚡",
    layout="wide"
)

css = Path("assets/style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

st.sidebar.subheader("Account")
st.sidebar.caption("Login to unlock Crypto forecasts.")

email = st.sidebar.text_input("Email")
team  = st.sidebar.text_input("Team")

if "token" not in st.session_state:
    st.session_state.token = None

colL, colR = st.sidebar.columns(2)

with colL:
    if st.button("Sign in", use_container_width=True):
        try:
            resp = api_login(email or "", team or "")
            st.session_state.token = resp.get("token")
            st.sidebar.success(f"Signed in • {resp.get('team')}")
        except:
            st.sidebar.error("Login failed — check email/team.")

with colR:
    if st.button("Sign out", use_container_width=True):
        st.session_state.token = None
        st.sidebar.info("Signed out.")

st.markdown(
    """
<div class="hero">
  <h1>HYBRID INTELLIGENCE SYSTEMS</h1>
  <div class="kicker">All arenas. Hybrid live. <b>Powered by LIPE</b>.</div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")
st.subheader("Choose your arena")

cards = [
    ("Crypto", "pages/1_Crypto_Flagship.py", "BTC/ETH • Bands • Strategy • Regime", "🔥"),
    ("Sports", None, "Edges • Odds • Momentum", "🏈"),
    ("Lottery", None, "GFW • Draws • Echo Mapping", "🎰"),
    ("Stocks", None, "Signals • Momentum • EOD", "📈"),
    ("Real Estate", None, "AVM • Macro • Trends", "🏠"),
]

cols = st.columns(3)
for i,(name, path, sub, emoji) in enumerate(cards):
    with cols[i % 3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f"### {emoji} {name}")
        st.caption(sub)

        if path:
            st.page_link(path, label="Enter", icon="➡️")
        else:
            st.button("Coming Soon", disabled=True)

        st.markdown("</div>", unsafe_allow_html=True)

st.caption("v1.0 • Streamlit • HIS Global Forecast OS")
