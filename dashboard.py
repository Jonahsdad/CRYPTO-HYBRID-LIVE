import streamlit as st
import requests, json
from datetime import datetime

st.set_page_config(page_title="Hybrid Intelligence Systems", page_icon="🧠", layout="wide")

# --- GATEWAY CONFIG ---
DEFAULT_API = "https://his-gateway.onrender.com"

# --- SIDE PANEL ---
with st.sidebar:
    st.title("⚙️ System")
    st.write("**Compute:** Remote API")
    st.text_input("API URL (Remote)", value=DEFAULT_API, key="api_url")

    st.markdown("### 🔍 Actions")
    if st.button("Ping Gateway"):
        try:
            r = requests.get(DEFAULT_API)
            st.json(r.json())
        except Exception as e:
            st.error(f"Ping failed: {e}")

    st.markdown("### 🎯 Get Forecast")
    signal_strictness = st.slider("Signal Strictness", 0, 100, 55)

# --- HEADER ---
st.markdown("## 🧠 Hybrid Intelligence Systems —")
st.markdown("### Core Engine")
st.caption("Powered by LIPE — Developed by Jesse Ray Landingham Jr")

# --- GATEWAY STATUS ---
try:
    resp = requests.get(DEFAULT_API)
    data = resp.json()
    st.success("🟢 Gateway Online")
    st.json(data)
except Exception as e:
    st.error(f"Gateway offline: {e}")

# --- CHOOSE ARENA ---
st.markdown("## Choose your arena")
arena = st.radio("Select Arena", ["🎰 Lottery", "💰 Crypto", "📈 Stocks"], horizontal=True)

st.markdown(f"### Home Arena: **{arena.replace('🎰','Lottery').replace('💰','Crypto').replace('📈','Stocks')}**")

# --- CONDITIONAL SECTIONS ---
if "Lottery" in arena:
    st.info("Lottery module: daily numbers, picks, entropy, RP overlays.")
    if st.button("Get Lottery Forecast"):
        payload = {"game": "pick4", "window": "last_30", "mode": "standard", "strictness": signal_strictness}
        try:
            r = requests.post(f"{DEFAULT_API}/v1/lipe/forecast", json=payload)
            st.json(r.json())
        except Exception as e:
            st.error(f"Request failed: {e}")

elif "Crypto" in arena:
    st.info("Crypto module: live market momentum, entropy, and top signals.")
    if st.button("Scan Crypto"):
        payload = {"universe": ["bitcoin","ethereum","solana","chainlink","avalanche-2"], "mode": "standard", "strictness": signal_strictness}
        try:
            r = requests.post(f"{DEFAULT_API}/v1/crypto/scan", json=payload)
            st.json(r.json())
        except Exception as e:
            st.error(f"Request failed: {e}")

elif "Stocks" in arena:
    st.info("Stocks module: intraday pressure, trend bias, and RP scoring.")
    if st.button("Scan Stocks"):
        payload = {"watchlist": ["AAPL","NVDA","MSFT","META","TSLA"], "mode": "standard", "strictness": signal_strictness}
        try:
            r = requests.post(f"{DEFAULT_API}/v1/stocks/scan", json=payload)
            st.json(r.json())
        except Exception as e:
            st.error(f"Request failed: {e}")
