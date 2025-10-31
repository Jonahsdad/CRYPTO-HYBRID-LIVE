# dashboard.py — Streamlit UI (Choose your arena)
import streamlit as st
import requests, json
from datetime import datetime

st.set_page_config(page_title="Hybrid Intelligence Systems", page_icon="🧠", layout="wide")

DEFAULT_API = "https://his-gateway.onrender.com"

# -------- Helpers --------
def get_json(url: str):
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, {"error": str(e)}

def post_json(url: str, payload: dict):
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, {"error": str(e), "payload": payload}

# -------- Sidebar --------
with st.sidebar:
    st.title("⚙️ System")
    st.write("**Compute**")
    st.radio("Mode", ["Remote API"], index=0, key="compute", horizontal=True)
    api_url = st.text_input("API URL (Remote)", value=DEFAULT_API, key="api_url")

    st.markdown("### Truth Filter")
    strictness = st.slider("Signal Strictness", 0, 100, 55)

    st.markdown("### Actions")
    colA, colB = st.columns(2)
    with colA:
        ping_clicked = st.button("⚡ Ping Gateway")
    with colB:
        run_clicked = st.button("🧠 Run")

# -------- Header --------
st.markdown("## 🧠 Hybrid Intelligence Systems —")
st.markdown("### Core Engine")
st.caption("Powered by LIPE — Developed by Jesse Ray Landingham Jr")

# -------- Gateway Status (with /status → / fallback) --------
ok, data = get_json(f"{api_url}/status")
if not ok:
    ok, data = get_json(api_url)  # fallback to "/"

if ok:
    st.success("🟢 Gateway Online")
else:
    st.error("🔴 Gateway Offline")
st.json(data)

# Manual ping action reuses the same logic
if 'ping_clicked' in locals() and ping_clicked:
    okp, datap = get_json(f"{api_url}/status")
    if not okp:
        okp, datap = get_json(api_url)
    st.subheader("Last Response")
    st.json(datap)

# -------- Choose your arena --------
st.markdown("## Choose your arena")
arena = st.radio("", ["🎰 Lottery", "💰 Crypto", "📈 Stocks"], horizontal=True, label_visibility="collapsed")

st.markdown("### Home Arena")
if "Lottery" in arena:
    st.write("Selected: **Lottery**")
    c1, c2, c3 = st.columns(3)
    if c1.button("🎯 Pick 3 Forecast"):
        payload = {"game": "pick3", "window": "last_30", "mode": "standard", "strictness": strictness}
        okf, resp = post_json(f"{api_url}/v1/lipe/forecast", payload)
        st.json(resp)
    if c2.button("🧠 Pick 4 Forecast"):
        payload = {"game": "pick4", "window": "last_30", "mode": "standard", "strictness": strictness}
        okf, resp = post_json(f"{api_url}/v1/lipe/forecast", payload)
        st.json(resp)
    if c3.button("🍀 Lucky Day Forecast"):
        payload = {"game": "luckyd", "window": "last_30", "mode": "standard", "strictness": strictness}
        okf, resp = post_json(f"{api_url}/v1/lipe/forecast", payload)
        st.json(resp)

elif "Crypto" in arena:
    st.write("Selected: **Crypto**")
    if st.button("🔎 Scan Crypto"):
        payload = {"universe": ["bitcoin","ethereum","solana","chainlink","avalanche-2"], "mode": "standard", "strictness": strictness}
        okc, resp = post_json(f"{api_url}/v1/crypto/scan", payload)
        st.json(resp)

else:
    st.write("Selected: **Stocks**")
    if st.button("📊 Scan Stocks"):
        payload = {"watchlist": ["AAPL","NVDA","MSFT","META","TSLA"], "mode": "standard", "strictness": strictness}
        oks, resp = post_json(f"{api_url}/v1/stocks/scan", payload)
        st.json(resp)
