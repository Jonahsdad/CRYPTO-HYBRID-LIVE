# app.py (Streamlit UI)
import os, requests, json
import streamlit as st
from datetime import datetime, timezone

DEFAULT_API = os.getenv("HIS_API_URL", "https://his-gateway.onrender.com")

# --------- Session Defaults ----------
st.set_page_config(page_title="HIS — Core Engine", page_icon="🧠", layout="wide")
SS = st.session_state
SS.setdefault("arena", "Lottery")
SS.setdefault("api_url", DEFAULT_API)
SS.setdefault("compute", "Remote API")
SS.setdefault("strictness", 55)
SS.setdefault("last_json", {})

def ts():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")

# --------- HTTP helpers ----------
def safe_get(url, **kw):
    try:
        r = requests.get(url, timeout=kw.pop("timeout", 12))
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, {"error": str(e)}

def safe_post(url, payload, **kw):
    try:
        r = requests.post(url, json=payload, timeout=kw.pop("timeout", 18))
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, {"error": str(e)}

def gateway_status():
    return safe_get(f"{SS.api_url}/status")

def run_lottery(game="pick4", window="last_30", mode="standard"):
    payload = {"game": game, "window": window, "mode": mode, "strictness": SS.strictness}
    ok, data = safe_post(f"{SS.api_url}/v1/lipe/forecast", payload)
    if not ok and "404" in str(data.get("error","")):
        ok, data = safe_post(f"{SS.api_url}/forecast", payload)
    return ok, data

def run_crypto(universe=None, mode="standard"):
    if universe is None:
        universe = ["BTC","ETH","SOL","LINK","AVAX","RNDR","TIA"]
    payload = {"universe": universe, "mode": mode, "strictness": SS.strictness}
    return safe_post(f"{SS.api_url}/v1/crypto/scan", payload)

def run_stocks(watchlist=None, mode="standard"):
    if watchlist is None:
        watchlist = ["AAPL","NVDA","MSFT","META","TSLA","AMZN"]
    payload = {"watchlist": watchlist, "mode": mode, "strictness": SS.strictness}
    return safe_post(f"{SS.api_url}/v1/stocks/scan", payload)

# --------- Sidebar (Action Panel) ----------
with st.sidebar:
    st.caption("System")
    st.radio("Compute", ["Local (in-app)", "Remote API"],
             index=1 if SS.compute == "Remote API" else 0, key="compute")
    st.text_input("API URL (Remote)", value=SS.api_url, key="api_url")
    st.caption("Truth Filter")
    st.slider("Signal Strictness", 0, 100, SS.strictness, key="strictness")

    st.divider()
    st.caption("Actions")

    col1, col2 = st.columns(2)
    ping_clicked = col1.button("⚡ Ping Gateway")
    go_clicked = col2.button("🚀 Run")

    st.caption("Last Response")
    if SS.last_json:
        st.json(SS.last_json)

# --------- Handle Actions ----------
gw_ok, gw_data = gateway_status()
if ping_clicked:
    gw_ok, gw_data = gateway_status()
    SS.last_json = {"status": "ok" if gw_ok else "error", "data": gw_data}

if go_clicked:
    if SS.arena == "Lottery":
        ok, data = run_lottery(game="pick4", window="last_30", mode="standard")
    elif SS.arena == "Crypto":
        ok, data = run_crypto()
    else:
        ok, data = run_stocks()
    SS.last_json = data if ok else {"status":"error","data":data}

# --------- Header ----------
st.markdown("## 🧠 Hybrid Intelligence Systems —\n### Core Engine")
st.caption("Powered by LIPE — Developed by Jesse Ray Landingham Jr")

badge = "🟢 Gateway Online" if gw_ok else "🔴 Gateway Offline"
bg = "#123e2b" if gw_ok else "#3e1b1b"
st.markdown(
    f"<div style='background:{bg};padding:8px 12px;border-radius:8px;color:white'>{badge}</div>",
    unsafe_allow_html=True,
)
st.caption(f"Gateway URL: {SS.api_url}")
st.caption(f"Checked: {ts()}")
st.json({"ok": gw_ok, "data": gw_data} if gw_ok else {"ok": False, "error":"Status route not reachable"})

# --------- Choose your arena (main canvas) ----------
st.markdown("## Choose your arena")
c1, c2, c3 = st.columns(3)
if c1.button("🏛️ Lottery"): SS.arena = "Lottery"
if c2.button("💰 Crypto"):   SS.arena = "Crypto"
if c3.button("📈 Stocks"):   SS.arena = "Stocks"

st.markdown("### Home Arena")
st.write(f"Selected: **{SS.arena}**")

# --------- Per-arena content ----------
if SS.arena == "Lottery":
    a, b, c = st.columns(3)
    if a.button("🎯 Pick 3 Forecast"): SS.last_json = run_lottery("pick3")[1]
    if b.button("🧠 Pick 4 Forecast"): SS.last_json = run_lottery("pick4")[1]
    if c.button("🍀 Lucky Day Forecast"): SS.last_json = run_lottery("ldl")[1]

    st.info("Lottery module: daily numbers, picks, entropy, RP overlays.")
    if SS.last_json:
        st.subheader("Forecast Result")
        st.json(SS.last_json)

elif SS.arena == "Crypto":
    a, b = st.columns(2)
    if a.button("🔎 Scan Top Coins"): SS.last_json = run_crypto()[1]
    if b.button("🧭 NBC Mode Scan"):  SS.last_json = run_crypto(mode="nbc")[1]

    st.info("Crypto module: live signals, entropy, NBC overlays.")
    if SS.last_json:
        st.subheader("Crypto Scan")
        st.json(SS.last_json)

elif SS.arena == "Stocks":
    a, b = st.columns(2)
    if a.button("🔍 Scan Watchlist"): SS.last_json = run_stocks()[1]
    if b.button("🧭 NBC Mode Scan"):  SS.last_json = run_stocks(mode="nbc")[1]

    st.info("Stocks module: momentum, factor overlays, pressure index.")
    if SS.last_json:
        st.subheader("Stocks Scan")
        st.json(SS.last_json)
