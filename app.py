# app.py
import os, time, requests, json
import streamlit as st
from datetime import datetime, timezone

# ---------- Defaults ----------
DEFAULT_API = os.getenv("HIS_API_URL", "https://crypto-hybrid-live-2.onrender.com")

if "arena" not in st.session_state:
    st.session_state.arena = "Lottery"
if "last_json" not in st.session_state:
    st.session_state.last_json = {}
if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API
if "compute" not in st.session_state:
    st.session_state.compute = "Remote API"
if "strictness" not in st.session_state:
    st.session_state.strictness = 55

# ---------- API helpers ----------
def ts():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")

def get_status(api_base: str):
    try:
        r = requests.get(f"{api_base}/status", timeout=12)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, {"ok": False, "error": str(e)}

def post_forecast(api_base: str, game="pick4", window="last_30", mode="standard", strictness=55):
    payload = {
        "game": game,
        "window": window,
        "mode": mode,
        "strictness": strictness
    }
    try:
        # prefer v1 route if present
        url = f"{api_base}/v1/lipe/forecast"
        r = requests.post(url, json=payload, timeout=18)
        if r.status_code == 404:
            # fall back to /forecast if older gateway
            url = f"{api_base}/forecast"
            r = requests.post(url, json=payload, timeout=18)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, {"ok": False, "error": str(e), "note":"Forecast endpoint not reachable"}

# ---------- Sidebar (ACTION PANEL) ----------
with st.sidebar:
    st.caption("System")
    st.radio(
        "Compute",
        ["Local (in-app)", "Remote API"],
        index=1 if st.session_state.compute == "Remote API" else 0,
        key="compute",
        help="Remote API uses your Render gateway. Local runs inside this app (dev only).",
    )

    st.text_input(
        "API URL (Remote)",
        value=st.session_state.api_url,
        key="api_url",
        help="Your Render gateway base URL",
    )

    st.caption("Truth Filter")
    st.slider("Signal Strictness", 0, 100, st.session_state.strictness, key="strictness",
              help="Controls how strict the forecast filter is.")

    st.divider()
    st.caption("Actions")

    colA, colB = st.columns(2)
    ping_clicked = colA.button("⚡ Ping Gateway")
    forecast_clicked = colB.button("🔮 Get Forecast for Home")

    st.caption("API URL")
    st.code(st.session_state.api_url, language="text")

    # small live-out pane
    if st.session_state.last_json:
        st.json(st.session_state.last_json)

# ---------- Handle actions ----------
gateway_ok, status_json = get_status(st.session_state.api_url)
if ping_clicked:
    gateway_ok, status_json = get_status(st.session_state.api_url)
    st.session_state.last_json = {"status": "ok" if gateway_ok else "error", "data": status_json}

if forecast_clicked:
    ok, data = post_forecast(
        st.session_state.api_url,
        game="pick4",
        window="last_30",
        mode="standard",
        strictness=st.session_state.strictness,
    )
    st.session_state.last_json = data if ok else {"status":"error","data": data}

# ---------- Header & gateway badge ----------
st.markdown("## 🧠 Hybrid Intelligence Systems —\n### Core Engine")
st.caption("Powered by LIPE — Developed by Jesse Ray Landingham Jr")

badge = "🟢 Gateway Online" if gateway_ok else "🔴 Gateway Offline"
bg = "#113d2b" if gateway_ok else "#3d1111"
st.markdown(
    f"""
    <div style='background:{bg};padding:8px 12px;border-radius:8px;color:#d9f8e6 if gateway_ok else #ffdada'>
    {badge}
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(f"Gateway URL: {st.session_state.api_url}")
st.caption(f"Checked: {ts()}")

# show latest payload in main area
st.json({"ok": gateway_ok, "data": status_json} if gateway_ok else {"ok": False, "error": "Status route not reachable"})

# ---------- Choose your arena ----------
st.markdown("## Choose your arena")
c1, c2, c3 = st.columns(3)
if c1.button("🏛️ Lottery"):
    st.session_state.arena = "Lottery"
if c2.button("💰 Crypto"):
    st.session_state.arena = "Crypto"
if c3.button("📈 Stocks"):
    st.session_state.arena = "Stocks"

st.markdown("### Home Arena")
st.write(f"Selected: **{st.session_state.arena}**")

# Simple per-arena placeholder sections (replace with your real modules)
if st.session_state.arena == "Lottery":
    st.info("Lottery module: daily numbers, picks, entropy, RP overlays.")
elif st.session_state.arena == "Crypto":
    st.info("Crypto module: live pricing, signals, overlays.")
elif st.session_state.arena == "Stocks":
    st.info("Stocks module: charts, momentum, factor overlays.")
