# main.py — Streamlit UI for Hybrid Intelligence Systems (HIS)
import os
import time
import json
from datetime import datetime

import requests
import pandas as pd
import streamlit as st

# ---------- Page Setup ----------
st.set_page_config(
    page_title="Hybrid Intelligence Systems",
    page_icon="🧠",
    layout="wide"
)

# ---------- Config / Secrets ----------
GATEWAY_URL = st.secrets.get("GATEWAY_URL", os.getenv("GATEWAY_URL", ""))
ENV_NAME    = st.secrets.get("ENV", "dev")
SERVICE     = st.secrets.get("SERVICE", "CRYPTO-HYBRID-LIVE")
OWNER       = st.secrets.get("OWNER", "Owner")

if not GATEWAY_URL:
    st.error("❌ GATEWAY_URL is missing. Add it in `.streamlit/secrets.toml`.")
    st.stop()

# ---------- Helpers ----------
def get_json(url: str, params=None, timeout=10):
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)

def status_badge(ok: bool):
    return "🟢 OK" if ok else "🔴 DOWN"

# ---------- Header ----------
left, right = st.columns([3, 2])
with left:
    st.markdown("### 🧠 Hybrid Intelligence Systems")
    st.caption(f"Powered by LIPE • Owner: **{OWNER}** • Service: **{SERVICE}** • Env: **{ENV_NAME}**")
with right:
    st.code(GATEWAY_URL, language="bash")

st.divider()

# ---------- System Status Card ----------
st.subheader("System Status")

health_url = f"{GATEWAY_URL}/health"
health_data, health_err = get_json(health_url, timeout=8)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Gateway", status_badge(health_err is None))
with col2:
    st.metric("Endpoint", "/health")
with col3:
    st.metric("Checked", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))

if health_err:
    st.error(f"Health check failed: {health_err}")
else:
    st.success("Gateway is reachable.")
    st.json(health_data)

st.divider()

# ---------- Quick Tests ----------
st.subheader("Quick Test")

with st.form("test_form", clear_on_submit=False):
    msg = st.text_input("Message to echo via /api/test?msg=", "hello LIPE")
    submitted = st.form_submit_button("Run Test")
    if submitted:
        test_url = f"{GATEWAY_URL}/api/test"
        data, err = get_json(test_url, params={"msg": msg})
        if err:
            st.error(f"/api/test failed: {err}")
        else:
            st.success("Test returned:")
            st.json(data)

st.divider()

# ---------- LIPE Forecast Wire (safe placeholder) ----------
st.subheader("LIPE Forecast (wire test)")
st.caption("This calls /v1/lipe/forecast if your Gateway exposes it. If not, we fallback gracefully.")

lipe_params = {
    "game": st.selectbox("Game", ["pick4", "pick3", "take5", "custom"], index=0),
    "window": st.selectbox("Window", ["last_30", "last_60", "ytd"], index=0),
    "mode": st.selectbox("Mode", ["standard", "entropy", "echo"], index=0)
}

if st.button("Run Forecast"):
    forecast_url = f"{GATEWAY_URL}/v1/lipe/forecast"
    data, err = get_json(forecast_url, params=lipe_params, timeout=20)
    if err:
        st.warning("Forecast endpoint not available yet on Gateway. Showing placeholder.")
        st.json({
            "provider": "lipe-forecast",
            "requested": lipe_params,
            "note": "Add /v1/lipe/forecast to the Gateway to activate this panel."
        })
    else:
        st.success("Forecast received.")
        st.json(data)

st.divider()

# ---------- Diagnostics ----------
st.subheader("Diagnostics")
diag = {
    "time_utc": datetime.utcnow().isoformat(),
    "gateway_url": GATEWAY_URL,
    "env": ENV_NAME,
    "service": SERVICE,
    "owner": OWNER,
    "python": os.sys.version.split()[0]
}
st.code(json.dumps(diag, indent=2), language="json")

st.caption("Tip: set Streamlit start command on Render to "
           "`streamlit run main.py --server.port=$PORT --server.address=0.0.0.0`")
