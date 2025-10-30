import os, json, requests
from datetime import datetime
import streamlit as st

st.set_page_config(page_title="Hybrid Intelligence Systems — Core Engine", page_icon="🧬", layout="wide")

# ─── Gateway helpers ───────────────────────
def safe_get(url):
    try:
        r = requests.get(url, timeout=10)
        return r.ok, r.json() if r.ok else None, None
    except Exception as e:
        return False, None, str(e)

def safe_post(url, payload):
    try:
        r = requests.post(url, json=payload, timeout=20)
        return r.status_code, r.text
    except Exception as e:
        return 0, str(e)

def check_status(base):
    for path in ["/status", "/ping"]:
        ok, data, err = safe_get(base.rstrip("/") + path)
        if ok:
            return {"ok": True, "data": data}
    return {"ok": False, "error": err or "Status route not reachable"}

def run_forecast(base, strictness):
    payload = {"game": "pick4", "window": "last_30", "mode": "standard", "strictness": strictness}
    code, body = safe_post(base.rstrip("/") + "/v1/lipe/forecast", payload)
    return code, body

# ─── Sidebar ───────────────────────────────
with st.sidebar:
    st.caption("System")
    compute = st.radio("Compute", ["Local (in-app)", "Remote API"], index=1)
    api_url = st.text_input("API URL (Remote)", "https://his-gateway.onrender.com")
    signal_strictness = st.slider("Signal Strictness", 0, 100, 55)

    st.divider()
    st.caption("⚙️ Actions")

    if st.button("⚡ Ping Gateway", use_container_width=True):
        with st.spinner("Pinging..."):
            ok, data, err = safe_get(api_url.rstrip("/") + "/ping")
        if ok:
            st.toast("Gateway OK ✅")
            st.json(data)
        else:
            st.error(f"Ping failed: {err}")

    if st.button("🔮 Get Forecast for Home", use_container_width=True):
        with st.spinner("Running forecast..."):
            code, body = run_forecast(api_url, signal_strictness)
        if code == 200:
            st.toast("Forecast complete ✅")
            st.json(json.loads(body))
        elif code == 0:
            st.error(f"Network error: {body}")
        else:
            st.error(f"Forecast failed (HTTP {code})")
            st.code(body)

    st.caption("API URL")
    st.code(api_url, language="text")

# ─── Main ──────────────────────────────────
st.markdown("# 🧬 Hybrid Intelligence Systems — Core Engine")
st.markdown("_Powered by LIPE — Developed by Jesse Ray Landingham Jr_")

status = check_status(api_url)
is_ok = status["ok"]

if is_ok:
    st.success("● Gateway Online")
else:
    st.error("○ Gateway Offline")

st.write("**Gateway URL:**", api_url)
st.write("**Checked:**", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"))

st.code(json.dumps(status, indent=2), language="json")

st.markdown("## Choose your arena")
cols = st.columns(3)
for label in ["🎰 Lottery", "💰 Crypto", "📈 Stocks"]:
    with cols.pop(0):
        st.button(label)
