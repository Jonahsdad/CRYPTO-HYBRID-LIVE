# app.py
# Hybrid Intelligence Systems — Core Engine UI
# Frontend for LIPE / HIS
# -----------------------------------------------------

import os, time, json
import requests
import streamlit as st

# ---------- Config ----------
st.set_page_config(
    page_title="Hybrid Intelligence Systems — Core Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Where the UI reads the gateway URL from
GATEWAY_URL = st.secrets.get("GATEWAY_URL", os.getenv("GATEWAY_URL", "")).rstrip("/")

# ---------- Helpers ----------
def api_get(path: str, **kwargs):
    """GET helper with friendly errors."""
    url = f"{GATEWAY_URL}{path}"
    try:
        r = requests.get(url, timeout=15, **kwargs)
        ct = r.headers.get("content-type", "")
        if "application/json" in ct:
            return r.status_code, r.json()
        else:
            # Sometimes frameworks return text for debug; try to present it
            return r.status_code, {"raw": r.text}
    except Exception as e:
        return 0, {"error": str(e)}

def health_check():
    return api_get("/health")

def ping_gateway(msg: str = "hello LIPE"):
    return api_get(f"/api/test", params={"msg": msg})

def run_forecast(domain: str):
    # Your FastAPI stub: /api/forecast?domain=crypto|lottery|stocks|...
    return api_get("/api/forecast", params={"domain": domain})

# ---------- Sidebar ----------
st.sidebar.header("🧠 Choose Your Arena")
arena = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🎲 Lottery",
        "💰 Crypto",
        "📈 Stocks",
        "🏆 Sports",
        "🏡 Real Estate",
        "🪙 Commodities",
        "🧍 Human Behavior",
        "🔮 Astrology",
    ],
    label_visibility="collapsed",
)

with st.sidebar.expander("⚙️ Control Panel", expanded=True):
    colA, colB = st.columns(2)
    if colA.button("🔄 Health"):
        st.session_state["last_health"] = health_check()
    if colB.button("📣 Ping"):
        st.session_state["last_ping"] = ping_gateway()

    st.selectbox  # silence linter, keeps layout nice
    sel_domain = st.selectbox(
        "Forecast Domain",
        ["crypto", "lottery", "stocks", "sports", "real_estate", "commodities"],
        index=0,
    )
    if st.button("🧠 Run Forecast"):
        st.session_state["last_forecast"] = (sel_domain, run_forecast(sel_domain))

st.sidebar.markdown("---")
st.sidebar.caption(f"Gateway: `{GATEWAY_URL or 'NOT SET'}`")

# ---------- Header ----------
st.title("🧬 Hybrid Intelligence Systems — Core Engine")
st.markdown("#### Powered by LIPE · Developed by Jesse Ray Landingham Jr")

# ---------- Status / Quick Start panel ----------
status_col, quick_col = st.columns([1, 1])

with status_col:
    code, data = health_check()
    ok = (code == 200) and isinstance(data, dict) and data.get("status") in ("ok", "running", "healthy")
    if ok:
        st.success("🟢 Gateway Online")
    else:
        st.error("🔴 Gateway Offline or Unreachable")

    st.json(data)

with quick_col:
    st.markdown("### Quick Start")
    st.markdown("- Use the left sidebar to select an arena.")
    st.markdown("- Each page reads data via the Gateway (FastAPI).")
    st.text_input("Gateway URL", GATEWAY_URL, disabled=True)
    st.markdown(
        f"[Open Gateway Docs]({GATEWAY_URL}/docs)"
        if GATEWAY_URL
        else "_Set `GATEWAY_URL` in Streamlit **Secrets**._"
    )

st.divider()

# ---------- Arena Pages ----------
def render_domain_result(title: str, domain_key: str):
    st.subheader(title)
    st.caption(f"Domain key: `{domain_key}` · Source: `{GATEWAY_URL}/api/forecast?domain={domain_key}`")
    with st.spinner("Fetching forecast..."):
        code, payload = run_forecast(domain_key)
    if code == 200:
        st.success("Forecast OK")
        # Common expected fields: prediction(s), confidence/entropy, reasoning
        if isinstance(payload, dict):
            # Show headline if present
            headline = payload.get("headline")
            if headline:
                st.markdown(f"**{headline}**")
            # Metrics
            met1, met2, met3 = st.columns(3)
            met1.metric("Confidence", f"{payload.get('confidence', '—')}")
            met2.metric("Entropy", f"{payload.get('entropy', '—')}")
            met3.metric("Timestamp", f"{payload.get('ts', '—')}")
            # Body
            st.json(payload)
        else:
            st.write(payload)
    else:
        st.error(f"HTTP {code or '0'} from gateway")
        st.json(payload)

if arena == "🏠 Home":
    st.subheader("🏠 Home Arena")
    st.markdown(
        "Welcome to **Hybrid Intelligence Systems**. Use the sidebar to explore "
        "domains. The Control Panel lets you refresh health, ping the gateway, and run forecasts."
    )

    # Show last actions (so your new buttons feel alive)
    with st.expander("📜 Last Actions", expanded=False):
        if "last_health" in st.session_state:
            st.markdown("**Last Health:**")
            st.json({"status_code": st.session_state["last_health"][0], "payload": st.session_state["last_health"][1]})
        if "last_ping" in st.session_state:
            st.markdown("**Last Ping:**")
            st.json({"status_code": st.session_state["last_ping"][0], "payload": st.session_state["last_ping"][1]})
        if "last_forecast" in st.session_state:
            domain, (code, payload) = st.session_state["last_forecast"]
            st.markdown(f"**Last Forecast** for `{domain}`:")
            st.json({"status_code": code, "payload": payload})

elif arena == "🎲 Lottery":
    render_domain_result("🎲 Lottery — Forecast", "lottery")

elif arena == "💰 Crypto":
    render_domain_result("💰 Crypto — Forecast", "crypto")

elif arena == "📈 Stocks":
    render_domain_result("📈 Stocks — Forecast", "stocks")

elif arena == "🏆 Sports":
    render_domain_result("🏆 Sports — Forecast", "sports")

elif arena == "🏡 Real Estate":
    render_domain_result("🏡 Real Estate — Forecast", "real_estate")

elif arena == "🪙 Commodities":
    render_domain_result("🪙 Commodities — Forecast", "commodities")

elif arena == "🧍 Human Behavior":
    render_domain_result("🧍 Human Behavior — Forecast", "human_behavior")

elif arena == "🔮 Astrology":
    render_domain_result("🔮 Astrology — Forecast", "astrology")

# ---------- Footer ----------
st.divider()
st.caption("© 2025 Jesse Ray Landingham Jr — LIPE · HIS · All Rights Reserved")
