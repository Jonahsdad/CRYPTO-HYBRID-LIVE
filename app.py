# app.py
import os
import json
import time
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

import requests
import streamlit as st

# ────────────────────────────────────────────────────────────────────────────────
# Page / Theme
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid Intelligence Systems — Core Engine",
    page_icon="🧬",
    layout="wide",
)

# Minimal dark-friendly tweaks (optional)
st.markdown(
    """
    <style>
    .metric-ok {background: #1b4332; color:#d8f3dc; padding:6px 10px; border-radius:8px; font-weight:600;}
    .pill      {background: #111827; padding:6px 10px; border-radius:999px; border:1px solid #374151;}
    .muted     {opacity: .75;}
    .jsonbox   {background:#0b1220; border:1px solid #1f2937; border-radius:8px; padding:12px;}
    .arena-btn {width:100%; text-align:left;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
GATEWAY_URL_DEFAULT = os.getenv("HIS_GATEWAY_URL", "https://his-gateway.onrender.com")

def _safe_get(url: str, timeout: int = 15) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    try:
        r = requests.get(url, timeout=timeout)
        ok = r.ok
        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text}
        return ok, data, ""
    except Exception as e:
        return False, None, str(e)

def ping_gateway(base: str) -> Dict[str, Any]:
    url = base.rstrip("/") + "/ping"
    ok, data, err = _safe_get(url)
    if not ok:
        return {"status": "error", "error": err or data}
    return {"status": "ok", "data": data}

def gateway_status(base: str) -> Dict[str, Any]:
    # If your gateway has a /status route, use it; otherwise fall back to /ping
    for path in ("/status", "/ping"):
        ok, data, err = _safe_get(base.rstrip("/") + path)
        if ok:
            return {"status": "ok", "service": "HIS_Gateway", "env": "production", "ts": datetime.utcnow().isoformat()+"Z", "data": data}
    return {"status": "error", "service": "HIS_Gateway", "error": err or "Status route not reachable"}

def run_home_forecast(base: str, strictness: int) -> Dict[str, Any]:
    # Adjust to your real endpoint (POST/GET) as needed
    url = base.rstrip("/") + f"/forecast/home?strictness={strictness}"
    ok, data, err = _safe_get(url)
    if not ok:
        return {"status": "error", "error": err or data}
    return {"status": "ok", "result": data}

def status_pill(ok: bool) -> str:
    if ok:
        return '<span class="metric-ok">● Gateway Online</span>'
    return '<span class="pill">○ Gateway Offline</span>'

# ────────────────────────────────────────────────────────────────────────────────
# Sidebar (System • Truth Filter • Actions)  ←——  FULL FIX: Actions moved here
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.caption("System")
    compute = st.radio(
        "Compute",
        options=["Local (in-app)", "Remote API"],
        index=1,
    )

    api_url = st.text_input(
        "API URL (Remote)",
        value=st.session_state.get("api_url", GATEWAY_URL_DEFAULT),
        help="Your gateway base URL.",
    )
    st.session_state["api_url"] = api_url

    st.divider()

    st.caption("Truth Filter")
    signal_strictness = st.slider(
        "Signal Strictness",
        min_value=0, max_value=100, value=55,
        help="Controls which module controls appear on this page."
    )

    st.divider()

    # ── Actions now live in the sidebar
    with st.expander("⚙️ Actions", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            if st.button("⚡ Ping Gateway", use_container_width=True):
                with st.spinner("Pinging gateway…"):
                    res = ping_gateway(api_url)
                if res.get("status") == "ok":
                    st.toast("Gateway responded ✅", icon="✅")
                    st.json(res.get("data"))
                else:
                    st.error(f"Ping failed: {res.get('error')}")

        with c2:
            if st.button("🔮 Get Forecast for Home", use_container_width=True):
                with st.spinner("Running forecast…"):
                    res = run_home_forecast(api_url, signal_strictness)
                if res.get("status") == "ok":
                    st.toast("Forecast complete ✅", icon="🎯")
                    st.json(res.get("result"))
                else:
                    st.error(f"Forecast failed: {res.get('error')}")

        # Best cross-platform “copy” is Streamlit's built-in copy icon on code blocks
        st.caption("API URL")
        st.code(api_url, language="text")  # shows a copy button automatically

# ────────────────────────────────────────────────────────────────────────────────
# Main Header
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("# 🧬 Hybrid Intelligence Systems — Core Engine")
st.markdown("_Powered by LIPE — Developed by Jesse Ray Landingham Jr_")

# Gateway tiles
status = gateway_status(api_url)
is_online = status.get("status") == "ok"

left, mid, right = st.columns([1.2, 1, 1])
with left:
    st.markdown(status_pill(is_online), unsafe_allow_html=True)
with mid:
    st.markdown(
        f"""
        **Gateway URL**  
        <span class="pill">{api_url}</span>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        f"""
        **Checked**  
        <span class="pill">{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}</span>
        """,
        unsafe_allow_html=True,
    )

# Status JSON
st.markdown(" ")
st.markdown("**Status**")
st.markdown('<div class="jsonbox">', unsafe_allow_html=True)
st.code(json.dumps(
    {
        "status": status.get("status"),
        "service": status.get("service", "HIS_Gateway"),
        "env": status.get("env", "production"),
        "ts": status.get("ts", datetime.utcnow().isoformat()+"Z"),
        "data": status.get("data"),
        "error": status.get("error"),
    },
    indent=2,
), language="json")
st.markdown("</div>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# Choose your arena
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("## Choose your arena")

arenas = [
    ("🎰 Lottery", "Daily numbers, picks, entropy, risk modes"),
    ("💰 Crypto", "Live pricing, signals, overlays"),
    ("📈 Stocks", "Charts, momentum, factor overlays"),
    ("🧮 Options", "Chains, skew & IV views"),
    ("🏠 Real Estate", "Market tilt & projections"),
    ("🪙 Commodities", "Energy, metals, ag"),
    ("🏆 Sports", "Game signals and parlay edges"),
    ("🧠 Human Behavior", "Cognitive & sentiment lenses"),
    ("🔮 Astrology", "Playful probabilistic lens"),
]

rows = [arenas[i:i+3] for i in range(0, len(arenas), 3)]
for row in rows:
    cols = st.columns(3)
    for (label, blurb), col in zip(row, cols):
        with col:
            if st.button(label, key=f"arena_{label}", use_container_width=True):
                st.session_state["arena"] = label
            st.caption(blurb)

# ────────────────────────────────────────────────────────────────────────────────
# Home Arena (contextual content)
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("## Home Arena")
selected = st.session_state.get("arena", "🎰 Lottery")
st.write(f"Selected: **{selected}**")

# Example contextual content (swap with your real module loaders)
if selected == "🎰 Lottery":
    st.info("Lottery module: plug in your pick engines, entropy views, and RP overlays here.")
elif selected == "💰 Crypto":
    st.info("Crypto module: live pricing, signals, narrative overlays.")
# … extend for other arenas as you wire them.

# Footer
st.caption("Manage app")
