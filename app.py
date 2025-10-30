import os, json, requests
from datetime import datetime
import streamlit as st

# ── Page config
st.set_page_config(page_title="Hybrid Intelligence Systems — Core Engine", page_icon="🧬", layout="wide")

# ── Helpers
GATEWAY_URL_DEFAULT = os.getenv("HIS_GATEWAY_URL", "https://his-gateway.onrender.com")

def _safe_get(url, timeout=15):
    try:
        r = requests.get(url, timeout=timeout)
        ok, text = r.ok, r.text
        try:
            data = r.json()
        except Exception:
            data = {"raw": text}
        return ok, data, ""
    except Exception as e:
        return False, None, str(e)

def _post_json(url, payload, timeout=25):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return r.status_code, r.headers.get("content-type", ""), r.text
    except Exception as e:
        return 0, "", str(e)

def ping_gateway(base):
    return _safe_get(base.rstrip("/") + "/ping")

def gateway_status(base):
    for path in ("/status", "/ping"):
        ok, data, err = _safe_get(base.rstrip("/") + path)
        if ok: return {"status":"ok","service":"HIS_Gateway","env":"production","ts":datetime.utcnow().isoformat()+"Z","data":data}
    return {"status":"error","service":"HIS_Gateway","data":None,"error":err or "Status route not reachable"}

def run_home_forecast(base, strictness, game="pick4", window="last_30", mode="standard"):
    url = base.rstrip("/") + "/v1/lipe/forecast"
    payload = {"game": game, "window": window, "mode": mode, "strictness": strictness}
    return _post_json(url, payload)

# ── Sidebar (System • Truth Filter • Actions)
with st.sidebar:
    st.caption("System")
    compute = st.radio("Compute", ["Local (in-app)", "Remote API"], index=1)

    api_url = st.text_input("API URL (Remote)", value=st.session_state.get("api_url", GATEWAY_URL_DEFAULT), help="Gateway base URL")
    st.session_state["api_url"] = api_url

    st.divider()
    st.caption("Truth Filter")
    signal_strictness = st.slider("Signal Strictness", 0, 100, 55)
    st.session_state["signal_strictness"] = signal_strictness

    st.divider()
    with st.expander("⚙️ Actions", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            if st.button("⚡ Ping Gateway", use_container_width=True):
                with st.spinner("Pinging…"):
                    ok, data, err = ping_gateway(api_url)
                if ok: st.toast("Gateway responded ✅"); st.json(data)
                else:  st.error(f"Ping failed: {err or data}")

        with c2:
            if st.button("🔮 Get Forecast for Home", use_container_width=True):
                with st.spinner("Running forecast…"):
                    code, ctype, body = run_home_forecast(api_url, signal_strictness)
                if code == 200 and ("json" in ctype.lower() or body.strip().startswith("{")):
                    try: st.toast("Forecast complete ✅"); st.json(json.loads(body))
                    except Exception: st.code(body, language="json")
                elif code in (404, 405, 501):
                    st.error("Forecast failed: endpoint not found on Gateway.")
                elif code == 0:
                    st.error(f"Network error: {body}")
                else:
                    st.error(f"Gateway HTTP {code}")
                    try: st.code(json.dumps(json.loads(body), indent=2), language="json")
                    except Exception: st.code(body or "<empty>", language="json")

    st.caption("API URL")
    st.code(api_url, language="text")  # built-in copy button

# ── Header
st.markdown("# 🧬 Hybrid Intelligence Systems — Core Engine")
st.markdown("_Powered by LIPE — Developed by Jesse Ray Landingham Jr_")

# ── Gateway tiles & status JSON
status = gateway_status(api_url)
is_online = status.get("status") == "ok"
left, mid, right = st.columns([1.2, 1, 1])
with left:
    st.markdown("**Gateway**")
    st.success("● Gateway Online") if is_online else st.error("○ Gateway Offline")
with mid:
    st.markdown("**Gateway URL**")
    st.link_button(api_url, api_url, help="Open gateway")
with right:
    st.markdown("**Checked**")
    st.code(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ'), language="text")

st.markdown("**Status**")
st.code(json.dumps({
    "status": status.get("status"),
    "service": status.get("service", "HIS_Gateway"),
    "env": status.get("env", "production"),
    "ts": status.get("ts", datetime.utcnow().isoformat()+"Z"),
    "data": status.get("data"),
    "error": status.get("error"),
}, indent=2), language="json")

# ── Choose your arena
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

# ── Home arena
st.markdown("## Home Arena")
selected = st.session_state.get("arena", "🎰 Lottery")
st.write(f"Selected: **{selected}**")
if selected == "🎰 Lottery":
    st.info("Lottery module: plug in your pick engines, entropy views, and RP overlays here.")
