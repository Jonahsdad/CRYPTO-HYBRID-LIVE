import streamlit as st
import requests
import toml
from datetime import datetime

# =========================
#   CONFIG
# =========================
st.set_page_config(
    page_title="Hybrid Intelligence Systems — Core Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Resolve GATEWAY_URL from .streamlit/secrets.toml, Streamlit Cloud secrets, or env
def get_gateway_url() -> str:
    # 1) local .streamlit/secrets.toml (Render deploy also mounts this file)
    try:
        s = toml.load(".streamlit/secrets.toml")
        if s.get("GATEWAY_URL"):
            return s["GATEWAY_URL"].rstrip("/")
    except Exception:
        pass
    # 2) Streamlit Cloud/Render env secrets
    if "GATEWAY_URL" in st.secrets:
        return str(st.secrets["GATEWAY_URL"]).rstrip("/")
    # 3) fallback for local dev
    return "http://localhost:8000"

GATEWAY_URL = get_gateway_url()

ARENAS = [
    "🏠 Home",
    "🎲 Lottery",
    "💰 Crypto",
    "📈 Stocks",
    "🏆 Sports",
    "🏡 Real Estate",
    "🪙 Commodities",
    "🧍 Human Behavior",
    "🔮 Astrology",
]

# Map UI label -> domain value the gateway expects
DOMAIN_MAP = {
    "🏠 Home": "home",
    "🎲 Lottery": "lottery",
    "💰 Crypto": "crypto",
    "📈 Stocks": "stocks",
    "🏆 Sports": "sports",
    "🏡 Real Estate": "real_estate",
    "🪙 Commodities": "commodities",
    "🧍 Human Behavior": "human",
    "🔮 Astrology": "astrology",
}

# =========================
#   SIDEBAR
# =========================
st.sidebar.header("🧠 Choose Your Arena")
arena = st.sidebar.radio("Navigation", ARENAS, index=0)
st.sidebar.caption("Each page uses the HIS Gateway for data.")

# =========================
#   HELPERS
# =========================
def safe_get_json(url: str, timeout: int = 10):
    try:
        r = requests.get(url, timeout=timeout)
        ct = r.headers.get("content-type", "")
        # Some proxies return text; try json either way
        data = r.json() if "application/json" in ct or r.text.startswith("{") else {"raw": r.text}
        return r.status_code, data, None
    except Exception as e:
        return None, None, str(e)

def ping_gateway():
    return safe_get_json(f"{GATEWAY_URL}/api/test?msg=hello LIPE")

def check_health():
    return safe_get_json(f"{GATEWAY_URL}/health")

def get_forecast(domain: str):
    return safe_get_json(f"{GATEWAY_URL}/api/forecast?domain={domain}")

# =========================
#   HEADER
# =========================
st.title("🧬 Hybrid Intelligence Systems — Core Engine")
st.markdown("Powered by LIPE • Developed by **Jesse Ray Landingham Jr**")

# =========================
#   STATUS PANEL
# =========================
col1, col2, col3 = st.columns([1.2, 1, 1], vertical_alignment="center")

with col1:
    st.caption("System Status")
    hc_status, hc_data, hc_err = check_health()
    if hc_status == 200 and hc_data and hc_data.get("status") == "ok":
        st.success("🟢 Gateway Online")
    elif hc_err:
        st.error("🔴 Gateway Offline or Unreachable")
        st.code(hc_err, language="text")
    else:
        st.error(f"🔴 Gateway problem (HTTP {hc_status})")
        if hc_data:
            st.json(hc_data)

with col2:
    st.caption("Gateway URL")
    st.code(GATEWAY_URL, language="text")

with col3:
    st.caption("Checked")
    st.code(datetime.utcnow().isoformat(timespec="seconds") + "Z", language="text")

# =========================
#   ACTION BAR (on the panel)
# =========================
st.markdown("### ⚡ Actions")

ab1, ab2 = st.columns([1, 1], vertical_alignment="center")

with ab1:
    if st.button("📡 Ping Gateway", use_container_width=True):
        st.write("**Ping Result**")
        code, data, err = ping_gateway()
        if err:
            st.error(err)
        else:
            st.write(f"HTTP {code}")
            st.json(data)

with ab2:
    domain = DOMAIN_MAP[arena]
    if st.button(f"🎯 Get Forecast for {arena}", use_container_width=True):
        st.write(f"**Forecast: {arena}**")
        code, data, err = get_forecast(domain)
        if err:
            st.error(err)
        else:
            st.write(f"HTTP {code}")
            st.json(data)

st.divider()

# =========================
#   HOME ARENA
# =========================
if arena == "🏠 Home":
    st.subheader("🏠 Home Arena")
    st.write("Welcome to Hybrid Intelligence Systems. Use the sidebar to explore different domains.")
    # That old Imgur URL is dead; show a simple placeholder instead:
    st.image(
        "https://placehold.co/1200x400?text=HIS%20Core%20Engine%20%F0%9F%94%A5",
        caption="HIS Core Engine powered by LIPE — Tier 33 Intelligence Layer.",
        use_column_width=True,
    )

# (Other arenas can add their own UI blocks later; the forecast button already works.)
st.caption("© 2025 Jesse Ray Landingham Jr — All Rights Reserved.")
