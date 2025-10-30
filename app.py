import streamlit as st
import requests
import json
import time
import toml

# =====================================================
# HYBRID INTELLIGENCE SYSTEM — CORE ENGINE UI
# =====================================================

st.set_page_config(page_title="Hybrid Intelligence Systems — Core Engine", layout="wide")

st.sidebar.header("🧠 Choose Your Arena")
arena = st.sidebar.radio("Navigation", [
    "🏠 Home",
    "🎲 Lottery",
    "💰 Crypto",
    "📈 Stocks",
    "🏆 Sports",
    "🏡 Real Estate",
    "🪙 Commodities",
    "🧍 Human Behavior",
    "🔮 Astrology"
])

# =====================================================
# Load Secrets (Backend Config)
# =====================================================

try:
    secrets = toml.load(".streamlit/secrets.toml")
    GATEWAY_URL = secrets["GATEWAY_URL"]
except Exception:
    GATEWAY_URL = st.secrets.get("GATEWAY_URL", "")
    
st.title("🧬 Hybrid Intelligence Systems — Core Engine")
st.markdown("#### Powered by LIPE · Developed by Jesse Ray Landingham Jr")

# =====================================================
# Gateway Connection Check
# =====================================================

def check_gateway():
    try:
        r = requests.get(f"{GATEWAY_URL}/health", timeout=10)
        if r.status_code == 200:
            return True, r.json()
        else:
            return False, {"status": "offline", "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return False, {"status": "offline", "error": str(e)}

connected, response = check_gateway()

if connected:
    st.success("🟢 Gateway Online")
    st.json(response)
else:
    st.error("🚫 Gateway Offline or Unreachable")
    st.json(response)

st.markdown(f"**Gateway URL:** [{GATEWAY_URL}]({GATEWAY_URL})")

# =====================================================
# Home Arena
# =====================================================

if arena == "🏠 Home":
    st.subheader("🏠 Home Arena")
    st.markdown("Welcome to the Hybrid Intelligence Systems. Use the sidebar to explore different domains.")
    st.image("https://i.imgur.com/0V9H9qk.png", use_column_width=True)
    st.info("HIS Core Engine powered by LIPE — Tier 33 Intelligence Layer.")
    st.divider()
    st.caption("© 2025 Jesse Ray Landingham Jr — All Rights Reserved.")
