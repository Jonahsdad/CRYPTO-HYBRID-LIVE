# ================================================================
#  HYBRID INTELLIGENCE SYSTEMS — CORE DASHBOARD
#  File: main.py
#  Author: Jesse Ray Landingham Jr
#  Version: Stable Render/Streamlit Deployment
# ================================================================

import streamlit as st
import requests
import os
from datetime import datetime

# ------------------------------------------------
# Environment Setup
# ------------------------------------------------
st.set_page_config(
    page_title="Hybrid Intelligence Systems — LIPE Core",
    layout="wide",
    page_icon="🧠"
)

# Load environment variables or Streamlit secrets
GATEWAY_URL = os.getenv("GATEWAY_URL", st.secrets.get("GATEWAY_URL", "http://localhost:8000"))
HIS_KEY = os.getenv("HIS_KEY", st.secrets.get("HIS_KEY", "demo-key"))

# ------------------------------------------------
# Gateway Communication Layer
# ------------------------------------------------
def ping_gateway():
    """Check the gateway’s /health route."""
    try:
        res = requests.get(f"{GATEWAY_URL}/health", timeout=10)
        if res.status_code == 200:
            return res.json()
        else:
            return {"status": "error", "code": res.status_code}
    except Exception as e:
        return {"status": "offline", "error": str(e)}

def send_to_gateway(payload: dict):
    """Forward payload to gateway endpoint."""
    try:
        headers = {"x-his-key": HIS_KEY}
        res = requests.post(f"{GATEWAY_URL}/api/test", json=payload, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.json()
        else:
            return {"error": f"Gateway returned {res.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# ------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------
st.sidebar.title("🧭 Choose Your Arena")
section = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🎰 Lottery",
        "💰 Crypto",
        "📈 Stocks",
        "⚽ Sports",
        "🏡 Real Estate",
        "🪙 Commodities",
        "🧍‍♂️ Human Behavior",
        "🔮 Astrology"
    ]
)

# ------------------------------------------------
# Header
# ------------------------------------------------
st.title("🧠 Hybrid Intelligence Systems — Core Engine")
st.caption("Powered by LIPE · Developed by Jesse Ray Landingham Jr")

# ------------------------------------------------
# System Status Block
# ------------------------------------------------
with st.container():
    st.subheader("System Status")
    status = ping_gateway()
    if status.get("status") == "ok":
        st.success(f"✅ Gateway Active — {status.get('service', 'Unknown')} ({status.get('env', 'local')})")
    else:
        st.error("🚫 Gateway Offline or Unreachable")
        st.write(status)

    st.write(f"**Gateway URL:** `{GATEWAY_URL}`")

# ------------------------------------------------
# Quick Start Guide
# ------------------------------------------------
with st.expander("🚀 Quick Start"):
    st.markdown("""
    - Each page connects through the HIS Gateway (FastAPI backend).
    - The LIPE engine runs forecasts for lottery, crypto, and market arenas.
    - The Gateway URL above should match your Render deployment.
    - Status green = ready for live forecasting.
    """)

# ------------------------------------------------
# Core Arena Routing (Examples)
# ------------------------------------------------
if section == "🏠 Home":
    st.header("🏠 Home Arena")
    st.info("Welcome to Hybrid Intelligence Systems. Use the sidebar to explore different domains.")
    st.image("https://i.imgur.com/0V9Qh9K.png", use_container_width=True)

elif section == "💰 Crypto":
    st.header("💰 Crypto Forecasts")
    st.write("Example call to LIPE gateway:")
    data = send_to_gateway({"domain": "crypto", "timestamp": datetime.utcnow().isoformat()})
    st.json(data)

elif section == "🎰 Lottery":
    st.header("🎰 Lottery Arena")
    st.write("Draw data and predictions will appear here once LIPE is connected.")
    st.info("Future integration: Illinois Pick 3 & Pick 4 engine feed.")

elif section == "📈 Stocks":
    st.header("📈 Stock Market Arena")
    st.info("Real-time market integration coming soon. Streamlit & FRED API sync.")

elif section == "⚽ Sports":
    st.header("⚽ Sports Forecast Arena")
    st.info("Displays performance models and odds analysis.")

elif section == "🏡 Real Estate":
    st.header("🏡 Real Estate Intelligence")
    st.info("Dynamic housing metrics and predictive analytics for key markets.")

elif section == "🪙 Commodities":
    st.header("🪙 Commodities & Metals")
    st.info("Forecast models for gold, silver, oil, and emerging resources.")

elif section == "🧍‍♂️ Human Behavior":
    st.header("🧍‍♂️ Human Behavior Mapping")
    st.info("Behavioral pattern detection and LIPE emotional entropy mapping.")

elif section == "🔮 Astrology":
    st.header("🔮 Astrological Mapping")
    st.info("Archetype and planetary correlation layer for human forecasting.")

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.divider()
st.caption(f"© {datetime.now().year} Hybrid Intelligence Systems — All Rights Reserved.")
