# streamlit/streamlit_app.py
from __future__ import annotations
import os, requests
import streamlit as st

# --- MUST be first ---
st.set_page_config(page_title="HIS — Flagship", page_icon="⚡", layout="wide")

st.session_state.setdefault("api_base", os.environ.get("API_BASE_URL", "").rstrip("/"))
st.session_state.setdefault("connected", False)

st.sidebar.title("HYBRID INTELLIGENCE SYSTEMS")
st.sidebar.caption("Global Forecast OS • Powered by LIPE")

api_in = st.sidebar.text_input("API Base URL", st.session_state["api_base"] or "https://<your-render-url>")
colA, colB = st.sidebar.columns(2)
with colA:
    if st.button("Connect", use_container_width=True):
        st.session_state["api_base"] = api_in.rstrip("/")
        try:
            r = requests.get(f'{st.session_state["api_base"]}/healthz', timeout=8)
            r.raise_for_status()
            st.session_state["connected"] = True
            st.sidebar.success("Connected")
        except Exception as e:
            st.session_state["connected"] = False
            st.sidebar.error(f"Failed: {e}")
with colB:
    if st.button("Disconnect", use_container_width=True):
        st.session_state["connected"] = False

st.sidebar.markdown("---")
st.sidebar.page_link("pages/1_Crypto_Flagship.py", label="Crypto Flagship", icon="🟣")
st.sidebar.page_link("pages/2_Plans.py", label="Plans & Pricing", icon="💳")
st.sidebar.page_link("pages/3_Status.py", label="Status", icon="📊")

st.markdown("""
<div style="padding:8px 0 2px;">
  <h2 style="margin:0">HYBRID INTELLIGENCE SYSTEMS</h2>
  <div style="opacity:.8">All arenas. Hybrid live. <b>Powered by LIPE</b>.</div>
</div>
""", unsafe_allow_html=True)

st.write("Pick a page on the left to begin. Start with **Crypto Flagship**.")
