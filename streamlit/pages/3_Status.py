# streamlit/pages/3_Status.py
import streamlit as st, requests, pandas as pd
st.set_page_config(page_title="Status — HIS", page_icon="📊", layout="wide")

st.subheader("📊 System Status")
base = st.session_state.get("api_base") or ""
col = st.columns(2)

try:
    if base:
        r = requests.get(f"{base}/healthz", timeout=6)
        ok = r.json().get("ok")
        st.metric("API", "UP" if ok else "DOWN")
    else:
        st.warning("Set API Base URL in the sidebar.")
except Exception as e:
    st.error(f"Health failed: {e}")

st.caption("Upgrade later: publish latency, accuracy, model cards.")
