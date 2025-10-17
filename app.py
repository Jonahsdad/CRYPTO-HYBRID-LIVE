# ============================== PHASE 9 — DISPLAY OPTIMIZATION ==============================
# Crypto Hybrid Live — Phase 9 (UI Enhancement + 4x Visual Tabs + Smooth Flow)
# Full ready-to-paste file. Replaces your current app.py.
# ============================================================================================

import math, time, json, re, io, requests, pandas as pd, numpy as np, streamlit as st
from datetime import datetime, timezone

try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ---- Config ----
APP_NAME = "Crypto Hybrid Live — Phase 9 (Display Optimized)"
st.set_page_config(page_title=APP_NAME, layout="wide")

# ---- Custom CSS Styling ----
def _apply_css():
    st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-size: 18px !important;
    }
    /* ---- Massive Tab Buttons ---- */
    div[data-baseweb="tab-list"] button {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        margin-right: 1rem !important;
        background: linear-gradient(135deg, #22c55e, #15803d);
        color: white !important;
        border: 3px solid #16a34a !important;
        transition: all .3s ease-in-out;
        transform: scale(1.0);
    }
    div[data-baseweb="tab-list"] button:hover {
        transform: scale(1.1);
        background: linear-gradient(135deg, #16a34a, #166534);
        box-shadow: 0 0 15px rgba(34,197,94,0.6);
    }
    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #4ade80, #16a34a);
        color: #111 !important;
        transform: scale(1.2);
        box-shadow: 0 0 25px rgba(74,222,128,0.9);
    }
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
    }
    .stTabs [role="tablist"] {
        justify-content: center !important;
        margin-bottom: 1.5rem;
    }
    .phase-banner {
        font-size: 2rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #22c55e, #15803d);
        color: white;
        border-radius: 15px;
        padding: .5rem 0;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
_apply_css()

# ---- Demo dataset to simulate Phase 8 results ----
@st.cache_data(ttl=60)
def fake_data():
    names=["Bitcoin","Ethereum","Solana","Avalanche","Cardano","Dogecoin","Chainlink","Polkadot"]
    syms=["BTC","ETH","SOL","AVAX","ADA","DOGE","LINK","DOT"]
    data=[]
    for n,s in zip(names,syms):
        data.append({
            "name":n,"symbol":s,
            "current_price":np.random.uniform(0.1,50000),
            "truth_full":np.random.uniform(0,1),
            "raw_heat":np.random.uniform(0,1),
            "fusion_v2":np.random.uniform(0,1),
            "divergence":np.random.uniform(-.5,.5)
        })
    return pd.DataFrame(data)

df=fake_data()

# ---- Banner ----
st.markdown(f'<div class="phase-banner">🟢 {APP_NAME}</div>', unsafe_allow_html=True)
st.caption("Enhanced Visual Layer • Bigger Tabs • Smooth Metrics • Phase 9 UI")

# ---- KPIs ----
col1,col2,col3,col4 = st.columns(4)
col1.metric("Assets", len(df))
col2.metric("Avg Truth", f"{df['truth_full'].mean():.2f}")
col3.metric("Avg Fusion", f"{df['fusion_v2'].mean():.2f}")
col4.metric("Avg Raw", f"{df['raw_heat'].mean():.2f}")

# ---- TABS (Large Display) ----
tab1, tab2, tab3, tab4 = st.tabs(["🧭 TRUTH","🔥 RAW","🧠 FUSION","📈 MOVERS"])

with tab1:
    st.subheader("🧭 LIPE Truth Table (Top Signals)")
    st.dataframe(df.sort_values("truth_full", ascending=False), use_container_width=True, height=500)

with tab2:
    st.subheader("🔥 Raw Data Scan (Unfiltered Strength)")
    st.dataframe(df.sort_values("raw_heat", ascending=False), use_container_width=True, height=500)

with tab3:
    st.subheader("🧠 Fusion AI Score (Truth + Sentiment + Cross-Market)")
    st.dataframe(df.sort_values("fusion_v2", ascending=False), use_container_width=True, height=500)

with tab4:
    st.subheader("📈 Divergence Movers (Deviation Detector)")
    st.dataframe(df.sort_values("divergence", ascending=False), use_container_width=True, height=500)

st.markdown("---")
st.caption("© 2025 Crypto Hybrid Live | Phase 9 Display Optimization | Powered by LIPE Engine")
