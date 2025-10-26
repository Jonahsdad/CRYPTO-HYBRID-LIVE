# HYBRID INTELLIGENCE SYSTEMS - Neon UX
# app.py - full, drop-in (landing tiles + minimal sidebar)

import sys, os
sys.path.append(os.path.dirname(__file__))

import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

try:
    import yfinance as yf
except Exception:
    yf = None

# --- Engine Import ---
try:
    from lipe_core import LIPE
except Exception as e:
    LIPE = None
    _IMPORT_ERR = f"Local LIPE unavailable: {e}"
else:
    _IMPORT_ERR = None

# --- Core Settings ---
APP_TITLE = "HYBRID INTELLIGENCE SYSTEMS"
APP_TAGLINE = "One brain. Many frontiers."
API_URL_DEFAULT = "https://YOUR-FORECAST-API.example.com"
API_URL = os.getenv("LIPE_API_URL", API_URL_DEFAULT)

MODULES = [
    ("🎲", "Lottery", "Daily numbers, picks, entropy, risk modes"),
    ("💰", "Crypto", "Live pricing, signals, overlays"),
    ("📈", "Stocks", "Charts, momentum, factor overlays"),
    ("🧾", "Options", "Chains, quick IV views"),
    ("🏠", "Real Estate", "Market tilt & projections"),
    ("🛢️", "Commodities", "Energy, metals, ag"),
    ("🏈", "Sports", "Game signals & parlay entropy"),
    ("🧑‍🤝‍🧑", "Human Behavior", "Cohort trends & intent"),
    ("🔭", "Astrology", "Playful probabilistic lens"),
]

# --- Helpers ---
def dl_csv(rows: List[Dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

def parse_draws_text(s: str) -> List[int]:
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except Exception:
        return []

def parse_draws_csv(file) -> List[int]:
    try:
        df = pd.read_csv(file)
        for col in ["draw", "Draw", "number", "Number", "value", "Value"]:
            if col in df.columns:
                vals = [int(x) for x in df[col].dropna().tolist()]
                if vals:
                    return vals
        return []
    except Exception:
        return []

def api_health(url: str):
    try:
        r = requests.get(f"{url.rstrip('/')}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def api_forecast(url: str, game: str, draws: List[int], settings: Dict[str, Any]):
    r = requests.post(
        f"{url.rstrip('/')}/forecast",
        json={"game": game, "draws": draws, "settings": settings},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()

# --- App Config ---
st.set_page_config(page_title="HIS — Hybrid Intelligence Systems", layout="wide")

# --- Neon Header + Bio Banner ---
st.markdown(f"""
<div class="neon-hero">
  <div class="pill">Hybrid · Local/Remote</div>
  <div class="pill">Engine: LIPE-Core · Tier 33 · Active</div>
  <h1 class="hero-title">🧠 HYBRID INTELLIGENCE SYSTEMS</h1>

  <h3 class="signature">
    Powered by <span>JESSE RAY LANDINGHAM JR</span>
  </h3>

  <p class="hero-bio">
    Jesse Ray Landingham Jr — visionary architect of <strong>LIPE</strong>, the Living Intelligence Predictive Engine.<br>
    A polymath creator driven by precision, intuition, and truth.<br>
    Uniting data, emotion, and design to build systems that think, learn, and evolve beyond limits.
  </p>

  <p class="hero-tag">{APP_TAGLINE}</p>
</div>
""", unsafe_allow_html=True)

# --- Styling ---
st.markdown("""
<style>
:root {
  --accent: #8be9fd;
  --accent2: #ff79c6;
  --bg: #0e0f12;
  --card: #151722;
  --muted: #8a8fa3;
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #12141b, #0f0f14);
}
.neon-hero {
  padding: 32px 28px;
  border-radius: 18px;
  background: radial-gradient(1200px 300px at 20% -20%, rgba(139,233,253,.20), transparent),
              radial-gradient(1200px 300px at 80% -20%, rgba(255,121,198,.18), transparent),
              linear-gradient(180deg, #141620, #0f1118);
  border: 1px solid rgba(255,255,255,.06);
  box-shadow: 0 0 24px rgba(139,233,253,.06), inset 0 0 24px rgba(255,121,198,.04);
  text-align: center;
}
.hero-title {
  font-size: 36px;
  font-weight: 800;
  margin: 0 0 10px 0;
  color: #ffffff;
}
.signature {
  font-size: 18px;
  color: var(--accent);
  font-weight: 600;
  margin-top: -4px;
}
.signature span {
  color: var(--accent2);
  font-weight: 700;
}
.hero-bio {
  font-size: 15px;
  color: #c2c6d6;
  margin-top: 8px;
  line-height: 1.6;
}
.hero-tag {
  color: var(--muted);
  margin-top: 16px;
  font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("System")
compute = st.sidebar.radio("Compute", ["Local (in-app)", "Remote API"])
st.sidebar.text_input("API URL (for Remote)", API_URL)
truth_strict = st.sidebar.slider("Truth Filter", 0, 100, 60)
module = st.sidebar.selectbox("Module", [m[1] for m in MODULES])

# --- Example View ---
st.markdown(f"### {module} View")
st.write("Choose settings and data, then click Run Forecast.")
