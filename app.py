# HYBRID INTELLIGENCE SYSTEMS — Neon UX (Realistic Arena Tiles + Pro Charts + Hybrid Astrology)
# One-file drop-in (no external service files required).

import sys, os, time, math, random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# Optional markets
try:
    import yfinance as yf
except Exception:
    yf = None

# Optional local engine
try:
    from lipe_core import LIPE
except Exception as e:
    LIPE = None
    _IMPORT_ERR = f"Local LIPE unavailable: {e}"
else:
    _IMPORT_ERR = None

# ---------------------- App Config ----------------------
APP_TITLE = "HYBRID INTELLIGENCE SYSTEMS"
APP_TAGLINE = "One brain. Many frontiers."
API_URL_DEFAULT = "https://YOUR-FORECAST-API.example.com"
API_URL = os.getenv("LIPE_API_URL", API_URL_DEFAULT)

# Path helper for assets
BASE_DIR = os.path.dirname(__file__)
def asset(path: str) -> str:
    return os.path.join(BASE_DIR, path)

# Realistic icons (with emoji fallback if file not found)
ARENA_ICONS = {
    "Lottery":      ("assets/lottery.png",   "🎰"),
    "Crypto":       ("assets/crypto.png",    "💰"),
    "Stocks":       ("assets/stocks.png",    "📈"),
    "Options":      ("assets/options.png",   "🧾"),
    "Real Estate":  ("assets/realestate.png","🏡"),
    "Commodities":  ("assets/commodities.png","🛢️"),
    "Sports":       ("assets/sports.png",    "🏈"),
    "Human Behavior":("assets/behavior.png", "🧠"),
    "Astrology":    ("assets/astrology.png", "🌌"),
}

MODULES = [
    ("Lottery",        "Daily numbers, picks, entropy, risk modes"),
    ("Crypto",         "Live pricing, signals, overlays"),
    ("Stocks",         "Charts, momentum, factor overlays"),
    ("Options",        "Chains, quick IV views"),
    ("Real Estate",    "Market tilt and projections"),
    ("Commodities",    "Energy, metals, ag"),
    ("Sports",         "Game signals and parlay entropy"),
    ("Human Behavior", "Cohort trends and intent"),
    ("Astrology",      "Planetary cycles and symbolic overlays"),
]

st.set_page_config(page_title="HIS — Hybrid Intelligence Systems", layout="wide")

# ---------------------- CSS ----------------------
st.markdown("""
<style>
:root { --accent:#8be9fd; --accent2:#ff79c6; --gold:#f7d774; --bg:#0e0f12; --card:#151722; --muted:#8a8fa3; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#12141b,#0f0f14); }

.neon-hero{ padding:22px 24px;border-radius:16px;
  background: radial-gradient(900px 250px at 15% -20%, rgba(139,233,253,.16), transparent),
              radial-gradient(900px 250px at 85% -20%, rgba(255,121,198,.14), transparent),
              linear-gradient(180deg,#141620,#0f1118);
  border:1px solid rgba(255,255,255,.06);
  box-shadow:0 0 14px rgba(139,233,253,.06), inset 0 0 18px rgba(255,121,198,.04);
  text-align:center; }
.pill{ display:inline-block;padding:4px 10px;border-radius:100px;font-size:12px;
  background:rgba(139,233,253,.12);border:1px solid rgba(139,233,253,.25); margin:0 6px 6px 0;}
.hero-title{ font-size:32px;font-weight:800;margin:4px 0 4px 0;color:#fff; }
.signature{ font-size:16px;color:var(--muted);margin-top:4px; }
.signature span{ color:var(--gold);font-weight:800; }
.hero-tag{ color:var(--muted);margin-top:8px;font-size:14px; }

.tile{ background:var(--card);border:1px solid rgba(255,255,255,.06);
  border-radius:16px;padding:14px;cursor:pointer;
  transition:transform .12s, box-shadow .12s, border-color .12s; }
.tile:hover{ transform:translateY(-2px);
  box-shadow:0 8px 20px rgba(139,233,253,.10), 0 0 0 1px rgba(139,233,253,.12) inset;
  border-color:rgba(139,233,253,.25); }
.tile-title{ font-weight:800;margin:8px 0 4px 0;font-size:16px;}
.tile-desc{ color:var(--muted);font-size:13px;min-height:38px;}
.icon-wrap{ display:flex; align-items:center; gap:12px; }
.icon-img{ width:34px; height:34px; border-radius:10px; object-fit:cover; border:1px solid rgba(255,255,255,.1); }
.icon-fallback{ font-size:26px; }

hr{ border:none;height:1px;background:rgba(255,255,255,.06); }
</style>
""", unsafe_allow_html=True)

# ---------------------- Chart Theme (Pro) ----------------------
def set_chart_theme():
    plt.rcParams.update({
        "figure.figsize": (8.5, 4.2),
        "axes.facecolor": "#0f1116",
        "figure.facecolor": "#0f1116",
        "axes.edgecolor": "#2a2f3a",
        "axes.labelcolor": "#d7e1f1",
        "xtick.color": "#c6cfdd",
        "ytick.color": "#c6cfdd",
        "grid.color": "#2a2f3a",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "axes.grid": True,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.framealpha": 0.1,
        "legend.edgecolor": "#2a2f3a",
        "savefig.facecolor": "#0f1116"
    })
set_chart_theme()

PALETTE = ["#8be9fd","#f7d774","#ff79c6","#50fa7b","#bd93f9","#f1fa8c"]

def line_plot(df: pd.DataFrame, x, y, title: str, color_idx: int = 0):
    fig, ax = plt.subplots()
    ax.plot(df[x], df[y], PALETTE[color_idx % len(PALETTE)])
    ax.set_title(title); ax.set_xlabel(str(x)); ax.set_ylabel(str(y))
    st.pyplot(fig)

def multi_line_plot(series: Dict[str, pd.Series], title: str):
    fig, ax = plt.subplots()
    for i, (label, s) in enumerate(series.items()):
        ax.plot(s.index, s.values, label=label, color=PALETTE[i % len(PALETTE)])
    ax.legend(loc="upper left"); ax.set_title(title); ax.set_xlabel("Date")
    st.pyplot(fig)

# ---------------------- Helpers ----------------------
def dl_csv(rows: List[Dict[str, Any]]) -> bytes:
    if not rows: return b""
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

def parse_draws_text(s: str) -> List[int]:
    try: return [int(x.strip()) for x in s.split(",") if x.strip()]
    except Exception: return []

def parse_draws_csv(file) -> List[int]:
    try:
        df = pd.read_csv(file)
        for col in ["draw","Draw","number","Number","value","Value"]:
            if col in df.columns:
                vals = [int(x) for x in df[col].dropna().tolist()]
                if vals: return vals
        return []
    except Exception: return []

def api_health(url: str):
    try:
        r = requests.get(f"{url.rstrip('/')}/health", timeout=5); r.raise_for_status(); return r.json()
    except Exception: return None

def api_forecast(url: str, game: str, draws: List[int], settings: Dict[str, Any]):
    r = requests.post(f"{url.rstrip('/')}/forecast",
                      json={"game":game,"draws":draws,"settings":settings},
                      timeout=20)
    r.raise_for_status(); return r.json()

# ---------------------- Astro Sync (hidden meta-layer) ----------------------
def astro_get_influence() -> Dict[str, float]:
    day = datetime.utcnow().timetuple().tm_yday
    mars = abs(math.sin(day / 58.6))
    venus = abs(math.sin(day / 224.7))
    mercury = abs(math.sin(day / 87.97))
    jupiter = abs(math.sin(day / 433.0))
    saturn = abs(math.cos(day / 10759.0))
    cosmic = (mars + venus + mercury + jupiter + saturn) / 5.0
    return {
        "mars": round(mars,3),
        "venus": round(venus,3),
        "mercury": round(mercury,3),
        "jupiter": round(jupiter,3),
        "saturn": round(saturn,3),
        "cosmic_index": round(cosmic,3)
    }

def astro_adjust(value: float, strength: float = 0.5):
    data = astro_get_influence()
    modulation = 1.0 + ((data["cosmic_index"] - 0.5) * strength)
    return max(0.0, value * modulation), data

# ---------------------- Astrology Arena helpers (visible layer) ----------------------
def astro_positions(planet: str = "Mars", days_back: int = 30):
    base = datetime.utcnow()
    out = []
    for i in range(days_back):
        d = base - timedelta(days=i)
        degree = (math.sin(i/5.0) * 180.0 / math.pi) % 360.0
        retro = random.choice([True, False])
        out.append({"date": d.date(), "degree": round(degree,2), "retrograde": retro})
    return out

def astro_interpret(deg: float) -> str:
    if 0 <= deg < 90: return "New beginnings"
    if 90 <= deg < 180: return "Growth and tension"
    if 180 <= deg < 270: return "Reflection and correction"
    return "Completion and harvest"

# ---------------------- Sidebar ----------------------
st.sidebar.header("System")
engine_mode = st.sidebar.radio("Compute", ["Local (in-app)", "Remote API"], index=0)
api_url = st.sidebar.text_input("API URL (Remote)", value=API_URL)

if "truth_filter" not in st.session_state: st.session_state.truth_filter = 55
truth_filter = st.sidebar.slider("Truth Filter", 0, 100, st.session_state.truth_filter
