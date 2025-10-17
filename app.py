# ============================== PHASE 10.2 — HYBRID PANEL + STOCKS FIX (STABLE) ==============================
# Crypto Hybrid Live — Complete Stable Version
# -------------------------------------------------------------------------------------------------------------
# Includes:
# ✅ Fixed CSS triple-quotes
# ✅ Streamlit-safe indentation
# ✅ Stocks + Crypto + FX
# ✅ Side panel + Truth presets
# ✅ Full layout consistency
# -------------------------------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
import time
from datetime import datetime

# Optional dependencies
try:
    import plotly.express as px
except:
    px = None
try:
    import yfinance as yf
    YF_OK = True
except:
    YF_OK = False

# ---------------------------- SETTINGS ----------------------------
st.set_page_config(page_title="Crypto Hybrid Live — Phase 10.2", layout="wide")

# ---------------------------- THEME + CSS --------------------------
def _apply_css():
    dark = True
    base_bg = "#0d1117" if dark else "#ffffff"
    base_fg = "#e6edf3" if dark else "#111111"
    accent  = "#22c55e"
    ring    = "#16a34a"
    font_px = 18

    st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
        background:{base_bg};
        color:{base_fg};
        font-size:{font_px}px !important;
    }}
    div[data-baseweb="tab-list"] button {{
        font-size:{round(font_px*1.3)}px !important;
        font-weight:800 !important;
        border-radius:12px !important;
        padding:1rem 2rem !important;
        margin-right:1rem !important;
        background:linear-gradient(135deg, {accent}, #15803d);
        color:white !important;
        border:3px solid {ring} !important;
        transition:all .25s ease-in-out;
        transform:scale(1.0);
    }}
    div[data-baseweb="tab-list"] button:hover {{
        transform:scale(1.08);
        box-shadow:0 0 18px rgba(34,197,94,0.55);
    }}
    div[data-baseweb="tab-list"] button[aria-selected="true"] {{
        background:linear-gradient(135deg, #4ade80, {accent});
        color:#111 !important;
        transform:scale(1.16);
        box-shadow:0 0 28px rgba(74,222,128,0.9);
    }}
    [data-testid="stMetricValue"] {{
        font-size:{round(font_px*1.35)}px !important;
        font-weight:800 !important;
    }}
    .phase-banner {{
        font-size:{round(font_px*1.2)}px;
        font-weight:900;
        text-align:center;
        background:linear-gradient(90deg, {accent}, #15803d);
        color:white;
        border-radius:14px;
        padding:.5rem 0;
        margin-bottom:1rem;
    }}
    .explain {{
        border-left:5px solid {ring};
        background:rgba(34,197,94,0.08);
        padding:.75rem 1rem;
        border-radius:8px;
    }}
    </style>
    """, unsafe_allow_html=True)

_apply_css()

# ---------------------------- SIDEBAR ------------------------------
st.sidebar.header("🧭 Market Mode")
market_mode = st.sidebar.radio("Select Market", ["Crypto", "Stocks", "FX"], horizontal=True)

vs_currency = st.sidebar.selectbox("Currency (Crypto)", ["usd", "eur"], index=0)
topn = st.sidebar.slider("Top N Coins", 20, 250, 150, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Appearance")
font_size = st.sidebar.slider("Font size", 14, 24, 18)
contrast = st.sidebar.toggle("High contrast mode", value=False)

st.sidebar.markdown("---")
preset = st.sidebar.radio("Truth Preset", ["Balanced", "Momentum", "Liquidity", "Value"], index=0, horizontal=True)
st.sidebar.markdown("---")

st.sidebar.subheader("🕓 Live Mode")
auto_refresh = st.sidebar.toggle("Auto Refresh", value=False)
refresh_rate = st.sidebar.slider("Refresh every (sec)", 10, 120, 30, 5)
if auto_refresh:
    time.sleep(refresh_rate)

# ---------------------------- HELPERS ------------------------------
USER_AGENT = {"User-Agent": "Mozilla/5.0"}
COINGECKO_API = "https://api.coingecko.com/api/v3/coins/markets"

def safe_get(url, params=None):
    try:
        r = requests.get(url, params=params, headers=USER_AGENT, timeout=25)
        if r.status_code == 200:
            return r
    except Exception:
        pass
    return None

@st.cache_data(ttl=60)
def cg_markets(vs="usd", limit=150):
    params = {
        "vs_currency": vs,
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": "1h,24h,7d"
    }
    r = safe_get(COINGECKO_API, params)
    if r is None:
        return pd.DataFrame()
    return pd.DataFrame(r.json())

def _sig(x):
    if pd.isna(x): return 0.5
    return 1 / (1 + math.exp(-x / 10))

def _norm(w):
    s = sum(max(0, v) for v in w.values()) or 1.0
    return {k: max(0, v) / s for k, v in w.items()}

# ---------------------------- TRUTH + WEIGHTS -----------------------
DEFAULT_WEIGHTS = dict(w_vol=0.30, w_m24=0.25, w_m7=0.25, w_liq=0.20)
PRESETS = {
    "Balanced":  dict(w_vol=0.30, w_m24=0.25, w_m7=0.25, w_liq=0.20),
    "Momentum":  dict(w_vol=0.15, w_m24=0.45, w_m7=0.30, w_liq=0.10),
    "Liquidity": dict(w_vol=0.45, w_m24=0.20, w_m7=0.15, w_liq=0.20),
    "Value":     dict(w_vol=0.25, w_m24=0.20, w_m7=0.20, w_liq=0.35),
}

def lipe_truth(df, w):
    w = _norm(w)
    if "liquidity01" not in df: df["liquidity01"] = 0.0
    if "vol_to_mc" not in df:
        vol = df.get("total_volume", pd.Series(0, index=df.index))
        v01 = (vol - vol.min()) / (vol.max() - vol.min() + 1e-9)
        df["vol_to_mc"] = 2 * v01
    return (
        w["w_vol"] * (df["vol_to_mc"] / 2).clip(0, 1) +
        w["w_m24"] * df.get("momo_24h01", 0.5) +
        w["w_m7"]  * df.get("momo_7d01", 0.5) +
        w["w_liq"] * df["liquidity01"]
    ).clip(0, 1)

# ---------------------------- BUILD DATA ----------------------------
def build_crypto(vs="usd", limit=150):
    df = cg_markets(vs, limit)
    if df.empty:
        return df
    df["vol_to_mc"] = (df["total_volume"]/df["market_cap"]).replace([np.inf, -np.inf], np.nan).clip(0,2).fillna(0)
    df["momo_24h01"] = df["price_change_percentage_24h_in_currency"].apply(_sig)
    df["momo_7d01"] = df["price_change_percentage_7d_in_currency"].apply(_sig)
    mc = df["market_cap"].fillna(0)
    df["liquidity01"] = 0 if mc.max()==0 else (mc-mc.min())/(mc.max()-mc.min()+1e-9)
    df["truth_full"] = lipe_truth(df, DEFAULT_WEIGHTS)
    df["raw_heat"] = (0.5*(df["vol_to_mc"]/2).clip(0,1)+0.5*df["momo_24h01"]).clip(0,1)
    df["divergence"] = (df["raw_heat"]-df["truth_full"]).round(3)
    df["symbol"] = df["symbol"].str.upper()
    return df

def build_stocks(tickers):
    if not YF_OK:
        st.error("yfinance not installed.")
        return pd.DataFrame()
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    data = yf.download(tickers, period="6mo", interval="1d", auto_adjust=True, progress=False, threads=True)
    if data.empty:
        return pd.DataFrame()
    if "Adj Close" in data:
        prices = data["Adj Close"].ffill()
    else:
        prices = data.ffill()
    last = prices.iloc[-1]
    prev = prices.iloc[-2] if len(prices) > 1 else prices.iloc[-1]
    chg24 = (last/prev - 1.0)*100
    df = pd.DataFrame({
        "symbol": last.index,
        "current_price": last.values,
        "price_change_percentage_24h_in_currency": chg24.values,
        "market_cap": np.nan,
        "total_volume": np.nan
    })
    df["momo_24h01"] = df["price_change_percentage_24h_in_currency"].apply(_sig)
    df["momo_7d01"] = 0.5
    df["vol_to_mc"] = 0.5
    df["liquidity01"] = 0.5
    df["truth_full"] = lipe_truth(df, DEFAULT_WEIGHTS)
    df["raw_heat"] = (0.5*(df["vol_to_mc"]/2).clip(0,1)+0.5*df["momo_24h01"]).clip(0,1)
    df["divergence"] = (df["raw_heat"]-df["truth_full"]).round(3)
    return df

# ---------------------------- MAIN ---------------------------------
st.markdown('<div class="phase-banner">🟢 Crypto Hybrid Live — Phase 10.2 (Hybrid Panel + Stocks Fix)</div>', unsafe_allow_html=True)

if market_mode == "Crypto":
    df = build_crypto(vs_currency, topn)
elif market_mode == "Stocks":
    tickers = st.sidebar.text_input("Stock Tickers (comma-separated)", "AAPL,MSFT,NVDA,AMZN,GOOGL")
    df = build_stocks(tickers)
else:
    df = build_stocks("EURUSD=X,USDJPY=X")

if df.empty:
    st.error("No data loaded.")
    st.stop()

st.write("### 🔥 Market Snapshot")
st.dataframe(df[["symbol", "current_price", "price_change_percentage_24h_in_currency", "truth_full", "raw_heat", "divergence"]])

if px:
    st.write("### 📊 Truth vs Raw Heat")
    fig = px.scatter(df, x="truth_full", y="raw_heat", text="symbol", color="divergence")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)
