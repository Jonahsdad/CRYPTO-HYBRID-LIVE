# ============================== PHASE 10.1 HYBRID PANEL + STOCKS FIX ==============================
# Crypto Hybrid Live — Phase 10.1 (Sidebar + Big Tabs + Explain + Compare + Robust Stocks/FX)
# Paste this entire file into app.py (no spaces above this line)
# ================================================================================================

import math
import time
import json
import re
import io
import traceback
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timezone

# Optional plotting
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# Auto-install helpers (Streamlit Cloud sometimes misses deps)
def _ensure(pkgs):
    import importlib, subprocess, sys
    for p in pkgs:
        try:
            importlib.import_module(p)
        except Exception:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
            except Exception:
                pass

_ensure(["yfinance", "feedparser"])

# Finance + feeds
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

try:
    import feedparser
    FP_OK = True
except Exception:
    FP_OK = False

# Text polarity
try:
    from textblob import TextBlob
    def _polarity(t):
        try:
            return float(TextBlob(t).sentiment.polarity)
        except:
            return 0.0
except Exception:
    def _polarity(t):
        return 0.0

# ---- App config ----------------------------------------------------------
APP_NAME = "Crypto Hybrid Live — Phase 10.1 (Hybrid Panel + Stocks Fix)"
st.set_page_config(page_title=APP_NAME, layout="wide")
USER_AGENT = {"User-Agent": "Mozilla/5.0 (CHL-Phase10.1)"}
CG_PER_PAGE = 150

# ---- Cache Reset Button --------------------------------------------------
st.sidebar.markdown("### ⚙️ Maintenance")
if st.sidebar.button("🧹 Clear Cache & Restart"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.experimental_rerun()

# ---- Theme and CSS -------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
if "font_px" not in st.session_state:
    st.session_state["font_px"] = 18
if "contrast" not in st.session_state:
    st.session_state["contrast"] = False
if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = []

def _apply_css():
    dark = st.session_state["theme"] == "dark"
    base_bg = "#0d1117" if dark else "#ffffff"
    base_fg = "#e6edf3" if dark else "#111111"
    accent = "#22c55e" if not st.session_state["contrast"] else "#ffdd00"
    ring = "#16a34a" if not st.session_state["contrast"] else "#ffaa00"
    font_px = st.session_state["font_px"]

    st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
        background:{base_bg}; color:{base_fg}; font-size:{font_px}px !important;
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
        padding:.75rem 1rem; border-radius:8px;
    }}
    </style>
    """, unsafe_allow_html=True)

_apply_css()

# ---- Weights -------------------------------------------------------------
DEFAULT_WEIGHTS = dict(w_vol=0.30, w_m24=0.25, w_m7=0.25, w_liq=0.20)
PRESETS = {
    "Balanced":  dict(w_vol=0.30, w_m24=0.25, w_m7=0.25, w_liq=0.20),
    "Momentum":  dict(w_vol=0.15, w_m24=0.45, w_m7=0.30, w_liq=0.10),
    "Liquidity": dict(w_vol=0.45, w_m24=0.20, w_m7=0.15, w_liq=0.20),
    "Value":     dict(w_vol=0.25, w_m24=0.20, w_m7=0.20, w_liq=0.35),
}
FUSION_V2 = dict(w_truth=0.70, w_sent=0.15, w_xmkt=0.15)

def _norm(w):
    s = sum(max(0, v) for v in w.values()) or 1.0
    return {k: max(0, v) / s for k, v in w.items()}

def _sig(p):
    if pd.isna(p): return 0.5
    return 1 / (1 + math.exp(-float(p) / 10.0))

def lipe_truth(df, w):
    w = _norm(w or DEFAULT_WEIGHTS)
    if "liquidity01" not in df:
        df["liquidity01"] = 0.0
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

def mood_label(x):
    if x >= 0.8: return "🟢 EUPHORIC"
    if x >= 0.6: return "🟡 OPTIMISTIC"
    if x >= 0.4: return "🟠 NEUTRAL"
    return "🔴 FEARFUL"

# ---- Data Helpers -------------------------------------------------------
def safe_get(url, params=None, t=25):
    try:
        r = requests.get(url, params=params, headers=USER_AGENT, timeout=t)
        if r.status_code == 200: return r
    except Exception:
        pass
    return None

@st.cache_data(ttl=60)
def cg_markets(vs="usd", limit=150):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    p = {"vs_currency": vs, "order": "market_cap_desc",
         "per_page": limit, "page": 1,
         "sparkline": "false",
         "price_change_percentage": "1h,24h,7d"}
    r = safe_get(url, p)
    return pd.DataFrame(r.json()) if r else pd.DataFrame()

@st.cache_data(ttl=300)
def rss_sentiment():
    if not FP_OK: return 0.5, []
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://news.bitcoin.com/feed/"
    ]
    titles = []
    for f in feeds:
        try:
            d = feedparser.parse(f)
            for e in d.entries[:20]:
                titles.append(e.title)
        except Exception:
            continue
    if not titles: return 0.5, []
    pol = np.mean([_polarity(t) for t in titles])
    return float((pol + 1) / 2), titles[:30]

# ---- Robust yfinance loader --------------------------------------------
@st.cache_data(ttl=180, show_spinner=False)
def yf_multi_robust(ticks, period="6mo"):
    if not YF_OK:
        return pd.DataFrame(), {}, "yfinance import failed"
    try:
        if isinstance(ticks, str):
            ticks = [t.strip() for t in ticks.split(",") if t.strip()]
        ticks = [t.upper() for t in ticks]
        data = yf.download(ticks, period=period, interval="1d",
                           auto_adjust=True, progress=False, threads=True)
        if data is None or len(data) == 0:
            return pd.DataFrame(), {}, "empty result"
        adj = data.get("Adj Close", data)
        if isinstance(adj, pd.Series):
            adj = adj.to_frame(ticks[0])
        meta = {}
        for t in ticks:
            try:
                fi = yf.Ticker(t).fast_info
                meta[t] = {
                    "market_cap": getattr(fi, "market_cap", np.nan),
                    "last_price": getattr(fi, "last_price", np.nan),
                    "volume": getattr(fi, "last_volume", np.nan)
                }
            except Exception:
                meta[t] = {"market_cap": np.nan, "last_price": np.nan, "volume": np.nan}
        return adj, meta, ""
    except Exception as e:
        return pd.DataFrame(), {}, str(e)

def stocks_health_check():
    if not YF_OK: return False, "yfinance import failed"
    try:
        ping = yf.download("AAPL", period="5d", interval="1d",
                           auto_adjust=True, progress=False)
        if ping is None or ping.empty:
            return False, "Yahoo returned empty"
        return True, "OK"
    except Exception as e:
        return False, str(e)

# ---- Header -------------------------------------------------------------
st.markdown(f'<div class="phase-banner">🟢 {APP_NAME}</div>', unsafe_allow_html=True)

# ---- Sidebar Panel ------------------------------------------------------
st.sidebar.header("🧭 Market")
market = st.sidebar.radio("Mode", ["Crypto", "Stocks", "FX"], horizontal=True)
vs_currency = st.sidebar.selectbox("Currency (Crypto)", ["usd"], index=0)
topn = st.sidebar.slider("Top N (Crypto)", 20, 250, CG_PER_PAGE, 10)

st.sidebar.subheader("🎨 Appearance")
theme_pick = st.sidebar.radio("Theme", ["Dark", "Light"],
                              index=0 if st.session_state["theme"] == "dark" else 1,
                              horizontal=True)
st.session_state["theme"] = "dark" if theme_pick == "Dark" else "light"
st.session_state["contrast"] = st.sidebar.toggle("High-contrast mode", value=False)
st.session_state["font_px"] = st.sidebar.slider("Font size", 14, 24,
                                                st.session_state["font_px"], 1)
_apply_css()

st.sidebar.subheader("🧭 Truth Preset")
preset = st.sidebar.radio("Preset", list(PRESETS.keys()), index=0, horizontal=True)
w_edit = dict(PRESETS[preset])
for k in list(w_edit.keys()):
    w_edit[k] = st.sidebar.slider(k, 0.0, 1.0, float(w_edit[k]), 0.01)
w_edit = _norm(w_edit)

stocks_in = ""
fx_in = ""
if market == "Stocks":
    stocks_in = st.sidebar.text_area("Tickers",
                                     value="AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA")
if market == "FX":
    fx_in = st.sidebar.text_area("FX pairs",
                                 value="EURUSD=X,USDJPY=X,GBPUSD=X,AUDUSD=X,USDCAD=X")

st.sidebar.markdown("---")
st.sidebar.subheader("🟢 Live Mode")
live = st.sidebar.toggle("Auto-refresh", value=False)
every = st.sidebar.slider("Refresh every (sec)", 10, 120, 30, 5)
if live: time.sleep(every)

st.sidebar.markdown("---")
st.sidebar.subheader("⭐ Watchlist")
add_w = st.sidebar.text_input("Add symbol (BTC, ETH, etc)")
if st.sidebar.button("Add to watchlist") and add_w.strip():
    sym = add_w.strip().upper()
    if sym not in st.session_state["watchlist"]:
        st.session_state["watchlist"].append(sym)
if st.session_state["watchlist"]:
    st.sidebar.write(", ".join(st.session_state["watchlist"]))

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Source Status")
st.sidebar.write("✅ CoinGecko")
st.sidebar.write("✅" if YF_OK else "⚠️", "Yahoo Finance")
st.sidebar.write("✅" if FP_OK else "⚠️", "RSS")
if market in ("Stocks", "FX"):
    ok, msg = stocks_health_check()
    st.sidebar.write(("⚠️", "✅")[ok], f"Stocks health: {msg}")

# ---- Data Builders ------------------------------------------------------
def build_crypto():
    df = cg_markets(vs_currency, topn)
    if df.empty: return df
    df["vol_to_mc"] = (df["total_volume"] / df["market_cap"]).replace(
        [np.inf, -np.inf], np.nan).clip(0, 2).fillna(0)
    df["momo_24h01"] = df["price_change_percentage_24h_in_currency"].apply(_sig)
    df["momo_7d01"] = df["price_change_percentage_7d_in_currency"].apply(_sig)
    mc = df["market_cap"].fillna(0)
    df["liquidity01"] = 0 if mc.max() == 0 else (mc - mc.min()) / (mc.max() - mc.min() + 1e-9)
    df["truth_full"] = lipe_truth(df, w_edit)
    df["raw_heat"] = (0.5 * (df["vol_to_mc"] / 2).clip(0, 1) +
                      0.5 * df["momo_24h01"]).clip(0, 1)
    df["divergence"] = (df["raw_heat"] - df["truth_full"]).round(3)
    df["symbol"] = df["symbol"].str.upper()
    return df

def build_from_yf(tickers):
    prices, meta, err = yf_multi_robust(tickers)
    if err: st.warning(f"Yahoo issue: {err}")
    if prices.empty: return pd.DataFrame()
    last = prices.ffill().iloc[-1]
    prev = prices.ffill().iloc[-2] if len(prices) >= 2 else prices.ffill().iloc[-1]
    chg24 = (last / prev - 1.0) * 100.0
    rows = []
    for t in prices.columns:
        rows.append({
            "symbol": t.upper(),
            "current_price": float(last.get(t, np.nan)),
            "price_change_percentage_24h_in_currency": float(chg24.get(t, np.nan)),
            "market_cap": float(meta.get(t, {}).get("market_cap", np.nan)),
            "total_volume": float(meta.get(t, {}).get("volume", np.nan))
        })
    df = pd.DataFrame(rows)
    df["momo_24h01"] = df["price_change_percentage_24h_in_currency"].apply(_sig)
    df["momo_7d01"] = 0.5
    if df["market_cap"].notna().sum() > 0:
        mc = df["market_cap"].fillna(0)
        df["liquidity01"] = 0 if mc.max() == 0 else (mc - mc.min()) / (mc.max() - mc.min() + 1e-9)
        df["vol_to_mc"] = (df["total_volume"] / df["market_cap"]).replace(
            [np.inf, -np.inf], np.nan).clip(0, 2).fillna(0)
    else:
        v = df["total_volume"].fillna(0)
        df["liquidity01"] = 0 if v.max() == 0 else (v - v.min()) / (v.max() - v.min() + 1e-9)
        df["vol_to_mc"] = 2 * ((v - v.min()) / (v.max() - v.min() + 1e-9))
    df["truth_full"] = lipe_truth(df, w_edit)
    df["raw_heat"] = (0.5 * (df["vol_to_mc"] / 2).clip(0, 1) +
                      0.5 * df["momo_24h01"]).clip(0, 1)
    df["divergence"] = (df["raw_heat"] - df["truth_full"]).round(3)
    return df

if market == "Crypto":
    df = build_crypto()
elif market == "Stocks":
    df = build
