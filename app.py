# ============================ PHASE 10.5 — HYBRID PANEL (YF FALLBACK) =============================
# Crypto Hybrid Live — Stable build with automatic yfinance fallback (no crashes)
# ================================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
import time
from datetime import datetime, timezone

# ---------- Try optional libs (do NOT crash if missing) ----------
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

# ---------- App config ----------
st.set_page_config(page_title="Crypto Hybrid Live — Phase 10.5", layout="wide")
USER_AGENT = {"User-Agent": "Mozilla/5.0 (CHL-Phase10.5)"}
CG_API = "https://api.coingecko.com/api/v3/coins/markets"

# ---------- CSS (triple-quoted, safe) ----------
def _apply_css(font_px=18, high_contrast=False, dark=True):
    base_bg = "#0d1117" if dark else "#ffffff"
    base_fg = "#e6edf3" if dark else "#111111"
    accent  = "#22c55e" if not high_contrast else "#ffdd00"
    ring    = "#16a34a" if not high_contrast else "#ffaa00"

    st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
        background:{base_bg};
        color:{base_fg};
        font-size:{font_px}px !important;
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
    div[data-baseweb="tab-list"] button {{
        font-size:{round(font_px*1.25)}px !important;
        font-weight:800 !important;
        border-radius:12px !important;
        padding:.8rem 1.4rem !important;
        margin-right:1rem !important;
        background:linear-gradient(135deg, {accent}, #15803d);
        color:white !important;
        border:3px solid {ring} !important;
        transition:all .2s ease-in-out;
        transform:scale(1.0);
    }}
    div[data-baseweb="tab-list"] button[aria-selected="true"] {{
        background:linear-gradient(135deg, #4ade80, {accent});
        color:#111 !important;
        transform:scale(1.12);
        box-shadow:0 0 24px rgba(74,222,128,0.65);
    }}
    [data-testid="stMetricValue"] {{
        font-size:{round(font_px*1.35)}px !important;
        font-weight:800 !important;
    }}
    .status-chip {{
        display:inline-block; padding:.25rem .6rem; border-radius:999px; font-weight:700;
        margin-right:.5rem; color:#111;
    }}
    .ok {{ background:#86efac; }}  /* green */
    .warn {{ background:#facc15; }} /* yellow */
    .err {{ background:#fda4af; }}  /* red */
    </style>
    """, unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.header("🧭 Market Mode")
market_mode = st.sidebar.radio("Select Market", ["Crypto", "Stocks", "FX"], horizontal=True)

vs_currency = st.sidebar.selectbox("Currency (Crypto)", ["usd", "eur"], index=0)
topn = st.sidebar.slider("Top N Coins", 20, 250, 150, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Appearance")
font_size = st.sidebar.slider("Font size", 14, 24, 18)
contrast = st.sidebar.toggle("High contrast mode", value=False)
theme_dark = True  # keep dark by default for now
_apply_css(font_size, contrast, theme_dark)

st.sidebar.markdown("---")
st.sidebar.subheader("🧭 Truth Preset")
preset = st.sidebar.radio("Preset", ["Balanced", "Momentum", "Liquidity", "Value"], index=0, horizontal=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🕓 Live Mode")
auto_refresh = st.sidebar.toggle("Auto Refresh", value=False)
refresh_rate = st.sidebar.slider("Refresh every (sec)", 10, 120, 30, 5)
if auto_refresh: time.sleep(refresh_rate)

# ---------- Helpers ----------
def safe_get(url, params=None, t=25):
    try:
        r = requests.get(url, params=params, headers=USER_AGENT, timeout=t)
        if r.status_code == 200:
            return r
    except Exception:
        pass
    return None

@st.cache_data(ttl=60)
def cg_markets(vs="usd", limit=150):
    p = {"vs_currency":vs,"order":"market_cap_desc","per_page":limit,"page":1,
         "sparkline":"false","price_change_percentage":"1h,24h,7d"}
    r = safe_get(CG_API, p)
    return pd.DataFrame(r.json()) if r else pd.DataFrame()

def _sig(x):
    if pd.isna(x): return 0.5
    return 1/(1+math.exp(-float(x)/10.0))

def _norm(w):
    s = sum(max(0,v) for v in w.values()) or 1.0
    return {k:max(0,v)/s for k,v in w.items()}

DEFAULT_WEIGHTS = dict(w_vol=0.30, w_m24=0.25, w_m7=0.25, w_liq=0.20)
PRESETS = {
    "Balanced":  dict(w_vol=0.30, w_m24=0.25, w_m7=0.25, w_liq=0.20),
    "Momentum":  dict(w_vol=0.15, w_m24=0.45, w_m7=0.30, w_liq=0.10),
    "Liquidity": dict(w_vol=0.45, w_m24=0.20, w_m7=0.15, w_liq=0.20),
    "Value":     dict(w_vol=0.25, w_m24=0.20, w_m7=0.20, w_liq=0.35),
}

def lipe_truth(df, w):
    w=_norm(w)
    if "liquidity01" not in df: df["liquidity01"]=0.0
    if "vol_to_mc" not in df:
        vol=df.get("total_volume", pd.Series(0, index=df.index))
        v01=(vol-vol.min())/(vol.max()-vol.min()+1e-9)
        df["vol_to_mc"]=2*v01
    return (
        w["w_vol"]*(df["vol_to_mc"]/2).clip(0,1) +
        w["w_m24"]*df.get("momo_24h01",0.5) +
        w["w_m7"] *df.get("momo_7d01",0.5) +
        w["w_liq"]*df["liquidity01"]
    ).clip(0,1)

# ---------- Builders ----------
def build_crypto(vs="usd", limit=150, weights=DEFAULT_WEIGHTS):
    df=cg_markets(vs, limit)
    if df.empty: return df
    df["vol_to_mc"]=(df["total_volume"]/df["market_cap"]).replace([np.inf,-np.inf],np.nan).clip(0,2).fillna(0)
    df["momo_24h01"]=df["price_change_percentage_24h_in_currency"].apply(_sig)
    df["momo_7d01"]=df["price_change_percentage_7d_in_currency"].apply(_sig)
    mc=df["market_cap"].fillna(0)
    df["liquidity01"]=0 if mc.max()==0 else (mc-mc.min())/(mc.max()-mc.min()+1e-9)
    df["truth_full"]=lipe_truth(df, weights)
    df["raw_heat"]=(0.5*(df["vol_to_mc"]/2).clip(0,1)+0.5*df["momo_24h01"]).clip(0,1)
    df["divergence"]=(df["raw_heat"]-df["truth_full"]).round(3)
    df["symbol"]=df["symbol"].str.upper()
    return df

def _stocks_demo():
    # Realistic demo snapshot, clearly labeled
    rows = [
        ("AAPL", 227.4,  0.85),
        ("MSFT", 424.1,  0.65),
        ("NVDA", 114.7, -1.20),
        ("AMZN", 196.0,  0.10),
        ("GOOGL",174.8, -0.40),
    ]
    df=pd.DataFrame(rows, columns=["symbol","current_price","price_change_percentage_24h_in_currency"])
    df["total_volume"]=1_000_000
    df["market_cap"]=np.nan
    df["momo_24h01"]=df["price_change_percentage_24h_in_currency"].apply(_sig)
    df["momo_7d01"]=0.5
    df["vol_to_mc"]=0.5
    df["liquidity01"]=0.5
    df["truth_full"]=lipe_truth(df, DEFAULT_WEIGHTS)
    df["raw_heat"]=(0.5*(df["vol_to_mc"]/2).clip(0,1)+0.5*df["momo_24h01"]).clip(0,1)
    df["divergence"]=(df["raw_heat"]-df["truth_full"]).round(3)
    return df

def build_stocks_or_fx(tickers_csv, period="6mo"):
    status = "live"
    if not YF_OK:
        return _stocks_demo(), "demo"
    try:
        tickers=[t.strip().upper() for t in tickers_csv.split(",") if t.strip()]
        data = yf.download(tickers, period=period, interval="1d",
                           auto_adjust=True, progress=False, threads=True)
        if data is None or data.empty:
            return _stocks_demo(), "demo"
        prices = data.get("Adj Close", data).ffill()
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(tickers[0])
        last = prices.iloc[-1]
        prev = prices.iloc[-2] if len(prices)>1 else prices.iloc[-1]
        chg24 = (last/prev - 1.0)*100.0
        rows=[]
        for t in prices.columns:
            rows.append({"symbol":t.upper(),
                         "current_price":float(last.get(t,np.nan)),
                         "price_change_percentage_24h_in_currency":float(chg24.get(t,np.nan))})
        df=pd.DataFrame(rows)
        # Simple features (no cap/volume from yahoo fast_info to keep it robust)
        df["total_volume"]=np.nan
        df["market_cap"]=np.nan
        df["momo_24h01"]=df["price_change_percentage_24h_in_currency"].apply(_sig)
        df["momo_7d01"]=0.5
        df["vol_to_mc"]=0.5
        df["liquidity01"]=0.5
        df["truth_full"]=lipe_truth(df, DEFAULT_WEIGHTS)
        df["raw_heat"]=(0.5*(df["vol_to_mc"]/2).clip(0,1)+0.5*df["momo_24h01"]).clip(0,1)
        df["divergence"]=(df["raw_heat"]-df["truth_full"]).round(3)
        return df, status
    except Exception:
        return _stocks_demo(), "demo"

# ---------- Header ----------
st.markdown('<div class="phase-banner">🟢 Crypto Hybrid Live — Phase 10.5 (YF Fallback)</div>', unsafe_allow_html=True)

# ---------- Preset to weights ----------
weights = PRESETS.get(preset, DEFAULT_WEIGHTS)

# ---------- Market switch ----------
if market_mode == "Crypto":
    df = build_crypto(vs_currency, topn, weights)
    source_status = ("ok", "CoinGecko live")
elif market_mode == "Stocks":
    tickers = st.sidebar.text_input("Stock tickers (comma-separated)", "AAPL,MSFT,NVDA,AMZN,GOOGL")
    df, mode = build_stocks_or_fx(tickers)
    source_status = ("ok","Yahoo live") if mode=="live" else ("warn","Demo fallback (yfinance missing/unavailable)")
else:  # FX via Yahoo-style symbols
    pairs = st.sidebar.text_input("FX pairs (Yahoo symbols)", "EURUSD=X,USDJPY=X,GBPUSD=X")
    df, mode = build_stocks_or_fx(pairs)
    source_status = ("ok","Yahoo live") if mode=="live" else ("warn","Demo fallback (yfinance missing/unavailable)")

# ---------- Guard ----------
if df.empty:
    st.error("No data loaded.")
    st.stop()

# ---------- Status strip ----------
chip_class = {"ok":"ok","warn":"warn","err":"err"}.get(source_status[0],"warn")
st.markdown(f'<span class="status-chip {chip_class}">Source: {source_status[1]}</span>', unsafe_allow_html=True)

# ---------- KPIs ----------
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
col1,col2,col3,col4 = st.columns(4)
col1.metric("Assets", len(df))
col2.metric("Avg 24h %", f"{pd.to_numeric(df.get('price_change_percentage_24h_in_currency', pd.Series(0))).fillna(0).mean():.2f}%")
col3.metric("Avg Truth", f"{df['truth_full'].mean():.2f}")
col4.metric("Avg Raw", f"{df['raw_heat'].mean():.2f}")
st.caption(f"Last update: {now}")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["🧭 TRUTH","🔥 RAW","🧠 DIVERGENCE","📈 SCATTER"])

with tab1:
    st.subheader("🧭 LIPE Truth (ranked)")
    cols=[c for c in ["symbol","name","current_price","truth_full","liquidity01","price_change_percentage_24h_in_currency"] if c in df.columns]
    st.dataframe(df.sort_values("truth_full", ascending=False)[cols], use_container_width=True, height=560)

with tab2:
    st.subheader("🔥 Raw (momentum/volume)")
    cols=[c for c in ["symbol","current_price","price_change_percentage_24h_in_currency","vol_to_mc","raw_heat"] if c in df.columns]
    st.dataframe(df.sort_values("raw_heat", ascending=False)[cols], use_container_width=True, height=560)

with tab3:
    st.subheader("🧠 Divergence (Raw - Truth)")
    st.dataframe(df.sort_values("divergence", ascending=False)[["symbol","current_price","divergence"]], use_container_width=True, height=560)

with tab4:
    st.subheader("📈 Truth vs Raw Scatter")
    if PLOTLY_OK:
        fig = px.scatter(df, x="truth_full", y="raw_heat", text="symbol", color="divergence", color_continuous_scale="Turbo")
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Install plotly for scatter chart (optional).")

# ---------- Export ----------
st.markdown("### 📤 Export")
choice = st.selectbox("Choose table", ["Truth","Raw","Divergence"])
if choice=="Truth":
    out=df.sort_values("truth_full", ascending=False)
elif choice=="Raw":
    out=df.sort_values("raw_heat", ascending=False)
else:
    out=df.sort_values("divergence", ascending=False)
csv=out.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", csv, file_name=f"chl_{choice.lower()}.csv", mime="text/csv")

# ---------- Footer ----------
st.caption("Sources: CoinGecko (Crypto). Yahoo Finance or demo fallback (Stocks/FX). CHL Phase 10.5.")
