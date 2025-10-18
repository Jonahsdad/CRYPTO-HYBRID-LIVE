# -----------------------------------------------------------------------------
# Crypto Hybrid Live - Phase 14.3 (FULL)
# Powered by Jesse Ray Landingham Jr
# -----------------------------------------------------------------------------

import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Optional: stocks via yfinance (we handle if it's missing)
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

# ------------------------------- UI SETUP ------------------------------------

st.set_page_config(page_title="Crypto Hybrid Live — Phase 14.3", layout="wide")

CSS = """
<style>
/* big section headers */
.section-title { font-size: 1.5rem; font-weight: 700; margin: 0.4rem 0 0.2rem 0; }
/* pill tabs */
.badge { display:inline-block; padding:0.35rem 0.7rem; border-radius:999px; font-weight:700; }
.badge-raw { background:#221; color:#ff915e; border:1px solid #ff915e33; }
.badge-truth { background:#1c261c; color:#7dff96; border:1px solid #7dff9633; }
.badge-div { background:#1d2230; color:#8ecbff; border:1px solid #8ecbff33; }
.badge-hot { background:#2a1c1c; color:#ff6a6a; border:1px solid #ff6a6a33; }
.metric-box { border:1px solid #ffffff22; border-radius:12px; padding:0.8rem; background:#0e1117; }
.table-note { opacity:0.7; font-size:0.85rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("Navigation")
    nav = st.radio("Go to", ["Dashboard", "Crypto", "Stocks", "Scores", "Signal Center", "Export"], index=0)

    st.header("Appearance")
    fs = st.slider("Font size", 14, 24, 18)
    st.markdown(f"<style>html, body, [class*='css'] {{ font-size: {fs}px; }}</style>", unsafe_allow_html=True)

    st.header("Watchlist")
    wl_input = st.text_input("Add symbol (BTC, ETH, AAPL...)", value="")
    st.caption("Separate with commas. For stocks use tickers like AAPL, MSFT; for crypto use symbols or names.")

    st.header("Refresh")
    live = st.toggle("Auto Refresh", value=False, help="Re-run periodically")
    every = st.slider("Every (sec)", 10, 120, 30)

# ------------------------------- DATA LAYERS ---------------------------------

@st.cache_data(ttl=60, show_spinner="Loading crypto markets…")
def cg_markets(vs="usd", n=250):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    p = {
        "vs_currency": vs, "order": "market_cap_desc",
        "per_page": min(n, 250), "page": 1,
        "sparkline": "false", "price_change_percentage": "1h,24h,7d", "locale": "en"
    }
    r = requests.get(url, params=p, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df
    need = [
        "name","symbol","current_price","market_cap","total_volume",
        "price_change_percentage_1h_in_currency",
        "price_change_percentage_24h_in_currency",
        "price_change_percentage_7d_in_currency"
    ]
    for k in need:
        if k not in df.columns:
            df[k] = np.nan
    return df

def pct_sigmoid(pct):
    if pd.isna(pct): return 0.5
    x = float(pct)/10.0
    return 1.0/(1.0+math.exp(-x))

def score_crypto(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: 
        return df.copy()
    out = df.copy()
    out["vol_to_mc"] = (out["total_volume"]/out["market_cap"]).replace([np.inf,-np.inf],np.nan).clip(0,2).fillna(0)
    out["m1h"] = out["price_change_percentage_1h_in_currency"].apply(pct_sigmoid)
    out["m24"] = out["price_change_percentage_24h_in_currency"].apply(pct_sigmoid)
    out["m7d"] = out["price_change_percentage_7d_in_currency"].apply(pct_sigmoid)
    mc = out["market_cap"].fillna(0)
    out["liq01"] = 0 if mc.max()==0 else (mc-mc.min())/(mc.max()-mc.min()+1e-9)

    # Raw = quick heat (volume + 1h momentum)
    out["raw_heat"] = (0.5*(out["vol_to_mc"]/2).clip(0,1) + 0.5*out["m1h"].fillna(0.5)).clip(0,1)

    # Truth = slower, liquidity aware
    out["truth_full"] = (
        0.30*(out["vol_to_mc"]/2).clip(0,1) +
        0.25*out["m24"].fillna(0.5) +
        0.25*out["m7d"].fillna(0.5) +
        0.20*out["liq01"].fillna(0.0)
    ).clip(0,1)

    out["divergence"] = (out["raw_heat"] - out["truth_full"]).abs()
    return out

# Stocks via yfinance
@st.cache_data(ttl=120, show_spinner="Loading stocks…")
def yf_table(tickers: list[str]) -> pd.DataFrame:
    if not HAS_YF or not tickers:
        return pd.DataFrame()
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers=" ".join(tickers), period="5d", interval="1d", group_by="ticker", auto_adjust=True, threads=True, progress=False)
    rows = []
    for t in tickers:
        try:
            s = data[t]
            last = s.iloc[-1]["Close"]
            prev = s.iloc[-2]["Close"] if len(s)>=2 else np.nan
            pct24 = (last/prev-1)*100 if pd.notna(prev) else np.nan
            rows.append({"symbol": t, "current_price": float(last), "price_change_percentage_24h_in_currency": pct24})
        except Exception:
            rows.append({"symbol": t, "current_price": np.nan, "price_change_percentage_24h_in_currency": np.nan})
    df = pd.DataFrame(rows)
    df["market_cap"] = np.nan
    df["total_volume"] = np.nan
    df["price_change_percentage_1h_in_currency"] = np.nan
    df["price_change_percentage_7d_in_currency"] = np.nan
    df["name"] = df["symbol"]
    return df

# ------------------------------- VIEWS ---------------------------------------

def header_bar(title: str):
    st.markdown(f"<div class='section-title'>🟢 {title}</div>", unsafe_allow_html=True)
    st.caption("Truth → Noise • Live scores. Education only — not financial advice.")
    st.write(" ")

def metrics_bar(df_scored: pd.DataFrame, label: str):
    coins = len(df_scored)
    avg24 = df_scored.get("price_change_percentage_24h_in_currency", pd.Series(dtype=float)).mean()
    avg_truth = df_scored.get("truth_full", pd.Series(dtype=float)).mean()
    avg_raw = df_scored.get("raw_heat", pd.Series(dtype=float)).mean()
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f"<div class='metric-box'><b>Assets</b><br>{coins}</div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-box'><b>Avg 24h %</b><br>{avg24:0.2f}%</div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-box'><b>Avg Truth</b><br>{(avg_truth if not np.isnan(avg_truth) else 0):0.2f}</div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric-box'><b>Avg Raw</b><br>{(avg_raw if not np.isnan(avg_raw) else 0):0.2f}</div>", unsafe_allow_html=True)
    st.caption(f"Last update: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} • Mode: {label}")

def truth_raw_tabs(df_scored: pd.DataFrame, topn: int = 50):
    st.markdown(
        "<span class='badge badge-raw'>🔥 Raw</span> "
        "<span class='badge badge-truth'>🧭 Truth</span> "
        "<span class='badge badge-div'>|Δ| Divergence</span> "
        "<span class='badge badge-hot'>🚀 Movers</span>",
        unsafe_allow_html=True
    )
    st.write(" ")

    c1,c2,c3 = st.columns(3)

    with c1:
        st.subheader("Raw Wide Scan")
        cols = ["name","symbol","current_price","market_cap","total_volume","raw_heat"]
        st.dataframe(df_scored.sort_values("raw_heat", ascending=False).head(topn)[[c for c in cols if c in df_scored.columns]], use_container_width=True)

    with c2:
        st.subheader("Truth Filter")
        cols = ["name","symbol","current_price","market_cap","truth_full"]
        st.dataframe(df_scored.sort_values("truth_full", ascending=False).head(topn)[[c for c in cols if c in df_scored.columns]], use_container_width=True)

    with c3:
        st.subheader("Top Daily Gainers / Losers")
        if "price_change_percentage_24h_in_currency" in df_scored.columns:
            g = df_scored.sort_values("price_change_percentage_24h_in_currency", ascending=False).head(10)
            l = df_scored.sort_values("price_change_percentage_24h_in_currency", ascending=True).head(10)
            st.markdown("**Top Gainers**")
            st.dataframe(g[["name","symbol","current_price","price_change_percentage_24h_in_currency"]], use_container_width=True)
            st.markdown("**Top Losers**")
            st.dataframe(l[["name","symbol","current_price","price_change_percentage_24h_in_currency"]], use_container_width=True)
        else:
            st.info("No 24h % column available.")

def page_crypto():
    header_bar("Crypto Hybrid Live — Crypto")
    topn = st.sidebar.slider("Show top N (Crypto)", 50, 250, 150)
    df = cg_markets("usd", topn)
    df_scored = score_crypto(df)
    metrics_bar(df_scored, "Crypto")
    truth_raw_tabs(df_scored, topn=20)

def page_stocks():
    header_bar("Crypto Hybrid Live — Stocks")
    default = "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA"
    raw = st.sidebar.text_input("Stocks (comma-separated)", value=default)
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    if not HAS_YF:
        st.error("yfinance is not installed. Add `yfinance` to requirements.txt, then reboot the app.")
        return
    df = yf_table(tickers)
    if df.empty:
        st.warning("No data loaded. Check tickers.")
        return
    # Reuse crypto scorer (it tolerates missing cols)
    df_scored = score_crypto(df)
    metrics_bar(df_scored, "Stocks")
    truth_raw_tabs(df_scored, topn=min(20, len(df_scored)))

def page_dashboard():
    header_bar("Crypto Hybrid Live — Dashboard")
    # Quick glance: top crypto truth and raw
    df = score_crypto(cg_markets("usd", 200))
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Top Truth (Crypto)")
        st.dataframe(df.sort_values("truth_full", ascending=False).head(15)[["name","symbol","current_price","truth_full"]], use_container_width=True)
    with c2:
        st.subheader("Top Raw (Crypto)")
        st.dataframe(df.sort_values("raw_heat", ascending=False).head(15)[["name","symbol","current_price","raw_heat"]], use_container_width=True)

def page_scores():
    header_bar("Scores — Explainer")
    st.markdown("""
**Raw** ≈ fast heat from **Volume/MarketCap** and **1h momentum**.  
**Truth** ≈ slower, more stable score from **volume/MC**, **24h momentum**, **7d momentum**, and **market-cap liquidity**.  
**Divergence** = |Raw − Truth| (possible over-extension or mean reversion).
""")

def page_signal_center():
    header_bar("Signal Center (basic demo)")
    st.info("Build rule-based alerts here in the next phase (webhooks, email, Discord).")

def page_export():
    header_bar("Export")
    st.info("CSV export will be added here in the next phase.")

# ------------------------------- ROUTING -------------------------------------

if nav == "Dashboard":
    page_dashboard()
elif nav == "Crypto":
    page_crypto()
elif nav == "Stocks":
    page_stocks()
elif nav == "Scores":
    page_scores()
elif nav == "Signal Center":
    page_signal_center()
else:
    page_export()

# Optional auto-refresh loop (client-side re-run)
if live:
    st.caption(f"Auto refresh every {every}s is ON.")
    time.sleep(max(5, int(every)))
    st.rerun()
