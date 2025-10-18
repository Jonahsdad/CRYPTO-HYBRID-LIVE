# -----------------------------------------------------------------------------
# Crypto Hybrid Live — Phase 15 (FULL Hybrid Resurrection)
# Powered by Jesse Ray Landingham Jr
# -----------------------------------------------------------------------------
# Notes:
# - This is a full file (no compact mode).
# - ASCII-only header to avoid iPad “smart punctuation” issues.
# - Includes: Crypto (CoinGecko), Robust Stocks (yfinance), TRUTH/RAW/Δ with
#   color/emoji bars, Dashboard, Fusion view, Scores explainer, Signal Center shell,
#   CSV export, watchlist, auto-refresh, and light state persistence.
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Optional: yfinance for robust stocks (app still runs if missing)
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

PHASE_TAG = "PHASE 15 — FULL"

# ============================== PAGE CONFIG / THEME ===========================

st.set_page_config(
    page_title="Crypto Hybrid Live — Phase 15 (FULL)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS (no non-ASCII punctuation)
CSS = """
<style>
/* layout */
.block-container { padding-top: 0.6rem; padding-bottom: 2.0rem; }

/* top banner */
.phase-badge {
  padding: 10px; border-radius: 10px; background: #0f172a; border: 1px solid #334155;
  color: #7dfca3; font-weight: 800; text-align: center; margin-bottom: 8px;
}

/* section title */
.section-title { font-size: 24px; font-weight: 800; margin: 6px 0 6px 0; }

/* metric cards */
.metric-box {
  border: 1px solid #ffffff22; border-radius: 12px; padding: 0.8rem; background: #0e1117;
}

/* mini badges */
.badge { display:inline-block; padding: 6px 10px; border-radius: 999px; font-weight:700; margin-right:6px; }
.badge-raw   { background:#241c14; color:#ff9b63; border:1px solid #ff9b6333; }
.badge-truth { background:#172017; color:#7dff96; border:1px solid #7dff9633; }
.badge-div   { background:#161a22; color:#8ecbff; border:1px solid #8ecbff33; }
.badge-hot   { background:#231616; color:#ff7a7a; border:1px solid #ff7a7a33; }

/* tables */
.dataframe td, .dataframe th { font-size: 0.95em; }
.small-note { opacity: 0.75; font-size: 0.9rem; }

/* larger buttons */
.stButton>button { font-weight: 700; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(f"<div class='phase-badge'>✅ {PHASE_TAG}</div>", unsafe_allow_html=True)

# ============================== SIDEBAR ======================================

with st.sidebar:
    st.header("Navigation")
    nav = st.radio(
        "Go to",
        ["Dashboard", "Crypto", "Stocks", "Fusion", "Scores", "Signal Center", "Export"],
        index=0,
        key="nav_radio",
    )

    st.header("Appearance")
    font_size = st.slider("Font size", 14, 24, 18, key="font_size")
    st.markdown(
        f"<style>html, body, [class*='css'] {{ font-size: {font_size}px; }}</style>",
        unsafe_allow_html=True,
    )
    high_contrast = st.toggle("High contrast mode", value=False, key="hc")
    if high_contrast:
        st.markdown("<style>.metric-box{background:#0b0d12;border-color:#8ecbff44}</style>", unsafe_allow_html=True)

    st.header("Watchlist")
    wl = st.text_input(
        "Symbols (comma-separated)",
        value=st.session_state.get("wl", "BTC,ETH,SOL,AAPL,MSFT,NVDA,TSLA"),
        help="Crypto: symbols or names (BTC, ETH). Stocks: tickers (AAPL, MSFT).",
        key="watchlist",
    )
    st.session_state["wl"] = wl

    st.header("Refresh")
    auto = st.toggle("Auto refresh", value=False, help="Re-run periodically", key="auto")
    every = st.slider("Every (sec)", 10, 120, 30, key="every")

# ============================== UTILS / CORE LOGIC ===========================

def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def bar_emoji_01(v: float, on: str = "█", off: str = "░", slots: int = 10) -> str:
    try:
        v = float(v)
        v = 0.0 if np.isnan(v) else max(0.0, min(1.0, v))
    except Exception:
        v = 0.0
    filled = int(round(v * slots))
    return on * filled + off * (slots - filled)

def raw_fire(v: float) -> str:
    # fire intensity by score band
    if v >= 0.85: return "🔥🔥🔥"
    if v >= 0.65: return "🔥🔥"
    if v >= 0.45: return "🔥"
    return "·"

def truth_drop(v: float) -> str:
    if v >= 0.85: return "💧💧💧"
    if v >= 0.65: return "💧💧"
    if v >= 0.45: return "💧"
    return "·"

def pct_sigmoid(pct) -> float:
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return 0.5
    try:
        x = float(pct) / 10.0
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5

@st.cache_data(ttl=60, show_spinner="Loading CoinGecko…")
def fetch_cg_markets(vs: str = "usd", per_page: int = 250) -> pd.DataFrame:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    p = {
        "vs_currency": vs, "order": "market_cap_desc",
        "per_page": int(max(1, min(per_page, 250))), "page": 1,
        "sparkline": "false", "price_change_percentage": "1h,24h,7d", "locale": "en"
    }
    r = requests.get(url, params=p, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    for k in [
        "name","symbol","current_price","market_cap","total_volume",
        "price_change_percentage_1h_in_currency",
        "price_change_percentage_24h_in_currency",
        "price_change_percentage_7d_in_currency",
    ]:
        if k not in df.columns:
            df[k] = np.nan
    return df

def score_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df.copy()
    t = df.copy()
    t["vol_to_mc"] = ((t.get("total_volume", 0) / t.get("market_cap", np.nan))
                      .replace([np.inf, -np.inf], np.nan)).clip(0, 2).fillna(0)
    t["m1h"] = t.get("price_change_percentage_1h_in_currency", pd.Series(np.nan, index=t.index)).apply(pct_sigmoid)
    t["m24"] = t.get("price_change_percentage_24h_in_currency", pd.Series(np.nan, index=t.index)).apply(pct_sigmoid)
    t["m7d"] = t.get("price_change_percentage_7d_in_currency", pd.Series(np.nan, index=t.index)).apply(pct_sigmoid)
    mc = t.get("market_cap", pd.Series(0, index=t.index)).fillna(0)
    t["liq01"] = 0 if mc.max() == 0 else (mc - mc.min()) / (mc.max() - mc.min() + 1e-9)

    t["raw_heat"] = (0.5 * (t["vol_to_mc"] / 2).clip(0, 1) + 0.5 * t["m1h"].fillna(0.5)).clip(0, 1)
    t["truth_full"] = (
        0.30 * (t["vol_to_mc"] / 2).clip(0, 1) +
        0.25 * t["m24"].fillna(0.5) +
        0.25 * t["m7d"].fillna(0.5) +
        0.20 * t["liq01"].fillna(0.0)
    ).clip(0, 1)
    t["divergence"] = (t["raw_heat"] - t["truth_full"]).abs()

    # presentation columns
    t["RAW_BAR"]   = t["raw_heat"].apply(lambda v: raw_fire(v) + " " + bar_emoji_01(v))
    t["TRUTH_BAR"] = t["truth_full"].apply(lambda v: truth_drop(v) + " " + bar_emoji_01(v))
    t["DELTA_BAR"] = t["divergence"].apply(lambda v: bar_emoji_01(v, on="■", off="·"))

    return t

# ============================== STOCKS (ROBUST SNAPSHOT) =====================

@st.cache_data(ttl=120, show_spinner="Loading stocks…")
def yf_snapshot(tickers: List[str]) -> pd.DataFrame:
    if not HAS_YF or not tickers:
        return pd.DataFrame()
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(
            tickers=" ".join(tickers),
            period="5d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
    except Exception:
        return pd.DataFrame()

    rows = []
    for t in tickers:
        try:
            s = data[t]
            last = float(s.iloc[-1]["Close"])
            prev = float(s.iloc[-2]["Close"]) if len(s) >= 2 else np.nan
            pct24 = (last / prev - 1.0) * 100.0 if pd.notna(prev) else np.nan
            rows.append({
                "name": t, "symbol": t, "current_price": last,
                "price_change_percentage_24h_in_currency": pct24,
                "market_cap": np.nan, "total_volume": np.nan,
                "price_change_percentage_1h_in_currency": np.nan,
                "price_change_percentage_7d_in_currency": np.nan
            })
        except Exception:
            rows.append({
                "name": t, "symbol": t,
                "current_price": np.nan,
                "price_change_percentage_24h_in_currency": np.nan,
                "market_cap": np.nan, "total_volume": np.nan,
                "price_change_percentage_1h_in_currency": np.nan,
                "price_change_percentage_7d_in_currency": np.nan
            })
    return pd.DataFrame(rows)

# ============================== UI HELPERS ===================================

def header_block(title: str, caption: str = "") -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if caption: st.caption(caption)
    st.write("")

def kpi_row(df_scored: pd.DataFrame, label: str) -> None:
    coins = len(df_scored)
    avg24 = float(df_scored.get("price_change_percentage_24h_in_currency", pd.Series(dtype=float)).mean())
    avg_truth = float(df_scored.get("truth_full", pd.Series(dtype=float)).mean())
    avg_raw = float(df_scored.get("raw_heat", pd.Series(dtype=float)).mean())
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown("<div class='metric-box'><b>Assets</b><br>{}</div>".format(coins), unsafe_allow_html=True)
    with c2: st.markdown("<div class='metric-box'><b>Avg 24h %</b><br>{:.2f}%</div>".format(0 if np.isnan(avg24) else avg24), unsafe_allow_html=True)
    with c3: st.markdown("<div class='metric-box'><b>Avg Truth</b><br>{:.2f}</div>".format(0 if np.isnan(avg_truth) else avg_truth), unsafe_allow_html=True)
    with c4: st.markdown("<div class='metric-box'><b>Avg Raw</b><br>{:.2f}</div>".format(0 if np.isnan(avg_raw) else avg_raw), unsafe_allow_html=True)
    st.caption(f"{PHASE_TAG} • Last update: {now_utc_str()} • Mode: {label}")

def truth_raw_panels(df_scored: pd.DataFrame, topn: int = 25) -> None:
    st.markdown(
        "<span class='badge badge-raw'>RAW</span>"
        "<span class='badge badge-truth'>TRUTH</span>"
        "<span class='badge badge-div'>DELTA</span>"
        "<span class='badge badge-hot'>MOVERS</span>",
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("RAW — heat")
        cols = ["name","symbol","current_price","market_cap","total_volume","raw_heat","RAW_BAR"]
        have = [c for c in cols if c in df_scored.columns]
        st.dataframe(df_scored.sort_values("raw_heat", ascending=False).head(topn)[have], use_container_width=True)

    with c2:
        st.subheader("TRUTH — stability")
        cols = ["name","symbol","current_price","market_cap","truth_full","TRUTH_BAR"]
        have = [c for c in cols if c in df_scored.columns]
        st.dataframe(df_scored.sort_values("truth_full", ascending=False).head(topn)[have], use_container_width=True)

    with c3:
        st.subheader("MOVERS — 24h")
        if "price_change_percentage_24h_in_currency" in df_scored.columns:
            g = df_scored.sort_values("price_change_percentage_24h_in_currency", ascending=False).head(10)
            l = df_scored.sort_values("price_change_percentage_24h_in_currency", ascending=True).head(10)
            st.markdown("Top Gainers"); st.dataframe(g[["name","symbol","current_price","price_change_percentage_24h_in_currency","DELTA_BAR"]], use_container_width=True)
            st.markdown("Top Losers");  st.dataframe(l[["name","symbol","current_price","price_change_percentage_24h_in_currency","DELTA_BAR"]], use_container_width=True)
        else:
            st.info("No 24h % column available.")

# ============================== PAGES =========================================

def page_dashboard() -> None:
    header_block("Crypto Hybrid Live — Dashboard", "Glance across markets using Truth vs Raw.")
    df_crypto = score_table(fetch_cg_markets("usd", 200))
    kpi_row(df_crypto, "Crypto")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top TRUTH (Crypto)")
        st.dataframe(df_crypto.sort_values("truth_full", ascending=False).head(15)[["name","symbol","current_price","truth_full","TRUTH_BAR"]], use_container_width=True)
    with c2:
        st.subheader("Top RAW (Crypto)")
        st.dataframe(df_crypto.sort_values("raw_heat", ascending=False).head(15)[["name","symbol","current_price","raw_heat","RAW_BAR"]], use_container_width=True)

def page_crypto() -> None:
    header_block("Crypto", "Live CoinGecko with Truth vs Raw vs Delta.")
    topn = st.slider("Show top N (Crypto)", 50, 250, 150, key="crypto_topn")
    df = score_table(fetch_cg_markets("usd", topn))
    if df.empty:
        st.warning("No data received from CoinGecko.")
        return
    kpi_row(df, "Crypto")
    truth_raw_panels(df, topn=25)

def page_stocks() -> None:
    header_block("Stocks", "Robust yfinance snapshot, scored by the same lens.")
    default = "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA"
    raw = st.text_input("Tickers (comma-separated)", value=st.session_state.get("stock_input", default), key="stock_input")
    if not HAS_YF:
        st.error("yfinance is not installed on this deployment. Add `yfinance` to requirements.txt and reboot the app.")
        return
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    if not tickers:
        st.info("Enter at least one ticker.")
        return
    df0 = yf_snapshot(tickers)
    if df0.empty:
        st.warning("No stock data returned. Check tickers.")
        return
    df = score_table(df0)
    kpi_row(df, "Stocks")
    truth_raw_panels(df, topn=min(25, len(df)))

def page_fusion() -> None:
    header_block("Fusion", "Compare Crypto vs Stocks in one place.")
    # Crypto side
    dfc = score_table(fetch_cg_markets("usd", 120))
    dfc["universe"] = "CRYPTO"
    # Stocks side (use watchlist intersection)
    if HAS_YF:
        wl = st.session_state.get("wl", "")
        tick = [x.strip().upper() for x in wl.split(",") if x.strip() and x.isalpha()]
        dfs = score_table(yf_snapshot(tick)) if tick else pd.DataFrame()
        if dfs.empty:
            st.info("Stocks snapshot empty. Add tickers in the sidebar Watchlist.")
            dfs = pd.DataFrame(columns=dfc.columns)
        dfs["universe"] = "STOCKS"
    else:
        dfs = pd.DataFrame(columns=dfc.columns)
    # Union
    try:
        all_df = pd.concat([dfc, dfs], ignore_index=True).fillna(np.nan)
    except Exception:
        all_df = dfc.copy()
        all_df["universe"] = "CRYPTO"
    kpi_row(all_df, "Fusion")
    # Two panes filtered by universe
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Crypto — Top TRUTH")
        st.dataframe(all_df[all_df["universe"]=="CRYPTO"].sort_values("truth_full", ascending=False).head(20)[
            ["name","symbol","current_price","truth_full","TRUTH_BAR","RAW_BAR","DELTA_BAR"]
        ], use_container_width=True)
    with c2:
        st.subheader("Stocks — Top TRUTH")
        st.dataframe(all_df[all_df["universe"]=="STOCKS"].sort_values("truth_full", ascending=False).head(20)[
            ["name","symbol","current_price","truth_full","TRUTH_BAR","RAW_BAR","DELTA_BAR"]
        ], use_container_width=True)

def page_scores() -> None:
    header_block("Scores — Explainer")
    st.markdown("""
**RAW**: fast heat built from volume/marketcap and 1h momentum (scaled 0..1).  
**TRUTH**: slower, liquidity-aware blend of volume/MC, 24h momentum, 7d momentum, and market-cap liquidity (0..1).  
**DELTA**: absolute gap |RAW − TRUTH|; larger values may indicate overextension or mean reversion spots.

**Quick intuition (kid-level):**  
- RAW is like how much the crowd is shouting right now (loudness + short-term speed).  
- TRUTH is like the steady heartbeat of the coin/stock (stability + bigger context).  
- DELTA shows when the shouting does not match the heartbeat.
""")
    st.info("Weights are fixed in Phase 15. Next phases add user-configurable presets and saved scanners.")

def page_signal_center() -> None:
    header_block("Signal Center (preview)")
    st.markdown("""
This is where rules/alerts will live (Phase 16):
- Rules like: RAW > 0.7 and TRUTH rising and 24h% between +2 and +10
- Alerts to Email / Discord / Webhook
- Saved scans and backtest snapshots
""")
    st.warning("Coming next: rule builder and alert hooks.")

def page_export() -> None:
    header_block("Export")
    st.caption("Download current tables as CSV for your notebook or records.")
    dfc = score_table(fetch_cg_markets("usd", 200))
    st.download_button("Download Crypto CSV", data=dfc.to_csv(index=False).encode("utf-8"), file_name="crypto_truth_raw.csv", mime="text/csv")
    if HAS_YF:
        wl = st.session_state.get("wl","AAPL,MSFT,NVDA,TSLA")
        tick = [x.strip().upper() for x in wl.split(",") if x.strip()]
        dfs = score_table(yf_snapshot(tick)) if tick else pd.DataFrame()
        if not dfs.empty:
            st.download_button("Download Stocks CSV", data=dfs.to_csv(index=False).encode("utf-8"), file_name="stocks_truth_raw.csv", mime="text/csv")
        else:
            st.info("No stocks in snapshot. Add tickers in Watchlist on the sidebar.")
    else:
        st.info("yfinance not installed; install to enable Stocks export.")

# ============================== ROUTER ========================================

if nav == "Dashboard":
    page_dashboard()
elif nav == "Crypto":
    page_crypto()
elif nav == "Stocks":
    page_stocks()
elif nav == "Fusion":
    page_fusion()
elif nav == "Scores":
    page_scores()
elif nav == "Signal Center":
    page_signal_center()
else:
    page_export()

# ============================== AUTO REFRESH ==================================

if auto:
    st.caption(f"{PHASE_TAG} • Auto refresh every {int(every)}s")
    time.sleep(max(5, int(every)))
    st.rerun()
