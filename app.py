# -----------------------------------------------------------------------------
# Crypto Hybrid Live - Phase 17.1 (ASCII Safe)
# POWERED BY JESSE RAY LANDINGHAM JR
# -----------------------------------------------------------------------------
# What's in this build:
# - Robust Stocks via yfinance with batching + per-ticker fallback + retries
# - Crypto via CoinGecko
# - Truth / Raw / Confluence scoring with adjustable weights
# - Options chains (calls/puts)
# - Fusion page (Crypto + Stocks under same lens)
# - CSV upload for large watchlists
# - Export CSV
# - Clear-caches button
# - iPad-friendly layout and spacing
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Optional: stocks + options via yfinance (app still runs if missing)
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

PHASE_TAG  = "PHASE 17.1 - Robust Stocks + Confluence"
POWERED_BY = "POWERED BY JESSE RAY LANDINGHAM JR"

# ------------------------------ PAGE CONFIG / THEME ---------------------------

st.set_page_config(
    page_title="Crypto Hybrid Live - Phase 17.1",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.block-container { padding-top: 0.6rem; padding-bottom: 1.2rem; }

/* Banners */
.phase-badge {
  padding: 12px; border-radius: 12px;
  background: linear-gradient(90deg, #0f172a 0%, #0b2a1d 50%, #0f172a 100%);
  border: 1px solid #2f3a4a; color: #7dfca3; font-weight: 900;
  text-align: center; margin-bottom: 6px; font-size: 18px;
}
.powered-badge {
  padding: 10px; border-radius: 10px; background: #10161f; border: 1px solid #29434e;
  color: #9ad7ff; font-weight: 900; text-align: center; margin: 4px 0 12px 0; font-size: 16px;
}

/* Section titles & metric cards */
.section-title { font-size: 24px; font-weight: 800; margin: 6px 0 8px 0; }
.metric-box { border: 1px solid #ffffff22; border-radius: 12px; padding: 10px; background: #0e1117; }

/* Badge legend */
.badge { display:inline-block; padding: 6px 10px; border-radius: 999px; font-weight:700; margin-right:6px; }
.badge-raw   { background:#241c14; color:#ff9b63; border:1px solid #ff9b6333; }
.badge-truth { background:#172017; color:#7dff96; border:1px solid #7dff9633; }
.badge-conf  { background:#1a1a24; color:#ffd86b; border:1px solid #ffd86b33; }
.badge-div   { background:#161a22; color:#8ecbff; border:1px solid #8ecbff33; }

/* Tables */
.stDataFrame, .stDataEditor { border-radius: 10px; overflow: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(f"<div class='phase-badge'>✅ {PHASE_TAG}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='powered-badge'>{POWERED_BY}</div>", unsafe_allow_html=True)

# ------------------------------ SIDEBAR --------------------------------------

with st.sidebar:
    st.header("Navigation")
    nav = st.radio(
        "Go to",
        ["Dashboard", "Crypto", "Confluence", "Stocks", "Options", "Fusion", "Scores", "Export", "Settings"],
        index=0,
        key="nav_radio",
    )

    st.header("Appearance")
    font_size = st.slider("Font size", 14, 24, 18, key="font_size")
    st.markdown(
        f"<style>html, body, [class*='css'] {{ font-size: {font_size}px; }}</style>",
        unsafe_allow_html=True,
    )

    st.header("Truth Weights")
    w_vol = st.slider("Vol/Mcap", 0.0, 1.0, 0.30, 0.05, key="w_vol")
    w_24h = st.slider("24h Momentum", 0.0, 1.0, 0.25, 0.05, key="w_24h")
    w_7d  = st.slider("7d Momentum", 0.0, 1.0, 0.25, 0.05, key="w_7d")
    w_liq = st.slider("Liquidity/Size", 0.0, 1.0, 0.20, 0.05, key="w_liq")
    _sum = max(1e-9, w_vol+w_24h+w_7d+w_liq)
    W = dict(w_vol=w_vol/_sum, w_24h=w_24h/_sum, w_7d=w_7d/_sum, w_liq=w_liq/_sum)

    st.header("Auto Refresh")
    auto = st.toggle("Auto refresh", value=False, key="auto")
    every = st.slider("Every (sec)", 10, 120, 30, key="every", step=5)

# ------------------------------ UTILS ----------------------------------------

def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def pct_sigmoid(pct) -> float:
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return 0.5
    try:
        x = float(pct) / 10.0
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5

# ------------------------------ CRYPTO (CoinGecko) ---------------------------

@st.cache_data(ttl=60, show_spinner="Loading CoinGecko...")
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
    need = [
        "name","symbol","current_price","market_cap","total_volume",
        "price_change_percentage_1h_in_currency",
        "price_change_percentage_24h_in_currency",
        "price_change_percentage_7d_in_currency",
    ]
    for k in need:
        if k not in df.columns:
            df[k] = np.nan
    return df

# ------------------------------ SCORING --------------------------------------

def build_scores(df: pd.DataFrame, weights: Dict[str,float] | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    t = df.copy()

    t["total_volume"] = t.get("total_volume", pd.Series(np.nan, index=t.index))
    t["market_cap"]   = t.get("market_cap",   pd.Series(np.nan, index=t.index))
    t["vol_to_mc"]    = (t["total_volume"] / t["market_cap"]).replace([np.inf, -np.inf], np.nan).clip(0, 2).fillna(0)

    m1h = t.get("price_change_percentage_1h_in_currency",  pd.Series(np.nan, index=t.index)).apply(pct_sigmoid)
    m24 = t.get("price_change_percentage_24h_in_currency", pd.Series(np.nan, index=t.index)).apply(pct_sigmoid)
    m7d = t.get("price_change_percentage_7d_in_currency",  pd.Series(np.nan, index=t.index)).apply(pct_sigmoid)

    mc = t.get("market_cap", pd.Series(0, index=t.index)).fillna(0)
    t["liq01"] = 0 if mc.max() == 0 else (mc - mc.min()) / (mc.max() - mc.min() + 1e-9)

    # RAW
    t["raw_heat"] = (0.5 * (t["vol_to_mc"] / 2).clip(0, 1) + 0.5 * m1h.fillna(0.5)).clip(0, 1)

    # TRUTH
    if weights:
        w_vol = float(weights.get("w_vol", 0.30))
        w_24h = float(weights.get("w_24h", 0.25))
        w_7d  = float(weights.get("w_7d",  0.25))
        w_liq = float(weights.get("w_liq", 0.20))
        total = max(1e-9, w_vol + w_24h + w_7d + w_liq)
        w_vol, w_24h, w_7d, w_liq = w_vol/total, w_24h/total, w_7d/total, w_liq/total
    else:
        w_vol, w_24h, w_7d, w_liq = 0.30, 0.25, 0.25, 0.20

    t["truth_full"] = (
        w_vol * (t["vol_to_mc"] / 2).clip(0,1) +
        w_24h * m24.fillna(0.5) +
        w_7d  * m7d.fillna(0.5) +
        w_liq * t["liq01"].fillna(0.0)
    ).clip(0,1)

    # Confluence components
    t["consistency01"] = 1 - (m24.fillna(0.5) - m7d.fillna(0.5)).abs()
    t["agreement01"]   = 1 - (t["raw_heat"] - t["truth_full"]).abs()
    t["energy01"]      = (t["vol_to_mc"] / 2).clip(0,1)

    # Confluence (final)
    t["confluence01"] = (
        0.35 * t["truth_full"] +
        0.35 * t["raw_heat"]   +
        0.10 * t["consistency01"] +
        0.10 * t["agreement01"]   +
        0.05 * t["energy01"]      +
        0.05 * t["liq01"]
    ).clip(0,1)

    t["divergence"] = (t["raw_heat"] - t["truth_full"]).abs()

    # Badges
    def fire(v):
        if v >= 0.85: return "🔥🔥🔥"
        if v >= 0.65: return "🔥🔥"
        if v >= 0.45: return "🔥"
        return "."
    def drop(v):
        if v >= 0.85: return "💧💧💧"
        if v >= 0.65: return "💧💧"
        if v >= 0.45: return "💧"
        return "."
    def star(v):
        if v >= 0.85: return "⭐️⭐️⭐️"
        if v >= 0.65: return "⭐️⭐️"
        if v >= 0.45: return "⭐️"
        return "."

    t["RAW_BADGE"]        = t["raw_heat"].apply(fire)
    t["TRUTH_BADGE"]      = t["truth_full"].apply(drop)
    t["CONFLUENCE_BADGE"] = t["confluence01"].apply(star)
    return t

# ------------------------------ TABLE HELPERS ---------------------------------

def filter_controls(df_scored: pd.DataFrame, top_default: int = 150, key_prefix: str = "") -> pd.DataFrame:
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        topn = st.slider("Show Top N", 20, 250, top_default, step=10, key=f"{key_prefix}_topn")
    with c2:
        min_mc = st.number_input("Min Market Cap (USD)", min_value=0, value=0, step=1000000, key=f"{key_prefix}_minmc")
    with c3:
        min_truth = st.slider("Min TRUTH", 0.0, 1.0, 0.0, 0.05, key=f"{key_prefix}_mintruth")
    with c4:
        search = st.text_input("Search (name or symbol)", value="", key=f"{key_prefix}_search").strip().lower()

    out = df_scored.copy()
    if min_mc > 0 and "market_cap" in out.columns:
        out = out[out["market_cap"].fillna(0) >= min_mc]
    if min_truth > 0:
        out = out[out["truth_full"].fillna(0) >= min_truth]
    if search:
        mask = out["name"].str.lower().str.contains(search, na=False) | out["symbol"].str.lower().str.contains(search, na=False)
        out = out[mask]
    return out.sort_values("truth_full", ascending=False).head(topn)

def table_view(df: pd.DataFrame, cols: List[str]) -> None:
    have = [c for c in cols if c in df.columns]
    st.dataframe(
        df[have],
        use_container_width=True,
        hide_index=True,
        column_config={
            "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "market_cap": st.column_config.NumberColumn("Mkt Cap", format="$%d"),
            "total_volume": st.column_config.NumberColumn("Volume", format="$%d"),
            "price_change_percentage_24h_in_currency": st.column_config.NumberColumn("24h %", format="%.2f%%"),
            "raw_heat": st.column_config.ProgressColumn("RAW", min_value=0.0, max_value=1.0),
            "truth_full": st.column_config.ProgressColumn("TRUTH", min_value=0.0, max_value=1.0),
            "confluence01": st.column_config.ProgressColumn("Confluence", min_value=0.0, max_value=1.0),
            "divergence": st.column_config.ProgressColumn("Delta", min_value=0.0, max_value=1.0),
            "RAW_BADGE": st.column_config.TextColumn("Fire"),
            "TRUTH_BADGE": st.column_config.TextColumn("Drop"),
            "CONFLUENCE_BADGE": st.column_config.TextColumn("Star"),
        },
    )

def section_header(title: str, caption: str = "") -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if caption: st.caption(caption)

def kpi_row(df_scored: pd.DataFrame, label: str) -> None:
    n = len(df_scored)
    p24 = float(df_scored.get("price_change_percentage_24h_in_currency", pd.Series(dtype=float)).mean())
    tavg = float(df_scored.get("truth_full", pd.Series(dtype=float)).mean())
    ravg = float(df_scored.get("raw_heat", pd.Series(dtype=float)).mean())
    cavg = float(df_scored.get("confluence01", pd.Series(dtype=float)).mean())
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown("<div class='metric-box'><b>Assets</b><br>{}</div>".format(n), unsafe_allow_html=True)
    with c2: st.markdown("<div class='metric-box'><b>Avg 24h %</b><br>{:.2f}%</div>".format(0 if np.isnan(p24) else p24), unsafe_allow_html=True)
    with c3: st.markdown("<div class='metric-box'><b>Avg TRUTH</b><br>{:.2f}</div>".format(0 if np.isnan(tavg) else tavg), unsafe_allow_html=True)
    with c4: st.markdown("<div class='metric-box'><b>Avg RAW</b><br>{:.2f}</div>".format(0 if np.isnan(ravg) else ravg), unsafe_allow_html=True)
    with c5: st.markdown("<div class='metric-box'><b>Avg Confluence</b><br>{:.2f}</div>".format(0 if np.isnan(cavg) else cavg), unsafe_allow_html=True)
    st.caption(f"{PHASE_TAG} - {POWERED_BY} - Updated {now_utc_str()} - Mode: {label}")

# ------------------------------ STOCKS (Robust yfinance) ---------------------

def _parse_watchlist_text(text: str) -> List[str]:
    if not text: return []
    raw = [tok.strip().upper() for tok in text.replace("\n", ",").split(",")]
    seen, out = set(), []
    for t in raw:
        if t and t not in seen:
            out.append(t); seen.add(t)
    return out

def _read_csv_tickers(file) -> List[str]:
    try:
        df = pd.read_csv(file)
        for c in df.columns:
            if str(c).strip().lower() in ("ticker","tickers","symbol","symbols"):
                vals = [str(v).strip().upper() for v in df[c].tolist()]
                return [v for v in vals if v]
        return []
    except Exception:
        return []

def _history_close_pct(tk: "yf.Ticker", period="5d", interval="1d") -> Tuple[Optional[float], Optional[float]]:
    try:
        h = tk.history(period=period, interval=interval, auto_adjust=True)
        if h is None or h.empty: return None, None
        last = float(h["Close"].iloc[-1])
        prev = float(h["Close"].iloc[-2]) if len(h) >= 2 else None
        pct = (last/prev - 1.0)*100.0 if prev else None
        return last, pct
    except Exception:
        return None, None

@st.cache_data(ttl=180, show_spinner="Loading stocks...")
def yf_snapshot_robust(tickers: List[str], batch: int = 40, pause: float = 0.2, retries: int = 2) -> pd.DataFrame:
    if not HAS_YF or not tickers:
        return pd.DataFrame()
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers: return pd.DataFrame()

    rows = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i+batch]
        df_dl = None
        for attempt in range(retries+1):
            try:
                df_dl = yf.download(
                    tickers=" ".join(chunk),
                    period="5d",
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False,
                )
                break
            except Exception:
                time.sleep(pause)
                df_dl = None

        for t in chunk:
            parsed = False
            if df_dl is not None and not df_dl.empty:
                try:
                    if isinstance(df_dl.columns, pd.MultiIndex):
                        if t in df_dl.columns.levels[0]:
                            s = df_dl[t]
                            last = float(s["Close"].iloc[-1])
                            prev = float(s["Close"].iloc[-2]) if len(s) >= 2 else np.nan
                            pct24 = (last/prev - 1.0)*100.0 if pd.notna(prev) else np.nan
                            rows.append({"name": t, "symbol": t, "current_price": last,
                                         "price_change_percentage_24h_in_currency": pct24,
                                         "market_cap": np.nan, "total_volume": np.nan,
                                         "price_change_percentage_1h_in_currency": np.nan,
                                         "price_change_percentage_7d_in_currency": np.nan})
                            parsed = True
                    else:
                        if "Close" in df_dl.columns:
                            last = float(df_dl["Close"].iloc[-1])
                            prev = float(df_dl["Close"].iloc[-2]) if len(df_dl) >= 2 else np.nan
                            pct24 = (last/prev - 1.0)*100.0 if pd.notna(prev) else np.nan
                            rows.append({"name": t, "symbol": t, "current_price": last,
                                         "price_change_percentage_24h_in_currency": pct24,
                                         "market_cap": np.nan, "total_volume": np.nan,
                                         "price_change_percentage_1h_in_currency": np.nan,
                                         "price_change_percentage_7d_in_currency": np.nan})
                            parsed = True
                except Exception:
                    parsed = False

            if not parsed:
                last = pct24 = None
                for attempt in range(retries+1):
                    try:
                        tk = yf.Ticker(t)
                        last, pct24 = _history_close_pct(tk, period="5d", interval="1d")
                        if last is not None: break
                    except Exception:
                        pass
                    time.sleep(pause)
                rows.append({"name": t, "symbol": t,
                             "current_price": np.nan if last is None else last,
                             "price_change_percentage_24h_in_currency": np.nan if pct24 is None else pct24,
                             "market_cap": np.nan, "total_volume": np.nan,
                             "price_change_percentage_1h_in_currency": np.nan,
                             "price_change_percentage_7d_in_currency": np.nan})

    return pd.DataFrame(rows)

# ------------------------------ OPTIONS (yfinance) ----------------------------

@st.cache_data(ttl=180, show_spinner="Loading options...")
def load_options_chain(ticker: str, expiration: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not HAS_YF or not ticker or not expiration:
        return pd.DataFrame(), pd.DataFrame()
    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiration)
        calls = chain.calls.copy()
        puts  = chain.puts.copy()
        keep = ["contractSymbol","lastTradeDate","strike","lastPrice","bid","ask","change","percentChange","volume","openInterest","impliedVolatility"]
        calls = calls[[c for c in keep if c in calls.columns]]
        puts  = puts [[c for c in keep if c in puts.columns]]
        return calls, puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=120)
def list_expirations(ticker: str) -> List[str]:
    if not HAS_YF or not ticker: return []
    try:
        return list(yf.Ticker(ticker).options)
    except Exception:
        return []

# ------------------------------ PAGES ----------------------------------------

def section_header(title: str, caption: str = "") -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if caption: st.caption(caption)

def page_dashboard() -> None:
    section_header("Dashboard", "Top Confluence and TRUTH at a glance.")
    dfc = build_scores(fetch_cg_markets("usd", 200), W)
    kpi_row(dfc, "Crypto")
    st.markdown(
        "<span class='badge badge-raw'>RAW</span>"
        "<span class='badge badge-truth'>TRUTH</span>"
        "<span class='badge badge-conf'>CONFLUENCE</span>"
        "<span class='badge badge-div'>DELTA</span>",
        unsafe_allow_html=True,
    )
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Top Confluence (Crypto)")
        table_view(
            dfc.sort_values("confluence01", ascending=False).head(20),
            ["name","symbol","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence"]
        )
    with c2:
        st.subheader("Top TRUTH (Crypto)")
        table_view(
            dfc.sort_values("truth_full", ascending=False).head(20),
            ["name","symbol","current_price","TRUTH_BADGE","truth_full","RAW_BADGE","divergence"]
        )

def page_crypto() -> None:
    section_header("Crypto", "Interactive Truth/Raw/Confluence.")
    topn_pull = st.slider("Pull Top N from API", 50, 250, 200, step=50, key="cg_pull")
    df = build_scores(fetch_cg_markets("usd", topn_pull), W)
    if df.empty:
        st.warning("No data from CoinGecko.")
        return
    kpi_row(df, "Crypto")
    filt = filter_controls(df, top_default=150, key_prefix="crypto")
    st.subheader("Ranked by Confluence")
    table_view(
        filt.sort_values("confluence01", ascending=False),
        ["name","symbol","current_price","market_cap","price_change_percentage_24h_in_currency","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence"]
    )
    st.write("")
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Ranked by TRUTH")
        table_view(
            filt.sort_values("truth_full", ascending=False),
            ["name","symbol","current_price","market_cap","TRUTH_BADGE","truth_full","RAW_BADGE","divergence"]
        )
    with c2:
        st.subheader("Ranked by RAW")
        table_view(
            filt.sort_values("raw_heat", ascending=False),
            ["name","symbol","current_price","total_volume","RAW_BADGE","raw_heat","TRUTH_BADGE","divergence"]
        )

def page_confluence() -> None:
    section_header("Confluence", "RAW heat and TRUTH stability aligned.")
    topn_pull = st.slider("Pull Top N from API", 50, 250, 200, step=50, key="conf_pull")
    df = build_scores(fetch_cg_markets("usd", topn_pull), W)
    if df.empty:
        st.warning("No data from CoinGecko.")
        return
    kpi_row(df, "Confluence")
    filt = filter_controls(df, top_default=150, key_prefix="conf")
    st.subheader("Top Confluence Leaders")
    table_view(
        filt.sort_values("confluence01", ascending=False),
        ["name","symbol","current_price","market_cap","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence"]
    )

def page_stocks() -> None:
    section_header("Stocks", "Robust snapshot with Truth/Raw/Confluence.")
    if not HAS_YF:
        st.error("yfinance is not installed. Add `yfinance` to requirements.txt and reboot.")
        return

    default = "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA"
    raw = st.text_area("Tickers (comma/newline-separated)", value=st.session_state.get("stock_input", default), height=100, key="stock_input")
    uploaded = st.file_uploader("Or upload CSV with 'ticker' or 'symbol' column", type=["csv"])
    tickers: List[str] = _parse_watchlist_text(raw)
    if uploaded is not None:
        tickers = list(dict.fromkeys(tickers + _read_csv_tickers(uploaded)))

    if not tickers:
        st.info("Enter or upload at least one ticker.")
        return

    df0 = yf_snapshot_robust(tickers, batch=40, pause=0.2, retries=2)
    if df0.empty:
        st.warning("No stock data returned. Try fewer tickers or check symbols.")
        return

    df = build_scores(df0, W)
    kpi_row(df, "Stocks")
    filt = filter_controls(df, top_default=min(100, len(df)), key_prefix="stocks")
    st.subheader("Top Confluence (Stocks)")
    table_view(
        filt.sort_values("confluence01", ascending=False),
        ["name","symbol","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence"]
    )
    st.write("")
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Top TRUTH")
        table_view(
            filt.sort_values("truth_full", ascending=False),
            ["name","symbol","current_price","TRUTH_BADGE","truth_full","RAW_BADGE","divergence"]
        )
    with c2:
        st.subheader("Top RAW")
        table_view(
            filt.sort_values("raw_heat", ascending=False),
            ["name","symbol","current_price","RAW_BADGE","raw_heat","TRUTH_BADGE","divergence"]
        )

def page_options() -> None:
    section_header("Options", "Expirations and top chains.")
    if not HAS_YF:
        st.error("yfinance not installed. Add `yfinance` to requirements.txt and reboot.")
        return
    sym = st.text_input("Stock Ticker", value="AAPL", key="opt_sym").strip().upper()
    if not sym:
        st.info("Enter a ticker (e.g., AAPL)")
        return
    exps = list_expirations(sym)
    if not exps:
        st.warning("No options expirations returned.")
        return
    exp = st.selectbox("Expiration", options=exps, index=0)
    calls, puts = load_options_chain(sym, exp)
    c1,c2 = st.columns(2)
    keep = ["contractSymbol","strike","lastPrice","openInterest","volume","impliedVolatility"]
    with c1:
        st.subheader(f"{sym} Calls - {exp}")
        if calls.empty: st.info("No calls data.")
        else: table_view(calls.sort_values(["openInterest","volume"], ascending=False).head(15), keep)
    with c2:
        st.subheader(f"{sym} Puts - {exp}")
        if puts.empty: st.info("No puts data.")
        else: table_view(puts.sort_values(["openInterest","volume"], ascending=False).head(15), keep)

def page_fusion() -> None:
    section_header("Fusion", "Crypto vs Stocks under the same lens.")
    dfc = build_scores(fetch_cg_markets("usd", 120), W)
    dfc["universe"] = "CRYPTO"
    if HAS_YF:
        default = "AAPL,MSFT,NVDA,TSLA"
        raw = st.text_input("Stock Tickers for Fusion (comma-separated)", value=default, key="fusion_tickers")
        tick = [x.strip().upper() for x in raw.split(",") if x.strip()]
        dfs0 = yf_snapshot_robust(tick, batch=40, pause=0.2, retries=2) if tick else pd.DataFrame()
        dfs  = build_scores(dfs0, W) if not dfs0.empty else pd.DataFrame()
        if not dfs.empty:
            dfs["universe"] = "STOCKS"
    else:
        dfs = pd.DataFrame(columns=dfc.columns)

    try:
        both = pd.concat([dfc, dfs], ignore_index=True)
    except Exception:
        both = dfc.copy()

    kpi_row(both, "Fusion")
    st.subheader("Universal Leaders by Confluence")
    table_view(
        both.sort_values("confluence01", ascending=False).head(30),
        ["name","symbol","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence","universe"]
    )

def page_scores() -> None:
    section_header("Scores - Explainer")
    st.markdown("""
RAW (0..1) = crowd heat now (Vol/Mcap + 1h momentum).
TRUTH (0..1) = stability blend (Vol/Mcap, 24h, 7d, liquidity) with adjustable weights.
DELTA (0..1) = |RAW - TRUTH|.
CONFLUENCE (0..1) = fusion of RAW + TRUTH + agreement/consistency + energy + liquidity.
""")

def page_export() -> None:
    section_header("Export", "Download CSV snapshots.")
    dfc = build_scores(fetch_cg_markets("usd", 200), W)
    st.download_button("Download Crypto CSV", data=dt_to_csv(dfc), file_name="crypto_truth_raw_confluence.csv", mime="text/csv")
    if HAS_YF:
        default = "AAPL,MSFT,NVDA,TSLA"
        dfs0 = yf_snapshot_robust([x.strip().upper() for x in default.split(",") if x.strip()], batch=40, pause=0.2, retries=2)
        if not dfs0.empty:
            dfs = build_scores(dfs0, W)
            st.download_button("Download Stocks CSV", data=dt_to_csv(dfs), file_name="stocks_truth_raw_confluence.csv", mime="text/csv")
    st.caption("Exports reflect current session pulls; refresh to update.")

def dt_to_csv(df: pd.DataFrame) -> bytes:
    try:
        return df.to_csv(index=False).encode("utf-8")
    except Exception:
        return b""

def page_settings() -> None:
    section_header("Settings", "Session utilities")
    if st.button("Clear all caches (force fresh data)"):
        st.cache_data.clear()
        st.success("Caches cleared. Reloading...")
        time.sleep(0.5)
        st.rerun()

# ------------------------------ ROUTER ---------------------------------------

if nav == "Dashboard":
    page_dashboard()
elif nav == "Crypto":
    page_crypto()
elif nav == "Confluence":
    page_confluence()
elif nav == "Stocks":
    page_stocks()
elif nav == "Options":
    page_options()
elif nav == "Fusion":
    page_fusion()
elif nav == "Scores":
    page_scores()
elif nav == "Export":
    page_export()
else:
    page_settings()

# ------------------------------ AUTO REFRESH ---------------------------------
if auto:
    st.caption(f"{PHASE_TAG} - {POWERED_BY} - Auto refresh every {int(every)}s")
    time.sleep(max(5, int(every)))
    st.rerun()
