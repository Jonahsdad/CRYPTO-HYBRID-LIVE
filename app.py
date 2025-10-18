# -----------------------------------------------------------------------------
# Crypto Hybrid Live — Phase 15.3 (FULL Interactive)
# POWERED BY JESSE RAY LANDINGHAM JR
# -----------------------------------------------------------------------------
# What you get:
# - Big powered-by banner + phase tag
# - Crypto (CoinGecko) with TRUTH / RAW / DELTA
# - Interactive tables (filters, search, sorting)
# - Color progress bars + 🔥 fire (RAW) + 💧 drop (TRUTH) badges
# - Stocks (yfinance) + Options page (expirations + option chain top ranks)
# - Fusion (Crypto vs Stocks) + Export CSV + Settings
# - Works on iPad (ASCII-only header)
# -----------------------------------------------------------------------------

from __future__ import annotations
import math, time
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Optional: robust stocks via yfinance (app still runs if missing)
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

PHASE_TAG  = "PHASE 15.3 — FULL INTERACTIVE"
POWERED_BY = "POWERED BY JESSE RAY LANDINGHAM JR"

# ============================== PAGE CONFIG / THEME ===========================

st.set_page_config(
    page_title="Crypto Hybrid Live — Phase 15.3",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.block-container { padding-top: 0.6rem; padding-bottom: 2.0rem; }

/* Banners */
.phase-badge {
  padding: 12px; border-radius: 12px;
  background: linear-gradient(90deg, #0f172a 0%, #0b2a1d 50%, #0f172a 100%);
  border: 1px solid #2f3a4a; color: #7dfca3; font-weight: 900;
  text-align: center; margin-bottom: 8px; font-size: 18px;
}
.powered-badge {
  padding: 10px; border-radius: 10px; background: #10161f; border: 1px solid #29434e;
  color: #9ad7ff; font-weight: 900; text-align: center; margin: 6px 0 14px 0; font-size: 16px;
}

/* Section titles & metric cards */
.section-title { font-size: 24px; font-weight: 800; margin: 6px 0 10px 0; }
.metric-box { border: 1px solid #ffffff22; border-radius: 12px; padding: 0.8rem; background: #0e1117; }

/* Badges row */
.badge { display:inline-block; padding: 6px 10px; border-radius: 999px; font-weight:700; margin-right:6px; }
.badge-raw   { background:#241c14; color:#ff9b63; border:1px solid #ff9b6333; }
.badge-truth { background:#172017; color:#7dff96; border:1px solid #7dff9633; }
.badge-div   { background:#161a22; color:#8ecbff; border:1px solid #8ecbff33; }
.badge-hot   { background:#231616; color:#ff7a7a; border:1px solid #ff7a7a33; }

/* Data Editor tweaks */
.stDataFrame, .stDataEditor { border-radius: 10px; overflow: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(f"<div class='phase-badge'>✅ {PHASE_TAG}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='powered-badge'>{POWERED_BY}</div>", unsafe_allow_html=True)

# ============================== SIDEBAR ======================================

with st.sidebar:
    st.header("Navigation")
    nav = st.radio(
        "Go to",
        ["Dashboard", "Crypto", "Stocks", "Options", "Fusion", "Scores", "Export", "Settings"],
        index=0,
        key="nav_radio",
    )

    st.header("Appearance")
    font_size = st.slider("Font size", 14, 24, 18, key="font_size")
    st.markdown(
        f"<style>html, body, [class*='css'] {{ font-size: {font_size}px; }}</style>",
        unsafe_allow_html=True,
    )

    st.header("Auto Refresh")
    auto = st.toggle("Auto refresh", value=False, key="auto")
    every = st.slider("Every (sec)", 10, 120, 30, key="every", step=5)

# ============================== CORE HELPERS =================================

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

    # Emoji badges
    def fire(v):  # RAW
        if v >= 0.85: return "🔥🔥🔥"
        if v >= 0.65: return "🔥🔥"
        if v >= 0.45: return "🔥"
        return "·"
    def drop(v):  # TRUTH
        if v >= 0.85: return "💧💧💧"
        if v >= 0.65: return "💧💧"
        if v >= 0.45: return "💧"
        return "·"

    t["RAW_BADGE"]   = t["raw_heat"].apply(fire)
    t["TRUTH_BADGE"] = t["truth_full"].apply(drop)
    return t

# ============================== INTERACTIVE TABLES ============================

def crypto_interactive(df_scored: pd.DataFrame) -> None:
    # Filters row
    st.markdown("<div class='section-title'>Interactive Filters</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        topn = st.slider("Show Top N", 20, 250, 150, step=10, key="c_topn")
    with c2:
        min_mc = st.number_input("Min Market Cap (USD)", min_value=0, value=0, step=1000000, key="c_min_mc")
    with c3:
        min_truth = st.slider("Min TRUTH", 0.0, 1.0, 0.0, 0.05, key="c_min_truth")
    with c4:
        search = st.text_input("Search (name or symbol)", value="", key="c_search").strip().lower()

    filtered = df_scored.copy()
    if min_mc > 0 and "market_cap" in filtered.columns:
        filtered = filtered[filtered["market_cap"].fillna(0) >= min_mc]
    if min_truth > 0:
        filtered = filtered[filtered["truth_full"].fillna(0) >= min_truth]
    if search:
        mask = filtered["name"].str.lower().str.contains(search, na=False) | filtered["symbol"].str.lower().str.contains(search, na=False)
        filtered = filtered[mask]

    # Display with color progress bars using Streamlit's column configs
    st.markdown(
        "<span class='badge badge-raw'>RAW 🔥</span>"
        "<span class='badge badge-truth'>TRUTH 💧</span>"
        "<span class='badge badge-div'>DELTA ◆</span>",
        unsafe_allow_html=True,
    )

    show_cols = [
        "name","symbol","current_price","market_cap","total_volume",
        "price_change_percentage_24h_in_currency","RAW_BADGE","TRUTH_BADGE",
        "raw_heat","truth_full","divergence"
    ]
    have = [c for c in show_cols if c in filtered.columns]
    view = filtered.sort_values("truth_full", ascending=False).head(topn)[have]

    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "market_cap": st.column_config.NumberColumn("Mkt Cap", format="$%d"),
            "total_volume": st.column_config.NumberColumn("Volume", format="$%d"),
            "price_change_percentage_24h_in_currency": st.column_config.NumberColumn("24h %", format="%.2f%%"),
            "raw_heat": st.column_config.ProgressColumn("RAW Heat", help="Faster, crowd heat (0..1)", min_value=0.0, max_value=1.0),
            "truth_full": st.column_config.ProgressColumn("TRUTH", help="Stability blend (0..1)", min_value=0.0, max_value=1.0),
            "divergence": st.column_config.ProgressColumn("Delta", help="|RAW − TRUTH|", min_value=0.0, max_value=1.0),
            "RAW_BADGE": st.column_config.TextColumn("🔥"),
            "TRUTH_BADGE": st.column_config.TextColumn("💧"),
            "name": st.column_config.TextColumn("Name"),
            "symbol": st.column_config.TextColumn("Symbol"),
        },
    )

def truth_raw_triptych(df_scored: pd.DataFrame, title_suffix: str = "", topn:int=25) -> None:
    st.markdown(
        "<span class='badge badge-raw'>RAW 🔥</span>"
        "<span class='badge badge-truth'>TRUTH 💧</span>"
        "<span class='badge badge-div'>DELTA ◆</span>",
        unsafe_allow_html=True,
    )
    c1,c2,c3 = st.columns(3)

    with c1:
        st.subheader(f"RAW — Heat {title_suffix}")
        cols = ["name","symbol","current_price","market_cap","total_volume","RAW_BADGE","raw_heat"]
        have = [c for c in cols if c in df_scored.columns]
        st.dataframe(
            df_scored.sort_values("raw_heat", ascending=False).head(topn)[have],
            use_container_width=True, hide_index=True,
            column_config={
                "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "market_cap": st.column_config.NumberColumn("Mkt Cap", format="$%d"),
                "total_volume": st.column_config.NumberColumn("Volume", format="$%d"),
                "raw_heat": st.column_config.ProgressColumn("RAW Heat", min_value=0.0, max_value=1.0),
                "RAW_BADGE": st.column_config.TextColumn("🔥"),
            }
        )

    with c2:
        st.subheader(f"TRUTH — Stability {title_suffix}")
        cols = ["name","symbol","current_price","market_cap","TRUTH_BADGE","truth_full"]
        have = [c for c in cols if c in df_scored.columns]
        st.dataframe(
            df_scored.sort_values("truth_full", ascending=False).head(topn)[have],
            use_container_width=True, hide_index=True,
            column_config={
                "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "market_cap": st.column_config.NumberColumn("Mkt Cap", format="$%d"),
                "truth_full": st.column_config.ProgressColumn("TRUTH", min_value=0.0, max_value=1.0),
                "TRUTH_BADGE": st.column_config.TextColumn("💧"),
            }
        )

    with c3:
        st.subheader(f"MOVERS — 24h {title_suffix}")
        if "price_change_percentage_24h_in_currency" in df_scored.columns:
            g = df_scored.sort_values("price_change_percentage_24h_in_currency", ascending=False).head(10)
            l = df_scored.sort_values("price_change_percentage_24h_in_currency", ascending=True).head(10)
            st.markdown("Top Gainers")
            st.dataframe(
                g[["name","symbol","current_price","price_change_percentage_24h_in_currency","divergence"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "price_change_percentage_24h_in_currency": st.column_config.NumberColumn("24h %", format="%.2f%%"),
                    "divergence": st.column_config.ProgressColumn("Delta", min_value=0.0, max_value=1.0),
                }
            )
            st.markdown("Top Losers")
            st.dataframe(
                l[["name","symbol","current_price","price_change_percentage_24h_in_currency","divergence"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "price_change_percentage_24h_in_currency": st.column_config.NumberColumn("24h %", format="%.2f%%"),
                    "divergence": st.column_config.ProgressColumn("Delta", min_value=0.0, max_value=1.0),
                }
            )
        else:
            st.info("No 24h % column available.")

# ============================== STOCKS / OPTIONS ==============================

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

def options_chain(ticker: str, expiration: Optional[str]=None):
    if not HAS_YF: return None, None, []
    t = yf.Ticker(ticker)
    exps = t.options or []
    if not exps: return t, None, []
    if expiration and expiration in exps:
        exp = expiration
    else:
        exp = exps[0]
    try:
        ch = t.option_chain(exp)
        return t, exp, ch
    except Exception:
        return t, exp, []

# ============================== PAGES =========================================

def section_header(title: str, caption: str = "") -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if caption: st.caption(caption)

def kpi_row(df_scored: pd.DataFrame, label: str) -> None:
    n = len(df_scored)
    p24 = float(df_scored.get("price_change_percentage_24h_in_currency", pd.Series(dtype=float)).mean())
    tavg = float(df_scored.get("truth_full", pd.Series(dtype=float)).mean())
    ravg = float(df_scored.get("raw_heat", pd.Series(dtype=float)).mean())
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown("<div class='metric-box'><b>Assets</b><br>{}</div>".format(n), unsafe_allow_html=True)
    with c2: st.markdown("<div class='metric-box'><b>Avg 24h %</b><br>{:.2f}%</div>".format(0 if np.isnan(p24) else p24), unsafe_allow_html=True)
    with c3: st.markdown("<div class='metric-box'><b>Avg TRUTH</b><br>{:.2f}</div>".format(0 if np.isnan(tavg) else tavg), unsafe_allow_html=True)
    with c4: st.markdown("<div class='metric-box'><b>Avg RAW</b><br>{:.2f}</div>".format(0 if np.isnan(ravg) else ravg), unsafe_allow_html=True)
    st.caption(f"{PHASE_TAG} • {POWERED_BY} • Updated {now_utc_str()} • Mode: {label}")

# ---- Dashboard
def page_dashboard() -> None:
    section_header("Dashboard", "TRUTH vs RAW overview for Crypto.")
    dfc = score_table(fetch_cg_markets("usd", 200))
    kpi_row(dfc, "Crypto")
    truth_raw_triptych(dfc, title_suffix="(Crypto)")

# ---- Crypto
def page_crypto() -> None:
    section_header("Crypto", "Interactive TRUTH/RAW with filters + badges.")
    topn = st.slider("Pull Top N from API", 50, 250, 200, step=50, key="cg_pull")
    df = score_table(fetch_cg_markets("usd", topn))
    if df.empty:
        st.warning("No data received from CoinGecko.")
        return
    kpi_row(df, "Crypto")
    crypto_interactive(df)
    st.write("")
    st.subheader("Triptych View")
    truth_raw_triptych(df, title_suffix="(Crypto)", topn=25)

# ---- Stocks
def page_stocks() -> None:
    section_header("Stocks", "Robust snapshot scored by the same lens.")
    if not HAS_YF:
        st.error("yfinance is not installed in this deployment. Add `yfinance` to requirements.txt and reboot.")
        return

    default = "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA"
    raw = st.text_input("Tickers (comma-separated)", value=st.session_state.get("stock_input", default), key="stock_input")
    uploaded = st.file_uploader("Or upload CSV with a 'ticker' column", type=["csv"])
    tickers: List[str] = []
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            if "ticker" in df_up.columns:
                tickers = [str(x).strip().upper() for x in df_up["ticker"].tolist() if str(x).strip()]
        except Exception:
            st.warning("Could not parse CSV; falling back to input box.")
    if not tickers:
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

    if not tickers:
        st.info("Enter at least one ticker (or upload CSV).")
        return

    df0 = yf_snapshot(tickers)
    if df0.empty:
        st.warning("No stock data returned. Check tickers.")
        return
    df = score_table(df0)
    kpi_row(df, "Stocks")
    crypto_interactive(df)  # reusing same interactive UI for stocks
    st.write("")
    st.subheader("Triptych View")
    truth_raw_triptych(df, title_suffix="(Stocks)", topn=min(25, len(df)))

# ---- Options
def page_options() -> None:
    section_header("Options", "Expirations + Option Chains (top ranks).")
    if not HAS_YF:
        st.error("yfinance not installed. Add `yfinance` to requirements.txt and reboot.")
        return
    sym = st.text_input("Stock Ticker", value="AAPL", key="opt_sym").strip().upper()
    if not sym:
        st.info("Enter a ticker (e.g., AAPL)")
        return
    tkr, exp, chain = options_chain(sym)
    if tkr is None:
        st.warning("Could not fetch options metadata.")
        return
    exps = tkr.options or []
    if not exps:
        st.warning("No options expirations found.")
        return

    exp_sel = st.selectbox("Expiration", exps, index=min(0, len(exps)-1))
    _, exp_use, chain = options_chain(sym, expiration=exp_sel)
    if not chain:
        st.warning("No option chain data for this expiry.")
        return

    calls = chain.calls.copy()
    puts  = chain.puts.copy()
    # Rank by Open Interest and Volume
    for df in (calls, puts):
        for col in ["openInterest", "volume", "lastPrice", "strike", "impliedVolatility"]:
            if col not in df.columns: df[col] = np.nan
    calls_top = calls.sort_values(["openInterest","volume"], ascending=False).head(15)[
        ["contractSymbol","strike","lastPrice","openInterest","volume","impliedVolatility"]
    ]
    puts_top = puts.sort_values(["openInterest","volume"], ascending=False).head(15)[
        ["contractSymbol","strike","lastPrice","openInterest","volume","impliedVolatility"]
    ]

    c1,c2 = st.columns(2)
    with c1:
        st.subheader(f"{sym} Calls — {exp_use}")
        st.dataframe(calls_top, use_container_width=True, hide_index=True)
    with c2:
        st.subheader(f"{sym} Puts — {exp_use}")
        st.dataframe(puts_top, use_container_width=True, hide_index=True)

    st.caption("Note: Options data is informational only; not investment advice.")

# ---- Fusion
def page_fusion() -> None:
    section_header("Fusion", "Crypto vs Stocks in one lens.")
    dfc = score_table(fetch_cg_markets("usd", 120))
    dfc["universe"] = "CRYPTO"
    if HAS_YF:
        default = "AAPL,MSFT,NVDA,TSLA"
        raw = st.text_input("Stock Tickers for Fusion (comma-separated)", value=default, key="fusion_tickers")
        tick = [x.strip().upper() for x in raw.split(",") if x.strip()]
        dfs = score_table(yf_snapshot(tick)) if tick else pd.DataFrame()
        dfs["universe"] = "STOCKS"
    else:
        dfs = pd.DataFrame(columns=dfc.columns)
    try:
        both = pd.concat([dfc, dfs], ignore_index=True)
    except Exception:
        both = dfc.copy()

    kpi_row(both, "Fusion")
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Crypto — Top TRUTH")
        subset = both[both.get("universe","")=="CRYPTO"].sort_values("truth_full", ascending=False).head(20)
        st.dataframe(
            subset[["name","symbol","current_price","RAW_BADGE","TRUTH_BADGE","raw_heat","truth_full","divergence"]],
            use_container_width=True, hide_index=True,
            column_config={
                "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "raw_heat": st.column_config.ProgressColumn("RAW Heat", min_value=0.0, max_value=1.0),
                "truth_full": st.column_config.ProgressColumn("TRUTH", min_value=0.0, max_value=1.0),
                "divergence": st.column_config.ProgressColumn("Delta", min_value=0.0, max_value=1.0),
                "RAW_BADGE": st.column_config.TextColumn("🔥"),
                "TRUTH_BADGE": st.column_config.TextColumn("💧"),
            }
        )
    with c2:
        st.subheader("Stocks — Top TRUTH")
        subset = both[both.get("universe","")=="STOCKS"].sort_values("truth_full", ascending=False).head(20)
        st.dataframe(
            subset[["name","symbol","current_price","RAW_BADGE","TRUTH_BADGE","raw_heat","truth_full","divergence"]],
            use_container_width=True, hide_index=True,
            column_config={
                "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "raw_heat": st.column_config.ProgressColumn("RAW Heat", min_value=0.0, max_value=1.0),
                "truth_full": st.column_config.ProgressColumn("TRUTH", min_value=0.0, max_value=1.0),
                "divergence": st.column_config.ProgressColumn("Delta", min_value=0.0, max_value=1.0),
                "RAW_BADGE": st.column_config.TextColumn("🔥"),
                "TRUTH_BADGE": st.column_config.TextColumn("💧"),
            }
        )

# ---- Scores Explainer
def page_scores() -> None:
    section_header("Scores — Explainer")
    st.markdown("""
**RAW (0..1)** — fast/crowd heat from Vol/Mcap and 1h momentum.  
**TRUTH (0..1)** — stability blend: Vol/Mcap, 24h momentum, 7d momentum, liquidity.  
**DELTA (0..1)** — absolute gap |RAW − TRUTH| (potential overextension / snapback).
""")
    st.info("These weights are fixed in Phase 15. A weight editor + saved presets land next.")

# ---- Export
def page_export() -> None:
    section_header("Export", "Download CSV snapshots.")
    dfc = score_table(fetch_cg_markets("usd", 200))
    st.download_button("Download Crypto CSV", data=dfc.to_csv(index=False).encode("utf-8"),
                       file_name="crypto_truth_raw.csv", mime="text/csv")
    if HAS_YF:
        default = "AAPL,MSFT,NVDA,TSLA"
        tick = [x.strip().upper() for x in default.split(",") if x.strip()]
        dfs = score_table(yf_snapshot(tick)) if tick else pd.DataFrame()
        if not dfs.empty:
            st.download_button("Download Stocks CSV", data=dfs.to_csv(index=False).encode("utf-8"),
                               file_name="stocks_truth_raw.csv", mime="text/csv")
    st.caption("Exports reflect current session pulls; refresh to update.")

# ---- Settings
def page_settings() -> None:
    section_header("Settings", "Toggles that shape how you view data.")
    st.toggle("High-contrast metric cards", key="hc_cards")
    st.caption("More settings arrive with the preset editor in the next phase.")

# ============================== ROUTER ========================================

if nav == "Dashboard":
    page_dashboard()
elif nav == "Crypto":
    page_crypto()
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

# ============================== AUTO REFRESH ==================================
if auto:
    st.caption(f"{PHASE_TAG} • {POWERED_BY} • Auto refresh every {int(every)}s")
    time.sleep(max(5, int(every)))
    st.rerun()
