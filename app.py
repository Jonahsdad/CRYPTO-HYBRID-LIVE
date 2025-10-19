from __future__ import annotations

# ============================== IMPORTS ==============================
import math, time, io, re
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Optional stocks & options (app still runs w/o yfinance but stock pages will warn)
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False
    yf = None  # avoid NameError

PHASE_TAG  = "PHASE 18.0 — US LISTINGS + OPTIONS"
POWERED_BY = "POWERED BY JESSE RAY LANDINGHAM JR"

# ============================== PAGE CONFIG / THEME ===========================
st.set_page_config(
    page_title="Crypto Hybrid Live — Phase 18.0",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.block-container { padding-top: 0.6rem; padding-bottom: 1.2rem; }
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
.section-title { font-size: 24px; font-weight: 800; margin: 6px 0 8px 0; }
.metric-box { border: 1px solid #ffffff22; border-radius: 12px; padding: 10px; background: #0e1117; }
.badge { display:inline-block; padding: 6px 10px; border-radius: 999px; font-weight:700; margin-right:6px; }
.badge-raw   { background:#241c14; color:#ff9b63; border:1px solid #ff9b6333; }
.badge-truth { background:#172017; color:#7dff96; border:1px solid #7dff9633; }
.badge-conf  { background:#1a1a24; color:#ffd86b; border:1px solid #ffd86b33; }
.badge-div   { background:#161a22; color:#8ecbff; border:1px solid #8ecbff33; }
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
        [
            "Dashboard",
            "Crypto",
            "Confluence",
            "US Market (All Listings)",
            "S&P 500",
            "Options",
            "Fusion",
            "Export",
            "Scores",
            "Settings",
        ],
        index=0, key="nav_radio",
    )
    st.header("Appearance")
    font_size = st.slider("Font size", 14, 24, 18, key="font_size")
    st.markdown(
        f"<style>html, body, [class*='css'] {{ font-size: {font_size}px; }}</style>",
        unsafe_allow_html=True,
    )
    st.header("Auto Refresh")
    auto  = st.toggle("Auto refresh", value=False, key="auto")
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
        "vs_currency": vs,
        "order": "market_cap_desc",
        "per_page": int(max(1, min(per_page, 250))),
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": "1h,24h,7d",
        "locale": "en",
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

# ============================== SCORING ======================================
def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    t = df.copy()

    # Inputs
    t["total_volume"] = t.get("total_volume", pd.Series(np.nan, index=t.index))
    t["market_cap"]   = t.get("market_cap",   pd.Series(np.nan, index=t.index))
    t["vol_to_mc"]    = (t["total_volume"] / t["market_cap"]).replace([np.inf, -np.inf], np.nan).clip(0, 2).fillna(0)

    m1h = t.get("price_change_percentage_1h_in_currency",  pd.Series(np.nan, index=t.index)).apply(pct_sigmoid)
    m24 = t.get("price_change_percentage_24h_in_currency", pd.Series(np.nan, index=t.index)).apply(pct_sigmoid)
    m7d = t.get("price_change_percentage_7d_in_currency",  pd.Series(np.nan, index=t.index)).apply(pct_sigmoid)

    mc = t.get("market_cap", pd.Series(0, index=t.index)).fillna(0)
    t["liq01"] = 0 if mc.max() == 0 else (mc - mc.min()) / (mc.max() - mc.min() + 1e-9)

    # RAW & TRUTH
    t["raw_heat"] = (0.5 * (t["vol_to_mc"] / 2).clip(0, 1) + 0.5 * m1h.fillna(0.5)).clip(0, 1)
    t["truth_full"] = (
        0.30*(t["vol_to_mc"]/2).clip(0,1) +
        0.25*m24.fillna(0.5) +
        0.25*m7d.fillna(0.5) +
        0.20*t["liq01"].fillna(0.0)
    ).clip(0,1)

    # Confluence
    t["consistency01"] = 1 - (m24.fillna(0.5) - m7d.fillna(0.5)).abs()
    t["agreement01"]   = 1 - (t["raw_heat"] - t["truth_full"]).abs()
    t["energy01"]      = (t["vol_to_mc"] / 2).clip(0,1)
    t["confluence01"]  = (
        0.35*t["truth_full"] + 0.35*t["raw_heat"] +
        0.10*t["consistency01"] + 0.10*t["agreement01"] +
        0.05*t["energy01"] + 0.05*t["liq01"]
    ).clip(0,1)

    t["divergence"] = (t["raw_heat"] - t["truth_full"]).abs()

    def fire(v): return "🔥🔥🔥" if v>=0.85 else ("🔥🔥" if v>=0.65 else ("🔥" if v>=0.45 else "·"))
    def drop(v): return "💧💧💧" if v>=0.85 else ("💧💧" if v>=0.65 else ("💧" if v>=0.45 else "·"))
    def star(v): return "⭐️⭐️⭐️" if v>=0.85 else ("⭐️⭐️" if v>=0.65 else ("⭐️" if v>=0.45 else "·"))
    t["RAW_BADGE"]        = t["raw_heat"].apply(fire)
    t["TRUTH_BADGE"]      = t["truth_full"].apply(drop)
    t["CONFLUENCE_BADGE"] = t["confluence01"].apply(star)
    return t

# ============================== UI HELPERS ===================================
def section_header(title: str, caption: str = "") -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if caption:
        st.caption(caption)

def kpi_row(df_scored: pd.DataFrame, label: str) -> None:
    n = len(df_scored)
    p24 = float(df_scored.get("price_change_percentage_24h_in_currency", pd.Series(dtype=float)).mean())
    tavg = float(df_scored.get("truth_full", pd.Series(dtype=float)).mean())
    ravg = float(df_scored.get("raw_heat", pd.Series(dtype=float)).mean())
    cavg = float(df_scored.get("confluence01", pd.Series(dtype=float)).mean())
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(f"<div class='metric-box'><b>Assets</b><br>{n}</div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-box'><b>Avg 24h %</b><br>{0 if np.isnan(p24) else p24:.2f}%</div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-box'><b>Avg TRUTH</b><br>{0 if np.isnan(tavg) else tavg:.2f}</div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric-box'><b>Avg RAW</b><br>{0 if np.isnan(ravg) else ravg:.2f}</div>", unsafe_allow_html=True)
    with c5: st.markdown(f"<div class='metric-box'><b>Avg Confluence</b><br>{0 if np.isnan(cavg) else cavg:.2f}</div>", unsafe_allow_html=True)
    st.caption(f"{PHASE_TAG} • {POWERED_BY} • Updated {now_utc_str()} • Mode: {label}")

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
            "raw_heat": st.column_config.ProgressColumn("RAW Heat", min_value=0.0, max_value=1.0),
            "truth_full": st.column_config.ProgressColumn("TRUTH", min_value=0.0, max_value=1.0),
            "confluence01": st.column_config.ProgressColumn("Confluence", min_value=0.0, max_value=1.0),
            "divergence": st.column_config.ProgressColumn("Delta", min_value=0.0, max_value=1.0),
            "RAW_BADGE": st.column_config.TextColumn("🔥"),
            "TRUTH_BADGE": st.column_config.TextColumn("💧"),
            "CONFLUENCE_BADGE": st.column_config.TextColumn("⭐"),
            "name": st.column_config.TextColumn("Name"),
            "symbol": st.column_config.TextColumn("Symbol"),
            "ListingExchange": st.column_config.TextColumn("Exchange"),
            "ETF": st.column_config.TextColumn("ETF"),
        },
    )

# ============================== STOCKS UTILITIES ==============================
@st.cache_data(ttl=180, show_spinner="Pulling daily price snapshots…")
def yf_snapshot_daily(tickers: List[str]) -> pd.DataFrame:
    if not HAS_YF or not tickers:
        return pd.DataFrame()
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(
            tickers=" ".join(tickers), period="5d", interval="1d",
            group_by="ticker", auto_adjust=True, threads=True, progress=False,
        )
    except Exception:
        return pd.DataFrame()

    rows = []
    # yfinance returns a MultiIndex when multiple tickers, else single-level
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            try:
                s = data[t]
                last = float(s["Close"].dropna().iloc[-1])
                prev = float(s["Close"].dropna().iloc[-2]) if s["Close"].dropna().shape[0] >= 2 else np.nan
                pct24 = (last/prev - 1.0)*100.0 if pd.notna(prev) else np.nan
                rows.append({
                    "name": t, "symbol": t, "current_price": last,
                    "price_change_percentage_24h_in_currency": pct24,
                    "market_cap": np.nan, "total_volume": np.nan,
                    "price_change_percentage_1h_in_currency": np.nan,
                    "price_change_percentage_7d_in_currency": np.nan
                })
            except Exception:
                rows.append({
                    "name": t, "symbol": t, "current_price": np.nan,
                    "price_change_percentage_24h_in_currency": np.nan,
                    "market_cap": np.nan, "total_volume": np.nan,
                    "price_change_percentage_1h_in_currency": np.nan,
                    "price_change_percentage_7d_in_currency": np.nan
                })
    else:
        # Single ticker case
        try:
            last = float(data["Close"].dropna().iloc[-1])
            prev = float(data["Close"].dropna().iloc[-2]) if data["Close"].dropna().shape[0] >= 2 else np.nan
            pct24 = (last/prev - 1.0)*100.0 if pd.notna(prev) else np.nan
            t = tickers[0]
            rows.append({
                "name": t, "symbol": t, "current_price": last,
                "price_change_percentage_24h_in_currency": pct24,
                "market_cap": np.nan, "total_volume": np.nan,
                "price_change_percentage_1h_in_currency": np.nan,
                "price_change_percentage_7d_in_currency": np.nan
            })
        except Exception:
            pass
    return pd.DataFrame(rows)

# ============================== OPTIONS ======================================
@st.cache_data(ttl=120)
def list_expirations(ticker: str) -> List[str]:
    if not HAS_YF or not ticker:
        return []
    try:
        return list(yf.Ticker(ticker).options)
    except Exception:
        return []

@st.cache_data(ttl=180, show_spinner="Loading option chain…")
def load_options_chain(ticker: str, expiration: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not HAS_YF or not ticker or not expiration:
        return pd.DataFrame(), pd.DataFrame()
    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiration)
        calls, puts = chain.calls.copy(), chain.puts.copy()
        keep = [
            "contractSymbol","lastTradeDate","strike","lastPrice","bid","ask","change",
            "percentChange","volume","openInterest","impliedVolatility"
        ]
        calls = calls[[c for c in keep if c in calls.columns]]
        puts  = puts[[c for c in keep if c in puts.columns]]   # <-- fixed dot here
        return calls, puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

# ============================== UNIVERSE FEEDS ================================
# 1) S&P 500 (Wikipedia)
@st.cache_data(ttl=600, show_spinner="Loading S&P 500 list…")
def fetch_sp500_constituents() -> pd.DataFrame:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0].copy()
        df = df.rename(columns={"Symbol":"symbol","Security":"name","GICS Sector":"Sector"})
        df["symbol"] = df["symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
        return df[["symbol","name","Sector"]]
    except Exception:
        return pd.DataFrame(columns=["symbol","name","Sector"])

# 2) US Listings (NASDAQ Trader)
NASDAQ_LISTED_URL = "https://ftp.nasdaqtrader.com/dynamic/SYMBOL_DIRECTORY/nasdaqlisted.txt"
OTHER_LISTED_URL  = "https://ftp.nasdaqtrader.com/dynamic/SYMBOL_DIRECTORY/otherlisted.txt"

@st.cache_data(ttl=900, show_spinner="Loading US listings…")
def fetch_us_listings() -> pd.DataFrame:
    frames = []
    try:
        txt = requests.get(NASDAQ_LISTED_URL, timeout=30).text
        df = pd.read_csv(io.StringIO(txt), sep="|")   # <-- use io.StringIO
        # Remove footer rows
        df = df[~df["Symbol"].astype(str).str.contains("File Creation Time", na=False)]
        df = df.rename(columns={
            "Symbol":"symbol",
            "Security Name":"name",
            "Market Category":"MarketCategory",
            "Test Issue":"TestIssue",
            "Financial Status":"FinancialStatus",
            "ETF":"ETF",
        })
        df["ListingExchange"] = "NASDAQ"
        frames.append(df[["symbol","name","ListingExchange","ETF","TestIssue"]])
    except Exception:
        pass

    try:
        txt = requests.get(OTHER_LISTED_URL, timeout=30).text
        df = pd.read_csv(io.StringIO(txt), sep="|")   # <-- use io.StringIO
        df = df[~df["ACT Symbol"].astype(str).str.contains("File Creation Time", na=False)]
        df = df.rename(columns={
            "ACT Symbol":"symbol",
            "Security Name":"name",
            "Exchange":"ListingExchange",
            "ETF":"ETF",
            "Test Issue":"TestIssue",
        })
        frames.append(df[["symbol","name","ListingExchange","ETF","TestIssue"]])
    except Exception:
        pass

    if not frames:
        return pd.DataFrame(columns=["symbol","name","ListingExchange","ETF","TestIssue"])

    allu = pd.concat(frames, ignore_index=True)
    # Clean + filters
    allu["symbol"] = allu["symbol"].astype(str).str.upper()
    allu["ETF"] = allu["ETF"].astype(str).str.upper()
    allu["TestIssue"] = allu["TestIssue"].astype(str).str.upper()
    # Filter out test issues
    allu = allu[allu["TestIssue"] != "Y"].copy()
    # Some symbols contain ^ or $ or spaces; drop weirds
    allu = allu[allu["symbol"].str.match(r"^[A-Z0-9\.-]+$", na=False)]
    return allu.drop_duplicates(subset=["symbol"])

# ============================== PAGES =========================================
def section_header(title: str, caption: str = "") -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if caption:
        st.caption(caption)

def page_dashboard() -> None:
    section_header("Dashboard", "Top Confluence & Truth leaders (Crypto).")
    dfc = build_scores(fetch_cg_markets("usd", 200))
    kpi_row(dfc, "Crypto")
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Top Confluence (Crypto)")
        table_view(
            dfc.sort_values("confluence01", ascending=False).head(20),
            ["name","symbol","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE"]
        )
    with c2:
        st.subheader("Top TRUTH (Crypto)")
        table_view(
            dfc.sort_values("truth_full", ascending=False).head(20),
            ["name","symbol","current_price","TRUTH_BADGE","truth_full","RAW_BADGE","divergence"]
        )

def page_crypto() -> None:
    section_header("Crypto", "Interactive TRUTH / RAW / CONFLUENCE (CoinGecko).")
    topn_pull = st.slider("Pull Top N from API", 50, 250, 200, step=50, key="cg_pull")
    df = build_scores(fetch_cg_markets("usd", topn_pull))
    if df.empty:
        st.warning("No data from CoinGecko.")
        return
    kpi_row(df, "Crypto")
    # Simple interactive filters
    c1,c2,c3,c4 = st.columns([1,1,1,1])
    with c1: topn = st.slider("Show Top N", 20, 250, 150, step=10, key="cg_topn")
    with c2: min_truth = st.slider("Min TRUTH", 0.0, 1.0, 0.0, 0.05, key="cg_mintruth")
    with c3: search = st.text_input("Search", value="", key="cg_search").strip().lower()
    with c4: order = st.selectbox("Order by", ["confluence01","truth_full","raw_heat"], index=0, key="cg_order")
    out = df.copy()
    if min_truth>0:
        out = out[out["truth_full"]>=min_truth]
    if search:
        mask = out["name"].str.lower().str.contains(search, na=False) | out["symbol"].str.lower().str.contains(search, na=False)
        out = out[mask]
    out = out.sort_values(order, ascending=False).head(topn)
    table_view(out, ["name","symbol","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence"])

def page_confluence() -> None:
    section_header("Confluence", "When RAW heat and TRUTH stability agree strongly (Crypto).")
    df = build_scores(fetch_cg_markets("usd", 200))
    if df.empty:
        st.warning("No data from CoinGecko.")
        return
    kpi_row(df, "Confluence")
    table_view(df.sort_values("confluence01", ascending=False).head(50),
               ["name","symbol","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence"])

def page_sp500() -> None:
    section_header("S&P 500", "Live constituents → prices → RAW/TRUTH/CONFLUENCE.")
    if not HAS_YF:
        st.error("yfinance not installed. Add `yfinance` to requirements.txt and reboot."); return
    base = fetch_sp500_constituents()
    if base.empty:
        st.warning("Couldn’t fetch the S&P 500 list (Wikipedia unreachable). Try again later.")
        return
    st.caption(f"Constituents fetched: {len(base)} • Updated {now_utc_str()}")
    limit = st.slider("Tickers to snapshot", 50, len(base), 200, step=50)
    tickers = base["symbol"].tolist()[:limit]
    snap = yf_snapshot_daily(tickers)
    if snap.empty:
        st.warning("No prices returned. Try fewer tickers or retry.")
        return
    merged = snap.merge(base, on="symbol", how="left")
    scored = build_scores(merged)
    kpi_row(scored, "S&P 500")
    table_view(
        scored.sort_values("confluence01", ascending=False).head(100),
        ["name","symbol","Sector","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE"]
    )
    st.download_button("Download S&P Snapshot (CSV)", data=scored.to_csv(index=False).encode("utf-8"),
                       file_name="sp500_truth_raw_confluence.csv", mime="text/csv")

def page_us_all() -> None:
    section_header("US Market (All Listings)", "NASDAQ + NYSE + ARCA + AMEX (via NASDAQ Trader).")
    if not HAS_YF:
        st.error("yfinance not installed. Add `yfinance` to requirements.txt and reboot."); return

    uni = fetch_us_listings()
    if uni.empty:
        st.warning("Couldn’t fetch the US listings. NASDAQ feed might be unreachable. Try again.")
        return

    c1,c2,c3,c4 = st.columns([2,1,1,1])
    with c1: query = st.text_input("Search by symbol or name", value="", key="usall_q").strip().lower()
    with c2: exch  = st.selectbox("Exchange", options=["(All)"] + sorted(uni["ListingExchange"].dropna().unique().tolist()), index=0)
    with c3: include_etf = st.toggle("Include ETFs", value=False, key="usall_etf")
    with c4: limit = st.slider("Tickers to snapshot", 50, 1000, 200, step=50, key="usall_limit")

    dfu = uni.copy()
    if not include_etf and "ETF" in dfu.columns:
        dfu = dfu[dfu["ETF"] != "Y"]
    if exch != "(All)":
        dfu = dfu[dfu["ListingExchange"] == exch]
    if query:
        mask = dfu["symbol"].str.lower().str.contains(query, na=False) | dfu["name"].str.lower().str.contains(query, na=False)
        dfu = dfu[mask]

    st.caption(f"Universe matches: {len(dfu)} • Showing snapshot for first {min(limit, len(dfu))}")
    if len(dfu) == 0:
        st.info("No matches. Try different filters or include ETFs.")
        return

    # Snapshot selected subset
    tickers = dfu["symbol"].tolist()[:limit]
    snap = yf_snapshot_daily(tickers)
    if snap.empty:
        st.warning("No price data returned for selected set. Try fewer tickers or another exchange.")
        return

    merged = snap.merge(dfu, on="symbol", how="left")
    scored = build_scores(merged)
    kpi_row(scored, "US Market")

    cL, cR = st.columns([1,1])
    with cL:
        st.subheader("Top Confluence ⭐")
        table_view(
            scored.sort_values("confluence01", ascending=False).head(100),
            ["name","symbol","ListingExchange","ETF","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence"]
        )
    with cR:
        st.subheader("Top TRUTH 💧")
        table_view(
            scored.sort_values("truth_full", ascending=False).head(100),
            ["name","symbol","ListingExchange","ETF","current_price","TRUTH_BADGE","truth_full","RAW_BADGE","divergence"]
        )

    st.download_button(
        "Download US Snapshot (CSV)",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="us_listings_truth_raw_confluence.csv",
        mime="text/csv",
    )

def page_options() -> None:
    section_header("Options Explorer", "Pick a ticker → expiration → calls/puts.")
    if not HAS_YF:
        st.error("yfinance not installed. Add `yfinance` to requirements and reboot."); return

    # Use US listings for convenience dropdown (fallback to manual box if fetch fails)
    base = fetch_us_listings()
    default = "AAPL"
    if base.empty:
        sym = st.text_input("Ticker", value=default, key="opt_sym").strip().upper()
    else:
        sym = st.selectbox("Ticker", options=[default] + base["symbol"].tolist(), index=0).strip().upper()

    if not sym:
        st.info("Enter a ticker (e.g., AAPL)")
        return
    exps = list_expirations(sym)
    if not exps:
        st.warning("No options expirations returned."); return
    exp = st.selectbox("Expiration", options=exps, index=0)
    calls, puts = load_options_chain(sym, exp)

    c1,c2 = st.columns(2)
    with c1:
        st.subheader(f"{sym} Calls — {exp}")
        if calls.empty: st.info("No calls data.")
        else:
            keep = ["contractSymbol","strike","lastPrice","openInterest","volume","impliedVolatility"]
            table_view(calls.sort_values(["openInterest","volume"], ascending=False).head(25), keep)
    with c2:
        st.subheader(f"{sym} Puts — {exp}")
        if puts.empty: st.info("No puts data.")
        else:
            keep = ["contractSymbol","strike","lastPrice","openInterest","volume","impliedVolatility"]
            table_view(puts.sort_values(["openInterest","volume"], ascending=False).head(25), keep)

def page_fusion() -> None:
    section_header("Fusion", "Crypto vs US leaders by Confluence.")
    dfc = build_scores(fetch_cg_markets("usd", 120))
    dfc["universe"] = "CRYPTO"

    uni = fetch_us_listings()
    if HAS_YF and not uni.empty:
        tickers = uni[uni["ETF"] != "Y"]["symbol"].tolist()[:120] if "ETF" in uni.columns else uni["symbol"].tolist()[:120]
        dfs0 = yf_snapshot_daily(tickers)
        dfs  = build_scores(dfs0.merge(uni, on="symbol", how="left")) if not dfs0.empty else pd.DataFrame()
        dfs["universe"] = "US"
    else:
        dfs = pd.DataFrame(columns=dfc.columns)

    try:
        both = pd.concat([dfc, dfs], ignore_index=True)
    except Exception:
        both = dfc.copy()

    kpi_row(both, "Fusion")
    st.subheader("Universal Leaders by Confluence ⭐")
    table_view(
        both.sort_values("confluence01", ascending=False).head(40),
        ["name","symbol","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence","universe"]
    )

def page_export() -> None:
    section_header("Export", "One-click CSV downloads.")
    dfc = build_scores(fetch_cg_markets("usd", 200))
    st.download_button("Download Crypto CSV", data=dfc.to_csv(index=False).encode("utf-8"),
                       file_name="crypto_truth_raw_confluence.csv", mime="text/csv")

    uni = fetch_us_listings()
    if HAS_YF and not uni.empty:
        tickers = uni["symbol"].tolist()[:500]
        dfs0 = yf_snapshot_daily(tickers)
        if not dfs0.empty:
            dfs = build_scores(dfs0.merge(uni, on="symbol", how="left"))
            st.download_button("Download US Snapshot CSV", data=dfs.to_csv(index=False).encode("utf-8"),
                               file_name="us_listings_truth_raw_confluence.csv", mime="text/csv")

def page_scores() -> None:
    section_header("Scores — Explainer", "")
    st.markdown("""
**RAW (0..1)** — crowd heat now (volume/market-cap + 1h momentum).  
**TRUTH (0..1)** — stability blend (vol/mcap, 24h, 7d, liquidity).  
**DELTA (0..1)** — |RAW − TRUTH|.  
**CONFLUENCE (0..1)** — fusion of RAW+TRUTH with agreement (RAW~TRUTH), consistency (24h~7d), energy & liquidity.

**Read fast**  
- ⭐ High Confluence → hype and quality aligned (prime).  
- 🔥 High RAW, low 💧 TRUTH → hype spike (fragile).  
- 💧 High TRUTH, low 🔥 RAW → sleeper quality (crowd not there yet).
""")

def page_settings() -> None:
    section_header("Settings", "More personalization next phase.")
    st.toggle("High-contrast metric cards", key="hc_cards")
    st.caption("Weights editor & saved presets coming soon.")

# ============================== ROUTER ========================================
if nav == "Dashboard":
    page_dashboard()
elif nav == "Crypto":
    page_crypto()
elif nav == "Confluence":
    page_confluence()
elif nav == "US Market (All Listings)":
    page_us_all()
elif nav == "S&P 500":
    page_sp500()
elif nav == "Options":
    page_options()
elif nav == "Fusion":
    page_fusion()
elif nav == "Export":
    page_export()
elif nav == "Scores":
    page_scores()
else:
    page_settings()

# ============================== AUTO REFRESH ==================================
if auto:
    st.caption(f"{PHASE_TAG} • {POWERED_BY} • Auto refresh every {int(every)}s")
    time.sleep(max(5, int(every)))
    st.rerun()
