# app.py  — Crypto Hybrid Live (Phase 18.6)
# Full, self-contained Streamlit app.
# Includes: single-line hero, interactive metric buttons, Plotly fallback, Crypto + Stocks + Options.

from datetime import datetime, timezone
from typing import List, Tuple
import math, time, io, re

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ------------------------------ Optional deps ------------------------------
# yfinance (stocks & options) is optional
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

# Plotly (pretty charts) is optional; we fall back to st.bar_chart if missing
try:
    import plotly.express as px
    HAS_PX = True
except Exception:
    HAS_PX = False

# ------------------------------ Constants / Theme ------------------------------
PHASE_TAG  = "PHASE 18.6 — Robust Stocks + Confluence + Options"
BRAND      = "CRYPTO HYBRID LIVE — POWERED BY JESSE RAY LANDINGHAM JR"

st.set_page_config(
    page_title="Crypto Hybrid Live",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root {
  --panel:#0c111a;
  --panel2:#0d141d;
  --ink:#c9d6e9;
  --accent:#1ee0a1;
  --muted:#6e7c91;
}
.block-container{padding-top:.4rem;padding-bottom:1rem;max-width:1440px;}
/* HERO */
.hero {
  margin: 2px 0 8px 0; padding: 10px 12px; border-radius: 12px;
  background: linear-gradient(90deg, #06131b 0%, #0d1f17 55%, #0b0e13 100%);
  border: 1px solid #1e2a37; color: #9de9c8; font-weight: 900; letter-spacing:.2px;
  display:flex; align-items:center; justify-content:center; gap:10px;
}
.hero .dot{width:10px;height:10px;border-radius:50%;background:#20e07c;box-shadow:0 0 6px #20e07c;}
.hero .brand{font-size:16px;}
.kpi {background:#0b1119;border:1px solid #1c2734;border-radius:12px;padding:12px;color:var(--ink);}
.kpi b{color:#fff}
.badge {display:inline-flex;align-items:center;gap:6px;padding:6px 11px;border-radius:999px;font-weight:800;border:1px solid #ffffff1f}
.badge-raw{background:#1b1410;color:#ffad73;border-color:#ffad7338}
.badge-truth{background:#0f1a11;color:#82ff9e;border-color:#82ff9e3a}
.badge-conf{background:#121116;color:#ffd86b;border-color:#ffd86b38}
.badge-delta{background:#11161d;color:#8ecbff;border-color:#8ecbff38}
.section-title{font-size:24px;font-weight:900;margin:6px 0 6px}
.stTabs [data-baseweb="tab-list"]{gap:10px}
.small-note{color:#8aa0b8;font-size:.85rem}
.stDataFrame, .stDataEditor{border-radius:10px;overflow:hidden}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------ Sidebar --------------------------------------
with st.sidebar:
    st.header("Navigation")
    nav = st.radio(
        "Go to",
        ["Dashboard","Crypto","Confluence","US Market (All Listings)","S&P 500","Options","Fusion","Export","Scores","Settings"],
        index=0
    )

    st.header("Truth Weights")
    w_vol = st.slider("Vol/Mcap", 0.00, 1.00, 0.30, 0.01)
    w_m24 = st.slider("24h Momentum", 0.00, 1.00, 0.25, 0.01)
    w_m7  = st.slider("7d Momentum", 0.00, 1.00, 0.25, 0.01)
    w_liq = st.slider("Liquidity/Size", 0.00, 1.00, 0.20, 0.01)
    wsum  = max(1e-9, w_vol + w_m24 + w_m7 + w_liq)
    W = dict(vol = w_vol/wsum, m24 = w_m24/wsum, m7=w_m7/wsum, liq=w_liq/wsum)

    st.header("Appearance")
    font_size = st.slider("Font size", 14, 22, 18)
    st.markdown(f"<style>html, body, [class*='css']{{font-size:{font_size}px}}</style>", unsafe_allow_html=True)

    st.header("Auto Refresh")
    auto  = st.toggle("Auto refresh", value=False)
    every = st.slider("Every (sec)", 10, 120, 30, step=5)

# ------------------------------ Helpers ---------------------------------------
def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def pct_sigmoid(pct) -> float:
    if pct is None or (isinstance(pct, float) and np.isnan(pct)): return 0.5
    try:
        return 1.0 / (1.0 + math.exp(-(float(pct)/10.0)))
    except Exception:
        return 0.5

@st.cache_data(ttl=60, show_spinner="Loading CoinGecko…")
def fetch_cg_markets(vs: str="usd", per_page: int=200) -> pd.DataFrame:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    p = {"vs_currency": vs, "order": "market_cap_desc", "per_page": int(max(1, min(per_page, 250))),
         "page": 1, "sparkline": "false", "price_change_percentage": "1h,24h,7d", "locale": "en"}
    r = requests.get(url, params=p, timeout=25); r.raise_for_status()
    df = pd.DataFrame(r.json())
    cols = ["name","symbol","current_price","market_cap","total_volume",
            "price_change_percentage_1h_in_currency","price_change_percentage_24h_in_currency",
            "price_change_percentage_7d_in_currency"]
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    return df

def _normalize01(series: pd.Series) -> pd.Series:
    x = series.astype(float).replace([np.inf,-np.inf], np.nan)
    mn, mx = x.min(), x.max()
    if pd.isna(mn) or pd.isna(mx) or mx <= mn: return pd.Series(0.0, index=x.index)
    return (x-mn)/(mx-mn)

def build_scores(df: pd.DataFrame, w: dict) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    t = df.copy()
    t["vol_to_mc"] = (t.get("total_volume",0) / t.get("market_cap",1)).replace([np.inf,-np.inf],np.nan).clip(0,2).fillna(0)
    m1h = t.get("price_change_percentage_1h_in_currency", pd.Series(np.nan,index=t.index)).apply(pct_sigmoid).fillna(0.5)
    m24 = t.get("price_change_percentage_24h_in_currency", pd.Series(np.nan,index=t.index)).apply(pct_sigmoid).fillna(0.5)
    m7d = t.get("price_change_percentage_7d_in_currency", pd.Series(np.nan,index=t.index)).apply(pct_sigmoid).fillna(0.5)
    t["liq01"] = _normalize01(t.get("market_cap",0).fillna(0))

    # RAW (heat now)
    t["raw_heat"] = (0.5*(t["vol_to_mc"]/2).clip(0,1) + 0.5*m1h).clip(0,1)
    # TRUTH (stability / quality)
    t["truth_full"] = (w["vol"]*(t["vol_to_mc"]/2).clip(0,1) + w["m24"]*m24 + w["m7"]*m7d + w["liq"]*t["liq01"]).clip(0,1)
    # Confluence (agreement + consistency + energy + liquidity)
    t["consistency01"] = 1 - (m24 - m7d).abs()
    t["agreement01"]   = 1 - (t["raw_heat"] - t["truth_full"]).abs()
    t["energy01"]      = (t["vol_to_mc"]/2).clip(0,1)
    t["confluence01"]  = (0.35*t["truth_full"] + 0.35*t["raw_heat"] + 0.10*t["consistency01"]
                         +0.10*t["agreement01"] + 0.05*t["energy01"] + 0.05*t["liq01"]).clip(0,1)
    t["divergence"]    = (t["raw_heat"] - t["truth_full"]).abs()

    def fire(v):  return "🔥🔥🔥" if v>=.85 else ("🔥🔥" if v>=.65 else ("🔥" if v>=.45 else "·"))
    def drop(v):  return "💧💧💧" if v>=.85 else ("💧💧" if v>=.65 else ("💧" if v>=.45 else "·"))
    def star(v):  return "⭐️⭐️⭐️" if v>=.85 else ("⭐️⭐️" if v>=.65 else ("⭐️" if v>=.45 else "·"))
    t["RAW_BADGE"]        = t["raw_heat"].apply(fire)
    t["TRUTH_BADGE"]      = t["truth_full"].apply(drop)
    t["CONFLUENCE_BADGE"] = t["confluence01"].apply(star)
    return t

def kpi_row(df_scored: pd.DataFrame, label: str) -> None:
    n = len(df_scored)
    p24 = float(df_scored.get("price_change_percentage_24h_in_currency", pd.Series(dtype=float)).mean())
    tavg = float(df_scored.get("truth_full", pd.Series(dtype=float)).mean())
    ravg = float(df_scored.get("raw_heat", pd.Series(dtype=float)).mean())
    cavg = float(df_scored.get("confluence01", pd.Series(dtype=float)).mean())
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(f"<div class='kpi'><b>Assets</b><br>{n}</div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='kpi'><b>Avg 24h %</b><br>{0 if np.isnan(p24) else p24:.2f}%</div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='kpi'><b>Avg TRUTH</b><br>{0 if np.isnan(tavg) else tavg:.2f}</div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='kpi'><b>Avg RAW</b><br>{0 if np.isnan(ravg) else ravg:.2f}</div>", unsafe_allow_html=True)
    with c5: st.markdown(f"<div class='kpi'><b>Avg Confluence</b><br>{0 if np.isnan(cavg) else cavg:.2f}</div>", unsafe_allow_html=True)
    st.caption(f"{PHASE_TAG} • Updated {now_utc_str()} • Mode: {label}")

def table_view(df: pd.DataFrame, cols: List[str]) -> None:
    have = [c for c in cols if c in df.columns]
    st.dataframe(
        df[have], use_container_width=True, hide_index=True,
        column_config={
            "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "market_cap": st.column_config.NumberColumn("Mkt Cap", format="$%d"),
            "total_volume": st.column_config.NumberColumn("Volume", format="$%d"),
            "price_change_percentage_24h_in_currency": st.column_config.NumberColumn("24h %", format="%.2f%%"),
            "raw_heat": st.column_config.ProgressColumn("RAW Heat", min_value=0.0, max_value=1.0),
            "truth_full": st.column_config.ProgressColumn("TRUTH", min_value=0.0, max_value=1.0),
            "confluence01": st.column_config.ProgressColumn("Confluence", min_value=0.0, max_value=1.0),
            "divergence": st.column_config.ProgressColumn("Δ", min_value=0.0, max_value=1.0),
            "RAW_BADGE": st.column_config.TextColumn("🔥"),
            "TRUTH_BADGE": st.column_config.TextColumn("💧"),
            "CONFLUENCE_BADGE": st.column_config.TextColumn("⭐"),
            "name": st.column_config.TextColumn("Name"),
            "symbol": st.column_config.TextColumn("Symbol"),
            "Sector": st.column_config.TextColumn("Sector"),
            "ListingExchange": st.column_config.TextColumn("Exchange"),
            "ETF": st.column_config.TextColumn("ETF"),
        },
    )

# ------------------------------ Stocks / Options utils ------------------------
@st.cache_data(ttl=180, show_spinner="Pulling daily price snapshots…")
def yf_snapshot_daily(tickers: List[str]) -> pd.DataFrame:
    if not HAS_YF or not tickers: return pd.DataFrame()
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    try:
        data = yf.download(" ".join(tickers), period="5d", interval="1d", group_by="ticker",
                           auto_adjust=True, threads=True, progress=False)
    except Exception:
        return pd.DataFrame()

    rows = []
    for t in tickers:
        try:
            s = data[t]
            last = float(s.iloc[-1]["Close"])
            prev = float(s.iloc[-2]["Close"]) if len(s) >= 2 else np.nan
            pct24 = (last/prev - 1.0)*100.0 if pd.notna(prev) else np.nan
            rows.append({"name": t, "symbol": t, "current_price": last,
                         "price_change_percentage_24h_in_currency": pct24,
                         "market_cap": np.nan, "total_volume": np.nan,
                         "price_change_percentage_1h_in_currency": np.nan,
                         "price_change_percentage_7d_in_currency": np.nan})
        except Exception:
            rows.append({"name": t, "symbol": t, "current_price": np.nan,
                         "price_change_percentage_24h_in_currency": np.nan,
                         "market_cap": np.nan, "total_volume": np.nan,
                         "price_change_percentage_1h_in_currency": np.nan,
                         "price_change_percentage_7d_in_currency": np.nan})
    return pd.DataFrame(rows)

@st.cache_data(ttl=120)
def list_expirations(ticker: str) -> List[str]:
    if not HAS_YF or not ticker: return []
    try: return list(yf.Ticker(ticker).options)
    except Exception: return []

@st.cache_data(ttl=180, show_spinner="Loading option chain…")
def load_options_chain(ticker: str, expiration: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not HAS_YF or not ticker or not expiration: return pd.DataFrame(), pd.DataFrame()
    try:
        tk = yf.Ticker(ticker); chain = tk.option_chain(expiration)
        calls, puts = chain.calls.copy(), chain.puts.copy()
        keep = ["contractSymbol","strike","lastPrice","openInterest","volume","impliedVolatility"]
        calls = calls[[c for c in keep if c in calls.columns]]
        puts  = puts [[c for c in keep if c in puts.columns]]
        return calls, puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

# ------------- Universes (S&P + All US listings via NASDAQ Trader) ------------
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

NASDAQ_LISTED_URL = "https://ftp.nasdaqtrader.com/dynamic/SYMBOL_DIRECTORY/nasdaqlisted.txt"
OTHER_LISTED_URL  = "https://ftp.nasdaqtrader.com/dynamic/SYMBOL_DIRECTORY/otherlisted.txt"

@st.cache_data(ttl=900, show_spinner="Loading US listings…")
def fetch_us_listings() -> pd.DataFrame:
    frames = []
    try:
        txt = requests.get(NASDAQ_LISTED_URL, timeout=30).text
        df = pd.read_csv(io.StringIO(txt), sep="|")
        df = df[~df["Symbol"].str.contains("File Creation Time", na=False)]
        df = df.rename(columns={
            "Symbol":"symbol","Security Name":"name","Market Category":"MarketCategory",
            "Test Issue":"TestIssue","Financial Status":"FinancialStatus","ETF":"ETF"})
        df["ListingExchange"] = "NASDAQ"
        frames.append(df[["symbol","name","ListingExchange","ETF","TestIssue"]])
    except Exception:
        pass
    try:
        txt = requests.get(OTHER_LISTED_URL, timeout=30).text
        df = pd.read_csv(io.StringIO(txt), sep="|")
        df = df[~df["ACT Symbol"].str.contains("File Creation Time", na=False)]
        df = df.rename(columns={
            "ACT Symbol":"symbol","Security Name":"name","Exchange":"ListingExchange",
            "ETF":"ETF","Test Issue":"TestIssue"})
        frames.append(df[["symbol","name","ListingExchange","ETF","TestIssue"]])
    except Exception:
        pass

    if not frames:
        return pd.DataFrame(columns=["symbol","name","ListingExchange","ETF","TestIssue"])
    allu = pd.concat(frames, ignore_index=True)
    allu["symbol"] = allu["symbol"].astype(str).str.upper()
    allu["ETF"] = allu["ETF"].astype(str).str.upper()
    allu["TestIssue"] = allu["TestIssue"].astype(str).str.upper()
    allu = allu[allu["TestIssue"] != "Y"].copy()
    allu = allu[allu["symbol"].str.match(r"^[A-Z0-9\.-]+$", na=False)]
    return allu.drop_duplicates(subset=["symbol"])

# ------------------------------ HERO -----------------------------------------
st.markdown(
    f"""
<div class='hero'>
  <div class='dot'></div>
  <div class='brand'>{BRAND}</div>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------------------ Pages ----------------------------------------
def dashboard():
    st.markdown("<div class='section-title'>Dashboard</div>", unsafe_allow_html=True)
    df = build_scores(fetch_cg_markets("usd", 200), W)
    kpi_row(df, "Crypto")

    # interactive metric buttons (single row, sticky feel)
    cols = st.columns([1,1,1,1,6])
    if "focus_metric" not in st.session_state:
        st.session_state.focus_metric = "conf"
    if cols[0].button("🔥 RAW", type="secondary"):   st.session_state.focus_metric = "raw"
    if cols[1].button("💧 TRUTH", type="secondary"): st.session_state.focus_metric = "truth"
    if cols[2].button("⭐ CONFLUENCE", type="secondary"): st.session_state.focus_metric = "conf"
    if cols[3].button("⚡ Δ (RAW→TRUTH)", type="secondary"): st.session_state.focus_metric = "delta"

    metric_map = {
        "raw":   ("raw_heat",     "RAW Heat (0..1)"),
        "truth": ("truth_full",   "TRUTH (0..1)"),
        "conf":  ("confluence01", "Confluence (0..1)"),
        "delta": ("divergence",   "Δ = |RAW − TRUTH| (0..1)"),
    }
    mcol, mtitle = metric_map[st.session_state.focus_metric]
    top = df.sort_values(mcol, ascending=False).head(25)

    if HAS_PX:
        fig = px.bar(top, x="symbol", y=mcol, hover_data=["name","current_price"], title=f"Top by {mtitle}", height=320)
        fig.update_layout(margin=dict(l=0,r=0,t=28,b=6), xaxis_title=None, yaxis_title=None, bargap=.18)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.bar_chart(top.set_index("symbol")[mcol], height=320)

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Top Confluence (Crypto)")
        table_view(df.sort_values("confluence01", ascending=False).head(20),
                   ["name","symbol","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE"])
    with c2:
        st.subheader("Top TRUTH (Crypto)")
        table_view(df.sort_values("truth_full", ascending=False).head(20),
                   ["name","symbol","current_price","TRUTH_BADGE","truth_full","RAW_BADGE","divergence"])

def page_crypto():
    st.markdown("<div class='section-title'>Crypto</div>", unsafe_allow_html=True)
    pull = st.slider("Pull Top N from API", 50, 250, 200, 50)
    df = build_scores(fetch_cg_markets("usd", pull), W)
    if df.empty: st.warning("No data from CoinGecko."); return
    kpi_row(df, "Crypto")
    c1,c2,c3,c4 = st.columns(4)
    with c1: topn = st.slider("Show Top N", 20, 250, 150, 10)
    with c2: min_truth = st.slider("Min TRUTH", 0.0, 1.0, 0.0, 0.05)
    with c3: search = st.text_input("Search", value="").strip().lower()
    with c4: order  = st.selectbox("Order by", ["confluence01","truth_full","raw_heat"], index=0)
    out = df.copy()
    if min_truth>0: out = out[out["truth_full"]>=min_truth]
    if search:
        mask = out["name"].str.lower().str.contains(search, na=False) | out["symbol"].str.lower().str.contains(search, na=False)
        out = out[mask]
    out = out.sort_values(order, ascending=False).head(topn)
    table_view(out, ["name","symbol","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence"])

def page_confluence():
    st.markdown("<div class='section-title'>Confluence</div>", unsafe_allow_html=True)
    df = build_scores(fetch_cg_markets("usd", 200), W)
    if df.empty: st.warning("No data."); return
    kpi_row(df, "Confluence")
    table_view(df.sort_values("confluence01", ascending=False).head(60),
               ["name","symbol","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence"])

def page_sp500():
    st.markdown("<div class='section-title'>S&P 500</div>", unsafe_allow_html=True)
    if not HAS_YF:
        st.error("yfinance not installed. Add `yfinance` to requirements.txt and reboot."); return
    base = fetch_sp500_constituents()
    if base.empty: st.warning("Couldn’t fetch S&P list."); return
    st.caption(f"Constituents: {len(base)} • Updated {now_utc_str()}")
    limit = st.slider("Tickers to snapshot", 50, len(base), 200, 50)
    tickers = base["symbol"].tolist()[:limit]
    snap = yf_snapshot_daily(tickers)
    if snap.empty: st.warning("No prices returned. Try fewer tickers."); return
    merged = snap.merge(base, on="symbol", how="left")
    scored = build_scores(merged, W)
    kpi_row(scored, "S&P 500")
    table_view(scored.sort_values("confluence01", ascending=False).head(100),
               ["name","symbol","Sector","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE"])

def page_us_all():
    st.markdown("<div class='section-title'>US Market (All Listings)</div>", unsafe_allow_html=True)
    if not HAS_YF:
        st.error("yfinance not installed. Add `yfinance` to requirements.txt and reboot."); return
    uni = fetch_us_listings()
    if uni.empty: st.warning("Couldn’t fetch US listings. Try again."); return
    c1,c2,c3,c4 = st.columns([2,1,1,1])
    with c1: query = st.text_input("Search by symbol or name", value="").strip().lower()
    with c2: exch  = st.selectbox("Exchange", ["(All)"] + sorted(uni["ListingExchange"].dropna().unique().tolist()))
    with c3: include_etf = st.toggle("Include ETFs", value=False)
    with c4: limit = st.slider("Tickers to snapshot", 50, 1000, 200, 50)
    dfu = uni.copy()
    if not include_etf and "ETF" in dfu.columns:
        dfu = dfu[dfu["ETF"] != "Y"]
    if exch != "(All)":
        dfu = dfu[dfu["ListingExchange"] == exch]
    if query:
        mask = dfu["symbol"].str.lower().str.contains(query, na=False) | dfu["name"].str.lower().str.contains(query, na=False)
        dfu = dfu[mask]
    st.caption(f"Universe matches: {len(dfu)} • Snapshot first {min(limit, len(dfu))}")
    if len(dfu)==0: st.info("No matches."); return
    tickers = dfu["symbol"].tolist()[:limit]
    snap = yf_snapshot_daily(tickers)
    if snap.empty: st.warning("No price data returned for selected set."); return
    merged = snap.merge(dfu, on="symbol", how="left")
    scored = build_scores(merged, W)
    kpi_row(scored, "US Market")
    cL,cR = st.columns(2)
    with cL:
        st.subheader("Top Confluence ⭐")
        table_view(scored.sort_values("confluence01", ascending=False).head(100),
                   ["name","symbol","ListingExchange","ETF","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence"])
    with cR:
        st.subheader("Top TRUTH 💧")
        table_view(scored.sort_values("truth_full", ascending=False).head(100),
                   ["name","symbol","ListingExchange","ETF","current_price","TRUTH_BADGE","truth_full","RAW_BADGE","divergence"])

def page_options():
    st.markdown("<div class='section-title'>Options Explorer</div>", unsafe_allow_html=True)
    if not HAS_YF:
        st.error("yfinance not installed. Add `yfinance` to requirements.txt and reboot."); return
    base = fetch_us_listings()
    default = "AAPL"
    sym = st.selectbox("Ticker", options=[default] + (base["symbol"].tolist() if not base.empty else []), index=0)
    exps = list_expirations(sym)
    if not exps: st.warning("No options expirations returned."); return
    exp = st.selectbox("Expiration", options=exps, index=0)
    calls, puts = load_options_chain(sym, exp)
    c1,c2 = st.columns(2)
    with c1:
        st.subheader(f"{sym} Calls — {exp}")
        keep = ["contractSymbol","strike","lastPrice","openInterest","volume","impliedVolatility"]
        if calls.empty: st.info("No calls data.")
        else: table_view(calls.sort_values(["openInterest","volume"], ascending=False).head(25), keep)
    with c2:
        st.subheader(f"{sym} Puts — {exp}")
        keep = ["contractSymbol","strike","lastPrice","openInterest","volume","impliedVolatility"]
        if puts.empty: st.info("No puts data.")
        else: table_view(puts.sort_values(["openInterest","volume"], ascending=False).head(25), keep)

def page_fusion():
    st.markdown("<div class='section-title'>Fusion</div>", unsafe_allow_html=True)
    dfc = build_scores(fetch_cg_markets("usd", 120), W); dfc["universe"]="CRYPTO"
    uni = fetch_us_listings()
    if HAS_YF and not uni.empty:
        tickers = uni[(uni["ETF"]!="Y")]["symbol"].tolist()[:120] if "ETF" in uni.columns else uni["symbol"].tolist()[:120]
        dfs0 = yf_snapshot_daily(tickers)
        dfs = build_scores(dfs0.merge(uni, on="symbol", how="left"), W) if not dfs0.empty else pd.DataFrame()
        dfs["universe"]="US"
    else:
        dfs = pd.DataFrame(columns=dfc.columns)
    both = pd.concat([dfc, dfs], ignore_index=True) if not dfs.empty else dfc.copy()
    kpi_row(both, "Fusion")
    table_view(both.sort_values("confluence01", ascending=False).head(40),
               ["name","symbol","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence","universe"])

def page_export():
    st.markdown("<div class='section-title'>Export</div>", unsafe_allow_html=True)
    dfc = build_scores(fetch_cg_markets("usd", 200), W)
    st.download_button("Download Crypto CSV", data=dfc.to_csv(index=False).encode("utf-8"),
                       file_name="crypto_truth_raw_confluence.csv", mime="text/csv")
    uni = fetch_us_listings()
    if HAS_YF and not uni.empty:
        tickers = uni["symbol"].tolist()[:500]
        dfs0 = yf_snapshot_daily(tickers)
        if not dfs0.empty:
            dfs = build_scores(dfs0.merge(uni, on="symbol", how="left"), W)
            st.download_button("Download US Snapshot CSV", data=dfs.to_csv(index=False).encode("utf-8"),
                               file_name="us_listings_truth_raw_confluence.csv", mime="text/csv")

def page_scores():
    st.markdown("<div class='section-title'>Scores — Explainer</div>", unsafe_allow_html=True)
    st.markdown("""
**RAW (0–1)** — crowd heat **now** (volume/market-cap + 1h momentum).  
**TRUTH (0–1)** — stability/quality (vol/mcap, 24h, 7d, liquidity).  
**Δ (0–1)** — |RAW − TRUTH| (gap/mismatch).  
**CONFLUENCE (0–1)** — fusion: RAW + TRUTH + agreement (RAW≈TRUTH) + consistency (24h≈7d) + energy + liquidity.

**Read fast**
- ⭐ High **Confluence** → hype and quality aligned (prime).
- 🔥 High **RAW**, low 💧 **TRUTH** → hype spike (fragile).
- 💧 High **TRUTH**, low 🔥 **RAW** → sleeper quality (crowd not there yet).
""")

def page_settings():
    st.markdown("<div class='section-title'>Settings</div>", unsafe_allow_html=True)
    st.toggle("High-contrast metric cards", value=False)
    st.caption("More personalization & presets coming.")

# ------------------------------ Router ---------------------------------------
if nav == "Dashboard":               dashboard()
elif nav == "Crypto":                page_crypto()
elif nav == "Confluence":            page_confluence()
elif nav == "US Market (All Listings)": page_us_all()
elif nav == "S&P 500":               page_sp500()
elif nav == "Options":               page_options()
elif nav == "Fusion":                page_fusion()
elif nav == "Export":                page_export()
elif nav == "Scores":                page_scores()
else:                                page_settings()

# ------------------------------ Auto refresh ---------------------------------
if auto:
    st.caption(f"{PHASE_TAG} • Auto refresh {int(every)}s")
    time.sleep(max(5, int(every)))
    st.rerun()
