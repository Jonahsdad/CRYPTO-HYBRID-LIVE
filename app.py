py — Phase 19.0 (Hero 3×, robust pills, multi-provider stocks, safe fallbacks)

import math, time, json
from datetime import datetime, timezone
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ----------------------------- Optional providers -----------------------------
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

PROVIDERS = dict(ALPHAVANTAGE_API_KEY=None, FINNHUB_API_KEY=None)
if "providers" in st.secrets:
    for k in PROVIDERS.keys():
        try:
            PROVIDERS[k] = st.secrets["providers"].get(k)
        except Exception:
            pass

PHASE_TAG  = "PHASE 19.0 — Hero 3× • Pills-as-Buttons • Multi-Source Stocks"
APP_NAME   = "CRYPTO HYBRID LIVE"
POWERED_BY = "POWERED BY JESSE RAY LANDINGHAM JR"

# ----------------------------- PAGE CONFIG ------------------------------------
st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state="expanded")

# ----------------------------- CSS --------------------------------------------
CSS = f"""
<style>
.block-container {{ padding-top: .25rem; padding-bottom: 1.1rem; }}

.hero-wrap {{
  position: sticky; top: 0; z-index: 999;
  background: linear-gradient(90deg, #07111d 0%, #0c2a1f 50%, #07111d 100%);
  border: 1px solid #1e3446; border-radius: 16px;
  margin: 6px 0 12px 0; padding: 18px 22px;
  box-shadow: 0 10px 26px rgba(0,0,0,.40);
}}
.hero-line {{
  display:flex; align-items:center; justify-content:center; gap:16px;
  white-space:nowrap; letter-spacing:.2px; font-weight:900; line-height:1.1;
}}
.hero-title   {{ font-size: clamp(28px, 3.2vw, 44px); color:#eafff6; text-transform:uppercase; }}
.hero-dot     {{ color:#33e29a; font-size: clamp(20px, 2.0vw, 28px); }}
.hero-powered {{
  font-size: clamp(16px, 1.6vw, 22px); text-transform:uppercase;
  color:#94d8ff; font-weight:800; background:#0a1220;
  border:1px solid #254662; padding:6px 10px; border-radius:12px;
}}

.section-title {{ font-size:24px; font-weight:800; margin: 6px 0 8px 0; }}
.metric-box {{
  border: 1px solid #ffffff22; border-radius:12px; padding: 12px; background:#0e1117; text-align:center;
}}

.pills {{ display:flex; gap:10px; flex-wrap:wrap; margin: 8px 0 10px 0; }}
.pills > div {{ flex: 1 1 220px; }}  /* equal widths on large, wraps nicely on small */
.pill {{
  width:100%; height:46px; border-radius:12px; border:1px solid #ffffff26;
  display:flex; align-items:center; justify-content:center;
  font-weight:800; user-select:none; white-space:nowrap;
}}
.pill i {{ margin-right:10px; }}
.pill-raw   {{ background:#241c14; color:#ffb185; }}
.pill-truth {{ background:#182219; color:#93ffb2; }}
.pill-conf  {{ background:#1a1a24; color:#ffe08a; }}
.pill-delta {{ background:#121a25; color:#a7d3ff; }}

.stDataFrame, .stDataEditor {{ border-radius: 10px; overflow: hidden; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------- HERO -------------------------------------------
st.markdown(
    f"""
    <div class="hero-wrap">
      <div class="hero-line">
        <span class="hero-title">{APP_NAME}</span>
        <span class="hero-dot">•</span>
        <span class="hero-powered">{POWERED_BY}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- SIDEBAR ----------------------------------------
with st.sidebar:
    st.header("Navigation")
    nav = st.radio(
        "Go to",
        ["Dashboard","Crypto","Confluence","US Market (All Listings)","S&P 500","Options","Scores","Settings"],
        index=0)

    st.header("Truth Weights")
    w_vol = st.slider("Vol/Mcap",       0.0, 1.0, 0.30, 0.05)
    w_m24 = st.slider("24h Momentum",   0.0, 1.0, 0.25, 0.05)
    w_m7  = st.slider("7d Momentum",    0.0, 1.0, 0.25, 0.05)
    w_liq = st.slider("Liquidity/Size", 0.0, 1.0, 0.20, 0.05)

    st.header("Auto Refresh")
    auto  = st.toggle("Auto refresh", value=False)
    every = st.slider("Every (sec)", 10, 120, 30, step=5)

# ----------------------------- CORE HELPERS -----------------------------------
def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def logistic_pct(x):
    try:
        return 1/(1+math.exp(-(float(x)/10.0)))
    except Exception:
        return 0.5

@st.cache_data(ttl=60, show_spinner="CoinGecko…")
def fetch_cg(vs="usd", n=200) -> pd.DataFrame:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    p = dict(vs_currency=vs, order="market_cap_desc", per_page=max(1,min(n,250)),
             page=1, sparkline="false", price_change_percentage="1h,24h,7d")
    r = requests.get(url, params=p, timeout=30); r.raise_for_status()
    df = pd.DataFrame(r.json())
    cols = ["name","symbol","current_price","market_cap","total_volume",
            "price_change_percentage_1h_in_currency",
            "price_change_percentage_24h_in_currency",
            "price_change_percentage_7d_in_currency"]
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    return df

def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    t = df.copy()
    vol = t.get("total_volume", pd.Series(0,index=t.index)).fillna(0)
    mc  = t.get("market_cap",   pd.Series(0,index=t.index)).fillna(0)
    t["vol_to_mc"] = (vol/mc).replace([np.inf,-np.inf],np.nan).clip(0,2).fillna(0)
    t["liq01"] = 0 if mc.max()==0 else (mc-mc.min())/(mc.max()-mc.min()+1e-9)

    m1h = t.get("price_change_percentage_1h_in_currency",  pd.Series(np.nan,index=t.index)).apply(logistic_pct)
    m24 = t.get("price_change_percentage_24h_in_currency", pd.Series(np.nan,index=t.index)).apply(logistic_pct)
    m7  = t.get("price_change_percentage_7d_in_currency",  pd.Series(np.nan,index=t.index)).apply(logistic_pct)

    t["raw_heat"]   = (0.5*(t["vol_to_mc"]/2).clip(0,1) + 0.5*m1h.fillna(0.5)).clip(0,1)
    t["truth_full"] = (w_vol*(t["vol_to_mc"]/2).clip(0,1) + w_m24*m24.fillna(0.5) +
                       w_m7*m7.fillna(0.5) + w_liq*t["liq01"].fillna(0)).clip(0,1)

    consistency = 1 - (m24.fillna(0.5) - m7.fillna(0.5)).abs()
    agreement   = 1 - (t["raw_heat"] - t["truth_full"]).abs()
    energy      = (t["vol_to_mc"]/2).clip(0,1)

    t["confluence01"] = (0.35*t["truth_full"] + 0.35*t["raw_heat"] + 0.10*consistency +
                         0.10*agreement + 0.05*energy + 0.05*t["liq01"]).clip(0,1)
    t["divergence"] = (t["raw_heat"] - t["truth_full"]).abs()
    return t

def kpis(df: pd.DataFrame, label: str):
    n   = len(df)
    p24 = float(df.get("price_change_percentage_24h_in_currency", pd.Series(dtype=float)).mean())
    tavg= float(df.get("truth_full",  pd.Series(dtype=float)).mean())
    ravg= float(df.get("raw_heat",   pd.Series(dtype=float)).mean())
    cavg= float(df.get("confluence01", pd.Series(dtype=float)).mean())
    a,b,c,d,e = st.columns(5)
    with a: st.markdown(f"<div class='metric-box'><b>Assets</b><br>{n}</div>", unsafe_allow_html=True)
    with b: st.markdown(f"<div class='metric-box'><b>Avg 24h %</b><br>{0 if np.isnan(p24) else p24:.2f}%</div>", unsafe_allow_html=True)
    with c: st.markdown(f"<div class='metric-box'><b>Avg TRUTH</b><br>{0 if np.isnan(tavg) else tavg:.2f}</div>", unsafe_allow_html=True)
    with d: st.markdown(f"<div class='metric-box'><b>Avg RAW</b><br>{0 if np.isnan(ravg) else ravg:.2f}</div>", unsafe_allow_html=True)
    with e: st.markdown(f"<div class='metric-box'><b>Avg Confluence</b><br>{0 if np.isnan(cavg) else cavg:.2f}</div>", unsafe_allow_html=True)
    st.caption(f"{PHASE_TAG} • Updated {now_utc()} • Mode: {label}")

def table(df: pd.DataFrame, cols: List[str]):
    have = [c for c in cols if c in df.columns]
    st.dataframe(
        df[have], use_container_width=True, hide_index=True,
        column_config={
            "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "market_cap": st.column_config.NumberColumn("Mkt Cap", format="$%d"),
            "total_volume": st.column_config.NumberColumn("Volume", format="$%d"),
            "price_change_percentage_24h_in_currency": st.column_config.NumberColumn("24h %", format="%.2f%%"),
            "raw_heat": st.column_config.ProgressColumn("RAW", min_value=0.0, max_value=1.0),
            "truth_full": st.column_config.ProgressColumn("TRUTH", min_value=0.0, max_value=1.0),
            "confluence01": st.column_config.ProgressColumn("CONF", min_value=0.0, max_value=1.0),
            "divergence": st.column_config.ProgressColumn("Δ", min_value=0.0, max_value=1.0),
            "name": st.column_config.TextColumn("Name"),
            "symbol": st.column_config.TextColumn("Symbol"),
            "Sector": st.column_config.TextColumn("Sector"),
            "ListingExchange": st.column_config.TextColumn("Exchange"),
        }
    )

# ----------------------------- Pills-as-Buttons -------------------------------
if "rank_by" not in st.session_state:
    st.session_state["rank_by"] = "confluence01"

def pills_row():
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        if st.button("🔥 RAW", use_container_width=True):
            st.session_state["rank_by"] = "raw_heat"
    with c2:
        if st.button("💧 TRUTH", use_container_width=True):
            st.session_state["rank_by"] = "truth_full"
    with c3:
        if st.button("⭐ CONFLUENCE", use_container_width=True):
            st.session_state["rank_by"] = "confluence01"
    with c4:
        if st.button("⚡ Δ (RAW→TRUTH)", use_container_width=True):
            st.session_state["rank_by"] = "divergence"

# ----------------------------- Dashboard / Crypto -----------------------------
def page_dashboard():
    st.markdown("<div class='section-title'>Dashboard</div>", unsafe_allow_html=True)
    df = build_scores(fetch_cg("usd", 200))
    kpis(df, "Crypto")

    pills_row()
    order = st.session_state["rank_by"]
    top = df.sort_values(order, ascending=False).head(20)
    st.bar_chart(top.set_index("symbol")[order], use_container_width=True)

    L, R = st.columns(2)
    with L:
        st.subheader("Top Confluence (Crypto)")
        table(df.sort_values("confluence01", ascending=False).head(20),
              ["name","symbol","current_price","confluence01","truth_full","raw_heat","divergence"])
    with R:
        st.subheader("Top TRUTH (Crypto)")
        table(df.sort_values("truth_full", ascending=False).head(20),
              ["name","symbol","current_price","truth_full","raw_heat","confluence01","divergence"])

def page_crypto():
    st.markdown("<div class='section-title'>Crypto</div>", unsafe_allow_html=True)
    df = build_scores(fetch_cg("usd", 200))
    kpis(df, "Crypto")
    pills_row()
    order = st.session_state["rank_by"]
    table(df.sort_values(order, ascending=False),
          ["name","symbol","current_price","confluence01","truth_full","raw_heat","divergence"])

def page_confluence():
    st.markdown("<div class='section-title'>Confluence</div>", unsafe_allow_html=True)
    df = build_scores(fetch_cg("usd", 200))
    kpis(df, "Confluence")
    table(df.sort_values("confluence01", ascending=False).head(50),
          ["name","symbol","current_price","confluence01","truth_full","raw_heat","divergence"])

# ----------------------------- Stocks: multi-provider -------------------------
def _yf_snapshot_daily(tickers: List[str]) -> pd.DataFrame:
    if not HAS_YF or not tickers: return pd.DataFrame()
    T = [t.strip().upper() for t in tickers if t.strip()]
    if not T: return pd.DataFrame()
    try:
        data = yf.download(" ".join(T), period="5d", interval="1d",
                           group_by="ticker", auto_adjust=True, threads=True, progress=False)
    except Exception:
        return pd.DataFrame()
    rows = []
    for t in T:
        try:
            s = data[t]
            last = float(s.iloc[-1]["Close"])
            prev = float(s.iloc[-2]["Close"]) if len(s) >= 2 else np.nan
            pct24 = (last/prev - 1.0)*100.0 if pd.notna(prev) else np.nan
            rows.append({"name": t, "symbol": t, "current_price": last,
                         "price_change_percentage_24h_in_currency": pct24})
        except Exception:
            rows.append({"name": t, "symbol": t, "current_price": np.nan,
                         "price_change_percentage_24h_in_currency": np.nan})
    return pd.DataFrame(rows)

def _alpha_snapshot_daily(tickers: List[str], key: str) -> pd.DataFrame:
    if not key or not tickers: return pd.DataFrame()
    base = "https://www.alphavantage.co/query"
    out = []
    for t in tickers[:120]:
        try:
            p = {"function":"TIME_SERIES_DAILY_ADJUSTED","symbol":t,"apikey":key}
            r = requests.get(base, params=p, timeout=20); r.raise_for_status()
            js = r.json()
            ts = js.get("Time Series (Daily)", {})
            if not ts: continue
            dates = sorted(ts.keys())
            last = float(ts[dates[-1]]["5. adjusted close"])
            prev = float(ts[dates[-2]]["5. adjusted close"]) if len(dates)>=2 else np.nan
            pct24 = (last/prev - 1.0)*100.0 if pd.notna(prev) else np.nan
            out.append({"name":t,"symbol":t,"current_price":last,"price_change_percentage_24h_in_currency":pct24})
        except Exception:
            pass
    return pd.DataFrame(out)

def _finnhub_snapshot_daily(tickers: List[str], key: str) -> pd.DataFrame:
    if not key or not tickers: return pd.DataFrame()
    url = "https://finnhub.io/api/v1/quote"
    out = []
    for t in tickers[:120]:
        try:
            r = requests.get(url, params={"symbol":t,"token":key}, timeout=15); r.raise_for_status()
            js = r.json()
            last = float(js.get("c", np.nan)); prev = float(js.get("pc", np.nan))
            pct24 = (last/prev - 1.0)*100.0 if pd.notna(prev) else np.nan
            out.append({"name":t,"symbol":t,"current_price":last,"price_change_percentage_24h_in_currency":pct24})
        except Exception:
            pass
    return pd.DataFrame(out)

@st.cache_data(ttl=180, show_spinner="Stock snapshots…")
def stocks_snapshot(tickers: List[str]) -> Tuple[pd.DataFrame, str]:
    # Try yfinance → Alpha → Finnhub
    if HAS_YF:
        df = _yf_snapshot_daily(tickers)
        if not df.empty: return df, "yfinance"
    if PROVIDERS["ALPHAVANTAGE_API_KEY"]:
        df = _alpha_snapshot_daily(tickers, PROVIDERS["ALPHAVANTAGE_API_KEY"])
        if not df.empty: return df, "AlphaVantage"
    if PROVIDERS["FINNHUB_API_KEY"]:
        df = _finnhub_snapshot_daily(tickers, PROVIDERS["FINNHUB_API_KEY"])
        if not df.empty: return df, "Finnhub"
    return pd.DataFrame(), "none"

# Constituents
@st.cache_data(ttl=600, show_spinner="Loading S&P 500…")
def fetch_sp500() -> pd.DataFrame:
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
        df = pd.read_csv(pd.compat.StringIO(txt), sep="|")
        df = df[~df["Symbol"].str.contains("File Creation Time", na=False)]
        df = df.rename(columns={"Symbol":"symbol","Security Name":"name","ETF":"ETF"})
        df["ListingExchange"] = "NASDAQ"
        frames.append(df[["symbol","name","ListingExchange","ETF"]])
    except Exception:
        pass
    try:
        txt = requests.get(OTHER_LISTED_URL, timeout=30).text
        df = pd.read_csv(pd.compat.StringIO(txt), sep="|")
        df = df[~df["ACT Symbol"].str.contains("File Creation Time", na=False)]
        df = df.rename(columns={"ACT Symbol":"symbol","Security Name":"name","Exchange":"ListingExchange","ETF":"ETF"})
        frames.append(df[["symbol","name","ListingExchange","ETF"]])
    except Exception:
        pass
    if not frames:
        return pd.DataFrame(columns=["symbol","name","ListingExchange","ETF"])
    allu = pd.concat(frames, ignore_index=True)
    allu["symbol"] = allu["symbol"].astype(str).str.upper()
    allu["ETF"] = allu["ETF"].astype(str).str.upper()
    allu = allu[allu["symbol"].str.match(r"^[A-Z0-9\.-]+$", na=False)]
    return allu.drop_duplicates(subset=["symbol"])

# Pages
def page_sp500():
    st.markdown("<div class='section-title'>S&P 500</div>", unsafe_allow_html=True)
    base = fetch_sp500()
    if base.empty:
        st.warning("Couldn’t fetch S&P 500 constituents.")
        return
    limit = st.slider("Tickers to snapshot", 50, len(base), 200, step=50)
    tickers = base["symbol"].tolist()[:limit]
    snap, src = stocks_snapshot(tickers)
    if snap.empty:
        st.error("No stock prices from providers (yfinance / Alpha / Finnhub). Add a key or retry.")
        return
    st.caption(f"Provider: {src} • Constituents {len(base)} • {now_utc()}")
    scored = build_scores(snap.merge(base, on="symbol", how="left"))
    kpis(scored, "S&P 500")
    table(scored.sort_values("confluence01", ascending=False).head(100),
          ["name","symbol","Sector","current_price","confluence01","truth_full","raw_heat","divergence"])

def page_us_all():
    st.markdown("<div class='section-title'>US Market (All Listings)</div>", unsafe_allow_html=True)
    uni = fetch_us_listings()
    if uni.empty:
        st.warning("US listings unavailable.")
        return
    q = st.text_input("Search by symbol or name").strip().lower()
    exch = st.selectbox("Exchange", ["(All)"] + sorted(uni["ListingExchange"].dropna().unique().tolist()), index=0)
    include_etf = st.toggle("Include ETFs", value=False)
    limit = st.slider("Tickers to snapshot", 50, 1000, 200, step=50)

    dfu = uni.copy()
    if not include_etf and "ETF" in dfu.columns:
        dfu = dfu[dfu["ETF"] != "Y"]
    if exch != "(All)":
        dfu = dfu[dfu["ListingExchange"] == exch]
    if q:
        mask = dfu["symbol"].str.lower().str.contains(q, na=False) | dfu["name"].str.lower().str.contains(q, na=False)
        dfu = dfu[mask]

    if len(dfu)==0:
        st.info("No matches."); return

    tickers = dfu["symbol"].tolist()[:limit]
    snap, src = stocks_snapshot(tickers)
    if snap.empty:
        st.error("No stock prices from providers (yfinance / Alpha / Finnhub). Add a key or retry.")
        return
    st.caption(f"Provider: {src} • Universe matches {len(dfu)} • {now_utc()}")
    scored = build_scores(snap.merge(dfu, on="symbol", how="left"))
    kpis(scored, "US Market")
    table(scored.sort_values("confluence01", ascending=False).head(100),
          ["name","symbol","ListingExchange","current_price","confluence01","truth_full","raw_heat","divergence"])

# ----------------------------- Options (yfinance only) ------------------------
@st.cache_data(ttl=120)
def list_expirations(ticker: str) -> List[str]:
    if not HAS_YF or not ticker: return []
    try: return list(yf.Ticker(ticker).options)
    except Exception: return []

@st.cache_data(ttl=180, show_spinner="Option chain…")
def load_options_chain(ticker: str, expiration: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not HAS_YF or not ticker or not expiration: return pd.DataFrame(), pd.DataFrame()
    try:
        ch = yf.Ticker(ticker).option_chain(expiration)
        keep = ["contractSymbol","strike","lastPrice","openInterest","volume","impliedVolatility"]
        calls = ch.calls[[c for c in keep if c in ch.calls.columns]]
        puts  = ch.puts [[c for c in keep if c in ch.puts.columns]]
        return calls, puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

def page_options():
    st.markdown("<div class='section-title'>Options</div>", unsafe_allow_html=True)
    if not HAS_YF:
        st.error("Options require yfinance. Add `yfinance` to requirements.txt.")
        return
    base = fetch_us_listings()
    default = "AAPL"
    sym = st.selectbox("Ticker", options=[default] + (base["symbol"].tolist() if not base.empty else []), index=0)
    exps = list_expirations(sym)
    if not exps:
        st.warning("No expirations returned."); return
    exp = st.selectbox("Expiration", options=exps, index=0)
    calls, puts = load_options_chain(sym, exp)
    L,R = st.columns(2)
    with L:
        st.subheader(f"{sym} Calls — {exp}")
        if calls.empty: st.info("No calls data.")
        else:
            keep = ["contractSymbol","strike","lastPrice","openInterest","volume","impliedVolatility"]
            table(calls.sort_values(["openInterest","volume"], ascending=False).head(25), keep)
    with R:
        st.subheader(f"{sym} Puts — {exp}")
        if puts.empty: st.info("No puts data.")
        else:
            keep = ["contractSymbol","strike","lastPrice","openInterest","volume","impliedVolatility"]
            table(puts.sort_values(["openInterest","volume"], ascending=False).head(25), keep)

# ----------------------------- Scores / Settings ------------------------------
def page_scores():
    st.markdown("<div class='section-title'>Scores — Explainer</div>", unsafe_allow_html=True)
    st.markdown("""
**RAW (0..1)** — crowd heat now (volume/market-cap + 1h momentum).  
**TRUTH (0..1)** — stability (vol/mcap, 24h, 7d, liquidity).  
**Δ (0..1)** — |RAW − TRUTH| (gap).  
**CONFLUENCE (0..1)** — RAW & TRUTH agree + trend consistency + energy + liquidity.

**Read fast**
- ⭐ High Confluence → hype and quality aligned (prime)  
- 🔥 High RAW + low 💧 TRUTH → hype spike (fragile)  
- 💧 High TRUTH + low 🔥 RAW → sleeper quality (crowd not there yet)
""")

def page_settings():
    st.markdown("<div class='section-title'>Settings</div>", unsafe_allow_html=True)
    live = {
        "yfinance": HAS_YF,
        "AlphaVantage": bool(PROVIDERS["ALPHAVANTAGE_API_KEY"]),
        "Finnhub": bool(PROVIDERS["FINNHUB_API_KEY"]),
    }
    st.write("Providers:", live)
    st.caption("Add API keys under Settings → Secrets to enable more data sources.")

# ----------------------------- Router -----------------------------------------
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
elif nav == "Scores":
    page_scores()
else:
    page_settings()

# ----------------------------- Auto refresh -----------------------------------
if auto:
    st.caption(f"{PHASE_TAG} • Auto refresh every {int(every)}s")
    time.sleep(max(5, int(every)))
    st.rerun()
