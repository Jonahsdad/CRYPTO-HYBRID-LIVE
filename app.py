# build: HERO 3x, Truth Pills, Pill-driven chart, Multi-source Stocks, S&P500, US Market, Options
# POWERED BY JESSE RAY LANDINGHAM JR

from __future__ import annotations
import math, time, io, os
from datetime import datetime, timezone
from typing import List
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Optional yfinance (for options + last-resort prices)
HAS_YF = False
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

APP_NAME   = "CRYPTO HYBRID LIVE"
POWERED_BY = "POWERED BY JESSE RAY LANDINGHAM JR"
PHASE_TAG  = "Phase 19.5 — Hero 3× + Pills + Multi-Source Stocks + Options"

st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
:root{
  --bg:#0b1220; --panel:#0f172a; --ink:#dbeafe; --muted:#9fb3c8;
  --accent:#21d07a; --raw:#ff9b63; --truth:#66d5ff; --conf:#ffd86b; --delta:#b3a4ff;
}
.block-container{padding-top:1.4rem; padding-bottom:1.4rem; max-width:1400px;}
/* HERO */
.hero-wrap{position:sticky; top:0; z-index:9; padding:14px 0 14px 0;
  background:linear-gradient(180deg,rgba(11,18,32,.96),rgba(11,18,32,.85) 60%,rgba(11,18,32,0));}
.hero{
  border:1px solid #28425a; background:linear-gradient(90deg,#07211b 0%, #093e2b 50%, #07211b 100%);
  border-radius:16px; padding:22px 26px; color:#c6f6e1; font-weight:900; letter-spacing:.8px;
  font-size:34px; display:flex; align-items:center; justify-content:space-between; box-shadow:0 10px 30px #0008;
}
.hero small{color:#9ad7ff; font-size:16px; font-weight:800; letter-spacing:.6px}
/* KPIs */
.kpi{border:1px solid #23324a; background:#0f172a; border-radius:12px; padding:12px 14px; color:#dbeafe;}
.kpi b{font-size:13px; color:#9fb3c8;}
/* Pills */
.badge-row{display:flex; gap:12px; margin:8px 0 6px 0; flex-wrap:wrap;}
.pill{
  user-select:none; cursor:pointer; border-radius:14px; padding:14px 18px; font-weight:900;
  display:flex; align-items:center; gap:10px; font-size:18px; letter-spacing:.3px; border:1px solid #ffffff20;
  background:linear-gradient(180deg,#1a2333,#111827); color:#dbeafe; box-shadow: inset 0 0 0 1px #ffffff12;
}
.pill.active{box-shadow:0 6px 20px #0007, inset 0 0 0 1px #ffffff33}
.pill.raw{background:linear-gradient(180deg,#3a2414,#1b130d); border-color:#ff9b6335;}
.pill.truth{background:linear-gradient(180deg,#102433,#0b1a24); border-color:#66d5ff33;}
.pill.conf{background:linear-gradient(180deg,#2b2511,#1e1809); border-color:#ffd86b33;}
.pill.delta{background:linear-gradient(180deg,#231a37,#151026); border-color:#b3a4ff33;}
/* Legend + tables */
.legend{border:1px dashed #334155; background:#0e1624; border-radius:12px; padding:8px 12px; color:#9fb3c8; font-size:13px;}
.stDataFrame, .stDataEditor{border-radius:12px; overflow:hidden;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Navigation")
    nav = st.radio("Go to", [
        "Dashboard","Crypto","Confluence","US Market (All Listings)","S&P 500",
        "Options","Fusion","Export","Scores","Settings"
    ], index=0)
    st.header("Truth Weights")
    w_vol = st.slider("Vol/Mcap", 0.0, 1.0, 0.30, 0.01)
    w_m24 = st.slider("24h Momentum", 0.0, 1.0, 0.25, 0.01)
    w_m7  = st.slider("7d Momentum", 0.0, 1.0, 0.25, 0.01)
    w_liq = st.slider("Liquidity/Size", 0.0, 1.0, 0.20, 0.01)
    st.header("Auto Refresh")
    auto  = st.toggle("Auto refresh", value=False)
    every = st.slider("Every (sec)", 10, 120, 30, step=5)

def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def safe_sigmoid(p) -> float:
    if p is None or (isinstance(p, float) and np.isnan(p)): return 0.5
    try: return 1.0/(1.0+math.exp(-float(p)/10.0))
    except Exception: return 0.5

@st.cache_data(ttl=90, show_spinner="Loading crypto…")
def cg_markets(vs="usd", per_page=200) -> pd.DataFrame:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    p   = {"vs_currency":vs,"order":"market_cap_desc","per_page":int(max(1,min(per_page,250))),
           "page":1,"sparkline":"false","price_change_percentage":"1h,24h,7d","locale":"en"}
    r = requests.get(url, params=p, timeout=30); r.raise_for_status()
    df = pd.DataFrame(r.json())
    need = ["name","symbol","current_price","market_cap","total_volume",
            "price_change_percentage_1h_in_currency","price_change_percentage_24h_in_currency",
            "price_change_percentage_7d_in_currency"]
    for c in need:
        if c not in df.columns: df[c] = np.nan
    return df

def score(df: pd.DataFrame, wv=0.30, w24=0.25, w7=0.25, wl=0.20) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    t = df.copy()
    t["vol_to_mc"] = (t.get("total_volume",0)/t.get("market_cap",1)).replace([np.inf,-np.inf],np.nan).clip(0,2).fillna(0)
    m1  = t.get("price_change_percentage_1h_in_currency",  pd.Series(np.nan, index=t.index)).apply(safe_sigmoid)
    m24 = t.get("price_change_percentage_24h_in_currency", pd.Series(np.nan, index=t.index)).apply(safe_sigmoid)
    m7  = t.get("price_change_percentage_7d_in_currency",  pd.Series(np.nan, index=t.index)).apply(safe_sigmoid)
    mc  = t.get("market_cap",0).fillna(0)
    t["liq01"] = 0 if float(mc.max() or 0)==0 else ((mc-mc.min())/(mc.max()-mc.min()+1e-9)).clip(0,1)
    t["raw_heat"]   = (0.5*(t["vol_to_mc"]/2).clip(0,1) + 0.5*m1.fillna(0.5)).clip(0,1)
    t["truth_full"] = (wv*(t["vol_to_mc"]/2).clip(0,1) + w24*m24.fillna(0.5) + w7*m7.fillna(0.5) + wl*t["liq01"]).clip(0,1)
    t["consistency01"] = 1 - (m24.fillna(0.5) - m7.fillna(0.5)).abs()
    t["agreement01"]   = 1 - (t["raw_heat"] - t["truth_full"]).abs()
    t["energy01"]      = (t["vol_to_mc"]/2).clip(0,1)
    t["confluence01"]  = (0.35*t["truth_full"] + 0.35*t["raw_heat"] + 0.10*t["consistency01"] +
                          0.10*t["agreement01"] + 0.05*t["energy01"] + 0.05*t["liq01"]).clip(0,1)
    t["delta01"] = (t["raw_heat"] - t["truth_full"]).abs()
    fire = lambda v: "🔥🔥🔥" if v>=0.85 else ("🔥🔥" if v>=0.65 else ("🔥" if v>=0.45 else "·"))
    drop = lambda v: "💧💧💧" if v>=0.85 else ("💧💧" if v>=0.65 else ("💧" if v>=0.45 else "·"))
    star = lambda v: "⭐️⭐️⭐️" if v>=0.85 else ("⭐️⭐️" if v>=0.65 else ("⭐️" if v>=0.45 else "·"))
    t["RAW_BADGE"]  = t["raw_heat"].apply(fire)
    t["TRUTH_BADGE"]= t["truth_full"].apply(drop)
    t["CONF_BADGE"] = t["confluence01"].apply(star)
    return t

def _sec(name:str, default:str="") -> str:
    try: return st.secrets.get(name, default) or os.getenv(name, default)
    except Exception: return os.getenv(name, default)

@st.cache_data(ttl=120, show_spinner="Fetching stock prices…")
def get_stock_snapshot(tickers: List[str]) -> pd.DataFrame:
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers: return pd.DataFrame()

    # 0) IEX
    IEX = _sec("IEX_API_KEY")
    if IEX:
        try:
            syms = ",".join(tickers[:1000])
            r = requests.get("https://cloud.iexapis.com/stable/stock/market/batch",
                             params={"symbols":syms,"types":"quote","token":IEX}, timeout=25)
            r.raise_for_status()
            data = r.json(); rows=[]
            for t in tickers:
                q = data.get(t,{}).get("quote",{})
                if q:
                    rows.append({"symbol":t,"name":q.get("companyName",t),
                                 "current_price":q.get("latestPrice",np.nan),
                                 "price_change_percentage_24h_in_currency":np.nan,
                                 "market_cap":q.get("marketCap",np.nan),
                                 "total_volume":q.get("avgTotalVolume",np.nan)})
            if rows: return pd.DataFrame(rows)
        except Exception: pass

    # 1) Finnhub
    FINN = _sec("FINNHUB_API_KEY")
    if FINN:
        try:
            rows=[]
            for t in tickers[:400]:
                r = requests.get("https://finnhub.io/api/v1/quote", params={"symbol":t,"token":FINN}, timeout=10)
                if r.ok:
                    d=r.json()
                    last=float(d.get("c") or np.nan); prev=float(d.get("pc") or np.nan)
                    pct=(last/prev-1)*100 if (not np.isnan(last) and not np.isnan(prev) and prev>0) else np.nan
                    rows.append({"symbol":t,"name":t,"current_price":last,
                                 "price_change_percentage_24h_in_currency":pct,
                                 "market_cap":np.nan,"total_volume":np.nan})
            if rows: return pd.DataFrame(rows)
        except Exception: pass

    # 2) Alpha Vantage
    AV = _sec("ALPHAVANTAGE_API_KEY")
    if AV:
        try:
            rows=[]
            for t in tickers[:25]:
                r = requests.get("https://www.alphavantage.co/query",
                                 params={"function":"GLOBAL_QUOTE","symbol":t,"apikey":AV}, timeout=15)
                if r.ok:
                    g=r.json().get("Global Quote",{})
                    last=float(g.get("05. price") or np.nan)
                    prev=float(g.get("08. previous close") or np.nan)
                    pct=(last/prev-1)*100 if (not np.isnan(last) and not np.isnan(prev) and prev>0) else np.nan
                    rows.append({"symbol":t,"name":t,"current_price":last,
                                 "price_change_percentage_24h_in_currency":pct,
                                 "market_cap":np.nan,"total_volume":np.nan})
            if rows: return pd.DataFrame(rows)
        except Exception: pass

    # 3) yfinance (fallback)
    if HAS_YF:
        try:
            data = yf.download(" ".join(tickers[:400]), period="5d", interval="1d",
                               group_by="ticker", auto_adjust=True, threads=True, progress=False)
            rows=[]
            for t in tickers[:400]:
                try:
                    s=data[t]; last=float(s.iloc[-1]["Close"])
                    prev=float(s.iloc[-2]["Close"]) if len(s)>=2 else np.nan
                    pct=(last/prev-1)*100 if not np.isnan(prev) else np.nan
                    rows.append({"symbol":t,"name":t,"current_price":last,
                                 "price_change_percentage_24h_in_currency":pct,
                                 "market_cap":np.nan,"total_volume":np.nan})
                except Exception:
                    rows.append({"symbol":t,"name":t,"current_price":np.nan,
                                 "price_change_percentage_24h_in_currency":np.nan,
                                 "market_cap":np.nan,"total_volume":np.nan})
            return pd.DataFrame(rows)
        except Exception: pass

    return pd.DataFrame(columns=["symbol","name","current_price"])

@st.cache_data(ttl=600, show_spinner="Loading S&P 500…")
def sp500_list() -> pd.DataFrame:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0].rename(columns={"Symbol":"symbol","Security":"name","GICS Sector":"Sector"})
        df["symbol"]=df["symbol"].astype(str).str.upper().str.replace(".","-",regex=False)
        return df[["symbol","name","Sector"]]
    except Exception:
        return pd.DataFrame(columns=["symbol","name","Sector"])

NASDAQ_LISTED_URL = "https://ftp.nasdaqtrader.com/dynamic/SYMBOL_DIRECTORY/nasdaqlisted.txt"
OTHER_LISTED_URL  = "https://ftp.nasdaqtrader.com/dynamic/SYMBOL_DIRECTORY/otherlisted.txt"

@st.cache_data(ttl=900, show_spinner="Loading US listings…")
def us_listings() -> pd.DataFrame:
    frames=[]
    try:
        txt=requests.get(NASDAQ_LISTED_URL, timeout=30).text
        df=pd.read_csv(io.StringIO(txt), sep="|")
        df=df[~df["Symbol"].str.contains("File Creation Time", na=False)]
        df=df.rename(columns={"Symbol":"symbol","Security Name":"name","ETF":"ETF","Test Issue":"TestIssue"})
        df["ListingExchange"]="NASDAQ"; frames.append(df[["symbol","name","ListingExchange","ETF","TestIssue"]])
    except Exception: pass
    try:
        txt=requests.get(OTHER_LISTED_URL, timeout=30).text
        df=pd.read_csv(io.StringIO(txt), sep="|")
        df=df[~df["ACT Symbol"].str.contains("File Creation Time", na=False)]
        df=df.rename(columns={"ACT Symbol":"symbol","Security Name":"name","Exchange":"ListingExchange","ETF":"ETF","Test Issue":"TestIssue"})
        frames.append(df[["symbol","name","ListingExchange","ETF","TestIssue"]])
    except Exception: pass
    if not frames: return pd.DataFrame(columns=["symbol","name","ListingExchange","ETF","TestIssue"])
    allu=pd.concat(frames, ignore_index=True)
    allu["symbol"]=allu["symbol"].astype(str).str.upper()
    allu=allu[allu["TestIssue"].astype(str).str.upper()!="Y"]
    allu=allu[allu["symbol"].str.match(r"^[A-Z0-9\.-]+$", na=False)]
    return allu.drop_duplicates(subset=["symbol"])

def hero():
    st.markdown(
        f"""
        <div class="hero-wrap">
          <div class="hero">
            <div>{APP_NAME}</div>
            <div><small>{POWERED_BY}</small></div>
          </div>
        </div>
        """, unsafe_allow_html=True
    )

def kpis(df: pd.DataFrame, label:str):
    n=len(df)
    p24=float(df.get("price_change_percentage_24h_in_currency", pd.Series(dtype=float)).mean())
    tavg=float(df.get("truth_full", pd.Series(dtype=float)).mean())
    ravg=float(df.get("raw_heat", pd.Series(dtype=float)).mean())
    cavg=float(df.get("confluence01", pd.Series(dtype=float)).mean())
    c1,c2,c3,c4,c5=st.columns(5)
    with c1: st.markdown(f'<div class="kpi"><b>Assets</b><br>{n}</div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi"><b>Avg 24h %</b><br>{0 if np.isnan(p24) else p24:.2f}%</div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi"><b>Avg TRUTH</b><br>{0 if np.isnan(tavg) else tavg:.2f}</div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="kpi"><b>Avg RAW</b><br>{0 if np.isnan(ravg) else ravg:.2f}</div>', unsafe_allow_html=True)
    with c5: st.markdown(f'<div class="kpi"><b>Avg Confluence</b><br>{0 if np.isnan(cavg) else cavg:.2f}</div>', unsafe_allow_html=True)
    st.caption(f"{PHASE_TAG} • Updated {now_utc()} • Mode: {label}")

def legend_strip():
    st.markdown(
        """
        <div class="legend">
        🔥 RAW = crowd heat (vol/mcap + 1h) &nbsp;&nbsp;|&nbsp;&nbsp;
        💧 TRUTH = stability (vol/mcap + 24h + 7d + size) &nbsp;&nbsp;|&nbsp;&nbsp;
        ⭐ CONFLUENCE = RAW + TRUTH agree, consistent (24h~7d), energetic & liquid &nbsp;&nbsp;|&nbsp;&nbsp;
        ⚡ Δ = |RAW − TRUTH| (gap)
        </div>
        """, unsafe_allow_html=True
    )

def chart_bar(df: pd.DataFrame, metric:str, topn:int=20):
    import plotly.express as px
    key = {"raw":"raw_heat","truth":"truth_full","conf":"confluence01","delta":"delta01"}.get(metric,"confluence01")
    d=df.sort_values(key, ascending=False).head(topn)
    fig = px.bar(d, x="symbol", y=key, hover_name="name", height=360)
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)

def table(df:pd.DataFrame, cols:List[str], title:str):
    st.subheader(title)
    keep=[c for c in cols if c in df.columns]
    st.dataframe(
        df[keep], use_container_width=True, hide_index=True,
        column_config={
            "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "market_cap": st.column_config.NumberColumn("Mkt Cap", format="$%d"),
            "total_volume": st.column_config.NumberColumn("Volume", format="$%d"),
            "truth_full": st.column_config.ProgressColumn("TRUTH", min_value=0.0, max_value=1.0),
            "raw_heat": st.column_config.ProgressColumn("RAW", min_value=0.0, max_value=1.0),
            "confluence01": st.column_config.ProgressColumn("CONF", min_value=0.0, max_value=1.0),
            "delta01": st.column_config.ProgressColumn("Δ", min_value=0.0, max_value=1.0),
            "TRUTH_BADGE": st.column_config.TextColumn("💧"),
            "RAW_BADGE": st.column_config.TextColumn("🔥"),
            "CONF_BADGE": st.column_config.TextColumn("⭐"),
        },
    )

def pill_metric_state():
    if "metric" not in st.session_state: st.session_state.metric="conf"
    cols=st.columns(4)
    with cols[0]:
        if st.button("🔥 RAW"): st.session_state.metric="raw"
        st.markdown('<div class="pill raw'+(' active' if st.session_state.metric=="raw" else '')+'">🔥 RAW</div>', unsafe_allow_html=True)
    with cols[1]:
        if st.button("💧 TRUTH"): st.session_state.metric="truth"
        st.markdown('<div class="pill truth'+(' active' if st.session_state.metric=="truth" else '')+'">💧 TRUTH</div>', unsafe_allow_html=True)
    with cols[2]:
        if st.button("⭐ CONFLUENCE"): st.session_state.metric="conf"
        st.markdown('<div class="pill conf'+(' active' if st.session_state.metric=="conf" else '')+'">⭐ CONFLUENCE</div>', unsafe_allow_html=True)
    with cols[3]:
        if st.button("⚡ Δ (RAW↔TRUTH)"): st.session_state.metric="delta"
        st.markdown('<div class="pill delta'+(' active' if st.session_state.metric=="delta" else '')+'">⚡ Δ (RAW↔TRUTH)</div>', unsafe_allow_html=True)
    return st.session_state.metric

def page_dashboard():
    hero()
    dfc=score(cg_markets(per_page=200), w_vol,w_m24,w_m7,w_liq)
    kpis(dfc,"Crypto")
    legend_strip()
    def chart_bar(df: pd.DataFrame, metric: str, topn: int = 20):
    try:
        import plotly.express as px
    except ModuleNotFoundError:
        st.warning("Plotly not installed — charts disabled. Add `plotly>=5.22` to requirements.txt.")
        return
    key = {"raw": "raw_heat", "truth": "truth_full", "conf": "confluence01", "delta": "delta01"}.get(metric, "confluence01")
    d = df.sort_values(key, ascending=False).head(topn)
    fig = px.bar(d, x="symbol", y=key, hover_name="name", height=360)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    metric=pill_metric_state()
    chart_bar(dfc, metric, topn=22)
    c1,c2=st.columns(2)
    with c1:
        table(dfc.sort_values("confluence01",ascending=False).head(30),
              ["name","symbol","current_price","CONF_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE"], "Top Confluence (Crypto)")
    with c2:
        table(dfc.sort_values("truth_full",ascending=False).head(30),
              ["name","symbol","current_price","TRUTH_BADGE","truth_full","RAW_BADGE","delta01"], "Top TRUTH (Crypto)")

def page_crypto():
    hero(); legend_strip()
    topn=st.slider("Show Top N", 50, 250, 200, 10)
    order=st.selectbox("Order by", ["confluence01","truth_full","raw_heat","delta01"], index=0)
    df=score(cg_markets(per_page=topn), w_vol,w_m24,w_m7,w_liq)
    kpis(df,"Crypto"); chart_bar(df,"conf",20)
    table(df.sort_values(order,ascending=False).head(topn),
          ["name","symbol","current_price","CONF_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","delta01"], "Leaders")

def page_confluence():
    hero(); legend_strip()
    df=score(cg_markets(per_page=200), w_vol,w_m24,w_m7,w_liq)
    kpis(df,"Confluence"); chart_bar(df,"conf",20)
    table(df.sort_values("confluence01",ascending=False).head(60),
          ["name","symbol","current_price","CONF_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","delta01"], "Top Confluence")

def page_sp500():
    hero(); legend_strip()
    base=sp500_list()
    if base.empty: st.warning("Couldn’t fetch S&P list; try later."); return
    limit=st.slider("Tickers to snapshot", 50, len(base), 200, 50)
    tickers=base["symbol"].tolist()[:limit]
    snap=get_stock_snapshot(tickers)
    if snap.empty: st.error("No stock data (add IEX/FINNHUB/ALPHAVANTAGE secrets)."); return
    df=score(snap.merge(base,on="symbol",how="left"), w_vol,w_m24,w_m7,w_liq)
    kpis(df,"S&P 500"); chart_bar(df,"conf",20)
    table(df.sort_values("confluence01",ascending=False).head(100),
          ["name","symbol","Sector","current_price","CONF_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","delta01"], "Top S&P Confluence")

def page_us_all():
    hero(); legend_strip()
    uni=us_listings()
    if uni.empty: st.warning("US listings unavailable."); return
    c1,c2,c3,c4=st.columns([2,1,1,1])
    with c1: q=st.text_input("Search symbol/name","").strip().lower()
    with c2: exch=st.selectbox("Exchange", ["(All)"]+sorted(uni["ListingExchange"].dropna().unique().tolist()))
    with c3: include_etf=st.toggle("Include ETFs", False)
    with c4: limit=st.slider("Tickers to snapshot",50,1000,200,50)
    dfu=uni.copy()
    if not include_etf and "ETF" in dfu.columns: dfu=dfu[dfu["ETF"]!="Y"]
    if exch!="(All)": dfu=dfu[dfu["ListingExchange"]==exch]
    if q: dfu=dfu[dfu["symbol"].str.lower().str.contains(q) | dfu["name"].str.lower().str.contains(q)]
    st.caption(f"Universe matches: {len(dfu)} • Snapshot first {min(limit,len(dfu))}")
    if len(dfu)==0: return
    snap=get_stock_snapshot(dfu["symbol"].tolist()[:limit])
    if snap.empty: st.error("No stock data from providers."); return
    df=score(snap.merge(dfu,on="symbol",how="left"), w_vol,w_m24,w_m7,w_liq)
    kpis(df,"US Market"); chart_bar(df,"conf",20)
    cL,cR=st.columns(2)
    with cL:
        table(df.sort_values("confluence01",ascending=False).head(80),
              ["name","symbol","ListingExchange","ETF","current_price","CONF_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","delta01"], "Top Confluence ⭐")
    with cR:
        table(df.sort_values("truth_full",ascending=False).head(80),
              ["name","symbol","ListingExchange","ETF","current_price","TRUTH_BADGE","truth_full","RAW_BADGE","delta01"], "Top TRUTH 💧")

def page_options():
    hero()
    if not HAS_YF:
        st.error("Options require yfinance (add to requirements)."); return
    base=us_listings()
    sym=st.text_input("Ticker (e.g., AAPL)","AAPL").strip().upper() if base.empty else \
        st.selectbox("Ticker", options=["AAPL"]+base["symbol"].tolist(), index=0)
    if not sym: return
    try: exps=list(yf.Ticker(sym).options)
    except Exception: exps=[]
    if not exps: st.warning("No option expirations returned."); return
    exp=st.selectbox("Expiration", exps, index=0)
    try:
        chain=yf.Ticker(sym).option_chain(exp)
        calls,puts=chain.calls.copy(), chain.puts.copy()
    except Exception:
        calls,puts=pd.DataFrame(), pd.DataFrame()
    keep=["contractSymbol","strike","lastPrice","openInterest","volume","impliedVolatility"]
    c1,c2=st.columns(2)
    with c1:
        st.subheader(f"{sym} Calls — {exp}")
        if calls.empty: st.info("No calls data.")
        else: table(calls.sort_values(["openInterest","volume"],ascending=False).head(25),
                    [c for c in keep if c in calls.columns], "Calls")
    with c2:
        st.subheader(f"{sym} Puts — {exp}")
        if puts.empty: st.info("No puts data.")
        else: table(puts.sort_values(["openInterest","volume"],ascending=False).head(25),
                    [c for c in keep if c in puts.columns], "Puts")

def page_fusion():
    hero(); legend_strip()
    dfc=score(cg_markets(per_page=120), w_vol,w_m24,w_m7,w_liq); dfc["universe"]="CRYPTO"
    uni=us_listings()
    if not uni.empty:
        snap=get_stock_snapshot((uni[uni["ETF"]!="Y"]["symbol"] if "ETF" in uni.columns else uni["symbol"]).tolist()[:120])
        dfs=score(snap.merge(uni,on="symbol",how="left"), w_vol,w_m24,w_m7,w_liq) if not snap.empty else pd.DataFrame()
        dfs["universe"]="US"
    else:
        dfs=pd.DataFrame(columns=dfc.columns)
    both=pd.concat([dfc,dfs], ignore_index=True) if not dfs.empty else dfc
    kpis(both,"Fusion"); chart_bar(both,"conf",20)
    table(both.sort_values("confluence01",ascending=False).head(40),
          ["name","symbol","current_price","CONF_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","delta01","universe"], "Universal Confluence")

def page_export():
    hero()
    dfc=score(cg_markets(per_page=200), w_vol,w_m24,w_m7,w_liq)
    st.download_button("Download Crypto CSV", dfc.to_csv(index=False).encode("utf-8"),
                       file_name="crypto_truth_raw_conf.csv", mime="text/csv")
    uni=us_listings()
    if not uni.empty:
        snap=get_stock_snapshot(uni["symbol"].tolist()[:400])
        if not snap.empty:
            dfs=score(snap.merge(uni,on="symbol",how="left"), w_vol,w_m24,w_m7,w_liq)
            st.download_button("Download US Snapshot CSV", dfs.to_csv(index=False).encode("utf-8"),
                               file_name="us_truth_raw_conf.csv", mime="text/csv")

def page_scores():
    hero()
    st.markdown("### Scores — Explainer")
    st.markdown("""
- **🔥 RAW (0..1)** — crowd heat now *(volume/market-cap + 1h momentum)*
- **💧 TRUTH (0..1)** — stability *(vol/mcap + 24h + 7d + size)*
- **⭐ CONFLUENCE (0..1)** — RAW + TRUTH agree, consistent (24h~7d), energetic & liquid
- **⚡ Δ (0..1)** — absolute gap `|RAW − TRUTH|` *(higher = bigger mismatch)*
**Read fast:** ⭐ prime; 🔥>💧 spike; 💧>🔥 sleeper.
""")

def page_settings():
    hero(); st.caption("Saved presets & personalization coming soon.")

# Router
if   nav=="Dashboard": page_dashboard()
elif nav=="Crypto": page_crypto()
elif nav=="Confluence": page_confluence()
elif nav=="US Market (All Listings)": page_us_all()
elif nav=="S&P 500": page_sp500()
elif nav=="Options": page_options()
elif nav=="Fusion": page_fusion()
elif nav=="Export": page_export()
elif nav=="Scores": page_scores()
else: page_settings()

if auto:
    st.caption(f"{PHASE_TAG} • {POWERED_BY} • Auto refresh: {every}s")
    time.sleep(max(5, int(every))); st.rerun()
