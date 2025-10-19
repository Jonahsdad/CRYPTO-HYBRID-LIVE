# app.py — Crypto Hybrid Live v18.5
# Education only — not financial advice.

from __future__ import annotations

# ----- stdlib
import io, math, time
from datetime import datetime, timezone
from typing import List, Tuple, Optional

# ----- third-party
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# Optional stocks/options provider
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

PHASE_TAG  = "PHASE 18.5 — Sticky Hero + Interactive Buttons + Stocks Fallback"
APP_NAME   = "CRYPTO HYBRID LIVE"
POWERED_BY = "POWERED BY JESSE RAY LANDINGHAM JR"

# ============================== PAGE CONFIG ==================================
st.set_page_config(page_title="Crypto Hybrid Live", layout="wide", initial_sidebar_state="expanded")

# ============================== CSS ==========================================
st.markdown("""
<style>
:root{ --bg:#0b1220; --card:#0f172a; --edge:#20324a; --ink:#e7f0ff; --muted:#8aa4c6; }
.block-container{ padding-top:.6rem; padding-bottom:.6rem; }

/* Sticky single-line HERO (shared spot) */
.hero-wrap{ position:sticky; top:0; z-index:9999; margin:0 0 10px 0; }
.hero{
  padding:10px 14px; border-radius:12px;
  background:linear-gradient(90deg,#101b31 0%,#0b2a1d 55%,#101b31 100%);
  border:1px solid #20324a; box-shadow:0 2px 10px rgba(0,0,0,.25);
}
.hero-line{ width:100%; text-align:center; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  color:#e7f0ff; font-weight:900; letter-spacing:.3px; line-height:1.15; font-size:22px; margin:0; }
.hero-dash{ opacity:.85; padding:0 .35rem; }
.hero-pby{ color:#9ad7ff; font-weight:900; }

.metric-box{ border:1px solid #ffffff1a; border-radius:12px; padding:10px; background:#0f172a; color:#d6e2ff; }

.badge{ display:inline-block; padding:2px 8px; border-radius:999px; font-weight:800; font-size:11px; }
.badge-raw{   background:#241c14; color:#ff9b63; border:1px solid #ff9b6333; }
.badge-truth{ background:#172017; color:#7dff96; border:1px solid #7dff9633; }
.badge-conf{  background:#1a1a24; color:#ffd86b; border:1px solid #ffd86b33; }
.badge-delta{ background:#161a22; color:#8ecbff; border:1px solid #8ecbff33; }
</style>
""", unsafe_allow_html=True)

# ============================== HERO (always first) ===========================
def render_hero():
    components.html(
        f"""
        <div class="hero-wrap">
          <div class="hero">
            <p class="hero-line">
              <span>{APP_NAME}</span><span class="hero-dash"> — </span>
              <span class="hero-pby">{POWERED_BY}</span>
            </p>
          </div>
        </div>
        """,
        height=70, scrolling=False
    )
render_hero()

# ============================== SIDEBAR ======================================
with st.sidebar:
    st.header("Navigation")
    nav = st.radio(
        "Go to",
        ["Dashboard","Crypto","Confluence","US Market (All Listings)","S&P 500","Options","Fusion","Export","Scores","Settings"],
        index=0
    )

    st.header("Appearance")
    font_size = st.slider("Font size", 14, 24, 18)
    st.markdown(f"<style>html, body, [class*='css']{{font-size:{font_size}px}}</style>", unsafe_allow_html=True)

    st.header("Truth Weights")
    w_vol = st.slider("Vol/Mcap",       0.0, 1.0, 0.30, 0.05)
    w_m24 = st.slider("24h Momentum",   0.0, 1.0, 0.25, 0.05)
    w_m7  = st.slider("7d Momentum",    0.0, 1.0, 0.25, 0.05)
    w_liq = st.slider("Liquidity/Size", 0.0, 1.0, 0.20, 0.05)
    st.caption("Weights auto-normalize.")

    st.header("Auto Refresh")
    auto  = st.toggle("Auto refresh", value=False)
    every = st.slider("Every (sec)", 10, 120, 30, step=5)

# ============================== HELPERS ======================================
def now_utc_str() -> str: return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
def _sigmoid(x): 
    try: return 1/(1+math.exp(-float(x)))
    except Exception: return 0.5
def pct_sigmoid(pct): 
    if pct is None or (isinstance(pct,float) and np.isnan(pct)): return 0.5
    return _sigmoid(float(pct)/10.0)
def _normalize(*w): 
    s=max(sum(w),1e-9); return tuple(v/s for v in w)

# ============================== DATA (CRYPTO) ================================
@st.cache_data(ttl=60, show_spinner="Loading CoinGecko…")
def fetch_cg_markets(vs="usd", per_page=250) -> pd.DataFrame:
    url="https://api.coingecko.com/api/v3/coins/markets"
    p={"vs_currency":vs,"order":"market_cap_desc","per_page":int(max(1,min(per_page,250))),
       "page":1,"sparkline":"false","price_change_percentage":"1h,24h,7d","locale":"en"}
    r=requests.get(url,params=p,timeout=30); r.raise_for_status()
    df=pd.DataFrame(r.json())
    need=["name","symbol","current_price","market_cap","total_volume",
          "price_change_percentage_1h_in_currency","price_change_percentage_24h_in_currency",
          "price_change_percentage_7d_in_currency"]
    for k in need:
        if k not in df.columns: df[k]=np.nan
    return df

# ============================== SCORING ======================================
def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    t=df.copy()
    t["total_volume"]=t.get("total_volume",pd.Series(np.nan,index=t.index)).fillna(0)
    t["market_cap"]=t.get("market_cap",pd.Series(np.nan,index=t.index)).fillna(0)
    t["vol_to_mc"]=(t["total_volume"]/t["market_cap"]).replace([np.inf,-np.inf],np.nan).fillna(0).clip(0,2.0)

    m1h=t.get("price_change_percentage_1h_in_currency",pd.Series(np.nan,index=t.index)).apply(pct_sigmoid)
    m24=t.get("price_change_percentage_24h_in_currency",pd.Series(np.nan,index=t.index)).apply(pct_sigmoid)
    m7d=t.get("price_change_percentage_7d_in_currency",pd.Series(np.nan,index=t.index)).apply(pct_sigmoid)

    mc=t["market_cap"]
    t["liq01"]=((mc-mc.min())/(mc.max()-mc.min()+1e-9)).fillna(0)

    a,b,c,d=_normalize(w_vol,w_m24,w_m7,w_liq)
    t["raw_heat"]=(0.5*(t["vol_to_mc"]/2).clip(0,1)+0.5*m1h.fillna(0.5)).clip(0,1)
    t["truth_full"]=(a*(t["vol_to_mc"]/2).clip(0,1)+b*m24.fillna(0.5)+c*m7d.fillna(0.5)+d*t["liq01"]).clip(0,1)

    t["consistency01"]=1-(m24.fillna(0.5)-m7d.fillna(0.5)).abs()
    t["agreement01"]=1-(t["raw_heat"]-t["truth_full"]).abs()
    t["energy01"]=(t["vol_to_mc"]/2).clip(0,1)
    t["confluence01"]=(0.35*t["truth_full"]+0.35*t["raw_heat"]+0.10*t["consistency01"]+
                       0.10*t["agreement01"]+0.05*t["energy01"]+0.05*t["liq01"]).clip(0,1)
    t["divergence"]=(t["raw_heat"]-t["truth_full"]).abs()

    def fire(v): return "🔥🔥🔥" if v>=0.85 else ("🔥🔥" if v>=0.65 else ("🔥" if v>=0.45 else "·"))
    def drop(v): return "💧💧💧" if v>=0.85 else ("💧💧" if v>=0.65 else ("💧" if v>=0.45 else "·"))
    def star(v): return "⭐️⭐️⭐️" if v>=0.85 else ("⭐️⭐️" if v>=0.65 else ("⭐️" if v>=0.45 else "·"))
    t["RAW_BADGE"]=t["raw_heat"].apply(fire)
    t["TRUTH_BADGE"]=t["truth_full"].apply(drop)
    t["CONFLUENCE_BADGE"]=t["confluence01"].apply(star)
    return t

def section_header(title: str, caption: str="")->None:
    st.markdown(f"### {title}")
    if caption: st.caption(caption)

def kpi_row(df_scored: pd.DataFrame, label: str)->None:
    n=len(df_scored)
    p24=float(df_scored.get("price_change_percentage_24h_in_currency",pd.Series(dtype=float)).mean())
    tavg=float(df_scored.get("truth_full",pd.Series(dtype=float)).mean())
    ravg=float(df_scored.get("raw_heat",pd.Series(dtype=float)).mean())
    cavg=float(df_scored.get("confluence01",pd.Series(dtype=float)).mean())
    c1,c2,c3,c4,c5=st.columns(5)
    with c1: st.markdown(f"<div class='metric-box'><b>Assets</b><br>{n}</div>",unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-box'><b>Avg 24h %</b><br>{0 if np.isnan(p24) else p24:.2f}%</div>",unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-box'><b>Avg TRUTH</b><br>{0 if np.isnan(tavg) else tavg:.2f}</div>",unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric-box'><b>Avg RAW</b><br>{0 if np.isnan(ravg) else ravg:.2f}</div>",unsafe_allow_html=True)
    with c5: st.markdown(f"<div class='metric-box'><b>Avg Confluence</b><br>{0 if np.isnan(cavg) else cavg:.2f}</div>",unsafe_allow_html=True)
    st.caption(f"{PHASE_TAG} • {POWERED_BY} • Updated {now_utc_str()} • Mode: {label}")

def table_view(df: pd.DataFrame, cols: List[str])->None:
    have=[c for c in cols if c in df.columns]
    st.dataframe(
        df[have], use_container_width=True, hide_index=True,
        column_config={
            "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "market_cap": st.column_config.NumberColumn("Mkt Cap", format="$%d"),
            "total_volume": st.column_config.NumberColumn("Volume", format="$%d"),
            "price_change_percentage_24h_in_currency": st.column_config.NumberColumn("24h %", format="%.2f%%"),
            "raw_heat": st.column_config.ProgressColumn("RAW", min_value=0.0, max_value=1.0),
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

# ============================== STOCKS / OPTIONS ==============================
def _secret(key:str)->Optional[str]:
    try: return st.secrets.get(key)  # type: ignore
    except Exception: return None

ALPHA_KEY=_secret("ALPHAVANTAGE_API_KEY")

@st.cache_data(ttl=180, show_spinner="Alpha Vantage snapshots…")
def alpha_snapshot_daily(tickers: List[str]) -> pd.DataFrame:
    if not ALPHA_KEY or not tickers: return pd.DataFrame()
    rows=[]; base="https://www.alphavantage.co/query"
    for t in [x.strip().upper() for x in tickers if x.strip()]:
        try:
            params={"function":"TIME_SERIES_DAILY_ADJUSTED","symbol":t,"apikey":ALPHA_KEY,"datatype":"json"}
            r=requests.get(base,params=params,timeout=30); js=r.json()
            series=js.get("Time Series (Daily)",{})
            if not series: continue
            dates=sorted(series.keys())
            last=series[dates[-1]]; prev=series[dates[-2]] if len(dates)>=2 else None
            last_close=float(last["4. close"])
            pct24=(last_close/float(prev["4. close"])-1)*100 if prev else np.nan
            rows.append({"name":t,"symbol":t,"current_price":last_close,
                         "price_change_percentage_24h_in_currency":pct24,
                         "market_cap":np.nan,"total_volume":np.nan,
                         "price_change_percentage_1h_in_currency":np.nan,
                         "price_change_percentage_7d_in_currency":np.nan})
        except Exception:
            rows.append({"name":t,"symbol":t,"current_price":np.nan,
                         "price_change_percentage_24h_in_currency":np.nan,
                         "market_cap":np.nan,"total_volume":np.nan,
                         "price_change_percentage_1h_in_currency":np.nan,
                         "price_change_percentage_7d_in_currency":np.nan})
    return pd.DataFrame(rows)

@st.cache_data(ttl=180, show_spinner="yfinance snapshots…")
def yf_snapshot_daily(tickers: List[str]) -> pd.DataFrame:
    if not HAS_YF or not tickers: return pd.DataFrame()
    tickers=[t.strip().upper() for t in tickers if t.strip()]
    try:
        data=yf.download(" ".join(tickers),period="5d",interval="1d",
                         group_by="ticker",auto_adjust=True,threads=True,progress=False)
    except Exception:
        return pd.DataFrame()
    rows=[]
    for t in tickers:
        try:
            s=data[t]
            last=float(s.iloc[-1]["Close"])
            prev=float(s.iloc[-2]["Close"]) if len(s)>=2 else np.nan
            pct24=(last/prev-1.0)*100.0 if pd.notna(prev) else np.nan
            rows.append({"name":t,"symbol":t,"current_price":last,
                         "price_change_percentage_24h_in_currency":pct24,
                         "market_cap":np.nan,"total_volume":np.nan,
                         "price_change_percentage_1h_in_currency":np.nan,
                         "price_change_percentage_7d_in_currency":np.nan})
        except Exception:
            rows.append({"name":t,"symbol":t,"current_price":np.nan,
                         "price_change_percentage_24h_in_currency":np.nan,
                         "market_cap":np.nan,"total_volume":np.nan,
                         "price_change_percentage_1h_in_currency":np.nan,
                         "price_change_percentage_7d_in_currency":np.nan})
    return pd.DataFrame(rows)

def stocks_snapshot(tickers: List[str]) -> pd.DataFrame:
    if ALPHA_KEY: return alpha_snapshot_daily(tickers)
    if HAS_YF:    return yf_snapshot_daily(tickers)
    st.warning("No stock provider available. Add ALPHAVANTAGE_API_KEY (free) in Secrets or add 'yfinance' to requirements.")
    return pd.DataFrame()

@st.cache_data(ttl=120)
def list_expirations(ticker: str) -> List[str]:
    if not HAS_YF or not ticker: return []
    try: return list(yf.Ticker(ticker).options)
    except Exception: return []

@st.cache_data(ttl=180, show_spinner="Loading option chain…")
def load_options_chain(ticker: str, expiration: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not HAS_YF or not ticker or not expiration: return pd.DataFrame(), pd.DataFrame()
    try:
        tk=yf.Ticker(ticker); chain=tk.option_chain(expiration)
        keep=["contractSymbol","strike","lastPrice","openInterest","volume","impliedVolatility"]
        calls=chain.calls[[c for c in keep if c in chain.calls.columns]].copy()
        puts =chain.puts [[c for c in keep if c in chain.puts.columns]].copy()
        return calls, puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

# ============================== UNIVERSE FEEDS ================================
@st.cache_data(ttl=600, show_spinner="Loading S&P 500 list…")
def fetch_sp500_constituents() -> pd.DataFrame:
    try:
        tables=pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df=tables[0].copy()
        df=df.rename(columns={"Symbol":"symbol","Security":"name","GICS Sector":"Sector"})
        df["symbol"]=df["symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
        return df[["symbol","name","Sector"]]
    except Exception:
        return pd.DataFrame(columns=["symbol","name","Sector"])

NASDAQ_LISTED_URL="https://ftp.nasdaqtrader.com/dynamic/SYMBOL_DIRECTORY/nasdaqlisted.txt"
OTHER_LISTED_URL ="https://ftp.nasdaqtrader.com/dynamic/SYMBOL_DIRECTORY/otherlisted.txt"

@st.cache_data(ttl=900, show_spinner="Loading US listings…")
def fetch_us_listings() -> pd.DataFrame:
    frames=[]
    try:
        txt=requests.get(NASDAQ_LISTED_URL,timeout=30).text
        df=pd.read_csv(io.StringIO(txt),sep="|")
        df=df[~df["Symbol"].str.contains("File Creation Time",na=False)]
        df=df.rename(columns={"Symbol":"symbol","Security Name":"name","Market Category":"MarketCategory",
                              "Test Issue":"TestIssue","Financial Status":"FinancialStatus","ETF":"ETF"})
        df["ListingExchange"]="NASDAQ"
        frames.append(df[["symbol","name","ListingExchange","ETF","TestIssue"]])
    except Exception: pass
    try:
        txt=requests.get(OTHER_LISTED_URL,timeout=30).text
        df=pd.read_csv(io.StringIO(txt),sep="|")
        df=df[~df["ACT Symbol"].str.contains("File Creation Time",na=False)]
        df=df.rename(columns={"ACT Symbol":"symbol","Security Name":"name","Exchange":"ListingExchange",
                              "ETF":"ETF","Test Issue":"TestIssue"})
        frames.append(df[["symbol","name","ListingExchange","ETF","TestIssue"]])
    except Exception: pass

    if not frames: return pd.DataFrame(columns=["symbol","name","ListingExchange","ETF","TestIssue"])
    allu=pd.concat(frames,ignore_index=True)
    allu["symbol"]=allu["symbol"].astype(str).str.upper()
    allu["ETF"]=allu["ETF"].astype(str).str.upper()
    allu["TestIssue"]=allu["TestIssue"].astype(str).str.upper()
    allu=allu[allu["TestIssue"]!="Y"].copy()
    allu=allu[allu["symbol"].str.match(r"^[A-Z0-9\.-]+$",na=False)]
    return allu.drop_duplicates(subset=["symbol"])

# ============================== PAGES =========================================
def page_dashboard():
    section_header("Dashboard","Top Confluence & TRUTH leaders (Crypto).")
    df=build_scores(fetch_cg_markets("usd",200))
    if df.empty: st.warning("No data from CoinGecko."); return
    kpi_row(df,"Crypto")

    # Interactive buttons -> quick chart
    if "focus_metric" not in st.session_state: st.session_state.focus_metric="conf"
    c1,c2,c3,c4=st.columns(4)
    with c1:
        if st.button("🔥 RAW"):   st.session_state.focus_metric="raw"
    with c2:
        if st.button("💧 TRUTH"): st.session_state.focus_metric="truth"
    with c3:
        if st.button("⭐ CONFLUENCE"): st.session_state.focus_metric="conf"
    with c4:
        if st.button("⚡ Δ (RAW−TRUTH)"): st.session_state.focus_metric="delta"

    import plotly.express as px
    metric_map={"raw":("raw_heat","RAW Heat (0..1)"),
                "truth":("truth_full","TRUTH (0..1)"),
                "conf":("confluence01","Confluence (0..1)"),
                "delta":("divergence","Δ = |RAW − TRUTH| (0..1)")}
    mcol, mtitle = metric_map[st.session_state.focus_metric]
    top=df.sort_values(mcol,ascending=False).head(25)
    fig=px.bar(top,x="symbol",y=mcol,hover_data=["name","current_price"],title=f"Top by {mtitle}",height=320)
    fig.update_layout(margin=dict(l=0,r=0,t=28,b=6),xaxis_title=None,yaxis_title=None,bargap=.18)
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    cL,cR=st.columns(2)
    with cL:
        st.subheader("Top Confluence (Crypto)")
        table_view(df.sort_values("confluence01",ascending=False).head(20),
                   ["name","symbol","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE"])
    with cR:
        st.subheader("Top TRUTH (Crypto)")
        table_view(df.sort_values("truth_full",ascending=False).head(20),
                   ["name","symbol","current_price","TRUTH_BADGE","truth_full","RAW_BADGE","divergence"])

def page_crypto():
    section_header("Crypto","Interactive TRUTH / RAW / CONFLUENCE (CoinGecko).")
    topn_pull=st.slider("Pull Top N from API",50,250,200,step=50)
    df=build_scores(fetch_cg_markets("usd",topn_pull))
    if df.empty: st.warning("No data from CoinGecko."); return
    kpi_row(df,"Crypto")
    c1,c2,c3,c4=st.columns(4)
    with c1: topn=st.slider("Show Top N",20,250,150,step=10)
    with c2: min_truth=st.slider("Min TRUTH",0.0,1.0,0.0,0.05)
    with c3: search=st.text_input("Search",value="").strip().lower()
    with c4: order=st.selectbox("Order by",["confluence01","truth_full","raw_heat"],index=0)
    out=df.copy()
    if min_truth>0: out=out[out["truth_full"]>=min_truth]
    if search:
        mask=out["name"].str.lower().str.contains(search,na=False)|out["symbol"].str.lower().str.contains(search,na=False)
        out=out[mask]
    out=out.sort_values(order,ascending=False).head(topn)
    table_view(out,["name","symbol","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence"])

def page_confluence():
    section_header("Confluence","When RAW and TRUTH agree strongly (Crypto).")
    df=build_scores(fetch_cg_markets("usd",200))
    if df.empty: st.warning("No data from CoinGecko."); return
    kpi_row(df,"Confluence")
    table_view(df.sort_values("confluence01",ascending=False).head(50),
               ["name","symbol","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence"])

def page_sp500():
    section_header("S&P 500","Constituents → snapshot → RAW/TRUTH/CONFLUENCE.")
    base=fetch_sp500_constituents()
    if base.empty: st.warning("Couldn’t fetch S&P list."); return
    limit=st.slider("Tickers to snapshot",50,len(base),200,step=50)
    tickers=base["symbol"].tolist()[:limit]
    snap=stocks_snapshot(tickers)
    if snap.empty: st.warning("No prices returned (provider/rate limits)."); return
    merged=snap.merge(base,on="symbol",how="left")
    scored=build_scores(merged)
    kpi_row(scored,"S&P 500")
    table_view(scored.sort_values("confluence01",ascending=False).head(100),
               ["name","symbol","Sector","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE"])

def page_us_all():
    section_header("US Market (All Listings)","NASDAQ + NYSE + ARCA + AMEX.")
    uni=fetch_us_listings()
    if uni.empty: st.warning("Couldn’t fetch US listings."); return
    c1,c2,c3,c4=st.columns([2,1,1,1])
    with c1: query=st.text_input("Search (symbol/name)",value="").strip().lower()
    with c2: exch=st.selectbox("Exchange",["(All)"]+sorted(uni["ListingExchange"].dropna().unique().tolist()),index=0)
    with c3: include_etf=st.toggle("Include ETFs",value=False)
    with c4: limit=st.slider("Tickers to snapshot",50,1000,200,step=50)
    dfu=uni.copy()
    if not include_etf and "ETF" in dfu.columns: dfu=dfu[dfu["ETF"]!="Y"]
    if exch!="(All)": dfu=dfu[dfu["ListingExchange"]==exch]
    if query:
        mask=dfu["symbol"].str.lower().str.contains(query,na=False)|dfu["name"].str.lower().str.contains(query,na=False)
        dfu=dfu[mask]
    st.caption(f"Universe matches: {len(dfu)} • Snapshotting first {min(limit,len(dfu))}")
    tickers=dfu["symbol"].tolist()[:limit]
    snap=stocks_snapshot(tickers)
    if snap.empty: st.warning("No price data returned for selected set."); return
    merged=snap.merge(dfu,on="symbol",how="left")
    scored=build_scores(merged)
    kpi_row(scored,"US Market")
    cL,cR=st.columns(2)
    with cL:
        st.subheader("Top Confluence ⭐")
        table_view(scored.sort_values("confluence01",ascending=False).head(100),
                   ["name","symbol","ListingExchange","ETF","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence"])
    with cR:
        st.subheader("Top TRUTH 💧")
        table_view(scored.sort_values("truth_full",ascending=False).head(100),
                   ["name","symbol","ListingExchange","ETF","current_price","TRUTH_BADGE","truth_full","RAW_BADGE","divergence"])
    st.download_button("Download US Snapshot (CSV)",data=scored.to_csv(index=False).encode("utf-8"),
                       file_name="us_listings_truth_raw_confluence.csv",mime="text/csv")

def page_options():
    section_header("Options Explorer","Ticker → expiration → calls/puts (yfinance).")
    if not HAS_YF:
        st.error("Options needs yfinance. Add `yfinance` to requirements.txt and reboot."); return
    base=fetch_us_listings()
    sym=st.selectbox("Ticker",options=["AAPL"]+(base["symbol"].tolist() if not base.empty else []),index=0).strip().upper()
    if not sym: return
    exps=list_expirations(sym)
    if not exps: st.warning("No expirations returned."); return
    exp=st.selectbox("Expiration",options=exps,index=0)
    calls,puts=load_options_chain(sym,exp)
    c1,c2=st.columns(2)
    with c1:
        st.subheader(f"{sym} Calls — {exp}")
        if calls.empty: st.info("No calls data.")
        else: table_view(calls.sort_values(["openInterest","volume"],ascending=False).head(25),
                         ["contractSymbol","strike","lastPrice","openInterest","volume","impliedVolatility"])
    with c2:
        st.subheader(f"{sym} Puts — {exp}")
        if puts.empty: st.info("No puts data.")
        else: table_view(puts.sort_values(["openInterest","volume"],ascending=False).head(25),
                         ["contractSymbol","strike","lastPrice","openInterest","volume","impliedVolatility"])

def page_fusion():
    section_header("Fusion","Crypto vs US leaders by Confluence.")
    dfc=build_scores(fetch_cg_markets("usd",120)); dfc["universe"]="CRYPTO"
    uni=fetch_us_listings()
    if not uni.empty:
        tickers=(uni[uni["ETF"]!="Y"]["symbol"] if "ETF" in uni.columns else uni["symbol"]).tolist()[:120]
        dfs0=stocks_snapshot(tickers)
        dfs=build_scores(dfs0.merge(uni,on="symbol",how="left")) if not dfs0.empty else pd.DataFrame()
        dfs["universe"]="US"
    else:
        dfs=pd.DataFrame(columns=dfc.columns)
    try: both=pd.concat([dfc,dfs],ignore_index=True)
    except Exception: both=dfc.copy()
    kpi_row(both,"Fusion")
    table_view(both.sort_values("confluence01",ascending=False).head(40),
               ["name","symbol","current_price","CONFLUENCE_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence","universe"])

def page_export():
    section_header("Export","One-click CSV downloads.")
    dfc=build_scores(fetch_cg_markets("usd",200))
    st.download_button("Download Crypto CSV",data=dfc.to_csv(index=False).encode("utf-8"),
                       file_name="crypto_truth_raw_confluence.csv",mime="text/csv")
    uni=fetch_us_listings()
    if not uni.empty:
        tickers=uni["symbol"].tolist()[:400]
        dfs0=stocks_snapshot(tickers)
        if not dfs0.empty:
            dfs=build_scores(dfs0.merge(uni,on="symbol",how="left"))
            st.download_button("Download US Snapshot CSV",data=dfs.to_csv(index=False).encode("utf-8"),
                               file_name="us_listings_truth_raw_confluence.csv",mime="text/csv")

def page_scores():
    section_header("Scores — Explainer")
    st.markdown("""
**Lens glossary (quick)**  
- **🔥 RAW (0..1)** — crowd heat *now* (volume/market-cap + 1h momentum).  
- **💧 TRUTH (0..1)** — stability/quality (Vol/Mcap, 24h, 7d, Liquidity).  
- **⚡ Δ (0..1)** — mismatch `|RAW − TRUTH|` (bigger = bigger gap).  
- **⭐ Confluence (0..1)** — fusion of RAW+TRUTH + agreement (RAW≈TRUTH) + consistency (24h≈7d) + energy + liquidity.  

**Read fast**  
- ⭐ High Confluence → hype **and** quality aligned (prime).  
- 🔥 High RAW + low 💧 TRUTH → hype spike (fragile).  
- 💧 High TRUTH + low 🔥 RAW → sleeper quality (crowd not there yet).
""")

def page_settings():
    section_header("Settings","More personalization next phase.")
    st.toggle("High-contrast metric cards", key="hc_cards")
    st.caption("Weights editor & saved presets coming soon.")

# ============================== ROUTER ========================================
if nav == "Dashboard": page_dashboard()
elif nav == "Crypto": page_crypto()
elif nav == "Confluence": page_confluence()
elif nav == "US Market (All Listings)": page_us_all()
elif nav == "S&P 500": page_sp500()
elif nav == "Options": page_options()
elif nav == "Fusion": page_fusion()
elif nav == "Export": page_export()
elif nav == "Scores": page_scores()
else: page_settings()

# ============================== AUTO REFRESH ==================================
if auto:
    st.caption(f"{PHASE_TAG} • {POWERED_BY} • Auto refresh every {int(every)}s")
    time.sleep(max(5,int(every))); st.rerun()
