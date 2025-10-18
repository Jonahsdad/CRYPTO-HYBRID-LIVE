# -----------------------------------------------------------------------------
# Crypto Hybrid Live - Compact V1 (UNDER 200 LINES)
# Powered by Jesse Ray Landingham Jr
# -----------------------------------------------------------------------------

from __future__ import annotations
import math, time
from datetime import datetime, timezone
import numpy as np, pandas as pd, requests, streamlit as st

# Optional: yfinance (stocks); app still runs if unavailable
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

# ----------------------- APP CONFIG -----------------------
st.set_page_config(page_title="Crypto Hybrid Live — Compact", layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    st.header("Navigation")
    nav = st.radio("Go to", ["Dashboard","Crypto","Stocks","Scores"], index=0)
    st.header("Appearance")
    font = st.slider("Font size", 14, 24, 18)
    st.markdown(f"<style>html,body,[class*='css']{{font-size:{font}px}}</style>", unsafe_allow_html=True)
    st.header("Options")
    auto = st.toggle("Auto refresh", value=False)
    every = st.slider("Every (sec)", 10, 120, 30)

# ----------------------- HELPERS --------------------------
def now_utc(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

@st.cache_data(ttl=60, show_spinner="Loading crypto…")
def cg_markets(vs="usd", n=200) -> pd.DataFrame:
    url="https://api.coingecko.com/api/v3/coins/markets"
    p={"vs_currency":vs,"order":"market_cap_desc","per_page":min(n,250),"page":1,
       "sparkline":"false","price_change_percentage":"1h,24h,7d","locale":"en"}
    r=requests.get(url, params=p, timeout=30); r.raise_for_status()
    df=pd.DataFrame(r.json())
    need=["name","symbol","current_price","market_cap","total_volume",
          "price_change_percentage_1h_in_currency","price_change_percentage_24h_in_currency",
          "price_change_percentage_7d_in_currency"]
    for k in need:
        if k not in df.columns: df[k]=np.nan
    return df

def pct_sigmoid(x):
    if x is None or (isinstance(x,float) and np.isnan(x)): return 0.5
    try: return 1/(1+math.exp(-(float(x)/10.0)))
    except: return 0.5

def score_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df.copy()
    t=df.copy()
    t["vol_to_mc"]=((t.get("total_volume",0)/t.get("market_cap",np.nan))
                    .replace([np.inf,-np.inf],np.nan)).clip(0,2).fillna(0)
    t["m1h"]=t.get("price_change_percentage_1h_in_currency",pd.Series(np.nan,index=t.index)).apply(pct_sigmoid)
    t["m24"]=t.get("price_change_percentage_24h_in_currency",pd.Series(np.nan,index=t.index)).apply(pct_sigmoid)
    t["m7d"]=t.get("price_change_percentage_7d_in_currency",pd.Series(np.nan,index=t.index)).apply(pct_sigmoid)
    mc=t.get("market_cap",pd.Series(0,index=t.index)).fillna(0)
    t["liq01"]=0 if mc.max()==0 else (mc-mc.min())/(mc.max()-mc.min()+1e-9)
    t["raw_heat"]=(0.5*(t["vol_to_mc"]/2).clip(0,1)+0.5*t["m1h"].fillna(0.5)).clip(0,1)
    t["truth_full"]=(0.30*(t["vol_to_mc"]/2).clip(0,1)+0.25*t["m24"].fillna(0.5)+
                     0.25*t["m7d"].fillna(0.5)+0.20*t["liq01"].fillna(0.0)).clip(0,1)
    t["divergence"]=(t["raw_heat"]-t["truth_full"]).abs()
    return t

@st.cache_data(ttl=90, show_spinner="Loading stocks…")
def yf_snapshot(tickers_csv:str) -> pd.DataFrame:
    if not HAS_YF: return pd.DataFrame()
    tick=[x.strip().upper() for x in tickers_csv.split(",") if x.strip()]
    if not tick: return pd.DataFrame()
    try:
        data=yf.download(" ".join(tick), period="5d", interval="1d", group_by="ticker",
                         auto_adjust=True, threads=True, progress=False)
    except Exception: return pd.DataFrame()
    rows=[]
    for tck in tick:
        try:
            s=data[tck]; last=float(s.iloc[-1]["Close"])
            prev=float(s.iloc[-2]["Close"]) if len(s)>=2 else np.nan
            pct=(last/prev-1)*100 if pd.notna(prev) else np.nan
            rows.append({"name":tck,"symbol":tck,"current_price":last,
                         "price_change_percentage_24h_in_currency":pct,
                         "market_cap":np.nan,"total_volume":np.nan,
                         "price_change_percentage_1h_in_currency":np.nan,
                         "price_change_percentage_7d_in_currency":np.nan})
        except Exception:
            rows.append({"name":tck,"symbol":tck,"current_price":np.nan,
                         "price_change_percentage_24h_in_currency":np.nan,
                         "market_cap":np.nan,"total_volume":np.nan,
                         "price_change_percentage_1h_in_currency":np.nan,
                         "price_change_percentage_7d_in_currency":np.nan})
    return pd.DataFrame(rows)

def kpis(df: pd.DataFrame, label:str):
    a=len(df); p=df.get("price_change_percentage_24h_in_currency",pd.Series(dtype=float)).mean()
    T=df.get("truth_full",pd.Series(dtype=float)).mean(); R=df.get("raw_heat",pd.Series(dtype=float)).mean()
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Assets", a)
    c2.metric("Avg 24h %", 0 if np.isnan(p) else round(p,2))
    c3.metric("Avg Truth", 0 if np.isnan(T) else round(T,2))
    c4.metric("Avg Raw", 0 if np.isnan(R) else round(R,2))
    st.caption(f"Last update: {now_utc()} • Mode: {label}")

def truth_raw_blocks(df: pd.DataFrame, topn:int=20):
    c1,c2,c3=st.columns(3)
    with c1:
        st.subheader("RAW (heat)")
        cols=["name","symbol","current_price","market_cap","total_volume","raw_heat"]
        st.dataframe(df.sort_values("raw_heat",ascending=False).head(topn)[[c for c in cols if c in df]], use_container_width=True)
    with c2:
        st.subheader("TRUTH (stability)")
        cols=["name","symbol","current_price","market_cap","truth_full"]
        st.dataframe(df.sort_values("truth_full",ascending=False).head(topn)[[c for c in cols if c in df]], use_container_width=True)
    with c3:
        st.subheader("Movers (24h)")
        if "price_change_percentage_24h_in_currency" in df:
            g=df.sort_values("price_change_percentage_24h_in_currency",ascending=False).head(10)
            l=df.sort_values("price_change_percentage_24h_in_currency",ascending=True).head(10)
            st.markdown("**Top Gainers**"); st.dataframe(g[["name","symbol","current_price","price_change_percentage_24h_in_currency"]], use_container_width=True)
            st.markdown("**Top Losers**");  st.dataframe(l[["name","symbol","current_price","price_change_percentage_24h_in_currency"]], use_container_width=True)
        else:
            st.info("No 24h % available in this table.")

# ----------------------- PAGES ----------------------------
def page_dashboard():
    st.title("Crypto Hybrid Live — Compact")
    st.caption("Truth vs Raw lens. Education only — not financial advice.")
    df=score_table(cg_markets("usd",200))
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Top TRUTH (Crypto)")
        st.dataframe(df.sort_values("truth_full",ascending=False).head(15)[["name","symbol","current_price","truth_full"]], use_container_width=True)
    with c2:
        st.subheader("Top RAW (Crypto)")
        st.dataframe(df.sort_values("raw_heat",ascending=False).head(15)[["name","symbol","current_price","raw_heat"]], use_container_width=True)

def page_crypto():
    st.title("Crypto")
    topn=st.slider("Show top N", 50, 250, 150)
    df=score_table(cg_markets("usd",topn))
    kpis(df,"Crypto")
    truth_raw_blocks(df, topn=20)

def page_stocks():
    st.title("Stocks")
    default="AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA"
    raw=st.text_input("Tickers (comma-separated)", value=default)
    if not HAS_YF:
        st.error("yfinance not installed. Add `yfinance` to requirements.txt, then reboot.")
        return
    df0=yf_snapshot(raw)
    if df0.empty: st.warning("No stock data. Check tickers."); return
    df=score_table(df0)
    kpis(df,"Stocks")
    truth_raw_blocks(df, topn=min(20,len(df)))

def page_scores():
    st.title("Scores — Explainer")
    st.markdown("""
**RAW** = quick heat from Vol/Mcap and 1h momentum (0..1).  
**TRUTH** = slower, more stable blend: Vol/Mcap, 24h momentum, 7d momentum, liquidity (0..1).  
**DIVERGENCE** = |RAW − TRUTH| (potential overextension / mean reversion).
    """)

# ----------------------- ROUTER ---------------------------
if nav=="Dashboard": page_dashboard()
elif nav=="Crypto":  page_crypto()
elif nav=="Stocks":  page_stocks()
else:                page_scores()

# ----------------------- AUTO REFRESH ---------------------
if auto:
    st.caption(f"Auto refresh every {every}s is ON.")
    time.sleep(max(5,int(every)))
    st.rerun()
