# ========================== PHASE 8 — FUSION AI DASH ==========================
# Crypto Hybrid Live — Phase 8 (Fusion AI + Sentiment + Cross-Market + Backtests)
# Paste this entire file into app.py (no spaces above this line).
# ==============================================================================

# ---- Imports + safe fallbacks ------------------------------------------------
import math, time, json, re, io
import requests, pandas as pd, numpy as np, streamlit as st
from datetime import datetime, timezone

# Plotly (optional minis)
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# Auto-install helpers (Streamlit Cloud sometimes misses deps)
def _ensure(pkgs):
    import importlib, subprocess, sys
    for p in pkgs:
        try:
            importlib.import_module(p)
        except Exception:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
            except Exception:
                pass

_ensure(["yfinance","ccxt","cryptocompare","feedparser"])

# Finance + feeds (soft import)
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

try:
    import ccxt
    CCXT_OK = True
except Exception:
    CCXT_OK = False

try:
    import cryptocompare
    CC_OK = True
except Exception:
    CC_OK = False

try:
    import feedparser
    FP_OK = True
except Exception:
    FP_OK = False

# Sentiment (TextBlob) — safe fallback
try:
    from textblob import TextBlob
    def _polarity(t): 
        try: return float(TextBlob(t).sentiment.polarity)
        except: return 0.0
except Exception:
    def _polarity(t): return 0.0

# ---- App config --------------------------------------------------------------
APP_NAME = "Crypto Hybrid Live — Phase 8 (Fusion AI)"
st.set_page_config(page_title=APP_NAME, layout="wide")
USER_AGENT = {"User-Agent":"Mozilla/5.0 (CHL-Phase8)"}
CG_PER_PAGE = 150
HIST_DAYS = 90

# ---- Maintenance (clear cache button) ---------------------------------------
st.sidebar.markdown("### ⚙️ Maintenance")
if st.sidebar.button("🧹 Clear Cache & Restart"):
    st.cache_data.clear(); st.cache_resource.clear(); st.experimental_rerun()

# ---- Theme + CSS -------------------------------------------------------------
if "theme" not in st.session_state: st.session_state["theme"]="dark"
def _apply_css():
    dark = st.session_state["theme"]=="dark"
    base_bg = "#0d1117" if dark else "#ffffff"
    base_fg = "#e6edf3" if dark else "#111"
    accent  = "#23d18b" if dark else "#0b8f5a"
    st.markdown(f"""
    <style>
    html, body, [class*="css"] {{ background:{base_bg}; color:{base_fg}; font-size:18px; }}
    .stTabs [role="tablist"] button {{ font-size:1.25rem!important; font-weight:700!important; 
      margin-right:.6rem; border-radius:10px; background:#111; color:{accent}; border:1px solid #222; }}
    .stTabs [role="tablist"] button[aria-selected="true"] {{ background:{accent}; color:#000; transform:scale(1.04); }}
    [data-testid="stMetricValue"]{{font-size:2.2rem!important}}
    .pill {{ display:inline-block;padding:.1rem .55rem;border-radius:999px;background:#1d2633;margin-right:.35rem;font-size:.9rem; }}
    </style>
    """, unsafe_allow_html=True)
_apply_css()

# ---- Core weights (LIPE) -----------------------------------------------------
DEFAULT_WEIGHTS = dict(w_vol=0.30, w_m24=0.25, w_m7=0.25, w_liq=0.20)
PRESETS = {
    "Balanced":  dict(w_vol=0.30, w_m24=0.25, w_m7=0.25, w_liq=0.20),
    "Momentum":  dict(w_vol=0.15, w_m24=0.45, w_m7=0.30, w_liq=0.10),
    "Liquidity": dict(w_vol=0.45, w_m24=0.20, w_m7=0.15, w_liq=0.20),
    "Value":     dict(w_vol=0.25, w_m24=0.20, w_m7=0.20, w_liq=0.35),
}
FUSION_V2 = dict(w_truth=0.70, w_sent=0.15, w_xmkt=0.15)  # Fusion v2 blend

def _norm(w): 
    s = sum(max(0,v) for v in w.values()) or 1.0
    return {k:max(0,v)/s for k,v in w.items()}

def _sig(p):
    if pd.isna(p): return 0.5
    return 1/(1+math.exp(-float(p)/10.0))

def lipe_truth(df, w):
    w=_norm(w or DEFAULT_WEIGHTS)
    if "liquidity01" not in df: df["liquidity01"]=0.0
    if "vol_to_mc" not in df:
        vol=df.get("total_volume", pd.Series(0,index=df.index))
        v01=(vol-vol.min())/(vol.max()-vol.min()+1e-9)
        df["vol_to_mc"]=2*v01
    return (
        w["w_vol"]*(df["vol_to_mc"]/2).clip(0,1) +
        w["w_m24"]*df.get("momo_24h01",0.5) +
        w["w_m7"] *df.get("momo_7d01",0.5) +
        w["w_liq"]*df["liquidity01"]
    ).clip(0,1)

def mood_label(x):
    if x>=.8: return "🟢 EUPHORIC"
    if x>=.6: return "🟡 OPTIMISTIC"
    if x>=.4: return "🟠 NEUTRAL"
    return "🔴 FEARFUL"

# ---- Utilities ---------------------------------------------------------------
def safe_get(url, params=None, t=25):
    try:
        r=requests.get(url, params=params, headers=USER_AGENT, timeout=t)
        if r.status_code==200: return r
    except Exception: pass
    return None

@st.cache_data(ttl=60)
def cg_markets(vs="usd", limit=150):
    u="https://api.coingecko.com/api/v3/coins/markets"
    p={"vs_currency":vs,"order":"market_cap_desc","per_page":limit,"page":1,
       "sparkline":"false","price_change_percentage":"1h,24h,7d"}
    r=safe_get(u,p);  return pd.DataFrame(r.json()) if r else pd.DataFrame()

@st.cache_data(ttl=600)
def cg_chart(coin_id, vs="usd", days=90):
    u=f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    r=safe_get(u,{"vs_currency":vs,"days":days})
    if not r: return pd.DataFrame()
    arr=r.json().get("prices",[])
    if not arr: return pd.DataFrame()
    d=pd.DataFrame(arr, columns=["ts","price"]); d["ts"]=pd.to_datetime(d["ts"], unit="ms"); return d

@st.cache_data(ttl=300)
def rss_sentiment():
    """Simple crypto news sentiment from RSS (no keys)."""
    if not FP_OK: return 0.5, []
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://news.bitcoin.com/feed/"
    ]
    titles=[]
    for f in feeds:
        try:
            d=feedparser.parse(f)
            for e in d.entries[:20]:
                titles.append(e.title)
        except Exception:
            continue
    if not titles: return 0.5,[]
    pol = np.mean([_polarity(t) for t in titles])  # -1..1
    sent01 = float((pol+1)/2)                       # 0..1
    return sent01, titles[:30]

@st.cache_data(ttl=180)
def yf_multi(ticks, period="6mo"):
    if not YF_OK: return pd.DataFrame(), {}
    data = yf.download(ticks, period=period, interval="1d", auto_adjust=True, progress=False)
    frames=[]; meta={}
    for t in ticks:
        try: ser = data["Adj Close"][t].rename(t) if len(ticks)>1 else data["Adj Close"].rename(t)
        except: continue
        frames.append(ser)
        try: fi=yf.Ticker(t).fast_info
        except: fi={}
        meta[t]={"market_cap":getattr(fi,"market_cap",np.nan),
                 "last_price":getattr(fi,"last_price",np.nan),
                 "volume":getattr(fi,"last_volume",np.nan)}
    return pd.concat(frames,axis=1), meta

# ---- Header ------------------------------------------------------------------
st.title("🟢 "+APP_NAME)
st.caption("Fusion v2 = LIPE Truth + News Sentiment + Cross-Market drift • Backtests • Saved rules • Export")

# ---- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("🧭 Market")
    market = st.radio("Choose market", ["Crypto","Stocks","FX"], horizontal=True)
    vs_currency = st.selectbox("Currency (Crypto)", ["usd"], index=0)
    topn = st.slider("Show top N (Crypto)", 20, 250, CG_PER_PAGE, 10)

    st.subheader("Theme")
    theme_pick = st.radio("Theme", ["Dark","Light"], index=0 if st.session_state["theme"]=="dark" else 1, horizontal=True)
    st.session_state["theme"]="dark" if theme_pick=="Dark" else "light"; _apply_css()

    st.subheader("Truth Preset")
    preset = st.radio("Preset", list(PRESETS.keys()), index=0, horizontal=True)
    w_edit = dict(PRESETS[preset])
    st.subheader("Weights (Truth v1)")
    for k in list(w_edit.keys()):
        w_edit[k] = st.slider(k, 0.0, 1.0, float(w_edit[k]), 0.01)
    w_edit = _norm(w_edit)

    if market=="Stocks":
        st.subheader("Tickers")
        stocks_in = st.text_area("", value="AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA")
    if market=="FX":
        st.subheader("FX pairs")
        fx_in = st.text_area("", value="EURUSD=X,USDJPY=X,GBPUSD=X,AUDUSD=X,USDCAD=X")

    st.markdown("---")
    st.subheader("Live Mode")
    live = st.toggle("Auto-refresh", value=False)
    every = st.slider("Refresh every (sec)", 10, 120, 30, 5)
    if live: time.sleep(every)

# ---- Source status ------------------------------------------------------------
st.sidebar.markdown("### 🔍 Sources")
st.sidebar.write("✅", "CoinGecko")
st.sidebar.write("✅" if YF_OK else "⚠️", "Yahoo Finance")
st.sidebar.write("✅" if CCXT_OK else "⚠️", "CCXT")
st.sidebar.write("✅" if CC_OK else "⚠️", "CryptoCompare")
st.sidebar.write("✅" if FP_OK else "⚠️", "RSS/Feedparser")

# ---- Build DataFrames ---------------------------------------------------------
def build_crypto():
    df=cg_markets(vs_currency, topn)
    if df.empty: return df
    df["vol_to_mc"]=(df["total_volume"]/df["market_cap"]).replace([np.inf,-np.inf],np.nan).clip(0,2).fillna(0)
    df["momo_24h01"]=df["price_change_percentage_24h_in_currency"].apply(_sig)
    df["momo_7d01"]=df["price_change_percentage_7d_in_currency"].apply(_sig)
    mc=df["market_cap"].fillna(0)
    df["liquidity01"]=0 if mc.max()==0 else (mc-mc.min())/(mc.max()-mc.min()+1e-9)
    df["truth_full"]=lipe_truth(df,w_edit)
    df["raw_heat"]=(0.5*(df["vol_to_mc"]/2).clip(0,1)+0.5*df["momo_24h01"]).clip(0,1)
    df["divergence"]=(df["raw_heat"]-df["truth_full"]).round(3)
    df["symbol"]=df["symbol"].str.upper()
    return df

def build_yf(tickers):
    prices, meta=yf_multi(tickers)
    if prices.empty: return pd.DataFrame()
    last=prices.ffill().iloc[-1]
    chg24=(prices.ffill().iloc[-1]/prices.ffill().iloc[-2]-1.0)*100.0 if len(prices)>=2 else np.nan
    rows=[]
    for t in prices.columns:
        rows.append({
            "symbol":t.upper(),
            "current_price":float(last.get(t,np.nan)),
            "price_change_percentage_24h_in_currency": float(chg24.get(t,np.nan)) if isinstance(chg24,pd.Series) else float(chg24),
            "market_cap":float(meta.get(t,{}).get("market_cap",np.nan)),
            "total_volume":float(meta.get(t,{}).get("volume",np.nan)),
        })
    df=pd.DataFrame(rows)
    df["momo_24h01"]=df["price_change_percentage_24h_in_currency"].apply(_sig)
    df["momo_7d01"]=0.5
    if df["market_cap"].notna().sum()>0:
        mc=df["market_cap"].fillna(0)
        df["liquidity01"]=0 if mc.max()==0 else (mc-mc.min())/(mc.max()-mc.min()+1e-9)
        df["vol_to_mc"]=(df["total_volume"]/df["market_cap"]).replace([np.inf,-np.inf],np.nan).clip(0,2).fillna(0)
    else:
        v=df["total_volume"].fillna(0)
        df["liquidity01"]=0 if v.max()==0 else (v-v.min())/(v.max()-v.min()+1e-9)
        df["vol_to_mc"]=2*((v-v.min())/(v.max()-v.min()+1e-9))
    df["truth_full"]=lipe_truth(df,w_edit)
    df["raw_heat"]=(0.5*(df["vol_to_mc"]/2).clip(0,1)+0.5*df["momo_24h01"]).clip(0,1)
    df["divergence"]=(df["raw_heat"]-df["truth_full"]).round(3)
    return df

# choose market
if market=="Crypto":
    df = build_crypto()
elif market=="Stocks":
    df = build_yf([t.strip().upper() for t in (st.session_state.get("stocks_in","AAPL,MSFT").split(","))])
else:
    df = build_yf([t.strip().upper() for t in (st.session_state.get("fx_in","EURUSD=X,USDJPY=X").split(","))])

if df.empty:
    st.error("No data loaded."); st.stop()

# ---- Fusion v2: add sentiment + cross-market drift ---------------------------
news_sent, sample_titles = rss_sentiment()  # 0..1
xmkt = float(pd.to_numeric(df["price_change_percentage_24h_in_currency"], errors="coerce").fillna(0).mean())
xmkt01 = float((np.tanh(xmkt/5)+1)/2)
fw=_norm(FUSION_V2)
df["fusion_v2"] = (
    fw["w_truth"]*df["truth_full"] +
    fw["w_sent"] *news_sent +
    fw["w_xmkt"] *xmkt01
).clip(0,1)
df["mood"]=df["fusion_v2"].apply(mood_label)

# ---- KPIs --------------------------------------------------------------------
now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Assets", len(df))
c2.metric("Avg 24h Δ", f"{df['price_change_percentage_24h_in_currency'].mean():+.2f}%")
c3.metric("Avg Truth", f"{df['truth_full'].mean():.2f}")
c4.metric("Fusion v2", f"{df['fusion_v2'].mean():.2f}")
c5.metric("News Sent", f"{news_sent:.2f}")

# ---- Tabs --------------------------------------------------------------------
tab_fusion, tab_truth, tab_raw = st.tabs(
    ["🧠 Fusion v2","🧭 Truth","🔥 Raw"]
)
with tab_fusion:
    st.dataframe(df.sort_values("fusion_v2", ascending=False), use_container_width=True, height=640)
with tab_truth:
    st.dataframe(df.sort_values("truth_full", ascending=False), use_container_width=True, height=640)
with tab_raw:
    st.dataframe(df.sort_values("raw_heat", ascending=False), use_container_width=True, height=640)

st.markdown("<hr>", unsafe_allow_html=True)
st.caption(f"Sources: CoinGecko • RSS • yfinance • CHL Phase 8 © • {now}")
