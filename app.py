============================ PHASE 10 — HYBRID PANEL + PREMIUM DISPLAY ==============================
# Crypto Hybrid Live — Phase 10 (Sidebar Command Panel + Big Tabs + Explain Mode + Compare)
# Paste this entire file into app.py (no spaces above this line).
# ======================================================================================================

# ---- Imports + safe fallbacks -------------------------------------------------------------------------
import math, time, json, re, io
import requests, pandas as pd, numpy as np, streamlit as st
from datetime import datetime, timezone

# Optional plotting (used for small previews)
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

_ensure(["yfinance","feedparser"])

# Finance + feeds (soft import)
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

try:
    import feedparser
    FP_OK = True
except Exception:
    FP_OK = False

# Text polarity (safe fallback)
try:
    from textblob import TextBlob
    def _polarity(t):
        try: return float(TextBlob(t).sentiment.polarity)
        except: return 0.0
except Exception:
    def _polarity(t): return 0.0

# ---- App config ---------------------------------------------------------------------------------------
APP_NAME = "Crypto Hybrid Live — Phase 10 (Hybrid Panel)"
st.set_page_config(page_title=APP_NAME, layout="wide")
USER_AGENT = {"User-Agent":"Mozilla/5.0 (CHL-Phase10)"}
CG_PER_PAGE = 150

# ---- Cache Reset Button --------------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Maintenance")
if st.sidebar.button("🧹 Clear Cache & Restart"):
    st.cache_data.clear(); st.cache_resource.clear(); st.experimental_rerun()

# ---- Theme, Accessibility & CSS ------------------------------------------------------------------------
if "theme" not in st.session_state: st.session_state["theme"]="dark"
if "font_px" not in st.session_state: st.session_state["font_px"]=18
if "contrast" not in st.session_state: st.session_state["contrast"]=False
if "watchlist" not in st.session_state: st.session_state["watchlist"]=[]

def _apply_css():
    dark = st.session_state["theme"]=="dark"
    base_bg = "#0d1117" if dark else "#ffffff"
    base_fg = "#e6edf3" if dark else "#111"
    accent  = "#22c55e" if not st.session_state["contrast"] else "#ffdd00"
    ring    = "#16a34a" if not st.session_state["contrast"] else "#ffaa00"
    font_px = st.session_state["font_px"]

    st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
        background:{base_bg}; color:{base_fg}; font-size:{font_px}px !important;
    }}
    /* Big, prominent tabs */
    div[data-baseweb="tab-list"] button {{
        font-size: {round(font_px*1.3)}px !important;
        font-weight: 800 !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        margin-right: 1rem !important;
        background: linear-gradient(135deg, {accent}, #15803d);
        color: white !important;
        border: 3px solid {ring} !important;
        transition: all .25s ease-in-out;
        transform: scale(1.0);
    }}
    div[data-baseweb="tab-list"] button:hover {{
        transform: scale(1.08);
        box-shadow: 0 0 18px rgba(34,197,94,0.55);
    }}
    div[data-baseweb="tab-list"] button[aria-selected="true"] {{
        background: linear-gradient(135deg, #4ade80, {accent});
        color: #111 !important;
        transform: scale(1.16);
        box-shadow: 0 0 28px rgba(74,222,128,0.9);
    }}
    [data-testid="stMetricValue"] {{
        font-size: {round(font_px*1.35)}px !important;
        font-weight: 800 !important;
    }}
    .phase-banner {{
        font-size: {round(font_px*1.2)}px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, {accent}, #15803d);
        color: #111 if {st.session_state["contrast"]} else white;
        color: white;
        border-radius: 14px;
        padding: .5rem 0;
        margin-bottom: 1rem;
    }}
    .explain {{
        border-left: 5px solid {ring};
        background: rgba(34,197,94,0.08);
        padding: .75rem 1rem; border-radius: 8px;
    }}
    </style>
    """, unsafe_allow_html=True)
_apply_css()

# ---- LIPE Truth / Fusion --------------------------------------------------------------------------------
DEFAULT_WEIGHTS = dict(w_vol=0.30, w_m24=0.25, w_m7=0.25, w_liq=0.20)
PRESETS = {
    "Balanced":  dict(w_vol=0.30, w_m24=0.25, w_m7=0.25, w_liq=0.20),
    "Momentum":  dict(w_vol=0.15, w_m24=0.45, w_m7=0.30, w_liq=0.10),
    "Liquidity": dict(w_vol=0.45, w_m24=0.20, w_m7=0.15, w_liq=0.20),
    "Value":     dict(w_vol=0.25, w_m24=0.20, w_m7=0.20, w_liq=0.35),
}
FUSION_V2 = dict(w_truth=0.70, w_sent=0.15, w_xmkt=0.15)

def _norm(w):
    s=sum(max(0,v) for v in w.values()) or 1.0
    return {k:max(0,v)/s for k,v in w.items()}

def _sig(p):
    if pd.isna(p): return 0.5
    return 1/(1+math.exp(-float(p)/10.0))

def lipe_truth(df, w):
    w=_norm(w or DEFAULT_WEIGHTS)
    if "liquidity01" not in df: df["liquidity01"]=0.0
    if "vol_to_mc" not in df:
        vol=df.get("total_volume",pd.Series(0,index=df.index))
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

# ---- Helpers ------------------------------------------------------------------------------------------------
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
    r=safe_get(u,p); return pd.DataFrame(r.json()) if r else pd.DataFrame()

@st.cache_data(ttl=300)
def rss_sentiment():
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
    pol=np.mean([_polarity(t) for t in titles]); sent01=float((pol+1)/2)
    return sent01, titles[:30]

@st.cache_data(ttl=180)
def yf_multi(ticks, period="6mo"):
    if not YF_OK: return pd.DataFrame(), {}
    data=yf.download(ticks, period=period, interval="1d", auto_adjust=True, progress=False)
    frames=[]; meta={}
    for t in ticks:
        try: ser=data["Adj Close"][t].rename(t) if len(ticks)>1 else data["Adj Close"].rename(t)
        except: continue
        frames.append(ser)
        try: fi=yf.Ticker(t).fast_info
        except: fi={}
        meta[t]={"market_cap":getattr(fi,"market_cap",np.nan),
                 "last_price":getattr(fi,"last_price",np.nan),
                 "volume":getattr(fi,"last_volume",np.nan)}
    return pd.concat(frames,axis=1), meta

# ---- Header ------------------------------------------------------------------------------------------------
st.markdown(f'<div class="phase-banner">🟢 {APP_NAME}</div>', unsafe_allow_html=True)

# ---- Sidebar Command Panel ---------------------------------------------------------------------------------
st.sidebar.header("🧭 Market")
market = st.sidebar.radio("Mode", ["Crypto","Stocks","FX"], horizontal=True)
vs_currency = st.sidebar.selectbox("Currency (Crypto)", ["usd"], index=0)
topn = st.sidebar.slider("Top N (Crypto)", 20, 250, CG_PER_PAGE, 10)

st.sidebar.subheader("🎨 Appearance")
theme_pick = st.sidebar.radio("Theme", ["Dark","Light"], index=0 if st.session_state["theme"]=="dark" else 1, horizontal=True)
st.session_state["theme"]="dark" if theme_pick=="Dark" else "light"
st.session_state["contrast"]=st.sidebar.toggle("High-contrast mode", value=False)
st.session_state["font_px"]=st.sidebar.slider("Global font size", 14, 24, st.session_state["font_px"], 1)
_apply_css()

st.sidebar.subheader("🧭 Truth Preset")
preset = st.sidebar.radio("Preset", list(PRESETS.keys()), index=0, horizontal=True)
w_edit = dict(PRESETS[preset])
st.sidebar.caption("Adjust weights (they auto-normalize).")
for k in list(w_edit.keys()):
    w_edit[k]=st.sidebar.slider(k, 0.0, 1.0, float(w_edit[k]), 0.01)
w_edit=_norm(w_edit)

if market=="Stocks":
    stocks_in = st.sidebar.text_area("Tickers", value="AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA")
if market=="FX":
    fx_in = st.sidebar.text_area("FX pairs", value="EURUSD=X,USDJPY=X,GBPUSD=X,AUDUSD=X,USDCAD=X")

st.sidebar.markdown("---")
st.sidebar.subheader("🟢 Live Mode")
live = st.sidebar.toggle("Auto-refresh", value=False)
every = st.sidebar.slider("Refresh every (sec)", 10, 120, 30, 5)
if live: time.sleep(every)

st.sidebar.markdown("---")
st.sidebar.subheader("⭐ Watchlist")
add_w = st.sidebar.text_input("Add symbol (e.g., BTC, ETH)")
if st.sidebar.button("Add to watchlist") and add_w.strip():
    sym=add_w.strip().upper()
    if sym not in st.session_state["watchlist"]:
        st.session_state["watchlist"].append(sym)
if st.session_state["watchlist"]:
    st.sidebar.write(", ".join(st.session_state["watchlist"]))

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Source Status")
st.sidebar.write("✅ CoinGecko (public)")
st.sidebar.write("✅" if YF_OK else "⚠️", "Yahoo Finance")
st.sidebar.write("✅" if FP_OK else "⚠️", "RSS Sentiment")

# ---- Build Data (Crypto/Stocks/FX) -----------------------------------------------------------------------
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

# choose
if market=="Crypto":
    df = build_crypto()
elif market=="Stocks":
    if not YF_OK:
        st.error("yfinance not available. Ensure 'yfinance' is in requirements.txt"); st.stop()
    ticks=[t.strip().upper() for t in (stocks_in if 'stocks_in' in locals() else "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA").split(",") if t.strip()]
    df = build_yf(ticks)
else:
    if not YF_OK:
        st.error("yfinance not available. Ensure 'yfinance' is in requirements.txt"); st.stop()
    pairs=[t.strip().upper() for t in (fx_in if 'fx_in' in locals() else "EURUSD=X,USDJPY=X,GBPUSD=X,AUDUSD=X,USDCAD=X").split(",") if t.strip()]
    df = build_yf(pairs)

if df.empty:
    st.error("No data loaded."); st.stop()

# ---- Fusion v2 (adds sentiment + cross-market drift) ----------------------------------------------------
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

# ---- KPIs -------------------------------------------------------------------------------------------------
now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
k1,k2,k3,k4 = st.columns(4)
k1.metric("Assets", len(df))
k2.metric("Avg Truth", f"{df['truth_full'].mean():.2f}")
k3.metric("Avg Fusion", f"{df['fusion_v2'].mean():.2f}")
k4.metric("Avg Raw", f"{df['raw_heat'].mean():.2f}")
st.caption(f"Last update: {now}")

# ---- TOP BAR EXPLAINER -----------------------------------------------------------------------------------
with st.expander("📘 What do these scores mean? (Tap to learn)"):
    st.markdown("""
    **Truth (LIPE)** blends: Liquidity, 24-hour momentum, 7-day momentum, and volume/market-cap ratio.  
    **Raw** shows short-term fire: high volume relative to size + 24h move.  
    **Fusion v2** = 70% Truth + 15% News Sentiment + 15% Cross-Market drift.  
    **Divergence** = Raw − Truth (positive → hype spike; negative → steady undervalued).
    """)

# ---- BIG TABS (TRUTH / RAW / FUSION / MOVERS) ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🧭 TRUTH","🔥 RAW","🧠 FUSION","📈 MOVERS"])

with tab1:
    st.subheader("🧭 LIPE Truth (ranked)")
    cols=[c for c in ["symbol","name","current_price","market_cap","liquidity01","truth_full","divergence"] if c in df.columns]
    st.dataframe(df.sort_values("truth_full", ascending=False)[cols], use_container_width=True, height=600)

with tab2:
    st.subheader("🔥 Raw (momentum/volume)")
    cols=[c for c in ["symbol","name","current_price","total_volume","price_change_percentage_24h_in_currency","vol_to_mc","raw_heat"] if c in df.columns]
    st.dataframe(df.sort_values("raw_heat", ascending=False)[cols], use_container_width=True, height=600)

with tab3:
    st.subheader("🧠 Fusion v2 (Truth + sentiment + cross-market)")
    cols=[c for c in ["symbol","name","current_price","market_cap","fusion_v2","truth_full","divergence","mood"] if c in df.columns]
    st.dataframe(df.sort_values("fusion_v2", ascending=False)[cols], use_container_width=True, height=600)
    if sample_titles:
        with st.expander("News sample used for sentiment"):
            for t in sample_titles[:15]: st.write("•", t)

with tab4:
    st.subheader("📈 Divergence Movers")
    st.dataframe(df.sort_values("divergence", ascending=False)[["symbol","name","current_price","divergence"]], use_container_width=True, height=600)

# ---- EXPLAIN MODE ----------------------------------------------------------------------------------------
st.markdown("### 🤖 Explain Mode")
sel = st.selectbox("Pick a symbol to explain", ["(choose)"]+df["symbol"].tolist())
if sel and sel!="(choose)":
    r=df[df["symbol"]==sel].iloc[0]
    st.markdown(f"""
    <div class="explain">
      <b>{r.get('name', sel)} ({sel})</b><br/>
      • <b>Truth</b>: {r['truth_full']:.2f} — weighted blend of liquidity, 24h/7d momentum, and volume/size.<br/>
      • <b>Raw</b>: {r['raw_heat']:.2f} — short-term heat (volume vs. size + 24h move).<br/>
      • <b>Fusion</b>: {r['fusion_v2']:.2f} — Truth plus news sentiment and overall market drift.<br/>
      • <b>Divergence</b>: {r['divergence']:+.2f} — positive suggests hype spike; negative suggests quiet strength.<br/>
      <i>Tip:</i> High Truth + slightly negative Divergence can indicate undervalued builders; High Raw + low Truth can indicate pumpy risk.
    </div>
    """, unsafe_allow_html=True)

# ---- QUICK COMPARE ---------------------------------------------------------------------------------------
st.markdown("### 🆚 Quick Compare")
cA,cB=st.columns(2)
with cA: a = st.text_input("Symbol A", value="")
with cB: b = st.text_input("Symbol B", value="")
if st.button("Compare"):
    A=df[df["symbol"]==a.strip().upper()]
    B=df[df["symbol"]==b.strip().upper()]
    if A.empty or B.empty:
        st.warning("Enter two symbols that exist in the current table.")
    else:
        AA=A.iloc[0]; BB=B.iloc[0]
        cmp=pd.DataFrame([
            {"metric":"Truth","A":AA["truth_full"],"B":BB["truth_full"]},
            {"metric":"Raw","A":AA["raw_heat"],"B":BB["raw_heat"]},
            {"metric":"Fusion","A":AA["fusion_v2"],"B":BB["fusion_v2"]},
            {"metric":"Divergence","A":AA["divergence"],"B":BB["divergence"]},
            {"metric":"Price","A":AA["current_price"],"B":BB["current_price"]},
        ])
        st.dataframe(cmp, use_container_width=True)

# ---- EXPORT ------------------------------------------------------------------------------------------------
st.markdown("### 📤 Export")
choice = st.selectbox("Choose table to download", ["Fusion v2","Truth","Raw","Movers"])
if choice=="Fusion v2":
    out=df.sort_values("fusion_v2", ascending=False)
elif choice=="Truth":
    out=df.sort_values("truth_full", ascending=False)
elif choice=="Raw":
    out=df.sort_values("raw_heat", ascending=False)
else:
    out=df.sort_values("divergence", ascending=False)
csv=out.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", csv, file_name=f"chl_{choice.lower().replace(' ','_')}.csv", mime="text/csv")

# ---- Footer ------------------------------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.caption(f"Sources: CoinGecko • RSS • yfinance • CHL Phase 10 © • {now}")
