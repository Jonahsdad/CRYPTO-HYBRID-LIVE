# ====================== IMPORTS ======================
import math, time, random, re, json
import requests, pandas as pd, numpy as np, streamlit as st
from datetime import datetime, timezone
from io import StringIO
try:
    import plotly.express as px
    PLOTLY_OK=True
except Exception:
    PLOTLY_OK=False
try:
    import yfinance as yf
    YF_OK=True
except Exception:
    YF_OK=False
try:
    from textblob import TextBlob
    HAS_TEXTBLOB=True
except Exception:
    HAS_TEXTBLOB=False
    class TextBlob:
        def __init__(self,text):self.text=text
        @property
        def sentiment(self):return type("S",(),{"polarity":0.0})()
def _polarity_safe(t:str)->float:
    try:return float(TextBlob(t).sentiment.polarity)
    except Exception:return 0.0
APP_NAME="Crypto Hybrid Live — Phase 6 (Multi-Market)"
st.set_page_config(page_title=APP_NAME,layout="wide")
USER_AGENT={"User-Agent":"Mozilla/5.0 (CHL-Phase6)"}
FEATURES={"REDDIT":True,"DEFI_LLAMA":True,"ALERTS":True,"SNAPSHOT":True}
CG_PER_PAGE=150
HIST_DAYS=90
if "theme" not in st.session_state:st.session_state["theme"]="dark"
def apply_theme_css():
    dark=st.session_state.get("theme","dark")=="dark"
    base_bg="#0d1117" if dark else "#ffffff"
    base_fg="#e6edf3" if dark else "#111111"
    accent="#23d18b" if dark else "#0b8f5a"
    st.markdown(f"""
    <style>
    html,body,[class*="css"]{{font-size:18px;background:{base_bg};color:{base_fg};}}
    .stTabs [role="tablist"] button{{font-size:1.3rem!important;font-weight:700!important;margin-right:.7rem;border-radius:10px;background:#111;color:{accent};}}
    .stTabs [role="tablist"] button[aria-selected="true"]{{background:{accent};color:#000;transform:scale(1.05);}}
    [data-testid="stMetricValue"]{{font-size:2.3rem!important}}
    </style>
    """,unsafe_allow_html=True)
apply_theme_css()
DEFAULT_WEIGHTS=dict(w_vol=0.30,w_m24=0.25,w_m7=0.25,w_liq=0.20)
PRESETS={
"Balanced":dict(w_vol=0.30,w_m24=0.25,w_m7=0.25,w_liq=0.20),
"Momentum":dict(w_vol=0.15,w_m24=0.45,w_m7=0.30,w_liq=0.10),
"Liquidity":dict(w_vol=0.45,w_m24=0.20,w_m7=0.15,w_liq=0.20),
"Value":dict(w_vol=0.25,w_m24=0.20,w_m7=0.20,w_liq=0.35),
}
FUSION_WEIGHTS=dict(w_truth=0.70,w_social=0.15,w_tvl=0.10)
def _normalize_weights(w):
    s=sum(max(0,v) for v in w.values()) or 1.0
    return {k:max(0,v)/s for k,v in w.items()}
def pct_sigmoid(p):
    if pd.isna(p):return 0.5
    return 1/(1+math.exp(-float(p)/10))
def lipe_truth(df,w):
    w=_normalize_weights(w or DEFAULT_WEIGHTS)
    if "liquidity01" not in df:df["liquidity01"]=0.0
    if "vol_to_mc" not in df:
        vol=df.get("total_volume",pd.Series(0,index=df.index))
        v01=(vol-vol.min())/(vol.max()-vol.min()+1e-9)
        df["vol_to_mc"]=2*v01
    return (w["w_vol"]*(df["vol_to_mc"]/2)
          +w["w_m24"]*df.get("momo_24h01",.5)
          +w["w_m7"]*df.get("momo_7d01",.5)
          +w["w_liq"]*df["liquidity01"]).clip(0,1)
def mood_label(x):
    if x>=.8:return"🟢 EUPHORIC"
    if x>=.6:return"🟡 OPTIMISTIC"
    if x>=.4:return"🟠 NEUTRAL"
    return"🔴 FEARFUL"
def safe_get(u,p=None,t=25):
    try:
        r=requests.get(u,params=p,headers=USER_AGENT,timeout=t)
        if r.status_code==200:return r
    except:pass
    return None
@st.cache_data(ttl=60)
def fetch_markets_cg(vs="usd",limit=150):
    u="https://api.coingecko.com/api/v3/coins/markets"
    p={"vs_currency":vs,"order":"market_cap_desc","per_page":limit,"page":1,"sparkline":"false","price_change_percentage":"1h,24h,7d"}
    r=safe_get(u,p)
    return pd.DataFrame(r.json()) if r else pd.DataFrame()
@st.cache_data(ttl=600)
def cg_chart(cid,vs="usd",days=90):
    u=f"https://api.coingecko.com/api/v3/coins/{cid}/market_chart"
    r=safe_get(u,{"vs_currency":vs,"days":days})
    if not r:return pd.DataFrame()
    j=r.json().get("prices",[])
    if not j:return pd.DataFrame()
    d=pd.DataFrame(j,columns=["ts","price"]);d["ts"]=pd.to_datetime(d["ts"],unit="ms");return d
@st.cache_data(ttl=180)
def yf_multi(ticks,period="6mo"):
    if not YF_OK:return pd.DataFrame(),{}
    data=yf.download(ticks,period=period,interval="1d",auto_adjust=True,progress=False)
    frames=[];meta={}
    for t in ticks:
        try:ser=data["Adj Close"][t].rename(t) if len(ticks)>1 else data["Adj Close"].rename(t)
        except:continue
        frames.append(ser)
        try:i=yf.Ticker(t).fast_info
        except:i={}
        meta[t]={"market_cap":getattr(i,"market_cap",np.nan),"last_price":getattr(i,"last_price",np.nan),"volume":getattr(i,"last_volume",np.nan)}
    return pd.concat(frames,axis=1),meta
st.title("🟢 "+APP_NAME)
st.caption("Truth > Noise • Phase 6 adds Multi-Market (Crypto + Stocks + FX)")
with st.sidebar:
    st.header("🧭 Market")
    market=st.radio("Choose market",["Crypto","Stocks","FX"],horizontal=True)
    vs_currency=st.selectbox("Currency (Crypto)",["usd"])
    topn=st.slider("Show top N",20,250,150,10)
    st.subheader("Theme")
    theme=st.radio("Theme",["Dark","Light"],index=0 if st.session_state["theme"]=="dark" else 1,horizontal=True)
    st.session_state["theme"]="dark" if theme=="Dark" else"light";apply_theme_css()
    st.subheader("Truth Preset")
    preset=st.radio("Preset",list(PRESETS.keys()),horizontal=True)
    st.subheader("Weights")
    w_edit=dict(PRESETS[preset])
    for k in w_edit:w_edit[k]=st.slider(k,0.0,1.0,float(w_edit[k]),0.01)
    w_edit=_normalize_weights(w_edit)
    if market=="Stocks":
        st.subheader("Stocks tickers");stocks_in=st.text_area("",value="AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA")
    if market=="FX":
        st.subheader("FX pairs");fx_in=st.text_area("",value="EURUSD=X,USDJPY=X,GBPUSD=X,AUDUSD=X,USDCAD=X")
def build_crypto():
    df=fetch_markets_cg(vs_currency,topn)
    if df.empty:return df
    df["vol_to_mc"]=(df["total_volume"]/df["market_cap"]).clip(0,2).fillna(0)
    df["momo_24h01"]=df["price_change_percentage_24h_in_currency"].apply(pct_sigmoid)
    df["momo_7d01"]=df["price_change_percentage_7d_in_currency"].apply(pct_sigmoid)
    mc=df["market_cap"].fillna(0)
    df["liquidity01"]=(mc-mc.min())/(mc.max()-mc.min()+1e-9)
    df["truth_full"]=lipe_truth(df,w_edit)
    df["fusion_truth"]=(0.8*df["truth_full"]).clip(0,1)
    df["mood"]=df["truth_full"].apply(mood_label)
    return df
def build_yf(tickers):
    prices,meta=yf_multi(tickers)
    if prices.empty:return pd.DataFrame()
    last=prices.iloc[-1];chg24=(prices.iloc[-1]/prices.iloc[-2]-1)*100 if len(prices)>2 else 0
    df=pd.DataFrame({"symbol":prices.columns,"current_price":last.values,"price_change_percentage_24h_in_currency":chg24.values})
    df["momo_7d01"]=.5;df["momo_24h01"]=df["price_change_percentage_24h_in_currency"].apply(pct_sigmoid)
    df["liquidity01"]=.5;df["vol_to_mc"]=1
    df["truth_full"]=lipe_truth(df,w_edit)
    df["fusion_truth"]=df["truth_full"]
    df["mood"]=df["truth_full"].apply(mood_label)
    return df
if market=="Crypto":df=build_crypto()
elif market=="Stocks":df=build_yf([x.strip().upper() for x in stocks_in.split(",")])
else:df=build_yf([x.strip().upper() for x in fx_in.split(",")])
if df.empty:st.error("No data loaded");st.stop()
c1,c2,c3,c4=st.columns(4)
c1.metric("Assets",len(df))
c2.metric("Avg 24h Δ",f"{df['price_change_percentage_24h_in_currency'].mean():+.2f}%")
c3.metric("Avg Truth",f"{df['truth_full'].mean():.2f}")
c4.metric("Avg Fusion",f"{df['fusion_truth'].mean():.2f}")
tab1,tab2=st.tabs(["Fusion Truth","Raw"])
with tab1:
    st.subheader("Fusion Truth")
    st.dataframe(df.sort_values("fusion_truth",ascending=False)[["symbol","current_price","fusion_truth","mood"]],use_container_width=True,height=600)
with tab2:
    st.subheader("Raw")
    st.dataframe(df[["symbol","current_price","price_change_percentage_24h_in_currency"]],use_container_width=True,height=600)
st.markdown("<hr>",unsafe_allow_html=True)
st.caption("Sources: CoinGecko | yfinance • CHL Phase 6 © 2025")
