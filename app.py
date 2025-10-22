# =====================================================
# CRYPTO HYBRID LIVE — single-file Streamlit app
# Author: Jesse Ray Landingham Jr
# =====================================================

from __future__ import annotations
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Optional deps
try:
    import plotly.express as px
    HAS_PX = True
except Exception:
    HAS_PX = False

try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

# ----------------------- Page config -----------------------
st.set_page_config(
    page_title="Crypto Hybrid Live",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded",
)

# ----------------------- CSS -----------------------
st.markdown("""
<style>
.hero {
  margin:8px 0 18px 0;
  padding:14px 18px;
  border-radius:14px;
  border:1px solid #0d253a;
  background:radial-gradient(120% 160% at 0% 0%, #0f172a 0%, #052c3b 40%, #0b1f33 100%);
  color:#e6f1ff;
  box-shadow:0 8px 28px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
}
.hero h1{font-size:34px;margin:4px 0 2px 0;}
.hero .sub{opacity:.85;font-size:13px;}
.badge-row{display:flex;gap:10px;margin-top:8px;flex-wrap:wrap;}
.badge{display:inline-flex;align-items:center;gap:8px;
  padding:8px 12px;border-radius:10px;border:1px solid rgba(255,255,255,.08);
  background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,.02));
  color:#e6f1ff;font-weight:700;font-size:13px;}
.kpi{background:#0e1726;border:1px solid #14243a;border-radius:12px;padding:12px 14px;}
.kpi .label{opacity:.8;font-size:12px}
.kpi .value{font-weight:800;font-size:18px}
</style>
""", unsafe_allow_html=True)

# ----------------------- Hero -----------------------
st.markdown("""
<div class="hero">
  <h1>🚀 CRYPTO HYBRID LIVE</h1>
  <div class="sub">Powered by <b>Jesse Ray Landingham Jr</b></div>
  <div class="badge-row">
    <div class="badge">🔥 RAW</div>
    <div class="badge">💧 TRUTH</div>
    <div class="badge">⭐ CONFLUENCE</div>
    <div class="badge">⚡ Δ (RAW↔TRUTH)</div>
  </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# Helpers
# =====================================================
@st.cache_data(ttl=300)
def cg_top_market(per_page:int=20)->pd.DataFrame:
    url="https://api.coingecko.com/api/v3/coins/markets"
    params=dict(vs_currency="usd",order="market_cap_desc",per_page=per_page,page=1,
                sparkline=False,price_change_percentage="24h")
    try:
        r=requests.get(url,params=params,timeout=12)
        r.raise_for_status()
        d=pd.DataFrame(r.json())
        cols=["name","symbol","current_price","market_cap","price_change_percentage_24h"]
        d=d[cols].rename(columns={
            "name":"Name","symbol":"Symbol","current_price":"Price (USD)",
            "market_cap":"Market Cap","price_change_percentage_24h":"24h %"})
        return d
    except Exception as e:
        st.warning(f"⚠️ CoinGecko error: {e}. Using fallback.")
        return pd.DataFrame({
            "Name":["Bitcoin","Ethereum","Solana","XRP","Cardano"],
            "Symbol":["BTC","ETH","SOL","XRP","ADA"],
            "Price (USD)":[67000,3500,180,0.52,0.25],
            "Market Cap":[1.2e12,7.5e11,8.0e10,2.8e10,1.0e10],
            "24h %":[2.4,-1.2,0.6,-0.8,3.1],
        })

@st.cache_data(ttl=3600)
def load_sp500_universe()->pd.DataFrame:
    try:
        url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df=pd.read_html(url)[0]
        df=df.rename(columns={"Symbol":"Symbol","Security":"Security",
                              "GICS Sector":"GICS Sector",
                              "GICS Sub-Industry":"GICS Sub-Industry"})
        return df[["Symbol","Security","GICS Sector","GICS Sub-Industry"]]
    except Exception:
        st.warning("⚠️ Wikipedia unreachable – using fallback.")
        return pd.DataFrame({
            "Symbol":["AAPL","MSFT","GOOGL","AMZN","META"],
            "Security":["Apple","Microsoft","Alphabet","Amazon","Meta Platforms"],
            "GICS Sector":["Information Technology"]*5,
            "GICS Sub-Industry":["Consumer Electronics","Systems Software",
                                 "Interactive Media","E-Commerce","Social Media"]
        })

def attach_latest_prices(df:pd.DataFrame,n:int=10)->pd.DataFrame:
    out=df.copy()
    if out.empty: return out
    syms=out["Symbol"].astype(str).str.replace(".","-",regex=False).head(n)
    if HAS_YF:
        try:
            prices=[]
            for s in syms:
                try:
                    h=yf.Ticker(s).history(period="1d")
                    prices.append(float(h["Close"].iloc[-1]))
                except Exception:
                    prices.append(np.nan)
            out.loc[out.index[:len(prices)],"Latest Price"]=prices
            if out["Latest Price"].isna().all():
                raise RuntimeError("No yfinance prices.")
            return out
        except Exception as e:
            st.warning(f"⚠️ yfinance error: {e}. Using synthetic fallback.")
    out["Latest Price"]=np.random.default_rng(42).uniform(80,350,len(out)).round(2)
    return out

# =====================================================
# Views
# =====================================================
def view_crypto_dashboard():
    st.subheader("📊 Market Overview (Crypto)")
    df=cg_top_market(20)
    c1,c2,c3,c4=st.columns(4)
    c1.markdown(f'<div class="kpi"><div class="label">Assets</div>'
                f'<div class="value">{len(df)}</div></div>',unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi"><div class="label">Avg 24h %</div>'
                f'<div class="value">{df["24h %"].mean():.2f}%</div></div>',unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi"><div class="label">Top Price</div>'
                f'<div class="value">${df["Price (USD)"].max():,.0f}</div></div>',unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi"><div class="label">Median Cap</div>'
                f'<div class="value">${df["Market Cap"].median():,.0f}</div></div>',unsafe_allow_html=True)
    st.dataframe(df,use_container_width=True,height=380)
    if HAS_PX:
        fig=px.bar(df.head(10),x="Symbol",y="24h %",title="Top 10 — 24h Change (%)",color="24h %")
        st.plotly_chart(fig,use_container_width=True)

def view_sp500():
    st.subheader("🏛️ S&P 500 — Live Snapshot")
    base=load_sp500_universe()
    st.dataframe(base.head(20),use_container_width=True,height=360)
    priced=attach_latest_prices(base,10)
    if "Latest Price" in priced.columns:
        st.success("✅ Prices attached (yfinance or fallback).")
        st.dataframe(priced.head(10),use_container_width=True)
        if HAS_PX:
            fig=px.bar(priced.head(10),x="Symbol",y="Latest Price",title="Latest Prices (sample)")
            st.plotly_chart(fig,use_container_width=True)

def view_forecast():
    st.subheader("🧠 Forecast (coming soon)")
    st.write("Predictive modules (LIPE) attach here.")

def view_about():
    st.subheader("ℹ️ About")
    st.write("Crypto Hybrid Live blends RAW + TRUTH + CONFLUENCE for a clean market snapshot.")

# =====================================================
# Router
# =====================================================
with st.sidebar:
    st.header("Navigate")
    page=st.radio("",["Dashboard (Crypto)","S&P 500","Forecast","About"],index=0)

if page=="Dashboard (Crypto)":
    view_crypto_dashboard()
elif page=="S&P 500":
    view_sp500()
elif page=="Forecast":
    view_forecast()
else:
    view_about()
