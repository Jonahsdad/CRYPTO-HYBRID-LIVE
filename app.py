# =====================================================
# CRYPTO HYBRID LIVE — SINGLE FILE (STABLE FULL FIX)
# Phase 19.x  •  Hero 3x  •  Truth/Raw/Confluence  •  Chart
# Stocks (yfinance optional)  •  Options (yfinance optional)
# "Powered by JESSE RAY LANDINGHAM JR" co-hero
# =====================================================

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# -------- optional providers (app still runs if missing) --------
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

# ============================== CONFIG ===========================
APP_NAME   = "CRYPTO HYBRID LIVE"g
POWERED_BY = "POWERED BY JESSE RAY LANDINGHAM JR"
PHASE_TAG  = "PHASE 19.x — Hero 3× • Truth Pills • Multi-Source Stocks"

st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
/* Keep everything predictable and robust */
html, body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }

/* Big hero, pushed down so it never clips under the navbar */
.hero-wrap {
  margin-top: 36px;
  margin-bottom: 14px;
}
.hero {
  width: 100%;
  border-radius: 18px;
  padding: 28px 28px;
  background: linear-gradient(90deg, #061a12 0%, #0b3c2a 50%, #061a12 100%);
  border: 1px solid #1f3a2e;
  display: flex; justify-content: space-between; align-items: center;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35) inset, 0 4px 16px rgba(0,0,0,0.25);
}
.hero-title {
  font-weight: 950; letter-spacing: 1.2px;
  font-size: clamp(28px, 4.2vw, 54px); /* ~3× normal */
  color: #c7ffee; text-shadow: 0 2px 0 #073, 0 0 12px #0a6a4f;
  margin: 0;
}
.hero-powered {
  font-weight: 900; font-size: clamp(12px, 1.6vw, 18px);
  color: #9ad7ff; padding: 10px 14px; border-radius: 999px;
  background: rgba(8,23,36,0.7); border: 1px solid #24506b;
}

/* Pills under the KPIs */
.pills { display:flex; gap:10px; margin: 10px 0 8px 0; }
.pill {
  display:inline-flex; align-items:center; gap:8px;
  padding: 10px 14px; border-radius: 12px; font-weight: 800;
  border: 1px solid #ffffff20;
}
.p-raw  { background: linear-gradient(180deg,#2b180e,#1f140c); color:#ffad77; }
.p-tru  { background: linear-gradient(180deg,#11222a,#0d171d); color:#7bd8ff; }
.p-conf { background: linear-gradient(180deg,#2a200f,#1b1510); color:#ffd86b; }
.p-gap  { background: linear-gradient(180deg,#1e1830,#131022); color:#bfa3ff; }

/* Metric cards */
.metric {
  background:#0d1117; border:1px solid #ffffff22; border-radius:14px;
  padding:14px; height:100%;
}

/* Make Streamlit dataframes nicely rounded */
.stDataFrame, .stDataEditor { border-radius: 12px; overflow: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============================== UTIL =============================
def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def pct_sigmoid(pct) -> float:
    """Squash percentage to 0..1 (stable)."""
    try:
        x = float(pct) / 10.0
        return 1.0 / (1.0 + np.exp(-x))
    except Exception:
        return 0.5

@st.cache_data(ttl=60, show_spinner=False)
def cg_markets(vs: str = "usd", per_page: int = 200) -> pd.DataFrame:
    """CoinGecko markets with consistent columns."""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": vs, "order": "market_cap_desc", "per_page": int(max(1, min(250, per_page))),
        "page": 1, "sparkline": "false", "price_change_percentage": "1h,24h,7d", "locale": "en"
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    needed = [
        "name","symbol","current_price","market_cap","total_volume",
        "price_change_percentage_1h_in_currency",
        "price_change_percentage_24h_in_currency",
        "price_change_percentage_7d_in_currency"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    return df[needed]

def build_scores(df: pd.DataFrame,
                 w_volmc=0.30, w_m24=0.25, w_m7d=0.25, w_liq=0.20) -> pd.DataFrame:
    """Truth / Raw / Confluence."""
    if df is None or df.empty:
        return pd.DataFrame()

    t = df.copy()
    # base inputs
    t["vol_to_mc"] = (t["total_volume"] / t["market_cap"]).replace([np.inf, -np.inf], np.nan)
    t["vol_to_mc"] = t["vol_to_mc"].clip(lower=0, upper=2).fillna(0)
    m1h = t["price_change_percentage_1h_in_currency"].apply(pct_sigmoid).fillna(0.5)
    m24 = t["price_change_percentage_24h_in_currency"].apply(pct_sigmoid).fillna(0.5)
    m7d = t["price_change_percentage_7d_in_currency"].apply(pct_sigmoid).fillna(0.5)

    # liquidity proxy
    mc = t["market_cap"].fillna(0)
    t["liq01"] = 0 if mc.max() == 0 else ((mc - mc.min()) / (mc.max() - mc.min() + 1e-9)).clip(0,1)

    # RAW = crowd heat (volume/mcap + 1h)
    t["raw_heat"] = (0.5*(t["vol_to_mc"]/2) + 0.5*m1h).clip(0,1)

    # TRUTH = stability (vol/mcap + 24h + 7d + liquidity)
    t["truth_full"] = (w_volmc*(t["vol_to_mc"]/2) + w_m24*m24 + w_m7d*m7d + w_liq*t["liq01"]).clip(0,1)

    # Confluence = fusion of both with agreement/consistency/energy
    t["consistency01"] = (1 - (m24 - m7d).abs()).clip(0,1)
    t["agreement01"]   = (1 - (t["raw_heat"] - t["truth_full"]).abs()).clip(0,1)
    t["energy01"]      = (t["vol_to_mc"]/2).clip(0,1)
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
    t["CONF_BADGE"]       = t["confluence01"].apply(star)

    return t

# ============================== HERO ==============================
def hero() -> None:
    st.markdown(
        f"""
<div class="hero-wrap">
  <div class="hero">
     <div class="hero-title">{APP_NAME}</div>
     <div class="hero-powered">{POWERED_BY}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# ============================== KPI + PILLS =======================
def kpis(df_scored: pd.DataFrame, mode: str) -> None:
    c1,c2,c3,c4,c5 = st.columns([1,1,1,1,1])
    n = len(df_scored)
    p24 = float(df_scored["price_change_percentage_24h_in_currency"].mean()) if n else 0.0
    tavg = float(df_scored["truth_full"].mean()) if n else 0.0
    ravg = float(df_scored["raw_heat"].mean()) if n else 0.0
    cavg = float(df_scored["confluence01"].mean()) if n else 0.0
    with c1: st.markdown(f"<div class='metric'><b>Assets</b><br>{n}</div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric'><b>Avg 24h %</b><br>{p24:.2f}%</div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric'><b>Avg TRUTH</b><br>{tavg:.2f}</div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric'><b>Avg RAW</b><br>{ravg:.2f}</div>", unsafe_allow_html=True)
    with c5: st.markdown(f"<div class='metric'><b>Avg Confluence</b><br>{cavg:.2f}</div>", unsafe_allow_html=True)

    st.caption(f"{PHASE_TAG} • Updated {now_utc()} • Mode: {mode}")

    st.markdown(
        """
<div class="pills">
  <div class="pill p-raw">🔥 RAW <span style="opacity:.7">= crowd heat (vol/mcap + 1h)</span></div>
  <div class="pill p-tru">💧 TRUTH <span style="opacity:.7">= stability (vol/mcap + 24h + 7d + size)</span></div>
  <div class="pill p-conf">⭐ CONFLUENCE <span style="opacity:.7">= RAW + TRUTH agree, consistent & liquid</span></div>
  <div class="pill p-gap">⚡ Δ (RAW↔TRUTH) <span style="opacity:.7">= gap</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

# ============================== TABLE/CHART =======================
def table(df: pd.DataFrame, cols: List[str]) -> None:
    have = [c for c in cols if c in df.columns]
    st.dataframe(
        df[have], use_container_width=True, hide_index=True,
        column_config={
            "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "price_change_percentage_24h_in_currency": st.column_config.NumberColumn("24h %", format="%.2f%%"),
            "raw_heat": st.column_config.ProgressColumn("RAW", min_value=0.0, max_value=1.0),
            "truth_full": st.column_config.ProgressColumn("TRUTH", min_value=0.0, max_value=1.0),
            "confluence01": st.column_config.ProgressColumn("CONF", min_value=0.0, max_value=1.0),
            "divergence": st.column_config.ProgressColumn("Δ", min_value=0.0, max_value=1.0),
            "RAW_BADGE": st.column_config.TextColumn("🔥"),
            "TRUTH_BADGE": st.column_config.TextColumn("💧"),
            "CONF_BADGE": st.column_config.TextColumn("⭐"),
            "name": st.column_config.TextColumn("Name"),
            "symbol": st.column_config.TextColumn("Symbol"),
        },
    )

def leaders_chart(df_scored: pd.DataFrame, metric: str, topn: int = 18, title: str = "Leaders") -> None:
    """Compact bar chart for selected metric (safe if plotly missing)."""
    try:
        import plotly.express as px  # local import keeps startup fast & optional
        cols = ["name", "symbol", "current_price", metric]
        data = df_scored.sort_values(metric, ascending=False).head(topn)[cols]
        fig = px.bar(
            data, x="symbol", y=metric,
            hover_data=["name","current_price"] if "current_price" in data else ["name"],
        )
        fig.update_layout(height=320, title=title, margin=dict(l=8,r=8,t=42,b=8))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Chart unavailable ({e}).")

# ============================== STOCKS (Optional) =================
@st.cache_data(ttl=180)
def sp500_list() -> List[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        syms = df["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False).tolist()
        return syms
    except Exception:
        return ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA"]

@st.cache_data(ttl=180)
def yf_snapshot_daily(tickers: List[str]) -> pd.DataFrame:
    if not HAS_YF or not tickers: return pd.DataFrame()
    try:
        data = yf.download(" ".join(tickers), period="5d", interval="1d",
                           auto_adjust=True, group_by="ticker", threads=True, progress=False)
    except Exception:
        return pd.DataFrame()

    rows = []
    for t in tickers:
        try:
            s = data[t]
            last = float(s.iloc[-1]["Close"])
            prev = float(s.iloc[-2]["Close"]) if len(s) > 1 else np.nan
            pct  = (last/prev - 1.0)*100.0 if pd.notna(prev) else np.nan
            rows.append(dict(name=t, symbol=t, current_price=last,
                             price_change_percentage_24h_in_currency=pct,
                             market_cap=np.nan, total_volume=np.nan,
                             price_change_percentage_1h_in_currency=np.nan,
                             price_change_percentage_7d_in_currency=np.nan))
        except Exception:
            pass
    return pd.DataFrame(rows)

# ============================== PAGES ============================
def page_dashboard() -> None:
    hero()
    dfc = build_scores(cg_markets("usd", 200))
    kpis(dfc, "Crypto")

    # interactive chart selector
    metric = st.segmented_control(
        "Quick leaders",
        options=["confluence01","truth_full","raw_heat","divergence"],
        format_func=lambda x: {"confluence01":"Confluence ⭐","truth_full":"Truth 💧","raw_heat":"Raw 🔥","divergence":"Δ (RAW↔TRUTH)"}[x],
        selection_mode="single",
        key="metric_sel",
    )
    leaders_chart(dfc, metric, topn=18, title="Leaders")

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Top Confluence (Crypto)")
        table(
            dfc.sort_values("confluence01", ascending=False).head(20),
            ["name","symbol","current_price","CONF_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence"]
        )
    with c2:
        st.subheader("Top TRUTH (Crypto)")
        table(
            dfc.sort_values("truth_full", ascending=False).head(20),
            ["name","symbol","current_price","TRUTH_BADGE","truth_full","RAW_BADGE","divergence"]
        )

def page_crypto() -> None:
    hero()
    top_n = st.slider("Pull Top N (CoinGecko)", 50, 250, 200, step=50)
    df = build_scores(cg_markets("usd", top_n))
    kpis(df, "Crypto")

    c1,c2,c3,c4 = st.columns(4)
    with c1: show_n = st.slider("Show", 20, top_n, min(150, top_n), step=10)
    with c2: order  = st.selectbox("Order by", ["confluence01","truth_full","raw_heat","divergence"], index=0)
    with c3: min_truth = st.slider("Min TRUTH", 0.0, 1.0, 0.0, 0.05)
    with c4: q = st.text_input("Search symbol/name").strip().lower()

    out = df.copy()
    if min_truth>0: out = out[out["truth_full"]>=min_truth]
    if q:
        mask = out["name"].str.lower().str.contains(q, na=False) | out["symbol"].str.lower().str.contains(q, na=False)
        out = out[mask]
    out = out.sort_values(order, ascending=False).head(show_n)
    table(out, ["name","symbol","current_price","CONF_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence"])

def page_sp500() -> None:
    hero()
    st.subheader("S&P 500 Snapshot (optional provider)")
    if not HAS_YF:
        st.error("yfinance not installed. Add `yfinance` to requirements and redeploy.")
        return
    syms = sp500_list()
    n = st.slider("Tickers to snapshot", 50, len(syms), 200, step=50)
    snap = yf_snapshot_daily(syms[:n])
    if snap.empty:
        st.warning("No price data returned right now, try fewer tickers.")
        return
    scored = build_scores(snap)
    kpis(scored, "S&P 500")
    table(
        scored.sort_values("confluence01", ascending=False).head(100),
        ["name","symbol","current_price","CONF_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","divergence"]
    )

def page_options() -> None:
    hero()
    st.subheader("Options Explorer (yfinance)")
    if not HAS_YF:
        st.error("yfinance not installed. Add `yfinance` to requirements and redeploy.")
        return
    default = "AAPL"
    sym = st.text_input("Ticker", value=default).strip().upper()
    if not sym: return
    try:
        exps = yf.Ticker(sym).options
    except Exception:
        exps = []
    if not exps:
        st.info("No expirations available.")
        return
    exp = st.selectbox("Expiration", options=exps, index=0)
    try:
        chain = yf.Ticker(sym).option_chain(exp)
        calls, puts = chain.calls, chain.puts
    except Exception:
        calls, puts = pd.DataFrame(), pd.DataFrame()
    c1,c2 = st.columns(2)
    with c1:
        st.caption(f"{sym} Calls — {exp}")
        if calls.empty: st.info("No calls")
        else:
            keep = ["contractSymbol","strike","lastPrice","openInterest","volume","impliedVolatility"]
            table(calls[[c for c in keep if c in calls.columns]].sort_values("openInterest", ascending=False).head(30), keep)
    with c2:
        st.caption(f"{sym} Puts — {exp}")
        if puts.empty: st.info("No puts")
        else:
            keep = ["contractSymbol","strike","lastPrice","openInterest","volume","impliedVolatility"]
            table(puts[[c for c in keep if c in puts.columns]].sort_values("openInterest", ascending=False).head(30), keep)

def page_scores() -> None:
    hero()
    st.subheader("Score Lens — quick read")
    st.markdown(
        """
**🔥 RAW (0..1)** — crowd heat (volume/mcap + 1h momentum)  
**💧 TRUTH (0..1)** — stability (vol/mcap + 24h + 7d + liquidity/size)  
**⚡ Δ (0..1)** — |RAW − TRUTH| (gap)  
**⭐ CONFLUENCE (0..1)** — fusion of RAW + TRUTH with agreement (RAW≈TRUTH),
consistency (24h≈7d), energy & liquidity.

**Fast read**
- ⭐ High **Confluence** → hype & quality aligned (prime)
- 🔥 High **RAW**, low **TRUTH** → hype spike (fragile)
- 💧 High **TRUTH**, low **RAW** → sleeper quality (crowd not there yet)
"""
    )

# ============================== SIDEBAR & ROUTER ==================
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Dashboard","Crypto","S&P 500","Options","Scores"], index=0)

    st.header("Truth Weights")
    w1 = st.slider("Vol/Mcap", 0.0, 1.0, 0.30, 0.05)
    w2 = st.slider("24h Momentum", 0.0, 1.0, 0.25, 0.05)
    w3 = st.slider("7d Momentum", 0.0, 1.0, 0.25, 0.05)
    w4 = st.slider("Liquidity/Size", 0.0, 1.0, 0.20, 0.05)

    st.header("Auto Refresh")
    auto = st.toggle("Auto refresh", False)
    every = st.slider("Every (sec)", 10, 120, 30, step=5)

# apply weights globally by re-wrapping build_scores
orig_build_scores = build_scores
def build_scores(df: pd.DataFrame, w_volmc=w1, w_m24=w2, w_m7d=w3, w_liq=w4) -> pd.DataFrame:  # type: ignore
    return orig_build_scores(df, w_volmc, w_m24, w_m7d, w_liq)

# route
if page == "Dashboard":
    page_dashboard()
elif page == "Crypto":
    page_crypto()
elif page == "S&P 500":
    page_sp500()
elif page == "Options":
    page_options()
else:
    page_scores()

# auto refresh
if auto:
    st.caption(f"{PHASE_TAG} • Auto refresh every {every}s")
    time.sleep(max(5, int(every)))
    st.rerun()
