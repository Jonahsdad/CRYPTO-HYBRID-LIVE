# =====================================================
# CRYPTO HYBRID LIVE — FULL APP.PY (STABLE)
# Author: Jesse Ray Landingham Jr
# =====================================================

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# -------- Optional libraries (app still runs if missing) ----------
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

try:
    import plotly.express as px
    HAS_PX = True
except Exception:
    HAS_PX = False


# =================== APP CONSTANTS / PAGE CONFIG ==================
APP_NAME   = "CRYPTO HYBRID LIVE"
POWERED_BY = "POWERED BY JESSE RAY LANDINGHAM JR"
PHASE_TAG  = "Phase 19.x — Hero • Pills • Stocks+Options (fallback-safe)"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =================== THEME / HERO CSS =============================
CSS = """
<style>
:root {
  --bg: #0b111a;
  --card: #0e1622;
  --ink: #cfe8ff;
  --accent: #2dd4bf;
  --accent2: #60a5fa;
  --warn: #f59e0b;
  --bad: #ef4444;
  --ok: #22c55e;
}

.block-container { padding-top: 1.25rem; }

.hero-wrap {
  margin-top: 1.25rem;      /* pushes banner down so it isn't clipped */
  margin-bottom: 0.75rem;
}
.hero {
  background: radial-gradient(1200px 300px at 20% 0%, #0e3a2d 0%, #0b111a 50%),
              linear-gradient(90deg, #0b3b2d 0%, #0b2a5b 60%, #0b111a 100%);
  color: white;
  padding: 22px 26px;
  border-radius: 16px;
  border: 1px solid #1f2a3a;
  box-shadow: 0 8px 24px rgba(0,0,0,0.35), inset 0 0 60px rgba(45,212,191,0.10);
}
.hero h1 {
  margin: 0;
  font-weight: 900;
  letter-spacing: 0.02em;
  font-size: clamp(28px, 5vw, 44px);
}
.hero .sub {
  margin-top: 6px;
  font-weight: 700;
  color: #bfe8ff;
  font-size: clamp(14px, 2.4vw, 18px);
}

.metrics-4 .metric {
  background: var(--card);
  border: 1px solid #1f2a3a;
  border-radius: 14px;
  padding: 14px 16px;
  color: var(--ink);
}
.metrics-4 .metric b { font-size: 0.9rem; }
.metrics-4 .metric .n { font-size: 1.15rem; font-weight: 800; }

.badge-row { margin: 10px 0 6px 0; }
.pill {
  display:inline-flex; align-items:center; gap:8px;
  padding: 8px 12px; border-radius: 999px; margin-right: 8px;
  color:#fff; font-weight: 800; border: 1px solid #233146;
  background: linear-gradient(180deg, #192334, #121a27);
}
.pill .emo { font-size: 1.05rem; }
.pill.raw   { box-shadow: inset 0 -40px 80px rgba(255,140,66,.18); }
.pill.truth { box-shadow: inset 0 -40px 80px rgba(96,165,250,.22); }
.pill.conf  { box-shadow: inset 0 -40px 80px rgba(255,215,99,.20); }
.pill.delta { box-shadow: inset 0 -40px 80px rgba(148,163,184,.18); }

.btn-giant {
  display:inline-flex; align-items:center; gap:10px;
  padding: 14px 18px; border-radius: 12px; margin-right: 10px; margin-top: 6px;
  color:#fff; font-weight: 900; border: 1px solid #2b394d;
  background: linear-gradient(180deg, #1b2535, #111827);
}
.btn-giant .emo { font-size: 1.2rem; }
.btn-giant.raw   { background: linear-gradient(180deg, #2a1d14, #141013); border-color:#4a311c; }
.btn-giant.truth { background: linear-gradient(180deg, #112338, #0d1624); border-color:#1f3e66; }
.btn-giant.conf  { background: linear-gradient(180deg, #2b2610, #15130b); border-color:#4d3f1a; }
.btn-giant.delta { background: linear-gradient(180deg, #1e1f27, #10121a); border-color:#3a3f50; }

.table-note { color:#8aa8c7; font-size: 0.85rem; margin-top: 6px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =================== UTILITIES / SCORING ==========================
def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

@st.cache_data(ttl=120, show_spinner="Pulling CoinGecko…")
def fetch_cg_markets(vs="usd", per_page=200) -> pd.DataFrame:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": vs,
        "order": "market_cap_desc",
        "per_page": int(max(1, min(250, per_page))),
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": "1h,24h,7d",
        "locale": "en",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
    except Exception:
        # Safe fallback dataset (keeps app alive offline)
        df = pd.DataFrame({
            "name": ["Bitcoin","Ethereum","Tether","BNB","Solana","Cardano"],
            "symbol": ["btc","eth","usdt","bnb","sol","ada"],
            "current_price": [108000, 3950, 1.00, 560, 170, 0.40],
            "market_cap": [2.1e12, 4.8e11, 1.1e11, 8.5e10, 7.8e10, 1.3e10],
            "total_volume": [3.6e10, 1.2e10, 4.0e10, 3.5e9, 5.0e9, 6.8e8],
            "price_change_percentage_1h_in_currency": [0.2, -0.1, 0.0, 0.1, 0.4, -0.3],
            "price_change_percentage_24h_in_currency": [2.2, -1.5, 0.0, 0.8, 5.1, -0.6],
            "price_change_percentage_7d_in_currency": [4.5, -2.3, 0.0, 1.3, 11.0, -2.4],
        })
    # normalize columns we use
    for c in ["name","symbol","current_price","market_cap","total_volume",
              "price_change_percentage_1h_in_currency",
              "price_change_percentage_24h_in_currency",
              "price_change_percentage_7d_in_currency"]:
        if c not in df.columns:
            df[c] = np.nan
    return df

def _s(x):  # smooth sigmoid for pct → 0..1
    try:
        return 1.0 / (1.0 + math.exp(- float(x) / 10.0))
    except Exception:
        return 0.5

def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    t = df.copy()

    t["total_volume"] = t.get("total_volume", pd.Series(np.nan, index=t.index)).fillna(0)
    t["market_cap"]   = t.get("market_cap", pd.Series(np.nan, index=t.index)).fillna(0)
    t["vol_mc"]       = (t["total_volume"] / t["market_cap"]).replace([np.inf,-np.inf], np.nan).clip(0, 2).fillna(0)

    m1h = t.get("price_change_percentage_1h_in_currency",  pd.Series(np.nan, index=t.index)).apply(_s).fillna(0.5)
    m24 = t.get("price_change_percentage_24h_in_currency", pd.Series(np.nan, index=t.index)).apply(_s).fillna(0.5)
    m7d = t.get("price_change_percentage_7d_in_currency",  pd.Series(np.nan, index=t.index)).apply(_s).fillna(0.5)

    mc = t["market_cap"]
    if mc.max() > 0:
        liq01 = (mc - mc.min()) / (mc.max() - mc.min() + 1e-9)
    else:
        liq01 = pd.Series(0, index=t.index)

    t["raw_heat"]   = (0.5*(t["vol_mc"]/2).clip(0,1) + 0.5*m1h).clip(0,1)
    t["truth_full"] = (0.30*(t["vol_mc"]/2).clip(0,1) +
                       0.25*m24 + 0.25*m7d + 0.20*liq01).clip(0,1)

    t["consistency01"] = 1 - (m24 - m7d).abs()
    t["agreement01"]   = 1 - (t["raw_heat"] - t["truth_full"]).abs()
    t["energy01"]      = (t["vol_mc"]/2).clip(0,1)

    t["confluence01"]  = (0.35*t["truth_full"] + 0.35*t["raw_heat"] +
                          0.10*t["consistency01"] + 0.10*t["agreement01"] +
                          0.05*t["energy01"] + 0.05*liq01).clip(0,1)

    t["delta01"] = (t["raw_heat"] - t["truth_full"]).abs()

    def fire(v): return "🔥🔥🔥" if v>=0.85 else ("🔥🔥" if v>=0.65 else ("🔥" if v>=0.45 else "·"))
    def drop(v): return "💧💧💧" if v>=0.85 else ("💧💧" if v>=0.65 else ("💧" if v>=0.45 else "·"))
    def star(v): return "⭐️⭐️⭐️" if v>=0.85 else ("⭐️⭐️" if v>=0.65 else ("⭐️" if v>=0.45 else "·"))
    t["RAW_BADGE"]   = t["raw_heat"].apply(fire)
    t["TRUTH_BADGE"] = t["truth_full"].apply(drop)
    t["CONF_BADGE"]  = t["confluence01"].apply(star)
    return t

# =================== SMALL VISUAL HELPERS =========================
def kpi_row(df_scored: pd.DataFrame, label: str) -> None:
    n = int(len(df_scored)) if df_scored is not None else 0
    p24 = float(df_scored.get("price_change_percentage_24h_in_currency", pd.Series(dtype=float)).mean())
    tavg = float(df_scored.get("truth_full", pd.Series(dtype=float)).mean())
    ravg = float(df_scored.get("raw_heat", pd.Series(dtype=float)).mean())
    cavg = float(df_scored.get("confluence01", pd.Series(dtype=float)).mean())

    c1,c2,c3,c4,c5 = st.columns([1,1,1,1,1])
    with c1: st.markdown(f"<div class='metric'><b>Assets</b><div class='n'>{n}</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric'><b>Avg 24h %</b><div class='n'>{0 if np.isnan(p24) else p24:.2f}%</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric'><b>Avg TRUTH</b><div class='n'>{0 if np.isnan(tavg) else tavg:.2f}</div></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric'><b>Avg RAW</b><div class='n'>{0 if np.isnan(ravg) else ravg:.2f}</div></div>", unsafe_allow_html=True)
    with c5: st.markdown(f"<div class='metric'><b>Avg Confluence</b><div class='n'>{0 if np.isnan(cavg) else cavg:.2f}</div></div>", unsafe_allow_html=True)
    st.caption(f"{PHASE_TAG} • Updated {now_utc()} • Mode: {label}")

def chart_bar(df: pd.DataFrame, metric: str, topn: int = 20, title: str = "") -> None:
    cols_ok = ["name", "symbol", "current_price", metric]
    if df.empty or not set(cols_ok).issubset(df.columns):
        st.info("Not enough data to plot.")
        return
    show = df[cols_ok].sort_values(metric, ascending=False).head(max(5, min(40, int(topn))))

    if HAS_PX:
        fig = px.bar(show, x="symbol", y=metric, color=metric, hover_data=["name","current_price"], title=title)
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(show.set_index("symbol")[metric])

def table_view(df: pd.DataFrame, cols: List[str]) -> None:
    have = [c for c in cols if c in df.columns]
    st.dataframe(
        df[have],
        use_container_width=True,
        hide_index=True,
        column_config={
            "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "market_cap": st.column_config.NumberColumn("Mkt Cap", format="$%d"),
            "total_volume": st.column_config.NumberColumn("Volume", format="$%d"),
            "raw_heat": st.column_config.ProgressColumn("RAW", min_value=0.0, max_value=1.0),
            "truth_full": st.column_config.ProgressColumn("TRUTH", min_value=0.0, max_value=1.0),
            "confluence01": st.column_config.ProgressColumn("Confluence", min_value=0.0, max_value=1.0),
            "delta01": st.column_config.ProgressColumn("Δ", min_value=0.0, max_value=1.0),
            "RAW_BADGE": st.column_config.TextColumn("🔥"),
            "TRUTH_BADGE": st.column_config.TextColumn("💧"),
            "CONF_BADGE": st.column_config.TextColumn("⭐"),
        },
    )

# =================== STOCKS / OPTIONS HELPERS =====================
@st.cache_data(ttl=600)
def fetch_sp500_constituents() -> pd.DataFrame:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        df = df.rename(columns={"Symbol":"symbol","Security":"name","GICS Sector":"Sector"})
        df["symbol"] = df["symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
        return df[["symbol","name","Sector"]]
    except Exception:
        return pd.DataFrame(columns=["symbol","name","Sector"])

@st.cache_data(ttl=900)
def yf_snapshot_daily(tickers: List[str]) -> pd.DataFrame:
    if not HAS_YF or not tickers: return pd.DataFrame()
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    try:
        hist = yf.download(" ".join(tickers), period="5d", interval="1d",
                           group_by="ticker", auto_adjust=True, threads=True, progress=False)
    except Exception:
        return pd.DataFrame()
    rows = []
    for t in tickers:
        try:
            s = hist[t]
            last = float(s.iloc[-1]["Close"])
            prev = float(s.iloc[-2]["Close"]) if len(s) >= 2 else np.nan
            pct24 = (last/prev - 1.0)*100.0 if pd.notna(prev) else np.nan
            rows.append({"name": t, "symbol": t, "current_price": last,
                         "price_change_percentage_24h_in_currency": pct24})
        except Exception:
            rows.append({"name": t, "symbol": t, "current_price": np.nan,
                         "price_change_percentage_24h_in_currency": np.nan})
    return pd.DataFrame(rows)

@st.cache_data(ttl=120)
def list_expirations(ticker: str) -> List[str]:
    if not HAS_YF or not ticker: return []
    try:
        return list(yf.Ticker(ticker).options)
    except Exception:
        return []

@st.cache_data(ttl=180)
def load_options_chain(ticker: str, expiration: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not HAS_YF or not ticker or not expiration: return pd.DataFrame(), pd.DataFrame()
    try:
        ch = yf.Ticker(ticker).option_chain(expiration)
        keep = ["contractSymbol","strike","lastPrice","bid","ask","change","percentChange",
                "volume","openInterest","impliedVolatility"]
        calls = ch.calls[[c for c in keep if c in ch.calls.columns]]
        puts  = ch.puts [[c for c in keep if c in ch.puts.columns]]
        return calls, puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

# =================== HERO =====================================================
def hero() -> None:
    st.markdown("<div class='hero-wrap'><div class='hero'>"
                f"<h1>{APP_NAME}</h1>"
                f"<div class='sub'>{POWERED_BY}</div></div></div>", unsafe_allow_html=True)

# =================== PAGES ====================================================
def page_dashboard() -> None:
    hero()

    # Truth weights (visual only for now; next phase will wire to sliders)
    with st.sidebar:
        st.header("Truth Weights")
        st.slider("Vol/Mcap", 0.10, 0.50, 0.30, 0.01)
        st.slider("24h Momentum", 0.05, 0.50, 0.25, 0.01)
        st.slider("7d Momentum", 0.05, 0.50, 0.25, 0.01)
        st.slider("Liquidity/Size", 0.05, 0.50, 0.20, 0.01)
        st.header("Auto Refresh")
        auto = st.toggle("Auto refresh", value=False, key="auto")
        every = st.slider("Every (sec)", 10, 120, 30, step=5)

    dfc = build_scores(fetch_cg_markets("usd", 200))
    kpi_row(dfc, "Crypto")

    # Explainer row
    st.markdown(
        "<div class='badge-row'>"
        "<span class='pill raw'><span class='emo'>🔥</span> RAW <small>crowd heat (vol/mcap + 1h)</small></span>"
        "<span class='pill truth'><span class='emo'>💧</span> TRUTH <small>stability (vol/mcap + 24h + 7d + size)</small></span>"
        "<span class='pill conf'><span class='emo'>⭐</span> CONFLUENCE <small>RAW & TRUTH agree + consistency + energy</small></span>"
        "<span class='pill delta'><span class='emo'>⚡</span> Δ <small>|RAW − TRUTH|</small></span>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Big selectable buttons
    cols = st.columns([1,1,1,1])
    label_map = {
        0: ("raw_heat",     "🔥 RAW — leaders right now"),
        1: ("truth_full",   "💧 TRUTH — durable quality"),
        2: ("confluence01", "⭐ CONFLUENCE — hype + quality aligned"),
        3: ("delta01",      "⚡ Δ (RAW↔TRUTH) — divergence"),
    }
    for i, (key, text) in label_map.items():
        with cols[i]:
            st.button(text, key=f"btn_{key}")

    # Which button is last clicked?
    metric = "confluence01"
    for key in ["raw_heat", "truth_full", "confluence01", "delta01"]:
        if st.session_state.get(f"btn_{key}"):
            metric = key

    chart_bar(dfc, metric, topn=22, title="Leaders")

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Top Confluence (Crypto)")
        table_view(
            dfc.sort_values("confluence01", ascending=False).head(30),
            ["name","symbol","current_price","CONF_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","delta01"]
        )
    with c2:
        st.subheader("Top TRUTH (Crypto)")
        table_view(
            dfc.sort_values("truth_full", ascending=False).head(30),
            ["name","symbol","current_price","TRUTH_BADGE","truth_full","RAW_BADGE","delta01"]
        )

    if auto:
        st.caption(f"Auto-refresh in {int(every)}s…")
        import time
        time.sleep(max(5, int(every)))
        st.rerun()

def page_crypto() -> None:
    hero()
    df = build_scores(fetch_cg_markets("usd", 250))
    kpi_row(df, "Crypto")
    order = st.selectbox("Order by", ["confluence01","truth_full","raw_heat","delta01"], index=0)
    topn  = st.slider("Show Top N", 20, 250, 100, step=10)
    table_view(df.sort_values(order, ascending=False).head(topn),
               ["name","symbol","current_price","CONF_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","delta01"])

def page_sp500() -> None:
    hero()
    st.subheader("S&P 500")
    if not HAS_YF:
        st.warning("`yfinance` not installed. Add `yfinance` to requirements.txt and reboot.")
        return
    base = fetch_sp500_constituents()
    if base.empty:
        st.info("Could not fetch S&P 500 constituents right now.")
        return
    limit = st.slider("Tickers to snapshot", 50, len(base), 200, step=50)
    snap  = yf_snapshot_daily(base["symbol"].tolist()[:limit])
    if snap.empty:
        st.info("No price data returned yet. Try fewer tickers and re-run.")
        return
    scored = build_scores(snap.merge(base, on="symbol", how="left"))
    kpi_row(scored, "S&P 500")
    chart_bar(scored, "confluence01", topn=25, title="S&P leaders by Confluence")
    table_view(scored.sort_values("confluence01", ascending=False).head(80),
               ["name","symbol","Sector","current_price","CONF_BADGE","confluence01","RAW_BADGE","TRUTH_BADGE","delta01"])

def page_options() -> None:
    hero()
    st.subheader("Options Explorer")
    if not HAS_YF:
        st.warning("`yfinance` not installed. Add it to requirements.txt.")
        return
    base = fetch_sp500_constituents()
    sym = st.selectbox("Ticker", options=(["AAPL"] + base["symbol"].tolist()) if not base.empty else ["AAPL"])
    exps = list_expirations(sym)
    if not exps:
        st.info("No expirations returned."); return
    exp = st.selectbox("Expiration", options=exps)
    calls, puts = load_options_chain(sym, exp)
    c1,c2 = st.columns(2)
    with c1:
        st.caption(f"{sym} Calls — {exp}")
        table_view(
            calls.sort_values(["openInterest","volume"], ascending=False).head(25),
            ["contractSymbol","strike","lastPrice","volume","openInterest","impliedVolatility"]
        )
    with c2:
        st.caption(f"{sym} Puts — {exp}")
        table_view(
            puts.sort_values(["openInterest","volume"], ascending=False).head(25),
            ["contractSymbol","strike","lastPrice","volume","openInterest","impliedVolatility"]
        )

def page_scores() -> None:
    hero()
    st.subheader("Scores — Explainer")
    st.markdown("""
**RAW (0..1)** — crowd heat now (volume/market-cap + 1h momentum).  
**TRUTH (0..1)** — stability blend (vol/mcap, 24h, 7d, liquidity/size).  
**Δ (0..1)** — |RAW − TRUTH| (the gap).  
**CONFLUENCE (0..1)** — fusion of RAW + TRUTH with **agreement** (RAW≈TRUTH), **consistency** (24h≈7d), **energy** (vol/mcap), and **liquidity**.

**Read fast:**  
- ⭐ High **Confluence** → hype and quality aligned (prime).  
- 🔥 High **RAW**, low **💧 TRUTH** → hype spike (fragile).  
- 💧 High **TRUTH**, low **🔥 RAW** → sleeper quality (crowd not there yet).
""")

def page_settings() -> None:
    hero()
    st.subheader("Settings")
    st.write("More personalization, presets, and weights editor land in the next phase.")

# =================== ROUTER ======================================
with st.sidebar:
    st.header("Navigation")
    nav = st.radio(
        "Go to",
        ["Dashboard","Crypto","S&P 500","Options","Scores","Settings"],
        index=0,
        key="nav_radio",
    )

if   nav == "Dashboard": page_dashboard()
elif nav == "Crypto":    page_crypto()
elif nav == "S&P 500":   page_sp500()
elif nav == "Options":   page_options()
elif nav == "Scores":    page_scores()
else:                    page_settings()
