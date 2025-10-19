# app.py — Crypto Hybrid Live (Phase 19.4)
# Hero 3× + sticky spacing + vivid Truth pills + interactive chart
# Stocks still use yfinance (optional). Next phase: multi-source + options hardening.

import math, time
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Optional stocks (app still runs without this)
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

PHASE_TAG  = "PHASE 19.4 — Hero 3× • Truth Pills • Chart"
APP_TITLE  = "CRYPTO HYBRID LIVE"
POWERED_BY = "POWERED BY JESSE RAY LANDINGHAM JR"

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_TITLE} — {PHASE_TAG}",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CSS (banner 3×, spacing, vivid pills, tidy tables)
# -----------------------------------------------------------------------------
CSS = f"""
<style>
/* Move content down so the hero never gets clipped under Streamlit top chrome */
.block-container {{ padding-top: 2.25rem; padding-bottom: 1.25rem; }}

/* HERO banner (one shared line) */
.hero-wrap {{
  position: relative;
  margin: 18px 0 18px 0;
}}
.hero {{
  width: 100%;
  max-width: 1400px;
  margin: 0 auto;
  padding: 26px 28px;
  border-radius: 18px;
  border: 1px solid #1f2a38;
  background: radial-gradient(circle at 10% 0%, #0c2f23 0%, #0a1b2d 45%, #0a1422 100%);
  box-shadow: 0 18px 40px rgba(0,0,0,.45), inset 0 0 80px rgba(59, 255, 160, .08);
}}
.hero h1 {{
  margin: 0;
  font-size: 42px;               /* ~3× default */
  line-height: 1.05;
  letter-spacing: .6px;
  font-weight: 900;
  color: #e8fff5;
  text-shadow: 0 2px 12px rgba(0,0,0,.35);
}}
.hero .sub {{
  margin-top: 6px;
  font-weight: 800;
  font-size: 16px;
  letter-spacing: .4px;
  color: #9fdcff;
}}

/* KPI cards */
.metric-box {{
  border: 1px solid #253143;
  background: #0e1522;
  border-radius: 14px;
  padding: 14px 16px;
}}
.metric-box b {{ color:#b9d2ff; }}
.metric-box div.val {{ font-size: 18px; font-weight: 800; color:#eaf2ff; }}

/* Truth Pills — large, vivid, pressable */
.pills {{
  display:flex; gap:14px; flex-wrap:wrap; margin: 6px 0 14px 0;
}}
.pill {{
  cursor:pointer; user-select:none;
  display:flex; align-items:center; gap:10px;
  padding: 14px 18px; border-radius: 14px; border:1px solid #1f2a38;
  background:#111827; color:#f8fbff; font-weight:900; letter-spacing:.3px;
  transition: transform .08s ease, box-shadow .12s ease, filter .12s ease;
}}
.pill:hover {{ transform: translateY(-1px); filter: brightness(1.06); }}
.pill .icon {{ font-size: 18px; }}
.pill .label {{ font-size: 15px; }}

.pill-raw {{ background: linear-gradient(90deg,#ff6b00,#ffc03d); border-color:#ffb15b; color:#1a0f00; }}
.pill-truth {{ background: linear-gradient(90deg,#28f1ff,#0077ff); border-color:#3fb2ff; }}
.pill-conf {{ background: linear-gradient(90deg,#ffd93a,#ffaf5f); border-color:#ffc86d; color:#201500; }}
.pill-delta {{ background: linear-gradient(90deg,#b388ff,#6138ff); border-color:#845bff; }}

.pill.active {{ box-shadow: 0 0 0 3px rgba(255,255,255,.08), inset 0 0 0 9999px rgba(0,0,0,.08); }}

/* Table polish */
.stDataFrame, .stDataEditor {{ border-radius: 10px; overflow: hidden; }}
.section-title {{ font-weight: 900; font-size: 22px; margin: 4px 0 8px 0; }}
.smallnote {{ color:#8aa0b8; font-size: 13px; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Navigation")
    nav = st.radio(
        "Go to",
        ["Dashboard", "Crypto", "Confluence", "US Market (All Listings)", "S&P 500", "Options", "Scores", "Settings"],
        index=0,
    )

    st.header("Truth Weights")
    w_vol = st.slider("Vol/Mcap", 0.0, 1.0, 0.30, step=0.05)
    w_m24 = st.slider("24h Momentum", 0.0, 1.0, 0.25, step=0.05)
    w_m7  = st.slider("7d Momentum", 0.0, 1.0, 0.25, step=0.05)
    w_liq = st.slider("Liquidity/Size", 0.0, 1.0, 0.20, step=0.05)

    st.header("Auto Refresh")
    auto  = st.toggle("Auto refresh", value=False)
    every = st.slider("Every (sec)", 10, 120, 30, step=5)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def safe_sigmoid(pct) -> float:
    """Map % to 0..1 safely"""
    try:
        x = float(pct)/10.0
        return 1.0/(1.0+math.exp(-x))
    except Exception:
        return 0.5

@st.cache_data(ttl=60, show_spinner="Loading CoinGecko…")
def fetch_cg(vs="usd", limit=200) -> pd.DataFrame:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    p = {
        "vs_currency": vs, "order": "market_cap_desc",
        "per_page": int(max(1, min(limit, 250))), "page": 1,
        "sparkline": "false", "price_change_percentage": "1h,24h,7d", "locale": "en",
    }
    r = requests.get(url, params=p, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    # ensure columns exist
    cols = [
        "name","symbol","current_price","market_cap","total_volume",
        "price_change_percentage_1h_in_currency",
        "price_change_percentage_24h_in_currency",
        "price_change_percentage_7d_in_currency",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def build_scores(df: pd.DataFrame, w=(0.30,0.25,0.25,0.20)) -> pd.DataFrame:
    """Return df with raw_heat, truth_full, confluence01, delta."""
    if df is None or df.empty:
        return pd.DataFrame()

    w_vol, w_m24, w_m7, w_liq = w
    t = df.copy()

    t["vol_to_mc"] = (t["total_volume"]/t["market_cap"]).replace([np.inf,-np.inf], np.nan).clip(0,2).fillna(0)
    m1h = t["price_change_percentage_1h_in_currency"].apply(safe_sigmoid)
    m24 = t["price_change_percentage_24h_in_currency"].apply(safe_sigmoid)
    m7d = t["price_change_percentage_7d_in_currency"].apply(safe_sigmoid)

    mc = t["market_cap"].fillna(0)
    denom = (mc.max()-mc.min()) or 1.0
    t["liq01"] = (mc-mc.min())/denom

    # RAW & TRUTH
    t["raw_heat"] = (0.5*(t["vol_to_mc"]/2).clip(0,1) + 0.5*m1h.fillna(0.5)).clip(0,1)
    t["truth_full"] = (
        w_vol*(t["vol_to_mc"]/2).clip(0,1) +
        w_m24*m24.fillna(0.5) +
        w_m7*m7d.fillna(0.5) +
        w_liq*t["liq01"].fillna(0)
    ).clip(0,1)

    # Confluence blend
    t["consistency01"] = 1 - (m24.fillna(0.5) - m7d.fillna(0.5)).abs()
    t["agreement01"]   = 1 - (t["raw_heat"] - t["truth_full"]).abs()
    t["energy01"]      = (t["vol_to_mc"]/2).clip(0,1)
    t["confluence01"]  = (0.35*t["truth_full"] + 0.35*t["raw_heat"] +
                          0.10*t["consistency01"] + 0.10*t["agreement01"] +
                          0.05*t["energy01"] + 0.05*t["liq01"]).clip(0,1)

    t["delta"] = (t["raw_heat"] - t["truth_full"]).abs()
    return t

def kpi_row(df: pd.DataFrame, label: str):
    n = len(df)
    p24 = float(df.get("price_change_percentage_24h_in_currency", pd.Series(dtype=float)).mean())
    tavg = float(df.get("truth_full", pd.Series(dtype=float)).mean())
    ravg = float(df.get("raw_heat", pd.Series(dtype=float)).mean())
    cavg = float(df.get("confluence01", pd.Series(dtype=float)).mean())

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(f"<div class='metric-box'><b>Assets</b><div class='val'>{n}</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-box'><b>Avg 24h %</b><div class='val'>{0 if np.isnan(p24) else p24:.2f}%</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-box'><b>Avg TRUTH</b><div class='val'>{0 if np.isnan(tavg) else tavg:.2f}</div></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric-box'><b>Avg RAW</b><div class='val'>{0 if np.isnan(ravg) else ravg:.2f}</div></div>", unsafe_allow_html=True)
    with c5: st.markdown(f"<div class='metric-box'><b>Avg Confluence</b><div class='val'>{0 if np.isnan(cavg) else cavg:.2f}</div></div>", unsafe_allow_html=True)
    st.caption(f"{PHASE_TAG} • Updated {now_utc()} • Mode: {label}")

def table(df: pd.DataFrame, cols: List[str]):
    have = [c for c in cols if c in df.columns]
    st.dataframe(
        df[have], use_container_width=True, hide_index=True,
        column_config={
            "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "price_change_percentage_24h_in_currency": st.column_config.NumberColumn("24h %", format="%.2f%%"),
            "raw_heat": st.column_config.ProgressColumn("RAW", min_value=0.0, max_value=1.0),
            "truth_full": st.column_config.ProgressColumn("TRUTH", min_value=0.0, max_value=1.0),
            "confluence01": st.column_config.ProgressColumn("CONF", min_value=0.0, max_value=1.0),
            "delta": st.column_config.ProgressColumn("Δ", min_value=0.0, max_value=1.0),
        }
    )

# -----------------------------------------------------------------------------
# Hero
# -----------------------------------------------------------------------------
st.markdown(
    f"""
<div class='hero-wrap'>
  <div class='hero'>
    <h1>{APP_TITLE} · <span class='sub'>{POWERED_BY}</span></h1>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Dashboard (hero + pills + chart + top tables)
# -----------------------------------------------------------------------------
def page_dashboard():
    # Pull + score
    base = build_scores(fetch_cg(limit=200), (w_vol,w_m24,w_m7,w_liq))
    kpi_row(base, "Crypto")

    # Active metric in session
    if "active_metric" not in st.session_state:
        st.session_state.active_metric = "raw_heat"

    # Pills row
    cols = st.columns([1,1,1,1])
    pill_specs = [
        ("raw_heat",   "🔥", "RAW",        "pill pill-raw"),
        ("truth_full", "💧", "TRUTH",      "pill pill-truth"),
        ("confluence01","⭐", "CONFLUENCE","pill pill-conf"),
        ("delta",      "⚡", "Δ (RAW→TRUTH)","pill pill-delta"),
    ]
    for i, spec in enumerate(pill_specs):
        key, icon, label, klass = spec
        active = " active" if st.session_state.active_metric == key else ""
        with cols[i]:
            # Render as HTML button-look; click proxy with real button below
            st.markdown(
                f"<div class='{klass}{active}'><span class='icon'>{icon}</span><span class='label'>{label}</span></div>",
                unsafe_allow_html=True,
            )
            if st.button(f"{label}", key=f"pill_{key}"):
                st.session_state.active_metric = key

    # ---- Chart for active metric (top 20 by that metric) ----
    metric = st.session_state.active_metric
    metric_title = {
        "raw_heat":"RAW heat (crowd momentum now)",
        "truth_full":"TRUTH (stability blend)",
        "confluence01":"CONFLUENCE (fusion score)",
        "delta":"Δ gap |RAW−TRUTH|",
    }[metric]

    top = base.sort_values(metric, ascending=False).head(20)
    x = top["symbol"].str.upper().tolist()
    y = top[metric].astype(float).tolist()

    # Simple altair-free bar via st.pyplot-free API using native chart
    st.bar_chart(pd.DataFrame({"symbol": x, metric: y}).set_index("symbol"), use_container_width=True)

    # ---- Tables ----
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-title'>Top Confluence (Crypto)</div>", unsafe_allow_html=True)
        table(base.sort_values("confluence01", ascending=False).head(20),
              ["name","symbol","current_price","confluence01","truth_full","raw_heat","delta"])
    with c2:
        st.markdown("<div class='section-title'>Top TRUTH (Crypto)</div>", unsafe_allow_html=True)
        table(base.sort_values("truth_full", ascending=False).head(20),
              ["name","symbol","current_price","truth_full","raw_heat","confluence01","delta"])

# -----------------------------------------------------------------------------
# Other pages (kept minimal; we’ll harden in the next phase)
# -----------------------------------------------------------------------------
def page_crypto():
    df = build_scores(fetch_cg(limit=200), (w_vol,w_m24,w_m7,w_liq))
    kpi_row(df, "Crypto")
    order = st.selectbox("Order by", ["confluence01","truth_full","raw_heat","delta"], index=0)
    table(df.sort_values(order, ascending=False).head(150),
          ["name","symbol","current_price","confluence01","truth_full","raw_heat","delta"])

@st.cache_data(ttl=180)
def yf_snapshot_daily(tickers: List[str]) -> pd.DataFrame:
    if not HAS_YF or not tickers: return pd.DataFrame()
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    try:
        data = yf.download(" ".join(tickers), period="5d", interval="1d",
                           group_by="ticker", auto_adjust=True, threads=True, progress=False)
    except Exception:
        return pd.DataFrame()

    rows = []
    for t in tickers:
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

def page_us_all():
    st.markdown("<div class='section-title'>US Market (All Listings)</div>", unsafe_allow_html=True)
    if not HAS_YF:
        st.error("yfinance not installed. Add `yfinance` to requirements.txt and reboot.")
        return
    # Simple input for now (we’ll wire multi-source + universe builder next phase)
    tickers = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,NVDA,AMZN,GOOGL").upper().split(",")
    snap = yf_snapshot_daily(tickers)
    if snap.empty:
        st.warning("No data returned. Try fewer tickers.")
        return
    scored = build_scores(snap, (w_vol,w_m24,w_m7,w_liq))
    kpi_row(scored, "US Stocks (snapshot)")
    table(scored.sort_values("confluence01", ascending=False),
          ["name","symbol","current_price","confluence01","truth_full","raw_heat","delta"])

def page_sp500():
    st.markdown("<div class='section-title'>S&P 500 (snapshot)</div>", unsafe_allow_html=True)
    if not HAS_YF:
        st.error("yfinance not installed. Add `yfinance` to requirements.txt and reboot.")
        return
    tickers = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","BRK-B","LLY","TSLA"]  # fast demo subset
    snap = yf_snapshot_daily(tickers)
    if snap.empty:
        st.warning("No data returned.")
        return
    scored = build_scores(snap, (w_vol,w_m24,w_m7,w_liq))
    kpi_row(scored, "S&P subset")
    table(scored.sort_values("confluence01", ascending=False),
          ["name","symbol","current_price","confluence01","truth_full","raw_heat","delta"])

def page_options():
    st.markdown("<div class='section-title'>Options (preview)</div>", unsafe_allow_html=True)
    st.info("Next phase we’ll wire robust options chains and scanners. (yfinance supported; multi-source coming.)")

def page_scores():
    st.markdown("<div class='section-title'>Scores — Fast Explainer</div>", unsafe_allow_html=True)
    st.markdown("""
- **🔥 RAW (0..1)** — crowd energy *now* (vol/market-cap + 1h momentum).  
- **💧 TRUTH (0..1)** — stability blend (vol/mcap + 24h + 7d + liquidity).  
- **⚡ Δ (0..1)** — absolute gap |RAW − TRUTH| (bigger gap = more disagreement).  
- **⭐ CONFLUENCE (0..1)** — fusion of RAW + TRUTH + agreement (RAW≈TRUTH), consistency (24h≈7d), energy, liquidity.

**Read it fast**
- ⭐ High **Confluence** → hype and quality **aligned** (prime radar).
- 🔥 High **RAW**, low 💧 **TRUTH** → hype **spike** (fragile; watch for fade).
- 💧 High **TRUTH**, low 🔥 **RAW** → **sleeper quality** (crowd not there yet).
""")

def page_settings():
    st.markdown("<div class='section-title'>Settings</div>", unsafe_allow_html=True)
    st.caption("Theme/presets/alerts live here soon.")

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
if nav == "Dashboard":
    page_dashboard()
elif nav == "Crypto":
    page_crypto()
elif nav == "Confluence":
    page_crypto()  # using same data, different sort/filter in future
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

# -----------------------------------------------------------------------------
# Auto-refresh
# -----------------------------------------------------------------------------
if auto:
    st.caption(f"{PHASE_TAG} • Auto refresh every {int(every)}s")
    time.sleep(max(5, int(every)))
    st.rerun()
