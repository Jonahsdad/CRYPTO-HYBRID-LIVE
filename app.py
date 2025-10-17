# ====================== IMPORTS ======================
import math
import time
import random
import re
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timezone
from io import StringIO

# Plotly (optional)
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# Sentiment (TextBlob with safe fallback)
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except Exception:
    HAS_TEXTBLOB = False
    class TextBlob:
        def __init__(self, text): self.text = text
        @property
        def sentiment(self): return type("S", (), {"polarity": 0.0})()

def _polarity_safe(text: str) -> float:
    try:
        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        return 0.0

# ====================== CONFIG ======================
APP_NAME = "Crypto Hybrid Live — Phase 4 (Pro)"
st.set_page_config(page_title=APP_NAME, layout="wide")
USER_AGENT = {"User-Agent": "Mozilla/5.0 (CHL-Phase4)"}

FEATURES = {
    "SPARKLINES": False,   # can enable later
    "REDDIT": True,        # r/CryptoCurrency hot (no key)
    "DEFI_LLAMA": True,    # TVL
    "CRYPTOPANIC": True,   # needs secret token; auto-skip if missing
    "ALERTS": True,
    "SNAPSHOT": True,
}

CG_PER_PAGE = 150

# ====================== BIGGER (but clean) UI + THEME ======================
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

def apply_theme_css():
    dark = st.session_state.get("theme", "dark") == "dark"
    base_bg = "#0d1117" if dark else "#ffffff"
    base_fg = "#e6edf3" if dark else "#111111"
    accent  = "#23d18b" if dark else "#0b8f5a"

    # Mild bump ~1.3x
    TAB_FONT   = "1.35rem"
    TABLE_FONT = "1.20rem"
    HDR_FONT   = "1.9rem"
    KPI_VAL    = "2.4rem"
    KPI_LBL    = "1.15rem"

    st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
        font-size: 18px; line-height: 1.6;
        background: {base_bg}; color: {base_fg};
    }}
    .stTabs [role="tablist"] button {{
        font-size: {TAB_FONT} !important; font-weight: 700 !important;
        padding: 0.8rem 1.6rem !important; border-radius: 10px !important;
        margin-right: 0.8rem !important; background-color: #111 !important;
        color: {accent} !important; border: 1px solid #222 !important;
        transition: transform .15s ease-in-out;
    }}
    .stTabs [role="tablist"] button[aria-selected="true"] {{
        background-color: {accent} !important; color: #000 !important;
        transform: scale(1.04);
    }}
    .stDataFrame, .stDataFrame table, .stDataFrame td, .stDataFrame th {{
        font-size: {TABLE_FONT} !important; line-height: 1.7rem !important;
    }}
    h2, h3, .stSubheader, .stMarkdown h3 {{
        font-size: {HDR_FONT} !important; color: {accent} !important; font-weight: 800 !important;
    }}
    [data-testid="stMetricValue"] {{ font-size: {KPI_VAL} !important; }}
    [data-testid="stMetricLabel"] {{ font-size: {KPI_LBL} !important; }}
    .pill {{display:inline-block; padding:.15rem .55rem; border-radius:999px; background:#1d2633; margin-right:.35rem; font-size:.9rem;}}
    </style>
    """, unsafe_allow_html=True)

apply_theme_css()

# ====================== LIPE CORE ======================
DEFAULT_WEIGHTS = dict(w_vol=0.30, w_m24=0.25, w_m7=0.25, w_liq=0.20)
PRESETS = {
    "Balanced": dict(w_vol=0.30, w_m24=0.25, w_m7=0.25, w_liq=0.20),
    "Momentum": dict(w_vol=0.15, w_m24=0.45, w_m7=0.30, w_liq=0.10),
    "Liquidity": dict(w_vol=0.45, w_m24=0.20, w_m7=0.15, w_liq=0.20),
    "Value": dict(w_vol=0.25, w_m24=0.20, w_m7=0.20, w_liq=0.35),
}
FUSION_WEIGHTS  = dict(w_truth=0.60, w_social=0.20, w_news=0.10, w_tvl=0.10)

def _normalize_weights(w):
    s = float(sum(max(0.0, v) for v in w.values())) or 1.0
    return {k: max(0.0, v)/s for k, v in w.items()}

def pct_sigmoid(pct):
    if pd.isna(pct): return 0.5
    x = float(pct)/10.0
    return 1/(1+math.exp(-x))

def lipe_truth(df, w):
    w = _normalize_weights(w or DEFAULT_WEIGHTS)
    truth = (
        w["w_vol"] * (df["vol_to_mc"]/2).clip(0,1).fillna(0.0) +
        w["w_m24"] * df["momo_24h01"].fillna(0.5) +
        w["w_m7"]  * df["momo_7d01"].fillna(0.5) +
        w["w_liq"] * df["liquidity01"].fillna(0.0)
    )
    return truth.clip(0,1)

def mood_label(x):
    if pd.isna(x): x = 0.5
    if x>=0.80: return "🟢 EUPHORIC"
    if x>=0.60: return "🟡 OPTIMISTIC"
    if x>=0.40: return "🟠 NEUTRAL"
    return "🔴 FEARFUL"

def entropy01_from_changes(p1h, p24h, p7d):
    arr = np.array([p for p in [p1h, p24h, p7d] if pd.notna(p)], dtype=float)
    if arr.size < 2: return 0.5
    s = np.std(arr); chaos01 = min(max(s/20.0, 0.0), 1.0)
    return float(1.0 - chaos01)

def ensure_profile():
    if "profile" not in st.session_state:
        st.session_state["profile"] = {
            "weights": dict(DEFAULT_WEIGHTS),
            "watchlist": [],
            "perf_mode": False,
            "saved_presets": {},  # name -> weights
        }
    return st.session_state["profile"]

# ====================== HEADER ======================
st.title("🟢 " + APP_NAME)
st.caption("Truth > Noise • Phase 4 adds **weight editor**, **quick filters**, **saved presets**, **bigger tables**, **column modes**")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Controls")
    vs_currency = st.selectbox("Currency", ["usd"], index=0)
    topn = st.slider("Show top N by market cap", 20, 250, CG_PER_PAGE, step=10)
    search = st.text_input("🔎 Search coin (name or symbol)").strip().lower()

    st.markdown("---")
    st.subheader("Theme")
    theme_pick = st.radio("Pick theme", ["Dark","Light"], index=0 if st.session_state["theme"]=="dark" else 1, horizontal=True)
    st.session_state["theme"] = "dark" if theme_pick=="Dark" else "light"
    apply_theme_css()

    st.markdown("---")
    st.subheader("Truth Presets")
    preset = st.radio("Preset", list(PRESETS.keys()), index=0, horizontal=True)

    st.markdown("---")
    st.subheader("Weight Editor")
    prof = ensure_profile()
    w_edit = dict(PRESETS[preset])  # start from preset
    # sliders (0..1)
    w_edit["w_vol"] = st.slider("Weight: Volume/MarketCap", 0.0, 1.0, float(w_edit["w_vol"]), 0.01)
    w_edit["w_m24"] = st.slider("Weight: Momentum 24h",    0.0, 1.0, float(w_edit["w_m24"]), 0.01)
    w_edit["w_m7"]  = st.slider("Weight: Momentum 7d",     0.0, 1.0, float(w_edit["w_m7"]),  0.01)
    w_edit["w_liq"] = st.slider("Weight: Liquidity",       0.0, 1.0, float(w_edit["w_liq"]), 0.01)
    w_edit = _normalize_weights(w_edit)

    cA, cB = st.columns(2)
    with cA:
        if st.button("💾 Save as Preset"):
            name = st.text_input("Preset name (type then click Save again)", key="save_name")
            st.stop() if not name else None
            prof["saved_presets"][name] = dict(w_edit)
            st.success(f"Saved preset **{name}**.")
    with cB:
        if st.button("♻️ Reset Weights"):
            prof["weights"] = dict(DEFAULT_WEIGHTS)
            st.success("Weights reset to default Balanced.")

    if prof["saved_presets"]:
        sel = st.selectbox("Load saved preset", ["(none)"] + list(prof["saved_presets"].keys()))
        if sel != "(none)":
            w_edit = dict(prof["saved_presets"][sel])
            st.info(f"Loaded saved preset **{sel}**.")

    st.markdown("---")
    st.subheader("Performance")
    prof["perf_mode"] = st.toggle("Performance Mode (skip Social/News/TVL)", value=prof["perf_mode"])
    st.caption("Use this if the app is slow; you still get LIPE Truth.")

    st.markdown("---")
    st.subheader("Quick Filters")
    q_top_gainers = st.checkbox("Show only top 24h gainers", value=False)
    q_top_losers  = st.checkbox("Show only top 24h losers", value=False)
    q_large_cap   = st.checkbox("Large caps (Top 50 MC)", value=False)
    q_small_cap   = st.checkbox("Smaller caps (Rank > 200)", value=False)

    st.markdown("---")
    st.subheader("Alerts")
    alert_truth = st.slider("Trigger: Fusion Truth ≥", 0.0, 1.0, 0.85, 0.01)
    alert_div   = st.slider("Trigger: |Raw - Truth| ≥", 0.0, 1.0, 0.30, 0.01)

# ====================== DATA: COINGECKO ======================
def safe_get(url, params=None, timeout=25, retries=3, backoff=0.6, headers=None):
    h = headers or USER_AGENT
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=h)
            if r.status_code == 200: return r
        except Exception: pass
        time.sleep(backoff*(2**i) + random.uniform(0,0.2))
    return None

@st.cache_data(ttl=60)
def fetch_markets(vs="usd", per_page=CG_PER_PAGE, spark=False):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    p = {
        "vs_currency": vs,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": 1,
        "sparkline": "true" if spark and FEATURES["SPARKLINES"] else "false",
        "price_change_percentage": "1h,24h,7d",
        "locale": "en",
    }
    r = safe_get(url, params=p, timeout=30)
    if not r: return pd.DataFrame([])
    try: return pd.DataFrame(r.json())
    except Exception: return pd.DataFrame([])

df = fetch_markets(vs_currency)
if df.empty:
    st.error("Could not load data from CoinGecko. Try again in a minute.")
    st.stop()

df = df.sort_values("market_cap", ascending=False).head(topn).copy()
for k in ["current_price","market_cap","total_volume",
          "price_change_percentage_1h_in_currency",
          "price_change_percentage_24h_in_currency",
          "price_change_percentage_7d_in_currency",
          "name","symbol","market_cap_rank"]:
    if k not in df.columns: df[k] = np.nan

# engineered features
df["vol_to_mc"] = (df["total_volume"]/df["market_cap"]).replace([np.inf,-np.inf],np.nan).clip(0,2).fillna(0)
df["momo_1h01"]  = df["price_change_percentage_1h_in_currency"].apply(pct_sigmoid)
df["momo_24h01"] = df["price_change_percentage_24h_in_currency"].apply(pct_sigmoid)
df["momo_7d01"]  = df["price_change_percentage_7d_in_currency"].apply(pct_sigmoid)
mc = df["market_cap"].fillna(0)
df["liquidity01"] = 0 if mc.max()==0 else (mc-mc.min())/(mc.max()-mc.min()+1e-9)

# Truth using edited weights
TRUTH_W = dict(w_edit)
df["raw_heat"]   = (0.5*(df["vol_to_mc"]/2).clip(0,1) + 0.5*df["momo_1h01"].fillna(0.5)).clip(0,1)
df["truth_full"] = lipe_truth(df, TRUTH_W)
df["divergence"] = (df["raw_heat"] - df["truth_full"]).round(3)
df["entropy01"]  = df.apply(lambda r: entropy01_from_changes(
    r.get("price_change_percentage_1h_in_currency"),
    r.get("price_change_percentage_24h_in_currency"),
    r.get("price_change_percentage_7d_in_currency")), axis=1)
df["mood"] = df["truth_full"].apply(mood_label)
df["COIN_SYM"] = df["symbol"].str.upper()

# ====================== OPTIONAL SOURCES ======================
perf_mode = ensure_profile()["perf_mode"]
use_social   = FEATURES["REDDIT"]     and not perf_mode
use_news     = FEATURES["CRYPTOPANIC"] and not perf_mode
use_tvl      = FEATURES["DEFI_LLAMA"] and not perf_mode

@st.cache_data(ttl=300, show_spinner=False)
def fetch_reddit_titles(limit=80):
    url = f"https://www.reddit.com/r/CryptoCurrency/hot.json?limit={limit}"
    r = safe_get(url, timeout=20)
    if not r: return []
    try:
        posts = r.json().get("data",{}).get("children",[])
        return [p["data"].get("title","") for p in posts]
    except Exception:
        return []

@st.cache_data(ttl=120, show_spinner=False)
def social_scores(symbols):
    titles = fetch_reddit_titles()
    if not titles:
        return pd.DataFrame(columns=["symbol","buzz","sentiment","social01"]).set_index("symbol")
    rows=[]
    for sym in symbols:
        pat = re.compile(rf"\b{re.escape(sym)}\b")
        hits = [t for t in titles if pat.search(t.upper())]
        buzz = len(hits)
        pol = float(np.mean([_polarity_safe(t) for t in hits])) if buzz>0 else 0.0
        rows.append({"symbol": sym, "buzz":buzz, "sentiment":pol})
    out = pd.DataFrame(rows).set_index("symbol")
    if out.empty:
        out["social01"] = []
        return out
    out["buzz01"] = (np.log1p(out["buzz"]) / (np.log1p(out["buzz"]).max() or 1)).fillna(0.0)
    out["sent01"] = ((out["sentiment"].clip(-1,1) + 1)/2.0).fillna(0.5)
    out["social01"] = (0.7*out["buzz01"] + 0.3*out["sent01"]).clip(0,1)
    return out[["buzz","sentiment","social01"]]

CP_KEY = st.secrets.get("CRYPTOPANIC_KEY") if hasattr(st, "secrets") else None

@st.cache_data(ttl=300, show_spinner=False)
def news_scores(symbols):
    if not CP_KEY:
        return pd.DataFrame(columns=["news_hits","news_sent","news01"]).set_index(pd.Index([], name="symbol"))
    url = "https://cryptopanic.com/api/v1/posts/"
    p   = {"auth_token": CP_KEY, "public": "true", "kind":"news", "filter":"hot"}
    r = safe_get(url, params=p, timeout=20)
    if not r:
        return
