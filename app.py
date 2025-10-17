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
APP_NAME = "Crypto Hybrid Live — Phase 4.1 (Pro + Reload)"
st.set_page_config(page_title=APP_NAME, layout="wide")
USER_AGENT = {"User-Agent": "Mozilla/5.0 (CHL-Phase4.1)"}

FEATURES = {
    "SPARKLINES": False,   # can enable later
    "REDDIT": True,        # r/CryptoCurrency hot (no key)
    "DEFI_LLAMA": True,    # TVL
    "CRYPTOPANIC": True,   # needs secret token; auto-skip if missing
    "ALERTS": True,
    "SNAPSHOT": True,
}

CG_PER_PAGE = 150

# ====================== BIGGER (clean) UI + THEME ======================
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
st.caption("Truth > Noise • Phase 4.1 adds **Reload data** and **Empty-table guard** on top of Pro features")

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
            st.success("Weights reset to Balanced default.")

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

# ====================== DATA HELPERS ======================
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

# ====================== PULL DATA ======================
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
TRUTH_W = dict(_normalize_weights(w_edit))
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
        return pd.DataFrame(columns=["news_hits","news_sent","news01"]).set_index(pd.Index([], name="symbol"))
    try:
        titles = [x.get("title","") for x in r.json().get("results", [])]
    except Exception:
        titles = []
    rows=[]
    for sym in symbols:
        pat = re.compile(rf"\b{re.escape(sym)}\b")
        hits = [t for t in titles if pat.search(t.upper())]
        n=len(hits)
        pol = float(np.mean([_polarity_safe(t) for t in hits])) if n>0 else 0.0
        rows.append({"symbol":sym,"news_hits":n,"news_sent":pol})
    out = pd.DataFrame(rows).set_index("symbol")
    if out.empty:
        out["news01"]=[]
        return out
    out["hit01"] = (np.log1p(out["news_hits"]) / (np.log1p(out["news_hits"]).max() or 1)).fillna(0.0)
    out["sent01"] = ((out["news_sent"].clip(-1,1) + 1)/2.0).fillna(0.5)
    out["news01"] = (0.7*out["hit01"] + 0.3*out["sent01"]).clip(0,1)
    return out[["news_hits","news_sent","news01"]]

@st.cache_data(ttl=600, show_spinner=False)
def fetch_defi_protocols():
    r = safe_get("https://api.llama.fi/protocols", timeout=30)
    if not r: return pd.DataFrame([])
    try: return pd.DataFrame(r.json())
    except Exception: return pd.DataFrame([])

def map_tvl_to_symbols(coins, llama):
    if llama.empty: return pd.DataFrame({"tvl_score01":[]}).set_index(pd.Index([], name="symbol"))
    llama = llama[["symbol","tvl"]].fillna({"symbol":""})
    llama["symbol"] = llama["symbol"].str.upper()
    agg = llama.groupby("symbol", as_index=True)["tvl"].sum()
    raw = agg.reindex(coins).fillna(0.0)
    if raw.max() <= 0:
        score = pd.Series(0.0, index=raw.index, name="tvl_score01")
    else:
        score = (np.log1p(raw) / np.log1p(raw.max())).clip(0,1)
        score.name = "tvl_score01"
    return score.to_frame()

# Join external
df = df.set_index("COIN_SYM")
if use_social: df = df.join(social_scores(df.index.tolist()), how="left")
if use_news:   df = df.join(news_scores(df.index.tolist()),   how="left")
if use_tvl:
    tvl = map_tvl_to_symbols(df.index.tolist(), fetch_defi_protocols())
    df = df.join(tvl, how="left")

# Fill external features defaults
for col, default in [("social01",0.0), ("news01",0.0), ("tvl_score01",0.0)]:
    if col not in df.columns: df[col] = default
    df[col] = df[col].fillna(default)

# Fusion
FW = _normalize_weights(FUSION_WEIGHTS)
df["fusion_truth"] = (
    FW["w_truth"]  * df["truth_full"].fillna(0.0) +
    FW["w_social"] * df["social01"] +
    FW["w_news"]   * df["news01"] +
    FW["w_tvl"]    * df["tvl_score01"]
).clip(0,1)
df["mood_fusion"] = df["fusion_truth"].apply(mood_label)

# Back to normal columns
df = df.reset_index().rename(columns={"COIN_SYM":"symbol_up"})

# ---------------------- Quick Filters ----------------------
if search:
    mask = df["name"].str.lower().str.contains(search) | df["symbol"].str.lower().str.contains(search)
    df = df[mask].copy()

if q_large_cap:  df = df.sort_values("market_cap_rank").head(50)
if q_small_cap:  df = df[df["market_cap_rank"] > 200]
if q_top_gainers: df = df.sort_values("price_change_percentage_24h_in_currency", ascending=False).head(50)
if q_top_losers:  df = df.sort_values("price_change_percentage_24h_in_currency", ascending=True).head(50)

# ====================== KPIs + RELOAD ======================
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Coins", len(df))
c2.metric("Avg 24h Δ", f"{df['price_change_percentage_24h_in_currency'].mean():+.2f}%")
c3.metric("Avg LIPE Truth", f"{df['truth_full'].mean():.2f}")
c4.metric("Avg Fusion Truth", f"{df['fusion_truth'].mean():.2f}")
st.caption(f"Last update: {now}")

# --- Reload button to force fresh API pull
reload = st.button("🔄 Reload data (bypass cache)")
if reload:
    fetch_markets.clear()              # clear cached function data
    df = fetch_markets(vs_currency)    # re-fetch immediately
    st.experimental_rerun()

# --- Show message if filters/search remove all rows (prevents blank screen)
if df.empty:
    st.warning(
        "⚠️ No coins match the current **filters or search term**.\n"
        "Try clearing the search box, turning off Quick Filters, or hitting Reload."
    )
    st.stop()

# ====================== COLUMN MODES ======================
SIMPLE_COLS_FUSION = ["name","symbol","current_price","fusion_truth","mood_fusion"]
PRO_COLS_FUSION    = ["name","symbol","current_price","market_cap","fusion_truth","truth_full","social01","news01","tvl_score01","divergence","mood_fusion"]
SIMPLE_COLS_RAW    = ["name","symbol","current_price","raw_heat","total_volume"]
PRO_COLS_RAW       = ["name","symbol","current_price","market_cap","total_volume","raw_heat"]
SIMPLE_COLS_TRUTH  = ["name","symbol","current_price","truth_full","mood"]
PRO_COLS_TRUTH     = ["name","symbol","current_price","market_cap","liquidity01","truth_full","divergence","entropy01","mood"]

mode_simple = st.toggle("🧸 Easy Mode (simple columns)", value=True)
HEIGHT = 600
cols_fusion = SIMPLE_COLS_FUSION if mode_simple else PRO_COLS_FUSION
cols_raw    = SIMPLE_COLS_RAW    if mode_simple else PRO_COLS_RAW
cols_truth  = SIMPLE_COLS_TRUTH  if mode_simple else PRO_COLS_TRUTH

# ====================== TABS ======================
tab_fusion, tab_raw, tab_truth, tab_move, tab_watch, tab_help = st.tabs(
    ["🧭 Fusion Truth","🔥 Raw","🧭 LIPE Truth","📉 Movers","⭐ Watchlist","❓ Explainer"]
)

with tab_fusion:
    st.subheader("🧭 Fusion Truth (LIPE + Social + News + TVL)")
    st.dataframe(df.sort_values("fusion_truth", ascending=False)[[c for c in cols_fusion if c in df.columns]].reset_index(drop=True),
                 use_container_width=True, height=HEIGHT)

with tab_raw:
    st.subheader("🔥 Raw Wide Scan")
    raw_col = "raw_heat"
    if raw_col not in df.columns:
        cands = [c for c in df.columns if c.startswith("raw_heat")]
        if cands: raw_col = cands[0]
    st.dataframe(df.sort_values(raw_col, ascending=False)[[c for c in cols_raw if c in df.columns]].reset_index(drop=True),
                 use_container_width=True, height=HEIGHT)

with tab_truth:
    st.subheader("🧭 LIPE Truth")
    st.markdown(
        "<span class='pill'>Volume/MC</span><span class='pill'>Momentum 24h</span>"
        "<span class='pill'>Momentum 7d</span><span class='pill'>Liquidity</span>",
        unsafe_allow_html=True)
    st.dataframe(df.sort_values("truth_full", ascending=False)[[c for c in cols_truth if c in df.columns]].reset_index(drop=True),
                 use_container_width=True, height=HEIGHT)

with tab_move:
    st.subheader("📉 24h Gainers / Losers")
    g = df.sort_values("price_change_percentage_24h_in_currency", ascending=False).head(15)
    l = df.sort_values("price_change_percentage_24h_in_currency", ascending=True).head(15)
    cA, cB = st.columns(2)
    cA.write("**Top Gainers**")
    cA.dataframe(g[["name","symbol","current_price","price_change_percentage_24h_in_currency"]].reset_index(drop=True),
                 use_container_width=True, height=int(HEIGHT/1.4))
    cB.write("**Top Losers**")
    cB.dataframe(l[["name","symbol","current_price","price_change_percentage_24h_in_currency"]].reset_index(drop=True),
                 use_container_width=True, height=int(HEIGHT/1.4))

with tab_watch:
    st.subheader("⭐ Watchlist")
    choices = ["(add…)"] + df["symbol"].tolist()
    pick = st.selectbox("Add a coin to your watchlist", choices, index=0)
    if pick != "(add…)":
        wl = set(ensure_profile()["watchlist"]); wl.add(pick.upper())
        st.session_state["profile"]["watchlist"] = sorted(list(wl))
        st.success(f"Added **{pick}** to watchlist.")
    wl = ensure_profile()["watchlist"]
    if wl:
        st.markdown("**Your watchlist**")
        st.dataframe(df[df["symbol"].str.upper().isin(wl)][["name","symbol","current_price","fusion_truth","truth_full","mood_fusion"]],
                     use_container_width=True, height=int(HEIGHT/1.3))
    else:
        st.info("No coins yet. Add some above!")

with tab_help:
    st.subheader("❓ Truth & Fusion — simple explainer")
    st.markdown("""
    **Truth Score** is like judging a car race:
    - **Volume/MC** = size of the crowd vs track size (busy = real interest)  
    - **Momentum 24h & 7d** = how fast cars sped up recently  
    - **Liquidity** = how wide the track is (easy to move)  
    We mix these into **0..1**. Green = strong, Red = weak.  
    **Fusion Truth** adds **Social buzz**, **News**, and **TVL** on top of Truth.
    """)
    st.info("Tip: use **Presets** and **Weight Editor** in the sidebar to shape the Truth Score your way.")

# ====================== FOCUS COIN (gauge) ======================
st.markdown("---")
st.subheader("🎯 Focus coin")
names = ["(none)"] + df["name"].head(50).tolist()
pick = st.selectbox("Pick a coin to inspect", names, index=0, key="focus_picker")
if pick != "(none)" and PLOTLY_OK:
    row = df[df["name"]==pick].head(1).to_dict("records")[0]
    st.success(f"**{pick}** → Fusion {row['fusion_truth']:.2f} ({row['mood_fusion']}) • Truth {row['truth_full']:.2f}")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(row["fusion_truth"]),
        number={'valueformat': '.2f'},
        gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#23d18b"}}
    ))
    fig.update_layout(height=220, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)
elif pick != "(none)":
    st.info("Plotly not installed yet; gauge hidden.")

# ====================== ALERTS ======================
if FEATURES["ALERTS"]:
    hits = df[(df["fusion_truth"] >= alert_truth) | (df["divergence"].abs() >= alert_div)]
    if len(hits):
        st.warning(f"🚨 {len(hits)} coins matched your rules")
        st.dataframe(hits.sort_values("fusion_truth", ascending=False)
                     [["name","symbol","fusion_truth","truth_full","divergence","mood_fusion"]],
                     use_container_width=True, height=int(HEIGHT/1.3))

# ====================== SNAPSHOT (CSV) ======================
def mk_snapshot():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_UTC")
    buf = StringIO()
    buf.write(f"Snapshot,{ts}\n\nTop by Fusion Truth\n")
    buf.write(df.sort_values('fusion_truth', ascending=False).head(25)
              [["name","symbol","current_price","market_cap","fusion_truth","truth_full","social01","news01","tvl_score01","divergence","mood_fusion"]].to_csv(index=False))
    buf.write("\nTop by Truth\n")
    buf.write(df.sort_values('truth_full', ascending=False).head(25)
              [["name","symbol","current_price","market_cap","liquidity01","truth_full","divergence","entropy01","mood"]].to_csv(index=False))
    buf.write("\nTop by Raw\n")
    raw_col = "raw_heat" if "raw_heat" in df.columns else next((c for c in df.columns if c.startswith("raw_heat")), "raw_heat")
    cols_raw_all = [c for c in ["name","symbol","current_price","market_cap","total_volume","raw_heat"] if c in df.columns]
    buf.write(df.sort_values(raw_col, ascending=False).head(25)[cols_raw_all].to_csv(index=False))
    return f"snapshot_{ts}.csv", buf.getvalue().encode("utf-8")

if FEATURES["SNAPSHOT"]:
    fn, payload = mk_snapshot()
    st.download_button("⬇️ Download Snapshot (Fusion + Truth + Raw)",
                       payload, file_name=fn, mime="text/csv")

# ====================== FOOTER ======================
st.markdown("""<hr style="margin-top: 1rem; margin-bottom: 0.5rem;">""", unsafe_allow_html=True)
st.caption("APIs OK if tables are showing • Source blend: CoinGecko + Reddit + (CryptoPanic if keyed) + DeFiLlama • CHL Phase 4.1 ©")
