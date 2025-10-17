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
APP_NAME = "Crypto Hybrid Live — Phase 2 (Fusion)"
st.set_page_config(page_title=APP_NAME, layout="wide")
USER_AGENT = {"User-Agent": "Mozilla/5.0 (CHL-Fusion)"}

FEATURES = {
    "SPARKLINES": False,   # enable later if desired
    "REDDIT": True,        # hot posts from r/CryptoCurrency (no key)
    "DEFI_LLAMA": True,    # TVL data
    "CRYPTOPANIC": True,   # needs secret token; auto-skip if missing
    "ALERTS": True,
    "SNAPSHOT": True,
}

CG_PER_PAGE = 150  # top by market cap

st.title("🟢 " + APP_NAME)
st.caption(
    "Truth > Noise • Fusion = LIPE Truth + Social (Reddit) + News + TVL. "
    "Education only — not financial advice. Data: CoinGecko • Reddit • DeFiLlama • CryptoPanic"
)

# ====================== LIPE CORE (Truth score) ======================
DEFAULT_WEIGHTS = dict(w_vol=0.30, w_m24=0.25, w_m7=0.25, w_liq=0.20)
FUSION_WEIGHTS  = dict(w_truth=0.60, w_social=0.20, w_news=0.10, w_tvl=0.10)

def _normalize_weights(w):
    s = float(sum(max(0.0, v) for v in w.values())) or 1.0
    return {k: max(0.0, v)/s for k, v in w.items()}

def pct_sigmoid(pct):
    if pd.isna(pct): return 0.5
    x = float(pct)/10.0
    return 1/(1+math.exp(-x))

def lipe_truth(df, w=DEFAULT_WEIGHTS):
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
    s = np.std(arr)
    chaos01 = min(max(s/20.0, 0.0), 1.0)
    return float(1.0 - chaos01)

# ====================== HELPERS ======================
def safe_get(url, params=None, timeout=25, retries=3, backoff=0.6, headers=None):
    h = headers or USER_AGENT
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=h)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(backoff*(2**i) + random.uniform(0,0.2))
    return None

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Controls")
    vs_currency = st.selectbox("Currency", ["usd"], index=0)
    topn = st.slider("Show top N by market cap", 20, 250, CG_PER_PAGE, step=10)
    search = st.text_input("🔎 Search coin (name or symbol)").strip().lower()

    st.markdown("---")
    st.subheader("Alerts")
    alert_truth = st.slider("Trigger: Fusion Truth ≥", 0.0, 1.0, 0.85, 0.01)
    alert_div = st.slider("Trigger: |Raw - Truth| ≥", 0.0, 1.0, 0.30, 0.01)

# ====================== DATA: COINGECKO ======================
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
          "name","symbol"]:
    if k not in df.columns: df[k] = np.nan

# engineered features
df["vol_to_mc"] = (df["total_volume"]/df["market_cap"]).replace([np.inf,-np.inf],np.nan).clip(0,2).fillna(0)
df["momo_1h01"]  = df["price_change_percentage_1h_in_currency"].apply(pct_sigmoid)
df["momo_24h01"] = df["price_change_percentage_24h_in_currency"].apply(pct_sigmoid)
df["momo_7d01"]  = df["price_change_percentage_7d_in_currency"].apply(pct_sigmoid)
mc = df["market_cap"].fillna(0)
df["liquidity01"] = 0 if mc.max()==0 else (mc-mc.min())/(mc.max()-mc.min()+1e-9)

df["raw_heat"]   = (0.5*(df["vol_to_mc"]/2).clip(0,1) + 0.5*df["momo_1h01"].fillna(0.5)).clip(0,1)
df["truth_full"] = lipe_truth(df)
df["divergence"] = (df["raw_heat"] - df["truth_full"]).round(3)
df["entropy01"]  = df.apply(lambda r: entropy01_from_changes(
    r.get("price_change_percentage_1h_in_currency"),
    r.get("price_change_percentage_24h_in_currency"),
    r.get("price_change_percentage_7d_in_currency")), axis=1)
df["mood"] = df["truth_full"].apply(mood_label)
df["COIN_SYM"] = df["symbol"].str.upper()

# ====================== SOCIAL: REDDIT ======================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_reddit_titles(limit=80):
    if not FEATURES["REDDIT"]: return []
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
        return pd.DataFrame(columns=["symbol","buzz","sentiment","social01"])
    rows=[]
    for sym in symbols:
        pat = re.compile(rf"\b{re.escape(sym)}\b")
        hits = [t for t in titles if pat.search(t.upper())]
        buzz = len(hits)
        pol = float(np.mean([_polarity_safe(t) for t in hits])) if buzz>0 else 0.0
        rows.append({"symbol": sym, "buzz":buzz, "sentiment":pol})
    out = pd.DataFrame(rows)
    if out.empty:
        out["social01"] = []
        return out
    out["buzz01"] = (np.log1p(out["buzz"]) / (np.log1p(out["buzz"]).max() or 1)).fillna(0.0)
    out["sent01"] = ((out["sentiment"].clip(-1,1) + 1)/2.0).fillna(0.5)
    out["social01"] = (0.7*out["buzz01"] + 0.3*out["sent01"]).clip(0,1)
    return out.set_index("symbol")[["buzz","sentiment","social01"]]

social = social_scores(df["COIN_SYM"].tolist())

# ====================== NEWS: CRYPTOPANIC (optional) ======================
CP_KEY = st.secrets.get("CRYPTOPANIC_KEY") if hasattr(st, "secrets") else None

@st.cache_data(ttl=300, show_spinner=False)
def news_scores(symbols):
    if not FEATURES["CRYPTOPANIC"] or not CP_KEY:
        return pd.DataFrame(columns=["news_hits","news_sent","news01"])
    url = "https://cryptopanic.com/api/v1/posts/"
    p   = {"auth_token": CP_KEY, "public": "true", "kind":"news", "filter":"hot"}
    r = safe_get(url, params=p, timeout=20)
    if not r:
        return pd.DataFrame(columns=["news_hits","news_sent","news01"])
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
    out = pd.DataFrame(rows)
    if out.empty:
        out["news01"]=[]
        return out
    out["hit01"] = (np.log1p(out["news_hits"]) / (np.log1p(out["news_hits"]).max() or 1)).fillna(0.0)
    out["sent01"] = ((out["news_sent"].clip(-1,1) + 1)/2.0).fillna(0.5)
    out["news01"] = (0.7*out["hit01"] + 0.3*out["sent01"]).clip(0,1)
    return out.set_index("symbol")[["news_hits","news_sent","news01"]]

news = news_scores(df["COIN_SYM"].tolist())

# ====================== TVL: DEFI LLAMA ======================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_defi_protocols():
    if not FEATURES["DEFI_LLAMA"]: return pd.DataFrame([])
    r = safe_get("https://api.llama.fi/protocols", timeout=30)
    if not r: return pd.DataFrame([])
    try: return pd.DataFrame(r.json())
    except Exception: return pd.DataFrame([])

def map_tvl_to_symbols(coins, llama):
    if llama.empty: return pd.DataFrame({"tvl_score01":[]})
    llama = llama[["symbol","tvl"]].fillna({"symbol":""})
    llama["symbol"] = llama["symbol"].str.upper()
    agg = llama.groupby("symbol", as_index=True)["tvl"].sum()
    # map each coin to its tvl and score
    raw = agg.reindex(coins).fillna(0.0)
    if raw.max() <= 0:
        score = pd.Series(0.0, index=raw.index, name="tvl_score01")
    else:
        score = (np.log1p(raw) / np.log1p(raw.max())).clip(0,1)
        score.name = "tvl_score01"
    return score.to_frame()

defi = fetch_defi_protocols()
tvl  = map_tvl_to_symbols(df["COIN_SYM"].tolist(), defi)

# ====================== FUSION JOIN (no suffix conflicts) ======================
# Put coin symbol in index to join without creating _x/_y columns
df = df.set_index("COIN_SYM")
if not social.empty: df = df.join(social, how="left")
if not news.empty:   df = df.join(news,   how="left")
if not tvl.empty:    df = df.join(tvl,    how="left")

# Fill external features
for col, default in [("social01",0.0), ("news01",0.0), ("tvl_score01",0.0)]:
    if col not in df.columns: df[col] = default
    df[col] = df[col].fillna(default)

# FUSION score
FW = _normalize_weights(FUSION_WEIGHTS)
df["fusion_truth"] = (
    FW["w_truth"]  * df["truth_full"].fillna(0.0) +
    FW["w_social"] * df["social01"] +
    FW["w_news"]   * df["news01"] +
    FW["w_tvl"]    * df["tvl_score01"]
).clip(0,1)
df["mood_fusion"] = df["fusion_truth"].apply(mood_label)

# restore regular index/columns for display
df = df.reset_index().rename(columns={"COIN_SYM":"symbol_up"})

# search filter
if search:
    mask = df["name"].str.lower().str.contains(search) | df["symbol"].str.lower().str.contains(search)
    df = df[mask].copy()

# ====================== KPIs ======================
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
avg_fusion = df["fusion_truth"].mean()
avg_truth  = df["truth_full"].mean()
avg_24h    = df["price_change_percentage_24h_in_currency"].mean()
c1,c2,c3,c4 = st.columns(4)
c1.metric("Coins", len(df))
c2.metric("Avg 24h Δ", f"{avg_24h:+.2f}%")
c3.metric("Avg LIPE Truth", f"{avg_truth:.2f}")
c4.metric("Avg Fusion Truth", f"{avg_fusion:.2f}")
st.caption(f"Last update: {now}")

# ====================== TABLES ======================
cols_fusion = ["name","symbol","current_price","market_cap","fusion_truth","truth_full","social01","news01","tvl_score01","divergence","mood_fusion"]
cols_raw    = ["name","symbol","current_price","market_cap","total_volume","raw_heat"]
cols_truth  = ["name","symbol","current_price","market_cap","liquidity01","truth_full","divergence","entropy01","mood"]

tab1, tab2, tab3, tab4 = st.tabs(["🧭 Fusion Truth","🔥 Raw","🧭 LIPE Truth","📉 Movers"])
with tab1:
    st.subheader("🧭 Fusion Truth (LIPE + Social + News + TVL)")
    st.dataframe(df.sort_values("fusion_truth", ascending=False)[[c for c in cols_fusion if c in df.columns]].reset_index(drop=True), use_container_width=True)

with tab2:
    st.subheader("🔥 Raw Wide Scan")
    raw_col = "raw_heat"
    if raw_col not in df.columns:
        cands = [c for c in df.columns if c.startswith("raw_heat")]
        if cands: raw_col = cands[0]
    st.dataframe(df.sort_values(raw_col, ascending=False)[[c for c in cols_raw if c in df.columns]].reset_index(drop=True), use_container_width=True)

with tab3:
    st.subheader("🧭 LIPE Truth")
    st.dataframe(df.sort_values("truth_full", ascending=False)[[c for c in cols_truth if c in df.columns]].reset_index(drop=True), use_container_width=True)

with tab4:
    st.subheader("📉 24h Gainers / Losers")
    g = df.sort_values("price_change_percentage_24h_in_currency", ascending=False).head(12)
    l = df.sort_values("price_change_percentage_24h_in_currency", ascending=True).head(12)
    cA, cB = st.columns(2)
    cA.write("**Top Gainers**")
    cA.dataframe(g[["name","symbol","current_price","price_change_percentage_24h_in_currency"]].reset_index(drop=True), use_container_width=True)
    cB.write("**Top Losers**")
    cB.dataframe(l[["name","symbol","current_price","price_change_percentage_24h_in_currency"]].reset_index(drop=True), use_container_width=True)

# ====================== FOCUS COIN (optional gauge) ======================
st.markdown("---")
st.subheader("🎯 Focus coin")
names = ["(none)"] + df["name"].head(50).tolist()
pick = st.selectbox("Pick a coin to inspect", names, index=0)
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
        st.dataframe(hits.sort_values("fusion_truth", ascending=False)[["name","symbol","fusion_truth","truth_full","divergence","mood_fusion"]], use_container_width=True)

# ====================== SNAPSHOT (CSV) ======================
def mk_snapshot():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_UTC")
    buf = StringIO()
    buf.write(f"Snapshot,{ts}\n\nTop by Fusion Truth\n")
    buf.write(df.sort_values('fusion_truth', ascending=False).head(25)[[c for c in cols_fusion if c in df.columns]].to_csv(index=False))
    buf.write("\nTop by Truth\n")
    buf.write(df.sort_values('truth_full', ascending=False).head(25)[[c for c in cols_truth if c in df.columns]].to_csv(index=False))
    buf.write("\nTop by Raw\n")
    buf.write(df.sort_values('raw_heat', ascending=False).head(25)[[c for c in cols_raw if c in df.columns]].to_csv(index=False))
    return f"snapshot_{ts}.csv", buf.getvalue().encode("utf-8")

if FEATURES["SNAPSHOT"]:
    fn, payload = mk_snapshot()
    st.download_button("⬇️ Download Snapshot (Fusion + Truth + Raw)", payload, file_name=fn, mime="text/csv")

# ====================== FOOTER ======================
st.markdown("""<hr style="margin-top: 1rem; margin-bottom: 0.5rem;">""", unsafe_allow_html=True)
st.caption("APIs OK if tables are showing • Source blend: CoinGecko + Reddit + (CryptoPanic if keyed) + DeFiLlama • CHL Fusion ©")
