
import math, requests, pandas as pd, numpy as np, streamlit as st
from datetime import datetime, timezone

# ---------- Page & Sidebar ----------
st.set_page_config(page_title="Crypto Hybrid Live — Core+", layout="wide")
st.title("🟢 Crypto Hybrid Live — Core+")
st.caption("Truth > Noise. Live market intelligence with customizable weights and insights.")

with st.sidebar:
    st.header("⚙️ Controls")
    vs_currency = st.selectbox("Currency", ["usd"], index=0)
    topn = st.slider("Show top N by market cap", 10, 250, 100, step=10)
    st.markdown("---")
    st.subheader("Weights (Truth Score)")
    w_vol = st.slider("Volume/MC weight", 0.0, 1.0, 0.30, 0.01)
    w_m24 = st.slider("Momentum 24h weight", 0.0, 1.0, 0.25, 0.01)
    w_m7  = st.slider("Momentum 7d weight", 0.0, 1.0, 0.25, 0.01)
    w_liq = st.slider("Liquidity weight", 0.0, 1.0, 0.20, 0.01)
    if abs((w_vol + w_m24 + w_m7 + w_liq) - 1.0) > 1e-6:
        st.info("Weights will be normalized to sum to 1.0")
    st.markdown("---")
    search = st.text_input("🔎 Search (name or symbol)").strip().lower()

# ---------- Helpers ----------
def normalize_weights(a, b, c, d):
    s = a + b + c + d
    if s <= 1e-9: return 0.25, 0.25, 0.25, 0.25
    return a/s, b/s, c/s, d/s

def pct_sigmoid(pct):
    if pd.isna(pct): return 0.5
    x = float(pct)/10.0
    return 1/(1+math.exp(-x))

def mood_label(truth):
    try:
        t = float(truth)
    except Exception:
        t = 0.5
    if t >= 0.80: return "🟢 EUPHORIC"
    if t >= 0.60: return "🟡 OPTIMISTIC"
    if t >= 0.40: return "🟠 NEUTRAL"
    return "🔴 FEARFUL"

@st.cache_data(ttl=60)
def fetch_markets(vs="usd", page=1, per_page=250):
    """Fetch top markets from CoinGecko with caching."""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    p = {
        "vs_currency": vs,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "sparkline": "false",
        "price_change_percentage": "1h,24h,7d",
        "locale": "en",
    }
    try:
        r = requests.get(url, params=p, timeout=30)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame([])

# ---------- Data load & features ----------
df = fetch_markets(vs_currency)
if df.empty:
    st.error("Could not load data from CoinGecko right now. Please refresh in a minute.")
    st.stop()

# Keep only top N by market cap to speed up UI
df = df.sort_values("market_cap", ascending=False).head(topn).copy()

# Ensure required columns exist
need = [
    "current_price","market_cap","total_volume",
    "price_change_percentage_1h_in_currency",
    "price_change_percentage_24h_in_currency",
    "price_change_percentage_7d_in_currency",
    "name","symbol"
]
for k in need:
    if k not in df.columns:
        df[k] = np.nan

# Feature engineering
df["vol_to_mc"] = (df["total_volume"]/df["market_cap"]).replace([np.inf,-np.inf],np.nan).clip(0,2).fillna(0)
df["momo_1h01"]  = df["price_change_percentage_1h_in_currency"].apply(pct_sigmoid)
df["momo_24h01"] = df["price_change_percentage_24h_in_currency"].apply(pct_sigmoid)
df["momo_7d01"]  = df["price_change_percentage_7d_in_currency"].apply(pct_sigmoid)
mc = df["market_cap"].fillna(0)
df["liquidity01"] = 0 if mc.max()==0 else (mc - mc.min())/(mc.max() - mc.min() + 1e-9)

# Scores
df["raw_heat"] = (0.5*(df["vol_to_mc"]/2).clip(0,1) + 0.5*df["momo_1h01"].fillna(0.5)).clip(0,1)

wv, wm24, wm7, wliq = normalize_weights(w_vol, w_m24, w_m7, w_liq)
df["truth_full"] = (
    wv*(df["vol_to_mc"]/2).clip(0,1) +
    wm24*df["momo_24h01"].fillna(0.5) +
    wm7*df["momo_7d01"].fillna(0.5) +
    wliq*df["liquidity01"].fillna(0.0)
).clip(0,1)

df["divergence"] = (df["raw_heat"] - df["truth_full"]).round(3)
df["mood"] = df["truth_full"].apply(mood_label)

# Optional search filter
if search:
    mask = df["name"].str.lower().str.contains(search) | df["symbol"].str.lower().str.contains(search)
    df = df[mask].copy()

now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
st.write(f"Last update: **{now}** • Coins: **{len(df)}** • Data: CoinGecko")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["🔥 Raw Scan", "🧭 Truth Filter", "📉 Movers", "🧠 Insights"])

with tab1:
    st.subheader("🔥 Raw Wide Scan (energy & 1h momentum)")
    cols = ["name","symbol","current_price","market_cap","total_volume","raw_heat"]
    st.dataframe(df.sort_values("raw_heat", ascending=False)[cols].reset_index(drop=True), use_container_width=True)

with tab2:
    st.subheader("🧭 Truth Filter (weighted & normalized)")
    cols = ["name","symbol","current_price","market_cap","liquidity01","truth_full","divergence","mood"]
    st.dataframe(df.sort_values("truth_full", ascending=False)[cols].reset_index(drop=True), use_container_width=True)

with tab3:
    st.subheader("📉 Top Daily Gainers / Losers (24h)")
    g = df.sort_values("price_change_percentage_24h_in_currency", ascending=False).head(12).copy()
    l = df.sort_values("price_change_percentage_24h_in_currency", ascending=True).head(12).copy()
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Top Gainers**")
        st.dataframe(g[["name","symbol","current_price","price_change_percentage_24h_in_currency"]].reset_index(drop=True), use_container_width=True)
    with c2:
        st.write("**Top Losers**")
        st.dataframe(l[["name","symbol","current_price","price_change_percentage_24h_in_currency"]].reset_index(drop=True), use_container_width=True)

with tab4:
    st.subheader("🧠 AI-Style Narrative (rule-based v1)")
    # Simple, fast human-readable summaries without external LLMs
    def insight_row(r):
        pieces = []
        # momentum feel
        m24 = r.get("price_change_percentage_24h_in_currency", 0)
        m7  = r.get("price_change_percentage_7d_in_currency", 0)
        if pd.notna(m24):
            pieces.append(f"24h {m24:+.1f}%")
        if pd.notna(m7):
            pieces.append(f"7d {m7:+.1f}%")
        volmc = r.get("vol_to_mc", 0)
        pieces.append("high vol/MC" if volmc>1 else "mod vol/MC")
        mood = r.get("mood","NEUTRAL")
        return f"{r['name']} ({r['symbol'].upper()}): Truth {r['truth_full']:.2f}, {', '.join(pieces)} → {mood}"
    top_insights = df.sort_values("truth_full", ascending=False).head(10)
    for _, row in top_insights.iterrows():
        st.write("• " + insight_row(row))

# ---------- Footer ----------
st.markdown("""<hr style="margin-top: 1rem; margin-bottom: 0.5rem;">""", unsafe_allow_html=True)
st.caption("Data source: CoinGecko • Educational analytics, not financial advice. © Crypto Hybrid Live")
