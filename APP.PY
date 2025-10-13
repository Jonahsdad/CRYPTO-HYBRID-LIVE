# app.py — Crypto Hybrid Live (Full, Streamlit Cloud)
# Top 250 coins from CoinGecko with:
# - Raw Wide Scan (uncut heat)
# - Truth Filter (Full): market + liquidity + dev + optional social/video/on-chain
# - Dev Pulse (CoinGecko developer_data for Top-N)
# - YouTube Pulse (if YT_KEY set in secrets)
# - Twitter mentions (if TW_BEARER set)
# - On-chain TX proxy via Covalent (if COVALENT_KEY set)
# - Entropy + Divergence + Alerts
# Designed to run on Streamlit Community Cloud, iPad-friendly.

import os, math, time, re, requests, pandas as pd, numpy as np, streamlit as st
from datetime import datetime, timezone
from textblob import TextBlob

st.set_page_config(page_title="Crypto Hybrid Live", layout="wide")
st.title("🟢 Crypto Hybrid Live — Full")
st.caption("Top 250 coins from CoinGecko. Auto-refresh ~60s. Market + Dev + optional Social/YouTube/On-chain + Entropy.")

# ─── Secrets (auto-off if missing) ─────────────────────────────────────────────
TW_BEARER    = os.environ.get("TW_BEARER", "").strip()
YT_KEY       = os.environ.get("YT_KEY", "").strip()
COVALENT_KEY = os.environ.get("COVALENT_KEY", "").strip()

ENABLE_TW  = bool(TW_BEARER)
ENABLE_YT  = bool(YT_KEY)
ENABLE_COV = bool(COVALENT_KEY)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def pct_sigmoid(pct):
    if pd.isna(pct): return 0.5
    x = float(pct) / 10.0  # ±10% → strong response
    return 1.0 / (1.0 + math.exp(-x))

def clip01(x):
    try: return max(0.0, min(1.0, float(x)))
    except: return 0.0

def safe_get(url, params=None, headers=None, timeout=25):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def quick_sent(text: str) -> float:
    try: return float(TextBlob(text or "").sentiment.polarity)
    except: return 0.0

def clean_query(name:str, symbol:str):
    base = f"{name} {symbol} crypto"
    return re.sub(r"\s+", " ", base).strip()

# ─── CoinGecko ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=45)
def cg_markets():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency":"usd", "order":"market_cap_desc", "per_page":250, "page":1,
        "sparkline":"false", "price_change_percentage":"1h,24h,7d", "locale":"en"
    }
    j = safe_get(url, params=params)
    return pd.DataFrame(j or [])

@st.cache_data(ttl=180)
def cg_coin_detail(coin_id: str):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    params = {
        "localization":"false", "tickers":"false",
        "market_data":"false", "community_data":"false",
        "developer_data":"true", "sparkline":"false"
    }
    return safe_get(url, params=params) or {}

def compute_dev01_from_devdata(dev):
    if not dev: return 0.5
    cc4w = float(dev.get("commit_count_4_weeks") or 0.0)
    forks = float(dev.get("forks") or 0.0)
    stars = float(dev.get("stars") or 0.0)
    cc = clip01(cc4w / 50.0)       # 50 commits/4w → ~1.0
    fk = clip01(forks / 500.0)
    stx= clip01(stars / 3000.0)
    return clip01(0.6*cc + 0.2*fk + 0.2*stx)

# ─── YouTube (optional) ───────────────────────────────────────────────────────
def yt_views_and_sentiment(name, symbol, max_items=3):
    if not ENABLE_YT:
        return 0.0, 0.5
    try:
        q = clean_query(name, symbol)
        s_url = "https://www.googleapis.com/youtube/v3/search"
        s_params = {"key":YT_KEY, "q":q, "part":"snippet", "type":"video", "maxResults":max_items, "relevanceLanguage":"en"}
        s = safe_get(s_url, s_params) or {}
        ids = [it["id"]["videoId"] for it in (s.get("items") or []) if it.get("id") and it["id"].get("videoId")]
        if not ids:
            return 0.0, 0.5
        v_url = "https://www.googleapis.com/youtube/v3/videos"
        v_params = {"key":YT_KEY, "id":",".join(ids), "part":"statistics,snippet"}
        v = safe_get(v_url, v_params) or {}
        view_sum, sents = 0.0, []
        for it in (v.get("items") or []):
            stats = it.get("statistics") or {}
            title = ((it.get("snippet") or {}).get("title") or "")
            view_sum += float(stats.get("viewCount") or 0.0)
            sents.append(quick_sent(title))
        view01 = clip01(view_sum / 300000.0)                 # 300k combined → ~1.0
        sent01 = clip01((np.mean(sents or [0.0]) + 1.0)/2.0) # -1..+1 → 0..1
        return view01, sent01
    except:
        return 0.0, 0.5

# ─── Twitter/X (optional) ─────────────────────────────────────────────────────
def tw_mentions_per15m(name, symbol):
    if not ENABLE_TW:
        return 0.0
    try:
        headers = {"Authorization": f"Bearer {TW_BEARER}"}
        query = f'("{name}" OR {symbol}) (crypto OR coin OR token) -is:retweet lang:en'
        url = "https://api.twitter.com/2/tweets/search/recent"
        params = {"query": query, "max_results": 50, "tweet.fields":"created_at"}
        j = safe_get(url, params=params, headers=headers) or {}
        cnt = len(j.get("data", []))
        return float(cnt) / 15.0
    except:
        return 0.0

# ─── On-chain (Covalent, ETH demo) ────────────────────────────────────────────
def covalent_tx_count_eth(contract_address):
    if not ENABLE_COV or not contract_address:
        return 0
    try:
        url = f"https://api.covalenthq.com/v1/1/address/{contract_address}/transactions_v3/"
        j = safe_get(url, params={"key": COVALENT_KEY, "no-logs":"true"}) or {}
        items = (j.get("data") or {}).get("items") or []
        return int(len(items))
    except:
        return 0

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Controls")
    topn    = st.slider("Rows per leaderboard", 10, 50, 20, 5)
    deep_n  = st.slider("Deep scan (Top-N for Dev/Social/On-chain)", 5, 50, 25, 5)
    filt    = st.text_input("Filter by name/symbol", "").strip().lower()
    st.caption(f"Twitter: {'ON' if ENABLE_TW else 'OFF'} • YouTube: {'ON' if ENABLE_YT else 'OFF'} • On-chain: {'ON' if ENABLE_COV else 'OFF'}")
    st.caption("Auto-refresh ~every 60s.")

# ─── Fetch + Features ─────────────────────────────────────────────────────────
try:
    df = cg_markets()
except Exception as e:
    st.error(f"CoinGecko error: {e}")
    st.stop()

need = [
    "id","symbol","name","current_price","market_cap","total_volume",
    "price_change_percentage_1h_in_currency",
    "price_change_percentage_24h_in_currency",
    "price_change_percentage_7d_in_currency",
    "last_updated"
]
for k in need:
    if k not in df.columns: df[k] = np.nan
df.rename(columns={
    "current_price":"price",
    "total_volume":"volume_24h",
    "price_change_percentage_1h_in_currency":"chg_1h",
    "price_change_percentage_24h_in_currency":"chg_24h",
    "price_change_percentage_7d_in_currency":"chg_7d",
}, inplace=True)

# Base features
df["vol_to_mc"] = (df["volume_24h"] / df["market_cap"]).replace([np.inf,-np.inf], np.nan).clip(0,2.0).fillna(0)
df["momo_1h01"]  = df["chg_1h"].apply(pct_sigmoid)
df["momo_24h01"] = df["chg_24h"].apply(pct_sigmoid)
df["momo_7d01"]  = df["chg_7d"].apply(pct_sigmoid)
mc = df["market_cap"].fillna(0)
df["liquidity01"] = (mc - mc.min()) / (mc.max() - mc.min() + 1e-9) if mc.max()>0 else 0.0

# Dev pulse for Top-N by market cap
df["dev01"] = 0.5
top_ids = df.sort_values("market_cap", ascending=False).head(deep_n)["id"].tolist()
dev_map = {}
for cid in top_ids:
    detail = cg_coin_detail(cid)
    dev_map[cid] = compute_dev01_from_devdata(detail.get("developer_data"))
df["dev01"] = df.apply(lambda r: dev_map.get(r["id"], 0.5), axis=1)

# Optional layers on Top-N
yt_view, yt_sent, tw_15m, tx_24h = {}, {}, {}, {}
if ENABLE_YT or ENABLE_TW or ENABLE_COV:
    for cid in top_ids:
        row = df[df["id"]==cid].iloc[0]
        if ENABLE_YT:
            v01, s01 = yt_views_and_sentiment(row["name"], row["symbol"])
            yt_view[cid] = v01; yt_sent[cid] = s01
        if ENABLE_TW:
            tw_15m[cid] = tw_mentions_per15m(row["name"], row["symbol"])
        if ENABLE_COV:
            detail = cg_coin_detail(cid)
            platforms = detail.get("platforms") or {}
            eth_contract = platforms.get("ethereum")
            tx_24h[cid] = covalent_tx_count_eth(eth_contract) if eth_contract else 0

df["yt_view01"] = df["id"].map(lambda x: yt_view.get(x, 0.0))
df["yt_sent01"] = df["id"].map(lambda x: yt_sent.get(x, 0.5))
df["tw_per15m"] = df["id"].map(lambda x: tw_15m.get(x, 0.0))
df["tx_24h"]    = df["id"].map(lambda x: tx_24h.get(x, 0))

# Entropy (volatility of momentum + optional sentiment dispersion)
entropy_base = df[["momo_1h01","momo_24h01","momo_7d01"]].std(axis=1).fillna(0.0)
if ENABLE_YT:
    df["entropy01"] = clip01((entropy_base + df[["yt_sent01"]].std(axis=1).fillna(0.0)) / 1.5)
else:
    df["entropy01"] = clip01(entropy_base)

# Raw Heat & Truth (Full) scores
df["raw_heat"] = (
    0.5 * (df["vol_to_mc"]/2.0).clip(0,1) +
    0.5 * df["momo_1h01"].fillna(0.5)
).clip(0,1)

df["truth_full"] = (
    0.20 * (df["vol_to_mc"]/2.0).clip(0,1) +
    0.20 * df["momo_24h01"].fillna(0.5) +
    0.15 * df["momo_7d01"].fillna(0.5) +
    0.15 * df["liquidity01"].fillna(0.0) +
    0.15 * df["dev01"].fillna(0.5) +
    0.05 * df["yt_view01"].fillna(0.0) +
    0.05 * (df["yt_sent01"].fillna(0.5)) +
    0.03 * clip01(df["tw_per15m"]/5.0) +
    0.02 * clip01(df["tx_24h"]/50.0)
).clip(0,1)

# Divergence: hype vs truth
df["divergence"] = (df["raw_heat"] - df["truth_full"]).round(3)

# Pretty columns
df["price_disp"] = df["price"].map(lambda v: f"${v:,.6f}" if v and v<1 else f"${v:,.2f}")
df["mc_disp"]    = df["market_cap"].map(lambda v: f"${v:,.0f}")
df["vol_disp"]   = df["volume_24h"].map(lambda v: f"${v:,.0f}")
df["chg1h_disp"] = df["chg_1h"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "—")
df["chg24h_disp"]= df["chg_24h"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "—")
df["chg7d_disp"] = df["chg_7d"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "—")

# Filter
filt = filt.strip().lower()
if filt:
    mask = df["name"].str.lower().str.contains(filt) | df["symbol"].str.lower().str.contains(filt)
    dfv = df[mask].copy()
else:
    dfv = df.copy()

# Header & alerts
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
st.write(
    f"Last update: **{now}** — Coins: **{len(dfv)}** — Deep scan: **Top {len(top_ids)}** "
    f"— Twitter: {'ON' if ENABLE_TW else 'OFF'} — YouTube: {'ON' if ENABLE_YT else 'OFF'} — On-chain: {'ON' if ENABLE_COV else 'OFF'}"
)

hot_truth = dfv[dfv["truth_full"] >= 0.75].sort_values("truth_full", ascending=False).head(5)
if not hot_truth.empty:
    st.success("⚡ Strong Truth signals: " + ", ".join(hot_truth["symbol"].str.upper().tolist()))

big_div = dfv[dfv["divergence"] >= 0.25].sort_values("divergence", ascending=False).head(5)
if not big_div.empty:
    st.warning("📣 Hype ≫ Truth (watch risk): " + ", ".join(big_div["symbol"].str.upper().tolist()))

# Three panels
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("🔥 Raw Wide Scan (uncut)")
    cols = ["name","symbol","price_disp","mc_disp","vol_disp","chg1h_disp","chg24h_disp","chg7d_disp","raw_heat"]
    st.dataframe(dfv.sort_values("raw_heat", ascending=False).head(topn)[cols], use_container_width=True)

with c2:
    st.subheader("🧭 Truth Filter (Full)")
    cols = ["name","symbol","price_disp","mc_disp","vol_disp","chg24h_disp","chg7d_disp",
            "dev01","yt_view01","yt_sent01","tw_per15m","tx_24h","entropy01","truth_full","divergence"]
    st.dataframe(dfv.sort_values("truth_full", ascending=False).head(topn)[cols], use_container_width=True)

with c3:
    st.subheader("📉 Top Daily Gainers / Losers")
    g = dfv.sort_values("chg_24h", ascending=False).head(topn)[["name","symbol","price_disp","chg24h_disp"]]
    l = dfv.sort_values("chg_24h", ascending=True).head(topn)[["name","symbol","price_disp","chg24h_disp"]]
    st.markdown("**Top Gainers (24h)**"); st.dataframe(g, use_container_width=True)
    st.markdown("**Top Losers (24h)**");  st.dataframe(l, use_container_width=True)

st.markdown("---")
st.subheader("All Coins (quick view)")
show_cols = ["name","symbol","price_disp","mc_disp","vol_disp","chg1h_disp","chg24h_disp","chg7d_disp",
             "vol_to_mc","liquidity01","dev01","yt_view01","yt_sent01","tw_per15m","tx_24h",
             "entropy01","truth_full","raw_heat","divergence"]
st.dataframe(dfv.sort_values("market_cap", ascending=False)[show_cols], use_container_width=True)

# Auto-refresh ~60s
time.sleep(60)
st.experimental_rerun()
