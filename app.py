# ====================== IMPORTS ======================
import math, time, random, re, requests, pandas as pd, numpy as np, streamlit as st
from datetime import datetime, timezone
from io import StringIO

# Plotly (safe import)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# Sentiment
from textblob import TextBlob

# ====================== CONFIG ======================
APP_NAME = "Crypto Hybrid Live — Phase 4 (LIPE: Multi-Source Fusion)"
BRAND_FOOTER = "Data: CoinGecko • Reddit • DeFi Llama • (opt) CryptoPanic • Educational only — not financial advice."
CG_PER_PAGE = 250
USER_AGENT = {"User-Agent": "Mozilla/5.0 (LIPE-Dashboard)"}

FEATURES = {
    "SPARKLINES": True,          # CoinGecko 7d sparkline
    "DEV_PULSE": True,           # CoinGecko developer_data (per-coin)
    "ALERTS": True,
    "SNAPSHOT": True,
    "URL_STATE": True,
    "WEIGHT_PRESETS": True,
    "SOCIAL_CARD": True,
    # NEW external sources:
    "REDDIT": True,              # r/CryptoCurrency hot posts (no key)
    "CRYPTOPANIC": True,         # headlines (needs key, but handled as optional)
    "DEFI_LLAMA": True           # protocol/chain TVL
}

st.set_page_config(page_title=APP_NAME, layout="wide")
st.title("🟢 " + APP_NAME)
st.caption("Truth > Noise • LIPE learns your preferences • Now fused with News, Social Buzz, and DeFi TVL (graceful fallbacks).")

# ---- CSS ----
st.markdown("""
<style>
html, body, [class*="css"] { font-size: 18px; line-height: 1.45; }
h1, h2, h3 { font-weight: 700; }
.stButton>button { padding: 0.6rem 1.0rem; font-size: 16px; }
[data-testid="stTable"] td, [data-testid="stTable"] th { font-size: 16px !important; padding: 10px 8px !important; }
.card {border:1px solid #2a2f3a; border-radius:10px; padding:14px; margin-bottom:10px;}
.badge {padding:3px 8px; border-radius:6px; background:#223; font-size:12px;}
.kpill {display:inline-block; padding:2px 8px; margin-right:6px; border-radius:999px; background:#17202a; font-size:12px;}
</style>
""", unsafe_allow_html=True)

# ====================== LIPE CORE ======================
DEFAULT_WEIGHTS = dict(w_vol=0.30, w_m24=0.25, w_m7=0.25, w_liq=0.20)
FUSION_WEIGHTS = dict(w_truth=0.60, w_news=0.15, w_social=0.15, w_tvl=0.10)  # used to compute Fusion Truth

def _normalize_weights(w):
    s = float(sum(max(0.0, v) for v in w.values())) or 1.0
    return {k: max(0.0, v)/s for k, v in w.items()}

def lipe_online_weight_update(user_w, feedback, lr=0.05):
    w = dict(user_w) if user_w else dict(DEFAULT_WEIGHTS)
    w['w_m24'] = w.get('w_m24', 0.25) + lr * float(feedback.get('momo24', 0))
    w['w_m7']  = w.get('w_m7',  0.25) + lr * float(feedback.get('momo7',  0))
    w['w_vol'] = w.get('w_vol', 0.30) + lr * float(feedback.get('vol',    0))
    w['w_liq'] = w.get('w_liq', 0.20) + lr * float(feedback.get('liq',    0))
    return _normalize_weights(w)

def lipe_explain_truth_row(r):
    bits = []
    p24 = r.get("price_change_percentage_24h_in_currency")
    p7  = r.get("price_change_percentage_7d_in_currency")
    if pd.notna(p24): bits.append(f"24h {p24:+.1f}%")
    if pd.notna(p7):  bits.append(f"7d {p7:+.1f}%")
    liq = r.get("liquidity01", 0)
    mood = r.get("mood", "")
    return f"Momentum: {', '.join(bits) if bits else 'n/a'} • Liquidity: {liq:.2f} • Mood: {mood}"

def lipe_score_truth(df, w):
    w = _normalize_weights(w or DEFAULT_WEIGHTS)
    truth = (
        w['w_vol'] * (df["vol_to_mc"]/2).clip(0,1).fillna(0.0) +
        w['w_m24'] * df["momo_24h01"].fillna(0.5) +
        w['w_m7']  * df["momo_7d01"].fillna(0.5) +
        w['w_liq'] * df["liquidity01"].fillna(0.0)
    )
    return truth.clip(0,1)

def mood_label(truth):
    t = float(truth) if pd.notna(truth) else 0.5
    if t >= 0.80: return "🟢 EUPHORIC"
    if t >= 0.60: return "🟡 OPTIMISTIC"
    if t >= 0.40: return "🟠 NEUTRAL"
    return "🔴 FEARFUL"

def pct_sigmoid(pct):
    if pd.isna(pct): return 0.5
    x = float(pct)/10.0
    return 1/(1+math.exp(-x))

def entropy01_from_changes(p1h, p24h, p7d):
    arr = np.array([p for p in [p1h, p24h, p7d] if pd.notna(p)], dtype=float)
    if arr.size < 2: return 0.5
    s = np.std(arr)
    chaos01 = min(max(s/20.0, 0.0), 1.0)
    return float(1.0 - chaos01)

def predictive_bias_label(m1h, m24, m7):
    m1h = 0.0 if pd.isna(m1h) else m1h
    m24 = 0.0 if pd.isna(m24) else m24
    m7  = 0.0 if pd.isna(m7)  else m7
    if m1h > 0 and m24 > 0 and m7 > 0:   return "🟢 Likely Up"
    if m1h < 0 and m24 < 0:              return "🔴 Cooling"
    if m24 > 5 and m7 > 10:              return "🟠 Overheated"
    return "⚪ Mixed"

def get_profile():
    if "_lipe_profile" not in st.session_state:
        st.session_state["_lipe_profile"] = {
            "weights": dict(DEFAULT_WEIGHTS),
            "watchlist": [],
            "mode": "simple",
            "tier": "free",
            "public_votes": {}
        }
    return st.session_state["_lipe_profile"]

def save_profile(p): st.session_state["_lipe_profile"] = p

def toggle_watch(symbol):
    p = get_profile()
    wl = set(p.get("watchlist", []))
    s = str(symbol).upper()
    wl.remove(s) if s in wl else wl.add(s)
    p["watchlist"] = sorted(list(wl))
    save_profile(p)

def get_params(): return st.query_params if FEATURES["URL_STATE"] else {}
def set_params(**kwargs):
    if FEATURES["URL_STATE"]:
        st.query_params.clear()
        for k, v in kwargs.items():
            if v is not None:
                st.query_params[k] = v

def safe_get(url, params=None, timeout=30, retries=3, backoff=0.6, headers=None):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=headers or USER_AGENT)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(backoff * (2 ** i) + random.uniform(0, 0.2))
    return None

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Controls")
    vs_currency = st.selectbox("Currency", ["usd"], index=0)
    topn = st.slider("Show top N by market cap", 20, 250, 100, step=10)
    search = st.text_input("🔎 Search coin (name or symbol)").strip().lower()

    st.markdown("---")
    simple_mode = st.toggle("🧸 Easy Mode (hide pro columns)", value=True)

    st.markdown("---")
    st.subheader("Truth Weight Presets")
    preset = st.selectbox("Preset", ["Balanced (default)", "Momentum", "Liquidity", "Value"], index=0)
    st.caption({
        "Balanced (default)":"Even mix of momentum + liquidity + stability.",
        "Momentum":"Chases recent winners harder (fast, risky).",
        "Liquidity":"Prefers bigger, deeper markets.",
        "Value":"Leans to under-loved coins with decent base."
    }[preset])

    st.markdown("---")
    st.subheader("Alerts (on-screen)")
    alert_truth = st.slider("Trigger: Truth ≥", 0.0, 1.0, 0.85, 0.01)
    alert_diverg = st.slider("Trigger: |Divergence| ≥", 0.0, 1.0, 0.30, 0.01)

    st.markdown("---")
    st.markdown("### 📧 Get the Daily Truth Brief")
    email = st.text_input("Your email")
    if st.button("Subscribe"):
        st.success("You’re on the list! (Wire to Beehiiv/Substack/Sheets later.)")

# ====================== DATA: COINGECKO ======================
@st.cache_data(ttl=60)
def fetch_markets(vs="usd", per_page=CG_PER_PAGE, spark=True):
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
    st.error("Could not load data from CoinGecko. Please refresh in a minute.")
    st.stop()

df = df.sort_values("market_cap", ascending=False).head(topn).copy()
need = [
    "id","name","symbol","current_price","market_cap","total_volume",
    "price_change_percentage_1h_in_currency",
    "price_change_percentage_24h_in_currency",
    "price_change_percentage_7d_in_currency"
]
for k in need:
    if k not in df.columns: df[k] = np.nan

# ====================== FEATURES (Market) ======================
df["vol_to_mc"] = (df["total_volume"]/df["market_cap"]).replace([np.inf,-np.inf],np.nan).clip(0,2).fillna(0)
df["momo_1h01"]  = df["price_change_percentage_1h_in_currency"].apply(pct_sigmoid)
df["momo_24h01"] = df["price_change_percentage_24h_in_currency"].apply(pct_sigmoid)
df["momo_7d01"]  = df["price_change_percentage_7d_in_currency"].apply(pct_sigmoid)
mc = df["market_cap"].fillna(0)
df["liquidity01"] = 0 if mc.max()==0 else (mc - mc.min())/(mc.max() - mc.min() + 1e-9)

# Raw & Truth
user_w = get_profile().get("weights", DEFAULT_WEIGHTS)
preset_map = {
    "Momentum": dict(w_vol=0.15, w_m24=0.45, w_m7=0.30, w_liq=0.10),
    "Liquidity":dict(w_vol=0.20, w_m24=0.20, w_m7=0.20, w_liq=0.40),
    "Value":    dict(w_vol=0.40, w_m24=0.20, w_m7=0.20, w_liq=0.20),
    "Balanced (default)": DEFAULT_WEIGHTS
}
base_w = preset_map.get(preset, DEFAULT_WEIGHTS)
user_w = {k: 0.7*base_w.get(k,0) + 0.3*user_w.get(k,0) for k in DEFAULT_WEIGHTS.keys()}
df["raw_heat"]   = (0.5*(df["vol_to_mc"]/2).clip(0,1) + 0.5*df["momo_1h01"].fillna(0.5)).clip(0,1)
df["truth_full"] = lipe_score_truth(df, user_w)
df["divergence"] = (df["raw_heat"] - df["truth_full"]).round(3)
df["entropy01"]  = df.apply(lambda r: entropy01_from_changes(
    r.get("price_change_percentage_1h_in_currency"),
    r.get("price_change_percentage_24h_in_currency"),
    r.get("price_change_percentage_7d_in_currency")), axis=1)
df["bias_24h"]   = df.apply(lambda r: predictive_bias_label(
    r.get("price_change_percentage_1h_in_currency"),
    r.get("price_change_percentage_24h_in_currency"),
    r.get("price_change_percentage_7d_in_currency")), axis=1)
df["mood"] = df["truth_full"].apply(mood_label)

# ====================== EXTERNAL: REDDIT (no key) ======================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_reddit_hot(limit=80):
    if not FEATURES["REDDIT"]:
        return []
    url = f"https://www.reddit.com/r/CryptoCurrency/hot.json?limit={limit}"
    r = safe_get(url, timeout=20)
    if not r: return []
    try:
        js = r.json()
        posts = js.get("data", {}).get("children", [])
        titles = [p["data"].get("title","") for p in posts]
        return titles
    except Exception:
        return []

@st.cache_data(ttl=120, show_spinner=False)
def social_buzz_scores(symbols):
    """Simple buzz & sentiment: count mentions in Reddit titles + TextBlob polarity."""
    titles = fetch_reddit_hot()
    if not titles: 
        return pd.DataFrame(columns=["symbol","buzz","sentiment","social01"])
    rows = []
    for sym in symbols:
        pat = re.compile(rf"\b{re.escape(sym.upper())}\b")
        hits = [t for t in titles if pat.search(t.upper())]
        buzz = len(hits)
        if buzz==0:
            rows.append({"symbol":sym.upper(), "buzz":0, "sentiment":0.0})
            continue
        pol = float(np.mean([TextBlob(t).sentiment.polarity for t in hits]))
        rows.append({"symbol":sym.upper(), "buzz":buzz, "sentiment":pol})
    out = pd.DataFrame(rows)
    if out.empty: 
        out["social01"] = []
        return out
    # normalize buzz (log) & sentiment to 0..1, combine 70/30
    out["buzz01"] = (np.log1p(out["buzz"]) / (np.log1p(out["buzz"]).max() or 1)).fillna(0.0)
    sent = out["sentiment"].clip(-1,1)
    out["sent01"] = (sent + 1)/2.0
    out["social01"] = (0.7*out["buzz01"] + 0.3*out["sent01"]).clip(0,1)
    return out[["symbol","buzz","sentiment","social01"]]

social = social_buzz_scores(df["symbol"].str.upper().tolist())

# ====================== EXTERNAL: CRYPTOPANIC (optional key) ======================
CP_KEY = st.secrets.get("CRYPTOPANIC_KEY") if hasattr(st, "secrets") else None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_cryptopanic_headlines(page_size=50):
    if not FEATURES["CRYPTOPANIC"]: return []
    if not CP_KEY: return []  # no key, skip quietly
    url = "https://cryptopanic.com/api/v1/posts/"
    p = {"auth_token": CP_KEY, "public": "true", "kind": "news", "filter": "hot", "currencies": "BTC,ETH,SOL,BNB,XRP,ADA,AVAX,DOGE,INJ,MATIC"}
    r = safe_get(url, params=p, timeout=20)
    if not r: return []
    try:
        js = r.json()
        results = js.get("results", [])
        titles = [x.get("title","") for x in results]
        return titles
    except Exception:
        return []

@st.cache_data(ttl=120, show_spinner=False)
def news_heat_scores(symbols):
    """Headline heat via CryptoPanic (if key); uses TextBlob sentiment; counts mentions."""
    titles = fetch_cryptopanic_headlines()
    if not titles: 
        return pd.DataFrame(columns=["symbol","news_hits","news_sent","news01"])
    rows = []
    for sym in symbols:
        pat = re.compile(rf"\b{re.escape(sym.upper())}\b")
        hits = [t for t in titles if pat.search(t.upper())]
        n = len(hits)
        if n==0:
            rows.append({"symbol":sym.upper(), "news_hits":0, "news_sent":0.0})
            continue
        pol = float(np.mean([TextBlob(t).sentiment.polarity for t in hits]))
        rows.append({"symbol":sym.upper(), "news_hits":n, "news_sent":pol})
    out = pd.DataFrame(rows)
    if out.empty: 
        out["news01"] = []
        return out
    out["hit01"] = (np.log1p(out["news_hits"]) / (np.log1p(out["news_hits"]).max() or 1)).fillna(0.0)
    out["sent01"] = ((out["news_sent"].clip(-1,1) + 1)/2.0).fillna(0.5)
    out["news01"] = (0.7*out["hit01"] + 0.3*out["sent01"]).clip(0,1)
    return out[["symbol","news_hits","news_sent","news01"]]

news = news_heat_scores(df["symbol"].str.upper().tolist())

# ====================== EXTERNAL: DEFI LLAMA (TVL) ======================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_defi_protocols():
    if not FEATURES["DEFI_LLAMA"]: return pd.DataFrame([])
    url = "https://api.llama.fi/protocols"
    r = safe_get(url, timeout=30)
    if not r: return pd.DataFrame([])
    try:
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame([])

defi = fetch_defi_protocols()
# Map coin symbol to a rough TVL score by fuzzy symbol match in protocols' "symbol" or name
def map_tvl_symbols(df_coins, df_llama):
    if df_llama.empty: 
        return pd.DataFrame(columns=["symbol","tvl_score01"])
    rows = []
    coins = df_coins["symbol"].str.upper().tolist()
    # build small index
    llama_rows = df_llama[["name","symbol","tvl"]].fillna({"symbol":""})
    for s in coins:
        # best-effort match: exact symbol match in protocols OR name contains coin name
        sub = llama_rows[(llama_rows["symbol"].str.upper()==s)]
        tvl = float(sub["tvl"].sum()) if len(sub) else 0.0
        rows.append({"symbol": s, "tvl_raw": tvl})
    out = pd.DataFrame(rows)
    if out["tvl_raw"].max()<=0:
        out["tvl_score01"] = 0.0
    else:
        out["tvl_score01"] = (np.log1p(out["tvl_raw"]) / np.log1p(out["tvl_raw"].max())).clip(0,1)
    return out[["symbol","tvl_score01"]]

tvlmap = map_tvl_symbols(df, defi)

# ====================== FUSION TRUTH ======================
# Merge external signals by symbol
df["SYMBOL_UP"] = df["symbol"].str.upper()
m = df.merge(social, left_on="SYMBOL_UP", right_on="symbol", how="left", suffixes=("","_soc"))
m = m.merge(news, left_on="SYMBOL_UP", right_on="symbol", how="left", suffixes=("","_news"))
m = m.merge(tvlmap, left_on="SYMBOL_UP", right_on="symbol", how="left", suffixes=("","_tvl"))

m["social01"] = m["social01"].fillna(0.0)
m["news01"]   = m["news01"].fillna(0.0)
m["tvl_score01"] = m["tvl_score01"].fillna(0.0)

FW = _normalize_weights(FUSION_WEIGHTS)
m["fusion_truth"] = (
    FW["w_truth"]  * m["truth_full"].fillna(0.0) +
    FW["w_news"]   * m["news01"] +
    FW["w_social"] * m["social01"] +
    FW["w_tvl"]    * m["tvl_score01"]
).clip(0,1)

# Keep back to df with new columns
keep_cols = list(df.columns) + ["social01","news01","tvl_score01","fusion_truth"]
df = m[keep_cols].copy()
df["mood_fusion"] = df["fusion_truth"].apply(mood_label)

# ====================== KPIs ======================
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
avg_truth   = df["truth_full"].mean()
avg_fusion  = df["fusion_truth"].mean()
avg_entropy = df["entropy01"].mean()
avg_24h     = df["price_change_percentage_24h_in_currency"].mean()

cA,cB,cC,cD,cE = st.columns(5)
with cA: st.metric("Coins", len(df))
with cB: st.metric("Avg 24h Δ", f"{avg_24h:+.2f}%")
with cC: st.metric("Avg Truth", f"{avg_truth:.2f}")
with cD: st.metric("Avg Fusion Truth", f"{avg_fusion:.2f}")
with cE: st.metric("Last update (UTC)", now)

st.markdown(
    f"<span class='kpill'>Pulse</span> Truth <b>{avg_truth:.2f}</b> • Fusion <b>{avg_fusion:.2f}</b> • "
    f"Entropy <b>{avg_entropy:.2f}</b> • 24h <b>{avg_24h:+.2f}%</b>",
    unsafe_allow_html=True
)

# ====================== Daily Brief ======================
def market_story():
    base_tone = "optimistic" if avg_truth>=0.6 else ("neutral" if avg_truth>=0.45 else "cautious")
    fusion_tone = "bullish" if avg_fusion>=0.6 else ("balanced" if avg_fusion>=0.45 else "wary")
    chaos = "calm" if avg_entropy>=0.6 else ("mixed" if avg_entropy>=0.4 else "chaotic")
    return (
        f"**Market mood:** {base_tone} / Fusion says {fusion_tone}. "
        f"Environment is {chaos}. Best setups are **High Fusion + Low Entropy**."
    )

with st.expander("🧠 Daily Truth Brief", expanded=True):
    st.write(market_story())

# ====================== Teach LIPE ======================
st.markdown("### 🧠 Teach LIPE (what you value)")
c1,c2,c3,c4 = st.columns(4)
profile = get_profile()
if c1.button("❤️ Momentum 24h"): profile["weights"] = lipe_online_weight_update(profile["weights"], {"momo24":+1}); save_profile(profile); st.rerun()
if c2.button("💚 Momentum 7d"):  profile["weights"] = lipe_online_weight_update(profile["weights"], {"momo7":+1});  save_profile(profile); st.rerun()
if c3.button("💙 Liquidity"):     profile["weights"] = lipe_online_weight_update(profile["weights"], {"liq":+1});     save_profile(profile); st.rerun()
if c4.button("🧡 Volume/MC"):    profile["weights"] = lipe_online_weight_update(profile["weights"], {"vol":+1});     save_profile(profile); st.rerun()
st.caption(f"Your LIPE weights → {profile['weights']}  • Fusion blend: {FUSION_WEIGHTS}")

# ====================== Search / Watchlist / Focus ======================
if search:
    mask = df["name"].str.lower().str.contains(search) | df["symbol"].str.lower().str.contains(search)
    df = df[mask].copy()

left, right = st.columns([0.58, 0.42])
with left:
    st.subheader("⭐ Watchlist")
    add_to_watch = st.selectbox("Add/remove coin by symbol", ["(choose)"] + sorted(df["symbol"].str.upper().unique().tolist()))
    if add_to_watch != "(choose)":
        toggle_watch(add_to_watch)
        st.success(f"Toggled {add_to_watch} in your watchlist.")
    wl = get_profile().get("watchlist", [])
    view_cols_simple = ["name","symbol","current_price","fusion_truth","truth_full","mood_fusion"]
    view_cols_pro    = ["name","symbol","current_price","market_cap","liquidity01","truth_full","fusion_truth","divergence","social01","news01","tvl_score01","entropy01","bias_24h","mood_fusion"]
    if wl:
        wldf = df[df["symbol"].str.upper().isin(wl)].sort_values("fusion_truth", ascending=False)
        st.dataframe(wldf[view_cols_simple if simple_mode else view_cols_pro], use_container_width=True)
    else:
        st.info("Your watchlist is empty. Add symbols above.")

with right:
    st.subheader("🎯 Focus coin")
    focus = st.selectbox("Pick a coin to inspect", ["(none)"] + df["name"].head(50).tolist())
    if focus != "(none)" and PLOTLY_OK:
        row = df[df["name"] == focus].head(1).to_dict("records")[0]
        st.success(
            f"**{focus}** → Fusion **{row['fusion_truth']:.2f}** ({row['mood_fusion']}) • "
            f"Truth {row['truth_full']:.2f} • 24h {row['price_change_percentage_24h_in_currency']:+.2f}% • "
            f"7d {row['price_change_percentage_7d_in_currency']:+.2f}% • Liquidity {row['liquidity01']:.2f}"
        )
        st.caption("Why: " + lipe_explain_truth_row(row))
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(row["fusion_truth"]),
            number={'valueformat': '.2f'},
            gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#23d18b"}}
        ))
        fig.update_layout(height=230, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ====================== Snapshot (CSV) ======================
def make_snapshot_csv(df_truth, df_fusion, df_raw):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_UTC")
    buf = StringIO()
    buf.write(f"Snapshot,{ts}\n\nTop by Fusion Truth\n")
    df_fusion.to_csv(buf, index=False)
    buf.write("\nTop by Truth (LIPE)\n")
    df_truth.to_csv(buf, index=False)
    buf.write("\nTop by Raw Heat\n")
    df_raw.to_csv(buf, index=False)
    return f"snapshot_{ts}.csv", buf.getvalue().encode("utf-8")

cols_simple_fusion = ["name","symbol","current_price","fusion_truth","mood_fusion"]
cols_pro_fusion    = ["name","symbol","current_price","market_cap","fusion_truth","truth_full","divergence","social01","news01","tvl_score01","entropy01","bias_24h","mood_fusion"]
truth_cols         = ["name","symbol","current_price","market_cap","liquidity01","truth_full","divergence","entropy01","bias_24h","mood_fusion"]
raw_cols           = ["name","symbol","current_price","market_cap","total_volume","raw_heat"]

top_fusion = df.sort_values("fusion_truth", ascending=False).head(25)[cols_pro_fusion if not simple_mode else cols_simple_fusion]
top_truth  = df.sort_values("truth_full",  ascending=False).head(25)[truth_cols]
top_raw    = df.sort_values("raw_heat",    ascending=False).head(25)[raw_cols]
fname, payload = make_snapshot_csv(top_truth, top_fusion, top_raw)
st.download_button("⬇️ Download Snapshot (Fusion + Truth + Raw)", payload, file_name=fname, mime="text/csv")

# ====================== TABS ======================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🧭 Fusion Truth", "🔥 Raw", "🧭 LIPE Truth", "📉 Movers", "📈 Mini Charts", "🗞️ News", "💬 Social", "🏦 TVL / DeFi"
])

with tab1:
    st.subheader("🧭 Fusion Truth (News + Social + TVL + LIPE)")
    st.dataframe(df.sort_values("fusion_truth", ascending=False)[top_fusion.columns].reset_index(drop=True), use_container_width=True)

with tab2:
    st.subheader("🔥 Raw Wide Scan")
    st.dataframe(df.sort_values("raw_heat", ascending=False)[raw_cols].reset_index(drop=True), use_container_width=True)

with tab3:
    st.subheader("🧭 LIPE Truth (weights applied)")
    st.dataframe(df.sort_values("truth_full", ascending=False)[truth_cols].reset_index(drop=True), use_container_width=True)

with tab4:
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

with tab5:
    st.subheader("📈 7-Day Mini Charts (Top 10 by Fusion Truth)")
    if not PLOTLY_OK or "sparkline_in_7d" not in df.columns:
        st.info("Mini charts need Plotly and sparkline data.")
    else:
        top10 = df.sort_values("fusion_truth", ascending=False).head(10)
        for _, r in top10.iterrows():
            prices = (r.get("sparkline_in_7d") or {}).get("price", [])
            if not prices: continue
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=prices, mode="lines", name=str(r["symbol"]).upper()))
            fig.update_layout(
                title=f"{r['name']} ({str(r['symbol']).upper()}) • Fusion {r['fusion_truth']:.2f}",
                height=220, margin=dict(l=10,r=10,t=30,b=10), showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("🗞️ Latest Headlines (CryptoPanic)")
    if CP_KEY:
        titles = fetch_cryptopanic_headlines()
        if titles:
            st.write(pd.DataFrame({"headline": titles}))
        else:
            st.info("No hot headlines returned right now.")
    else:
        st.info("Add CRYPTOPANIC_KEY in Streamlit secrets to enable headline scoring & display.")

with tab7:
    st.subheader("💬 Social Buzz (Reddit Hot)")
    titles = fetch_reddit_hot()
    if titles:
        st.write(pd.DataFrame({"reddit_title": titles}))
        st.caption("Buzz & sentiment are already fused into the Fusion Truth score.")
    else:
        st.info("Reddit titles not available right now.")

with tab8:
    st.subheader("🏦 DeFi Llama — Protocols / TVL")
    if defi.empty:
        st.info("TVL data not available right now.")
    else:
        show = defi[["name","symbol","tvl","category","chains"]].sort_values("tvl", ascending=False).head(50)
        st.dataframe(show, use_container_width=True)
        st.caption("We map protocol symbols to coin symbols best-effort to create tvl_score01 → Fusion Truth.")

# ====================== Alerts ======================
if FEATURES["ALERTS"]:
    matches = df[(df["fusion_truth"] >= alert_truth) | (df["divergence"].abs() >= alert_diverg)]
    if len(matches):
        st.warning(f"🚨 {len(matches)} coins matched your alert rules (Fusion / Divergence)")
        st.dataframe(matches.sort_values("fusion_truth", ascending=False)[
            ["name","symbol","fusion_truth","truth_full","divergence","mood_fusion"]
        ], use_container_width=True)

# ====================== FOOTER ======================
st.markdown("""<hr style="margin-top: 1rem; margin-bottom: 0.5rem;">""", unsafe_allow_html=True)
api_status = "🟢 APIs OK" if not df.empty else "🔴 API issue"
st.caption(f"{api_status} • {BRAND_FOOTER}")
