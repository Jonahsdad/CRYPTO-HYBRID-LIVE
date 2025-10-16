
# ====================== IMPORTS ======================
import math, time, random, requests, pandas as pd, numpy as np, streamlit as st
from datetime import datetime, timezone
from io import StringIO

# Plotly (safe import)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ====================== LIPE (inline, 1-file drop-in) ======================
DEFAULT_WEIGHTS = dict(w_vol=0.30, w_m24=0.25, w_m7=0.25, w_liq=0.20)

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

def get_profile():
    if "_lipe_profile" not in st.session_state:
        st.session_state["_lipe_profile"] = {
            "weights": dict(DEFAULT_WEIGHTS),
            "watchlist": [],
            "mode": "simple",
            "tier": "free",
            "public_votes": {}  # symbol -> +1 / -1
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

# ====================== APP CONFIG / FLAGS ======================
APP_NAME = "Crypto Hybrid Live — Phase 3 (LIPE)"
BRAND_FOOTER = "Data: CoinGecko • Educational analytics — not financial advice. © Crypto Hybrid Live"
FEATURES = {
    "DEV_PULSE": True,
    "ENTROPY_BIAS": True,
    "ALERTS": True,
    "SNAPSHOT": True,
    "URL_STATE": True,         # st.query_params
    "WEIGHT_PRESETS": True,
    "SPARKLINES": True,        # 7d mini-charts
    "SOCIAL_CARD": True,       # shareable summary card (PNG fallback-friendly)
    "PUBLIC_VOTES": True       # simple up/down vote per coin (session-only demo)
}

st.set_page_config(page_title=APP_NAME, layout="wide")
st.title("🟢 " + APP_NAME)
st.caption("Truth > Noise • Personalized by LIPE • Easy Mode + Pro Tools • Heatmaps • Alerts • Snapshots • Dev Pulse • Viral Share")

# ---- CSS: bigger fonts, clean cards, badges ----
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

# ====================== SIDEBAR CONTROLS ======================
with st.sidebar:
    st.header("⚙️ Controls")
    vs_currency = st.selectbox("Currency", ["usd"], index=0)
    topn = st.slider("Show top N by market cap", 20, 250, 100, step=10)
    search = st.text_input("🔎 Search coin (name or symbol)").strip().lower()

    st.markdown("---")
    simple_mode = st.toggle("🧸 Easy Mode (hide pro columns)", value=True)

    st.markdown("---")
    if FEATURES["WEIGHT_PRESETS"]:
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
    st.markdown("— Or —")
    st.link_button("Join our Discord", "https://discord.gg/your-invite")
    st.link_button("Follow on X", "https://twitter.com/yourhandle")

# ====================== HELPERS ======================
def pct_sigmoid(pct):
    if pd.isna(pct): return 0.5
    x = float(pct)/10.0
    return 1/(1+math.exp(-x))

def mood_label(truth):
    t = float(truth) if pd.notna(truth) else 0.5
    if t >= 0.80: return "🟢 EUPHORIC"
    if t >= 0.60: return "🟡 OPTIMISTIC"
    if t >= 0.40: return "🟠 NEUTRAL"
    return "🔴 FEARFUL"

def entropy01_from_changes(p1h, p24h, p7d):
    arr = np.array([p for p in [p1h, p24h, p7d] if pd.notna(p)], dtype=float)
    if arr.size < 2:
        return 0.5
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

def get_params(): return st.query_params if FEATURES["URL_STATE"] else {}
def set_params(**kwargs):
    if FEATURES["URL_STATE"]:
        st.query_params.clear()
        for k, v in kwargs.items():
            if v is not None:
                st.query_params[k] = v

def safe_get(url, params=None, timeout=30, retries=3, backoff=0.6):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(backoff * (2 ** i) + random.uniform(0, 0.2))
    return None

# ====================== DATA ======================
@st.cache_data(ttl=60)
def fetch_markets(vs="usd", per_page=250, spark=True):
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

@st.cache_data(ttl=300, show_spinner=False)
def fetch_dev_pulse(ids):
    rows = []
    for cid in ids:
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{cid}"
            p = {
                "localization":"false","tickers":"false",
                "market_data":"false","community_data":"false",
                "developer_data":"true","sparkline":"false"
            }
            r = safe_get(url, params=p, timeout=25)
            if not r or r.status_code != 200: continue
            js = r.json()
            dev = js.get("developer_data", {}) or {}
            rows.append({
                "id": cid,
                "dev_stars": dev.get("stars", 0) or 0,
                "dev_forks": dev.get("forks", 0) or 0,
                "dev_subs": dev.get("subscribers", 0) or 0,
                "dev_pr_merged": dev.get("pull_requests_merged", 0) or 0,
                "dev_commit_4w": dev.get("commit_count_4_weeks", 0) or 0
            })
            time.sleep(0.25)
        except Exception:
            continue
    dfp = pd.DataFrame(rows)
    if dfp.empty: return dfp
    for c in ["dev_stars","dev_forks","dev_subs","dev_pr_merged","dev_commit_4w"]:
        m, M = dfp[c].min(), dfp[c].max()
        dfp[c+"_01"] = 0.0 if M <= m else (dfp[c]-m)/(M-m)
    dfp["dev_pulse01"] = (
        0.25*dfp["dev_stars_01"] + 0.20*dfp["dev_forks_01"] +
        0.15*dfp["dev_subs_01"] + 0.20*dfp["dev_pr_merged_01"] +
        0.20*dfp["dev_commit_4w_01"]
    ).clip(0,1)
    return dfp[["id","dev_pulse01","dev_stars","dev_forks","dev_subs","dev_pr_merged","dev_commit_4w"]]

# ====================== LOAD & FEATURES ======================
params = get_params()
profile = get_profile()

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

# Feature engineering
df["vol_to_mc"] = (df["total_volume"]/df["market_cap"]).replace([np.inf,-np.inf],np.nan).clip(0,2).fillna(0)
df["momo_1h01"]  = df["price_change_percentage_1h_in_currency"].apply(pct_sigmoid)
df["momo_24h01"] = df["price_change_percentage_24h_in_currency"].apply(pct_sigmoid)
df["momo_7d01"]  = df["price_change_percentage_7d_in_currency"].apply(pct_sigmoid)
mc = df["market_cap"].fillna(0)
df["liquidity01"] = 0 if mc.max()==0 else (mc - mc.min())/(mc.max() - mc.min() + 1e-9)

# Raw scan & LIPE Truth
df["raw_heat"] = (0.5*(df["vol_to_mc"]/2).clip(0,1) + 0.5*df["momo_1h01"].fillna(0.5)).clip(0,1)
user_w = profile.get("weights", DEFAULT_WEIGHTS)

# Apply preset (blend with learned weights)
preset_map = {
    "Momentum": dict(w_vol=0.15, w_m24=0.45, w_m7=0.30, w_liq=0.10),
    "Liquidity":dict(w_vol=0.20, w_m24=0.20, w_m7=0.20, w_liq=0.40),
    "Value":    dict(w_vol=0.40, w_m24=0.20, w_m7=0.20, w_liq=0.20),
    "Balanced (default)": DEFAULT_WEIGHTS
}
base_w = preset_map.get(preset, DEFAULT_WEIGHTS)
user_w = {k: 0.7*base_w.get(k,0) + 0.3*user_w.get(k,0) for k in DEFAULT_WEIGHTS.keys()}

df["truth_full"] = lipe_score_truth(df, user_w)
df["divergence"] = (df["raw_heat"] - df["truth_full"]).round(3)
df["mood"] = df["truth_full"].apply(mood_label)

# Entropy & bias
df["entropy01"] = df.apply(
    lambda r: entropy01_from_changes(
        r.get("price_change_percentage_1h_in_currency"),
        r.get("price_change_percentage_24h_in_currency"),
        r.get("price_change_percentage_7d_in_currency")
    ), axis=1
)
df["bias_24h"] = df.apply(
    lambda r: predictive_bias_label(
        r.get("price_change_percentage_1h_in_currency"),
        r.get("price_change_percentage_24h_in_currency"),
        r.get("price_change_percentage_7d_in_currency")
    ), axis=1
)

# Search & URL share
set_params(q=search or None, n=str(topn))
if search:
    mask = df["name"].str.lower().str.contains(search) | df["symbol"].str.lower().str.contains(search)
    df = df[mask].copy()

# ====================== KPIs / PULSE / STORY ======================
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
avg_truth   = df["truth_full"].mean()
avg_entropy = df["entropy01"].mean()
avg_24h     = df["price_change_percentage_24h_in_currency"].mean()

cA,cB,cC,cD = st.columns(4)
with cA: st.metric("Coins", len(df))
with cB: st.metric("Avg 24h Δ", f"{avg_24h:+.2f}%")
with cC: st.metric("Avg Truth", f"{avg_truth:.2f}")
with cD: st.metric("Last update (UTC)", now)

st.markdown(
    f"<span class='kpill'>Market Pulse</span> Truth <b>{avg_truth:.2f}</b> • "
    f"Entropy <b>{avg_entropy:.2f}</b> • 24h <b>{avg_24h:+.2f}%</b>",
    unsafe_allow_html=True
)

def market_story():
    tone = "optimistic" if avg_truth>=0.6 else ("neutral" if avg_truth>=0.45 else "cautious")
    chaos = "calm" if avg_entropy>=0.6 else ("mixed" if avg_entropy>=0.4 else "chaotic")
    return (
        f"**Today’s market mood:** {tone} and {chaos}. "
        f"Average 24h move is {avg_24h:+.2f}%. "
        f"Look for **High Truth + Low Entropy** as best setups."
    )

with st.expander("🧠 Daily Truth Brief", expanded=True):
    st.write(market_story())

# ====================== TEACH LIPE ======================
st.markdown("### 🧠 Teach LIPE (what you value)")
c1,c2,c3,c4 = st.columns(4)
if c1.button("❤️ Momentum 24h"): profile["weights"] = lipe_online_weight_update(profile["weights"], {"momo24":+1}); save_profile(profile); st.rerun()
if c2.button("💚 Momentum 7d"):  profile["weights"] = lipe_online_weight_update(profile["weights"], {"momo7":+1});  save_profile(profile); st.rerun()
if c3.button("💙 Liquidity"):     profile["weights"] = lipe_online_weight_update(profile["weights"], {"liq":+1});     save_profile(profile); st.rerun()
if c4.button("🧡 Volume/MC"):    profile["weights"] = lipe_online_weight_update(profile["weights"], {"vol":+1});     save_profile(profile); st.rerun()
st.caption(f"Your LIPE weights → {profile['weights']}")

# ====================== WATCHLIST & FOCUS ======================
left, right = st.columns([0.55, 0.45])
with left:
    st.subheader("⭐ Watchlist")
    add_to_watch = st.selectbox("Add/remove coin by symbol", ["(choose)"] + sorted(df["symbol"].str.upper().unique().tolist()))
    if add_to_watch != "(choose)":
        toggle_watch(add_to_watch)
        st.success(f"Toggled {add_to_watch} in your watchlist.")
    wl = get_profile().get("watchlist", [])
    if wl:
        wldf = df[df["symbol"].str.upper().isin(wl)].sort_values("truth_full", ascending=False)
        st.dataframe(
            (wldf[["name","symbol","current_price","truth_full","divergence","mood"]]
             if simple_mode else
             wldf[["name","symbol","current_price","market_cap","liquidity01","truth_full","divergence","entropy01","bias_24h","mood"]]
            ),
            use_container_width=True
        )
    else:
        st.info("Your watchlist is empty. Add symbols above.")

with right:
    st.subheader("🎯 Focus coin")
    focus = st.selectbox("Pick a coin to inspect", ["(none)"] + df["name"].head(50).tolist())
    if focus != "(none)" and PLOTLY_OK:
        row = df[df["name"] == focus].head(1).to_dict("records")[0]
        st.success(
            f"**{focus}** → Truth **{row['truth_full']:.2f}** ({row['mood']}) • "
            f"24h {row['price_change_percentage_24h_in_currency']:+.2f}% • "
            f"7d {row['price_change_percentage_7d_in_currency']:+.2f}% • "
            f"Liquidity {row['liquidity01']:.2f}"
        )
        st.caption("Why: " + lipe_explain_truth_row(row))
        # Truth Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(row["truth_full"]),
            number={'valueformat': '.2f'},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "#23d18b"},
                'steps': [
                    {'range': [0.0, 0.4], 'color': "#4b0000"},
                    {'range': [0.4, 0.6], 'color': "#3a2c00"},
                    {'range': [0.6, 0.8], 'color': "#1e3a00"},
                    {'range': [0.8, 1.0], 'color': "#0d3b2a"},
                ],
            }
        ))
        fig.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ====================== SNAPSHOT (CSV) ======================
def make_snapshot_csv(df_truth, df_raw):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_UTC")
    buf = StringIO()
    buf.write(f"Snapshot,{ts}\n\nTop by Truth\n")
    df_truth.to_csv(buf, index=False)
    buf.write("\nTop by Raw Heat\n")
    df_raw.to_csv(buf, index=False)
    return f"snapshot_{ts}.csv", buf.getvalue().encode("utf-8")

truth_cols_simple = ["name","symbol","current_price","truth_full","divergence","mood"]
truth_cols_pro    = ["name","symbol","current_price","market_cap","liquidity01","truth_full","divergence","entropy01","bias_24h","mood"]
raw_cols          = ["name","symbol","current_price","market_cap","total_volume","raw_heat"]

top_truth = df.sort_values("truth_full", ascending=False).head(25)[truth_cols_pro if not simple_mode else truth_cols_simple]
top_raw   = df.sort_values("raw_heat",   ascending=False).head(25)[raw_cols]
fname, payload = make_snapshot_csv(top_truth, top_raw)
st.download_button("⬇️ Download Snapshot (Truth + Raw CSV)", payload, file_name=fname, mime="text/csv")

# ====================== TABS ======================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔥 Raw (Just Buzz)", "🧭 Truth (Best Picks)", "📉 Movers (Up/Down)",
    "📈 Mini Charts", "🗺️ Heatmaps", "🧑‍💻 Dev Pulse", "📣 Social / Votes"
])

with tab1:
    st.subheader("🔥 Raw Wide Scan")
    cols = ["name","symbol","current_price","market_cap","total_volume","raw_heat"]
    st.dataframe(df.sort_values("raw_heat", ascending=False)[cols].reset_index(drop=True), use_container_width=True)
    st.download_button("⬇️ CSV (Raw Scan)", df[cols].to_csv(index=False).encode("utf-8"), "raw_scan.csv","text/csv")

with tab2:
    st.subheader("🧭 Truth Filter (weights applied)")
    cols = truth_cols_pro if not simple_mode else truth_cols_simple
    st.dataframe(
        df.sort_values("truth_full", ascending=False)[cols].reset_index(drop=True),
        use_container_width=True
    )
    st.download_button("⬇️ CSV (Truth Table)", df[cols].to_csv(index=False).encode("utf-8"), "truth_table.csv","text/csv")

with tab3:
    st.subheader("📉 Top Daily Gainers / Losers (24h)")
    g = df.sort_values("price_change_percentage_24h_in_currency", ascending=False).head(12).copy()
    l = df.sort_values("price_change_percentage_24h_in_currency", ascending=True).head(12).copy()
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Top Gainers**")
        st.dataframe(g[["name","symbol","current_price","price_change_percentage_24h_in_currency"]].reset_index(drop=True), use_container_width=True)
        st.download_button("⬇️ Gainers (CSV)", g.to_csv(index=False).encode("utf-8"), "gainers.csv","text/csv")
    with c2:
        st.write("**Top Losers**")
        st.dataframe(l[["name","symbol","current_price","price_change_percentage_24h_in_currency"]].reset_index(drop=True), use_container_width=True)
        st.download_button("⬇️ Losers (CSV)", l.to_csv(index=False).encode("utf-8"), "losers.csv","text/csv")

with tab4:
    st.subheader("📈 7-Day Mini Charts (Top 10 by Truth)")
    if not PLOTLY_OK or "sparkline_in_7d" not in df.columns:
        st.info("Mini charts need Plotly and sparkline data. Try again shortly.")
    else:
        top10 = df.sort_values("truth_full", ascending=False).head(10)
        for _, r in top10.iterrows():
            prices = (r.get("sparkline_in_7d") or {}).get("price", [])
            if not prices or not PLOTLY_OK: continue
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=prices, mode="lines", name=str(r["symbol"]).upper()))
            fig.update_layout(
                title=f"{r['name']} ({str(r['symbol']).upper()}) • Truth {r['truth_full']:.2f}",
                height=220, margin=dict(l=10,r=10,t=30,b=10), showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("🗺️ Heatmaps (Top 50 by Market Cap)")
    if not PLOTLY_OK:
        st.info("Plotly not available yet.")
    else:
        top50 = df.sort_values("market_cap", ascending=False).head(50)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Truth Heatmap**")
            try:
                fig = px.treemap(top50, path=[top50["symbol"].str.upper()], values="market_cap",
                                 color="truth_full", color_continuous_scale="Turbo")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info("Truth heatmap unavailable.")
                st.exception(e)
        with c2:
            st.markdown("**24h % Change Heatmap**")
            try:
                fig2 = px.treemap(top50, path=[top50["symbol"].str.upper()], values="market_cap",
                                  color="price_change_percentage_24h_in_currency",
                                  color_continuous_scale="Picnic")
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.info("24h change heatmap unavailable.")
                st.exception(e)

with tab6:
    st.subheader("🧑‍💻 Developer Pulse (Top 25 by Market Cap)")
    if FEATURES["DEV_PULSE"]:
        top_ids = df.sort_values("market_cap", ascending=False).head(25)["id"].tolist() if "id" in df.columns else []
        if not top_ids:
            st.info("Developer data requires CoinGecko coin IDs; not available in this dataset.")
        else:
            devdf = fetch_dev_pulse(top_ids)
            if devdf.empty:
                st.info("Developer data temporarily unavailable.")
            else:
                m = df.merge(devdf, on="id", how="left")
                cols = ["name","symbol","dev_pulse01","dev_commit_4w","dev_pr_merged","dev_stars","dev_forks","dev_subs"]
                if "name" not in m.columns: m["name"] = m["id"]
                if "symbol" not in m.columns: m["symbol"] = m["id"]
                m["dev_pulse01"] = m["dev_pulse01"].fillna(0.0)
                st.dataframe(
                    m.sort_values("dev_pulse01", ascending=False)[cols].reset_index(drop=True),
                    use_container_width=True
                )
                st.download_button("⬇️ Dev Pulse (CSV)", m[cols].to_csv(index=False).encode("utf-8"), "dev_pulse.csv","text/csv")
    else:
        st.info("Dev Pulse feature is disabled.")

with tab7:
    st.subheader("📣 Social / Votes")
    st.write("**Make it viral** — share a simple Top-5 Truth summary. Upvote/Downvote coins to teach public sentiment (demo, session-only).")

    # Public votes (session demo)
    if FEATURES["PUBLIC_VOTES"]:
        v = profile.get("public_votes", {})
        pick = st.selectbox("Pick a coin to vote", ["(choose)"] + df["symbol"].str.upper().tolist())
        c1, c2 = st.columns(2)
        if c1.button("👍 Upvote") and pick != "(choose)":
            v[pick] = v.get(pick, 0) + 1
            profile["public_votes"] = v; save_profile(profile)
        if c2.button("👎 Downvote") and pick != "(choose)":
            v[pick] = v.get(pick, 0) - 1
            profile["public_votes"] = v; save_profile(profile)
        if v:
            vv = pd.DataFrame([{"symbol":k, "votes":val} for k,val in v.items()]).sort_values("votes", ascending=False)
            st.dataframe(vv, use_container_width=True)
        else:
            st.info("No votes yet. Try upvoting a symbol.")

    # Top-5 Truth summary text + PNG card (if kaleido available)
    top5 = df.sort_values("truth_full", ascending=False).head(5)[["name","symbol","truth_full","mood","divergence"]]
    st.markdown("**Top-5 Truth (shareable):**")
    st.table(top5.assign(symbol=top5["symbol"].str.upper()))

    # Build a shareable Plotly card
    if PLOTLY_OK and FEATURES["SOCIAL_CARD"]:
        txt_lines = [f"{r['name']} ({str(r['symbol']).upper()}): Truth {r['truth_full']:.2f} • {r['mood']} • Div {r['divergence']:+.2f}"
                     for _, r in top5.iterrows()]
        card_text = "<br>".join(txt_lines) if txt_lines else "No data"
        fig = go.Figure()
        fig.add_annotation(
            x=0.01, y=0.95, xanchor="left", yanchor="top", align="left",
            text=f"<b>Crypto Hybrid Live — Top-5 Truth</b><br>{card_text}<br><br><i>Truth > Hype • Powered by LIPE</i>",
            showarrow=False
        )
        fig.update_layout(
            width=900, height=500, paper_bgcolor="#0E1117",
            margin=dict(l=20,r=20,t=20,b=20), xaxis={'visible': False}, yaxis={'visible': False}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Try PNG export (needs kaleido); fallback: CSV + copyable text
        try:
            png_bytes = fig.to_image(format="png", scale=2)  # requires kaleido
            st.download_button("🖼️ Download Share Card (PNG)", png_bytes, file_name="top5_truth_card.png", mime="image/png")
        except Exception:
            st.info("Add `kaleido` to requirements.txt for PNG export. For now, copy text or download CSV below.")

    st.download_button("⬇️ Download Top-5 (CSV)", top5.to_csv(index=False).encode("utf-8"), "top5_truth.csv","text/csv")

# ====================== ALERTS ======================
if FEATURES["ALERTS"]:
    matches = df[(df["truth_full"] >= alert_truth) | (df["divergence"].abs() >= alert_diverg)]
    if len(matches):
        st.warning(f"🚨 {len(matches)} coins matched your alert rules")
        st.dataframe(matches.sort_values("truth_full", ascending=False)[
            ["name","symbol","truth_full","divergence","mood"]
        ], use_container_width=True)

# ====================== FOOTER ======================
st.markdown("""<hr style="margin-top: 1rem; margin-bottom: 0.5rem;">""", unsafe_allow_html=True)
api_status = "🟢 API OK" if not df.empty else "🔴 API issue"
st.caption(f"{api_status} • {BRAND_FOOTER} • Truth = momentum + liquidity + stability. Explanations inside each card.")
