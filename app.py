# ====================== IMPORTS ======================
import math, time, random, requests, pandas as pd, numpy as np, streamlit as st
from datetime import datetime, timezone
from io import StringIO

# Plotly (safe import so the app never crashes if not installed yet)
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ====================== APP CONFIG / FEATURE FLAGS ======================
APP_NAME = "Crypto Hybrid Live — Ultimate"
BRAND_FOOTER = "Data: CoinGecko • Educational analytics — not financial advice. © Crypto Hybrid Live"
FEATURES = {
    "DEV_PULSE": True,          # CoinGecko developer_data (no key; gentle rate limit)
    "ENTROPY_BIAS": True,       # Stability (entropy) + next-24h bias classifier
    "ALERTS": True,             # On-screen alerts (threshold-based)
    "SNAPSHOT": True,           # One-click snapshot CSV (Top Truth + Top Raw)
    "URL_STATE": True,          # Shareable URL with current settings (uses st.query_params)
    "WEIGHT_PRESETS": True,     # Quick-select presets for Truth weights
}

# ====================== PAGE / SIDEBAR ======================
st.set_page_config(page_title=APP_NAME, layout="wide")
st.title("🟢 " + APP_NAME)
st.caption("Truth > Noise • Live market intelligence with weights, visuals, insights, Dev Pulse, entropy & predictive bias.")

with st.sidebar:
    st.header("⚙️ Controls")
    vs_currency = st.selectbox("Currency", ["usd"], index=0)
    topn = st.slider("Show top N by market cap", 20, 250, 100, step=10)
    search = st.text_input("🔎 Search coin (name or symbol)").strip().lower()

    # Weight presets
    if FEATURES["WEIGHT_PRESETS"]:
        st.markdown("#### Presets")
        preset = st.selectbox("Choose a preset",
                              ["Balanced (default)", "Momentum", "Liquidity", "Value"],
                              index=0)
        if preset == "Momentum":
            w_vol, w_m24, w_m7, w_liq = 0.15, 0.45, 0.30, 0.10
        elif preset == "Liquidity":
            w_vol, w_m24, w_m7, w_liq = 0.20, 0.20, 0.20, 0.40
        elif preset == "Value":
            w_vol, w_m24, w_m7, w_liq = 0.40, 0.20, 0.20, 0.20
        else:
            w_vol, w_m24, w_m7, w_liq = 0.30, 0.25, 0.25, 0.20
    else:
        w_vol, w_m24, w_m7, w_liq = 0.30, 0.25, 0.25, 0.20

    st.markdown("---")
    st.subheader("Weights (Truth Score)")
    w_vol = st.slider("Volume/MC", 0.0, 1.0, w_vol, 0.01)
    w_m24 = st.slider("Momentum 24h", 0.0, 1.0, w_m24, 0.01)
    w_m7  = st.slider("Momentum 7d",  0.0, 1.0, w_m7,  0.01)
    w_liq = st.slider("Liquidity",    0.0, 1.0, w_liq, 0.01)
    st.caption("Weights auto-normalize to 1.0")

    if FEATURES["ALERTS"]:
        st.markdown("---")
        st.subheader("Alerts (on-screen)")
        alert_truth = st.slider("Trigger: Truth ≥", 0.0, 1.0, 0.85, 0.01)
        alert_diverg = st.slider("Trigger: |Divergence| ≥", 0.0, 1.0, 0.30, 0.01)

# ====================== HELPERS ======================
def normalize_weights(*weights):
    s = sum(weights)
    if s <= 1e-9:
        return [1/len(weights)]*len(weights)
    return [w/s for w in weights]

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
    """Simple entropy proxy from dispersion in % changes (lower chaos = higher score)."""
    arr = np.array([p for p in [p1h, p24h, p7d] if pd.notna(p)], dtype=float)
    if arr.size < 2:
        return 0.5
    s = np.std(arr)
    chaos01 = min(max(s/20.0, 0.0), 1.0)  # assume typical std in 0..20%
    return float(1.0 - chaos01)

def predictive_bias_label(m1h, m24, m7):
    """Classify next-24h bias from short vs longer momentum."""
    m1h = 0.0 if pd.isna(m1h) else m1h
    m24 = 0.0 if pd.isna(m24) else m24
    m7  = 0.0 if pd.isna(m7)  else m7
    if m1h > 0 and m24 > 0 and m7 > 0:
        return "🟢 Likely Up"
    if m1h < 0 and m24 < 0:
        return "🔴 Cooling"
    if m24 > 5 and m7 > 10:
        return "🟠 Overheated"
    return "⚪ Mixed"

# --- URL state (modern Streamlit) ---
def get_params():
    return st.query_params if FEATURES["URL_STATE"] else {}

def set_params(**kwargs):
    if FEATURES["URL_STATE"]:
        st.query_params.clear()
        for k, v in kwargs.items():
            if v is not None:
                st.query_params[k] = v

def make_snapshot_csv(df_truth, df_raw):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_UTC")
    buf = StringIO()
    buf.write(f"Snapshot,{ts}\n\nTop by Truth\n")
    df_truth.to_csv(buf, index=False)
    buf.write("\nTop by Raw Heat\n")
    df_raw.to_csv(buf, index=False)
    return f"snapshot_{ts}.csv", buf.getvalue().encode("utf-8")

def safe_get(url, params=None, timeout=30, retries=3, backoff=0.6):
    """Retry with exponential backoff to be gentle with free APIs."""
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        sleep_s = backoff * (2 ** i) + random.uniform(0, 0.2)
        time.sleep(sleep_s)
    return None

# ====================== DATA FETCH ======================
@st.cache_data(ttl=60)
def fetch_markets(vs="usd", per_page=250):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    p = {
        "vs_currency": vs,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": "1h,24h,7d",
        "locale": "en",
    }
    r = safe_get(url, params=p, timeout=30)
    if not r:
        return pd.DataFrame([])
    try:
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame([])

@st.cache_data(ttl=300, show_spinner=False)
def fetch_dev_pulse(ids):
    """CoinGecko developer_data for a small list of coin IDs (no key)."""
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
            if not r or r.status_code != 200:
                continue
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
            time.sleep(0.25)  # respect free tier
        except Exception:
            continue
    dfp = pd.DataFrame(rows)
    if dfp.empty:
        return dfp
    # normalize 0–1
    for c in ["dev_stars","dev_forks","dev_subs","dev_pr_merged","dev_commit_4w"]:
        m, M = dfp[c].min(), dfp[c].max()
        dfp[c+"_01"] = 0.0 if M <= m else (dfp[c]-m)/(M-m)
    dfp["dev_pulse01"] = (
        0.25*dfp["dev_stars_01"] +
        0.20*dfp["dev_forks_01"] +
        0.15*dfp["dev_subs_01"] +
        0.20*dfp["dev_pr_merged_01"] +
        0.20*dfp["dev_commit_4w_01"]
    ).clip(0,1)
    return dfp[["id","dev_pulse01","dev_stars","dev_forks","dev_subs","dev_pr_merged","dev_commit_4w"]]

# ====================== LOAD & FEATURE ENGINEERING ======================
# Load URL params first
params = get_params()
if "q" in params and not search:
    try: search = params["q"][0] if isinstance(params["q"], list) else str(params["q"])
    except Exception: pass
if "n" in params:
    try: topn = int(params["n"][0] if isinstance(params["n"], list) else params["n"])
    except Exception: pass
for key, var in [("wv","w_vol"),("w24","w_m24"),("w7","w_m7"),("wl","w_liq")]:
    if key in params:
        try:
            val = params[key][0] if isinstance(params[key], list) else params[key]
            val = float(val)
            if var in locals(): locals()[var] = val
        except Exception:
            pass

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

# Entropy & Predictive Bias
if FEATURES["ENTROPY_BIAS"]:
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

# Save state to URL (shareable)
set_params(
    q=search or None,
    n=str(topn),
    wv=f"{w_vol:.2f}", w24=f"{w_m24:.2f}", w7=f"{w_m7:.2f}", wl=f"{w_liq:.2f}"
)

# Search filter
if search:
    mask = df["name"].str.lower().str.contains(search) | df["symbol"].str.lower().str.contains(search)
    df = df[mask].copy()

# ====================== KPIs / SNAPSHOT ======================
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
cA,cB,cC,cD = st.columns(4)
with cA: st.metric("Coins", len(df))
with cB: st.metric("Avg 24h Δ", f"{df['price_change_percentage_24h_in_currency'].mean():+.2f}%")
with cC: st.metric("Avg Truth", f"{df['truth_full'].mean():.2f}")
with cD: st.metric("Last update (UTC)", now)

truth_cols = ["name","symbol","current_price","market_cap","truth_full","divergence"]
if "entropy01" in df.columns: truth_cols.append("entropy01")
if "bias_24h" in df.columns: truth_cols.append("bias_24h")
truth_cols.append("mood")
raw_cols   = ["name","symbol","current_price","market_cap","total_volume","raw_heat"]

top_truth = df.sort_values("truth_full", ascending=False).head(25)[truth_cols]
top_raw   = df.sort_values("raw_heat",   ascending=False).head(25)[raw_cols]

if FEATURES["SNAPSHOT"]:
    fname, payload = make_snapshot_csv(top_truth, top_raw)
    st.download_button("⬇️ Download Snapshot (Truth + Raw)", payload, file_name=fname, mime="text/csv")

# ====================== TABS ======================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔥 Raw Scan", "🧭 Truth", "📉 Movers", "🗺️ Heatmap", "🧑‍💻 Dev Pulse", "🧠 Insights"
])

with tab1:
    st.subheader("🔥 Raw Wide Scan")
    cols = ["name","symbol","current_price","market_cap","total_volume","raw_heat"]
    st.dataframe(df.sort_values("raw_heat", ascending=False)[cols].reset_index(drop=True), use_container_width=True)
    st.download_button("⬇️ CSV (Raw Scan)", df[cols].to_csv(index=False).encode("utf-8"), "raw_scan.csv", "text/csv")

with tab2:
    st.subheader("🧭 Truth Filter (weights applied)")
    cols = ["name","symbol","current_price","market_cap","liquidity01","truth_full","divergence"]
    if "entropy01" in df.columns: cols.append("entropy01")
    if "bias_24h" in df.columns: cols.append("bias_24h")
    cols.append("mood")
    st.dataframe(df.sort_values("truth_full", ascending=False)[cols].reset_index(drop=True), use_container_width=True)
    st.download_button("⬇️ CSV (Truth)", df[cols].to_csv(index=False).encode("utf-8"), "truth_table.csv", "text/csv")

with tab3:
    st.subheader("📉 Top Daily Gainers / Losers (24h)")
    g = df.sort_values("price_change_percentage_24h_in_currency", ascending=False).head(12).copy()
    l = df.sort_values("price_change_percentage_24h_in_currency", ascending=True).head(12).copy()
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Top Gainers**")
        st.dataframe(g[["name","symbol","current_price","price_change_percentage_24h_in_currency"]].reset_index(drop=True), use_container_width=True)
        st.download_button("⬇️ Gainers (CSV)", g.to_csv(index=False).encode("utf-8"), "gainers.csv", "text/csv")
    with c2:
        st.write("**Top Losers**")
        st.dataframe(l[["name","symbol","current_price","price_change_percentage_24h_in_currency"]].reset_index(drop=True), use_container_width=True)
        st.download_button("⬇️ Losers (CSV)", l.to_csv(index=False).encode("utf-8"), "losers.csv", "text/csv")

with tab4:
    st.subheader("🗺️ Market Heatmap by Truth (Top 50 by Market Cap)")
    if not PLOTLY_OK:
        st.info("Plotly not available yet. Rebuilding… Try again shortly after dependencies install.")
    else:
        top50 = df.sort_values("market_cap", ascending=False).head(50)
        try:
            fig = px.treemap(
                top50,
                path=[top50["symbol"].str.upper()],
                values="market_cap",
                color="truth_full",
                color_continuous_scale="Turbo",
                title=None
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info("Heatmap unavailable at the moment.")
            st.exception(e)

with tab5:
    st.subheader("🧑‍💻 Developer Pulse (Top 25 by Market Cap)")
    if FEATURES["DEV_PULSE"]:
        top_ids = df.sort_values("market_cap", ascending=False).head(25)["id"].tolist() if "id" in df.columns else []
        if not top_ids:
            st.info("Developer data requires CoinGecko coin IDs; not available in the current dataset.")
        else:
            devdf = fetch_dev_pulse(top_ids)
            if devdf.empty:
                st.info("Developer data temporarily unavailable.")
            else:
                m = df.merge(devdf, on="id", how="left")
                cols = ["name","symbol","dev_pulse01","dev_commit_4w","dev_pr_merged","dev_stars","dev_forks","dev_subs"]
                if "name" not in m.columns:  # fallback safety
                    m["name"] = m["id"]
                if "symbol" not in m.columns:
                    m["symbol"] = m["id"]
                m["dev_pulse01"] = m["dev_pulse01"].fillna(0.0)
                st.dataframe(
                    m.sort_values("dev_pulse01", ascending=False)[cols].reset_index(drop=True),
                    use_container_width=True
                )
                st.download_button("⬇️ Dev Pulse (CSV)", m[cols].to_csv(index=False).encode("utf-8"), "dev_pulse.csv","text/csv")
    else:
        st.info("Dev Pulse feature is disabled.")

with tab6:
    st.subheader("🧠 Narrative (rule-based v1)")
    def explain(r):
        parts = []
        if pd.notna(r.get("price_change_percentage_24h_in_currency")):
            parts.append(f"24h {r['price_change_percentage_24h_in_currency']:+.1f}%")
        if pd.notna(r.get("price_change_percentage_7d_in_currency")):
            parts.append(f"7d {r['price_change_percentage_7d_in_currency']:+.1f}%")
        if "entropy01" in r and pd.notna(r["entropy01"]):
            parts.append("low chaos" if r["entropy01"] >= 0.6 else "high chaos")
        if "bias_24h" in r:
            parts.append(f"bias: {r['bias_24h']}")
        return f"{r['name']} ({r['symbol'].upper()}): Truth {r['truth_full']:.2f}, {', '.join(parts)} → {r['mood']}"
    for _, row in df.sort_values("truth_full", ascending=False).head(12).iterrows():
        st.write("• " + explain(row))

# ====================== ALERTS (ON-SCREEN) ======================
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
st.caption(f"{api_status} • {BRAND_FOOTER} • Terms & Privacy: add links in README")
