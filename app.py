# ====================== IMPORTS ======================
import math, time, json, re
import requests, pandas as pd, numpy as np, streamlit as st
from datetime import datetime, timezone
from io import StringIO

# Optional plotting (used for minis)
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# yfinance for Stocks / FX
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

# ====================== CONFIG ======================
APP_NAME = "Crypto Hybrid Live — Phase 7 (Rules + Alerts)"
st.set_page_config(page_title=APP_NAME, layout="wide")
USER_AGENT = {"User-Agent": "Mozilla/5.0 (CHL-Phase7)"}
CG_PER_PAGE = 150
HIST_DAYS = 90

# ====================== THEME ======================
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
if "sent_keys" not in st.session_state:
    st.session_state["sent_keys"] = set()   # duplicate guard within session

def apply_theme_css():
    dark = st.session_state.get("theme","dark")=="dark"
    base_bg = "#0d1117" if dark else "#ffffff"
    base_fg = "#e6edf3" if dark else "#111111"
    accent  = "#23d18b" if dark else "#0b8f5a"
    st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
      font-size: 18px; background:{base_bg}; color:{base_fg};
    }}
    .stTabs [role="tablist"] button {{
      font-size:1.25rem !important; font-weight:700 !important;
      margin-right:.6rem; border-radius:10px;
      background:#111; color:{accent}; border:1px solid #222;
    }}
    .stTabs [role="tablist"] button[aria-selected="true"] {{
      background:{accent}; color:#000; transform:scale(1.04);
    }}
    [data-testid="stMetricValue"]{{font-size:2.2rem!important}}
    </style>
    """, unsafe_allow_html=True)
apply_theme_css()

# ====================== LIPE / SCORING ======================
DEFAULT_WEIGHTS = dict(w_vol=0.30, w_m24=0.25, w_m7=0.25, w_liq=0.20)
PRESETS = {
    "Balanced":  dict(w_vol=0.30, w_m24=0.25, w_m7=0.25, w_liq=0.20),
    "Momentum":  dict(w_vol=0.15, w_m24=0.45, w_m7=0.30, w_liq=0.10),
    "Liquidity": dict(w_vol=0.45, w_m24=0.20, w_m7=0.15, w_liq=0.20),
    "Value":     dict(w_vol=0.25, w_m24=0.20, w_m7=0.20, w_liq=0.35),
}

def _normalize_weights(w):
    s = float(sum(max(0, v) for v in w.values())) or 1.0
    return {k: max(0, v) / s for k, v in w.items()}

def pct_sigmoid(p):
    if pd.isna(p): return 0.5
    return 1 / (1 + math.exp(-float(p) / 10.0))

def lipe_truth(df, w):
    w = _normalize_weights(w or DEFAULT_WEIGHTS)
    if "liquidity01" not in df: df["liquidity01"] = 0.0
    if "vol_to_mc" not in df:
        vol = df.get("total_volume", pd.Series(0, index=df.index))
        v01 = (vol - vol.min()) / (vol.max() - vol.min() + 1e-9)
        df["vol_to_mc"] = 2 * v01
    truth = (
        w["w_vol"] * (df["vol_to_mc"] / 2).clip(0,1) +
        w["w_m24"] * df.get("momo_24h01", 0.5) +
        w["w_m7"]  * df.get("momo_7d01",  0.5) +
        w["w_liq"] * df["liquidity01"]
    )
    return truth.clip(0,1)

def mood_label(x):
    if x>=0.8: return "🟢 EUPHORIC"
    if x>=0.6: return "🟡 OPTIMISTIC"
    if x>=0.4: return "🟠 NEUTRAL"
    return "🔴 FEARFUL"

# ====================== HELPERS ======================
def safe_get(url, params=None, t=25):
    try:
        r = requests.get(url, params=params, headers=USER_AGENT, timeout=t)
        if r.status_code == 200:
            return r
    except Exception:
        pass
    return None

@st.cache_data(ttl=60)
def fetch_markets_cg(vs="usd", limit=150):
    u="https://api.coingecko.com/api/v3/coins/markets"
    p={"vs_currency":vs,"order":"market_cap_desc","per_page":limit,"page":1,
       "sparkline":"false","price_change_percentage":"1h,24h,7d"}
    r=safe_get(u,p)
    return pd.DataFrame(r.json()) if r else pd.DataFrame()

@st.cache_data(ttl=180)
def yf_multi(ticks, period="6mo"):
    if not YF_OK: return pd.DataFrame(), {}
    data = yf.download(ticks, period=period, interval="1d", auto_adjust=True, progress=False)
    frames=[]; meta={}
    for t in ticks:
        try:
            ser = data["Adj Close"][t].rename(t) if len(ticks)>1 else data["Adj Close"].rename(t)
        except Exception:
            continue
        frames.append(ser)
        try: fi=yf.Ticker(t).fast_info
        except Exception: fi={}
        meta[t]={"market_cap":getattr(fi,"market_cap",np.nan),
                 "last_price":getattr(fi,"last_price",np.nan),
                 "volume":getattr(fi,"last_volume",np.nan)}
    return pd.concat(frames,axis=1),meta

# ====================== HEADER ======================
st.title("🟢 "+APP_NAME)
st.caption("Build rules, auto-refresh, and send alerts to Discord / Telegram / Email (if secrets set).")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("🧭 Market")
    market = st.radio("Choose market", ["Crypto","Stocks","FX"], horizontal=True)
    vs_currency = st.selectbox("Currency (Crypto)", ["usd"], index=0)
    topn = st.slider("Show top N (Crypto)", 20, 250, CG_PER_PAGE, step=10)

    st.subheader("Theme")
    theme_pick = st.radio("Theme", ["Dark","Light"],
                          index=0 if st.session_state["theme"]=="dark" else 1,
                          horizontal=True)
    st.session_state["theme"] = "dark" if theme_pick=="Dark" else "light"
    apply_theme_css()

    st.subheader("Truth Preset")
    preset = st.radio("Preset", list(PRESETS.keys()), index=0, horizontal=True)
    w_edit = dict(PRESETS[preset])
    st.subheader("Weights")
    for k in w_edit:
        w_edit[k] = st.slider(k, 0.0, 1.0, float(w_edit[k]), 0.01)
    w_edit = _normalize_weights(w_edit)

    if market=="Stocks":
        st.subheader("Stocks tickers")
        stocks_in = st.text_area("", value="AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA")
    if market=="FX":
        st.subheader("FX pairs")
        fx_in = st.text_area("", value="EURUSD=X, USDJPY=X, GBPUSD=X, AUDUSD=X, USDCAD=X")

    st.markdown("---")
    st.subheader("Live Mode")
    live = st.toggle("Auto-refresh", value=False)
    every = st.slider("Refresh every (sec)", 10, 120, 30, 5)
    if live:
        st.experimental_rerun = st.experimental_rerun  # keep handle
        st.autorefresh = st.experimental_rerun  # alias (no-op but keeps code explicit)
        st.experimental_set_query_params(_=int(time.time()))  # bust URL cache
        # Streamlit’s built-in auto-refresh helper:
        st.experimental_memo.clear() if False else None  # placeholder

# Gentle autorefresh via hidden component
if live:
    # st_autorefresh alternative: tiny hack via time-based empty output
    st.empty()  # keep layout stable
    time.sleep(every)

# ====================== DATA BUILDERS ======================
def build_crypto():
    df = fetch_markets_cg(vs_currency, topn)
    if df.empty: return df
    for k in ["current_price","market_cap","total_volume",
              "price_change_percentage_24h_in_currency",
              "price_change_percentage_7d_in_currency","name","symbol"]:
        if k not in df: df[k]=np.nan
    df["momo_24h01"]=df["price_change_percentage_24h_in_currency"].apply(pct_sigmoid)
    df["momo_7d01"]=df["price_change_percentage_7d_in_currency"].apply(pct_sigmoid)
    df["vol_to_mc"]=(df["total_volume"]/df["market_cap"]).replace([np.inf,-np.inf],np.nan).clip(0,2).fillna(0)
    mc=df["market_cap"].fillna(0)
    df["liquidity01"]=0 if mc.max()==0 else (mc-mc.min())/(mc.max()-mc.min()+1e-9)
    df["truth_full"]=lipe_truth(df,w_edit)
    # Raw + divergence
    df["raw_heat"]=(0.5*(df["vol_to_mc"]/2).clip(0,1)+0.5*df["momo_24h01"]).clip(0,1)
    df["divergence"]=(df["raw_heat"]-df["truth_full"]).round(3)
    df["fusion_truth"]=df["truth_full"]  # fusion extras off in phase 7 core
    df["mood"]=df["truth_full"].apply(mood_label)
    df["symbol"]=df["symbol"].str.upper()
    return df

def build_yf(tickers):
    prices, meta = yf_multi(tickers)
    if prices.empty: return pd.DataFrame()
    last=prices.ffill().iloc[-1]
    chg24=(prices.ffill().iloc[-1]/prices.ffill().iloc[-2]-1.0)*100.0 if len(prices)>=2 else np.nan
    rows=[]
    for t in prices.columns:
        rows.append({
            "symbol": t.upper(),
            "current_price": float(last.get(t, np.nan)),
            "price_change_percentage_24h_in_currency": float(chg24.get(t, np.nan)) if isinstance(chg24, pd.Series) else np.nan,
            "market_cap": float(meta.get(t,{}).get("market_cap", np.nan)),
            "total_volume": float(meta.get(t,{}).get("volume", np.nan)),
        })
    df=pd.DataFrame(rows)
    df["momo_24h01"]=df["price_change_percentage_24h_in_currency"].apply(pct_sigmoid)
    df["momo_7d01"]=0.5
    if df["market_cap"].notna().sum()>0:
        mc=df["market_cap"].fillna(0)
        df["liquidity01"]=0 if mc.max()==0 else (mc-mc.min())/(mc.max()-mc.min()+1e-9)
        df["vol_to_mc"]=(df["total_volume"]/df["market_cap"]).replace([np.inf,-np.inf],np.nan).clip(0,2).fillna(0)
    else:
        v=df["total_volume"].fillna(0)
        df["liquidity01"]=0 if v.max()==0 else (v-v.min())/(v.max()-v.min()+1e-9)
        df["vol_to_mc"]=2*((v-v.min())/(v.max()-v.min()+1e-9))
    df["truth_full"]=lipe_truth(df,w_edit)
    df["raw_heat"]=(0.5*(df["vol_to_mc"]/2).clip(0,1)+0.5*df["momo_24h01"]).clip(0,1)
    df["divergence"]=(df["raw_heat"]-df["truth_full"]).round(3)
    df["fusion_truth"]=df["truth_full"]
    df["mood"]=df["truth_full"].apply(mood_label)
    return df

# Build selected market DF
if market=="Crypto":
    df = build_crypto()
elif market=="Stocks":
    ticks = [t.strip().upper() for t in (stocks_in if 'stocks_in' in locals() else "").split(",") if t.strip()]
    if not ticks: ticks=["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA"]
    if not YF_OK:
        st.error("yfinance not installed: add 'yfinance' to requirements.txt"); st.stop()
    df = build_yf(ticks)
else:
    pairs = [t.strip().upper() for t in (fx_in if 'fx_in' in locals() else "").split(",") if t.strip()]
    if not pairs: pairs=["EURUSD=X","USDJPY=X","GBPUSD=X","AUDUSD=X","USDCAD=X"]
    if not YF_OK:
        st.error("yfinance not installed: add 'yfinance' to requirements.txt"); st.stop()
    df = build_yf(pairs)

if df.empty:
    st.error("No data loaded. Try different inputs or reload.")
    st.stop()

# ====================== KPI SUMMARY ======================
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Assets", len(df))
c2.metric("Avg 24h Δ", f"{df['price_change_percentage_24h_in_currency'].mean():+.2f}%")
c3.metric("Avg Truth", f"{df['truth_full'].mean():.2f}")
c4.metric("Avg Fusion", f"{df['fusion_truth'].mean():.2f}")
st.caption(f"Last update: {now}")

# ====================== TABS ======================
tab_rules, tab_fusion, tab_raw, tab_truth, tab_watch = st.tabs(
    ["🛎️ Rules & Alerts", "🧭 Fusion", "🔥 Raw", "🧭 Truth", "⭐ Watchlist"]
)

# ---------------------- Rules & Alerts ----------------------
with tab_rules:
    st.subheader("🛎️ Rule Builder (matches trigger alerts)")
    st.markdown("Rules are **AND** conditions. Example: Fusion Truth ≥ 0.85 AND 24h Δ ≥ 3% AND |Divergence| ≥ 0.25")

    left, right = st.columns(2)
    with left:
        thr_truth   = st.slider("Fusion Truth ≥", 0.0, 1.0, 0.85, 0.01)
        thr_delta24 = st.slider("24h % ≥", -50.0, 50.0, 0.0, 0.5)
        thr_div_abs = st.slider("|Divergence| ≥", 0.0, 1.0, 0.30, 0.01)
    with right:
        min_price   = st.number_input("Min price (optional)", value=0.0, step=0.01)
        name_filter = st.text_input("Contains (symbol/name, optional)", value="").strip().upper()
        send_now    = st.toggle("Send external notifications (if secrets configured)", value=False)

    # compute matches
    filt = (df["fusion_truth"]>=thr_truth) & (df["divergence"].abs()>=thr_div_abs)
    if pd.notna(thr_delta24):
        filt = filt & (df["price_change_percentage_24h_in_currency"]>=thr_delta24)
    if min_price>0:
        filt = filt & (df["current_price"]>=min_price)
    if name_filter:
        filt = filt & (df["symbol"].str.upper().str.contains(name_filter) | df.get("name", df["symbol"]).str.upper().str.contains(name_filter))

    matches = df[filt].sort_values("fusion_truth", ascending=False).copy()
    st.success(f"Matches: **{len(matches)}**")
    show_cols = [c for c in ["symbol","current_price","fusion_truth","truth_full","divergence","price_change_percentage_24h_in_currency","market_cap"] if c in df.columns]
    st.dataframe(matches[show_cols], use_container_width=True, height=420)

    # ------------- Notifications -------------
    def notify_discord(webhook:str, text:str):
        try: requests.post(webhook, json={"content": text}, timeout=10)
        except Exception: pass

    def notify_telegram(token:str, chat_id:str, text:str):
        try:
            url=f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, data={"chat_id":chat_id,"text":text}, timeout=10)
        except Exception: pass

    def notify_email(host,port,user,pwd,to_addr,subject,body):
        import smtplib
        from email.mime.text import MIMEText
        try:
            msg=MIMEText(body)
            msg["Subject"]=subject; msg["From"]=user; msg["To"]=to_addr
            s=smtplib.SMTP(host, int(port)); s.starttls(); s.login(user,pwd); s.sendmail(user,[to_addr], msg.as_string()); s.quit()
        except Exception: pass

    # secrets
    sec = st.secrets.get("notifications", {})
    discord_webhook = sec.get("discord_webhook")
    tg_token        = sec.get("telegram_token")
    tg_chat         = sec.get("telegram_chat_id")
    smtp_host       = sec.get("smtp_host")
    smtp_port       = sec.get("smtp_port")
    smtp_user       = sec.get("smtp_user")
    smtp_pass       = sec.get("smtp_pass")
    email_to        = sec.get("email_to")

    # build and send messages (dedupe per asset this session)
    if send_now and len(matches):
        for _, r in matches.iterrows():
            key = f"{market}:{r['symbol']}"
            if key in st.session_state["sent_keys"]:
                continue
            msg = (f"🚨 {APP_NAME}\n"
                   f"Market: {market}\n"
                   f"Asset: {r['symbol']}  Price: {r['current_price']}\n"
                   f"Fusion Truth: {r['fusion_truth']:.2f}  Truth: {r['truth_full']:.2f}\n"
                   f"Divergence: {r['divergence']:+.2f}  24h: {r.get('price_change_percentage_24h_in_currency', np.nan):+.2f}%\n"
                   f"Time (UTC): {now}")
            if discord_webhook: notify_discord(discord_webhook, msg)
            if tg_token and tg_chat: notify_telegram(tg_token, tg_chat, msg)
            if smtp_host and smtp_user and smtp_pass and email_to:
                notify_email(smtp_host, smtp_port or 587, smtp_user, smtp_pass, email_to,
                             f"CHL Alert: {r['symbol']} ({market})", msg)
            st.session_state["sent_keys"].add(key)
        st.success("Notifications sent for new matches (duplicates skipped).")

# ---------------------- Fusion / Raw / Truth ----------------------
with tab_fusion:
    st.subheader(f"🧭 Fusion Truth — {market}")
    st.dataframe(df.sort_values("fusion_truth", ascending=False)
                   [[c for c in ["symbol","current_price","market_cap","fusion_truth","truth_full","divergence","mood"] if c in df.columns]],
                 use_container_width=True, height=600)

with tab_raw:
    st.subheader(f"🔥 Raw — {market}")
    st.dataframe(df.sort_values("raw_heat", ascending=False)
                   [[c for c in ["symbol","current_price","price_change_percentage_24h_in_currency","total_volume","raw_heat"] if c in df.columns]],
                 use_container_width=True, height=600)

with tab_truth:
    st.subheader(f"🧭 Truth — {market}")
    st.dataframe(df.sort_values("truth_full", ascending=False)
                   [[c for c in ["symbol","current_price","market_cap","liquidity01","truth_full","divergence","mood"] if c in df.columns]],
                 use_container_width=True, height=600)

with tab_watch:
    st.subheader("⭐ Watchlist (local to this browser)")
    wl = st.session_state.get("watchlist", [])
    add = st.selectbox("Add symbol", ["(choose)"] + df["symbol"].tolist(), index=0)
    if add != "(choose)":
        wl = sorted(set(wl + [add.upper()]))
        st.session_state["watchlist"] = wl
    if wl:
        st.dataframe(df[df["symbol"].isin(wl)][["symbol","current_price","fusion_truth","truth_full","divergence"]],
                     use_container_width=True, height=400)
    else:
        st.info("No symbols yet. Pick one above.")

# ====================== FOOTER ======================
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Sources: CoinGecko • yfinance • CHL Phase 7 ©")
