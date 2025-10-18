# ====================================================================================
#  CRYPTO HYBRID LIVE — PHASE 13.7 (FULL)
#  Adds GLOBAL Truth / Raw / Divergence FILTER + Big KPI boards on every page
#  Unified Crypto + Stocks + FX + Sentiment + Scores + Signal Center + Export
#  SmartCache AutoClean™ built-in
#  Powered by Jesse Ray Landingham Jr
# ====================================================================================

import os, gc, tempfile, time, math, ssl, smtplib
from email.mime.text import MIMEText
from datetime import datetime, timezone, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import requests
import pytz

# Optional libs (app still runs if missing)
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

try:
    import feedparser
    FP_OK = True
except Exception:
    FP_OK = False

try:
    from textblob import TextBlob
    TB_OK = True
except Exception:
    TB_OK = False


# ========================= SmartCache AutoClean™ ====================================
def smart_cache_clean(max_age_minutes=120):
    try:
        cache_dir = tempfile.gettempdir()
        now = time.time()
        for root, _, files in os.walk(cache_dir):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    if os.stat(fp).st_mtime < now - (max_age_minutes * 60):
                        os.remove(fp)
                except Exception:
                    pass
        gc.collect()
    except Exception:
        pass

def ensure_periodic_clean():
    if "last_clean" not in st.session_state:
        smart_cache_clean()
        st.session_state["last_clean"] = time.time()
    elif time.time() - st.session_state["last_clean"] > 900:
        smart_cache_clean()
        st.session_state["last_clean"] = time.time()


# ========================= App Config & Styles ======================================
APP_TITLE = "Crypto Hybrid Live — Phase 13.7"
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

def apply_css(font_px=18, high_contrast=False, dark=True):
    base_bg = "#0d1117" if dark else "#ffffff"
    base_fg = "#e6edf3" if dark else "#111111"
    accent  = "#22c55e" if not high_contrast else "#ffdd00"
    grad_to = "#15803d" if not high_contrast else "#ffaa00"
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            background:{base_bg}; color:{base_fg}; font-size:{font_px}px;
        }}
        .phase-banner {{
            font-weight:900; text-align:center; color:white;
            background:linear-gradient(90deg,{accent},{grad_to});
            border-radius:14px; padding:.6rem .8rem; margin:0 0 1rem 0;
        }}
        .chip {{ display:inline-block; padding:.25rem .6rem; border-radius:999px; font-weight:700; margin-right:.35rem; }}
        .ok {{ background:#86efac; color:#111; }} .warn {{ background:#fde047; color:#111; }} .err {{ background:#fda4af; color:#111; }}
        .kpi-card {{
            background: rgba(148,163,184,.08);
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 16px 18px;
            text-align: center;
        }}
        .kpi-title {{ font-size: 14px; opacity: .9; margin: 0 0 6px 0; }}
        .kpi-value {{ font-size: 42px; font-weight: 900; margin: 0; }}
        .kpi-sub   {{ font-size: 14px; opacity: .75; margin: 6px 0 0 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_css()


# ========================= Secrets / Keys (optional) =================================
ALPHA_V_KEY   = st.secrets.get("ALPHA_VANTAGE_KEY", None)
DISCORD_HOOK  = st.secrets.get("DISCORD_WEBHOOK_URL", "")
SMTP_HOST     = st.secrets.get("SMTP_HOST", "")
SMTP_PORT     = int(st.secrets.get("SMTP_PORT", "587")) if st.secrets.get("SMTP_PORT") else 587
SMTP_USER     = st.secrets.get("SMTP_USER", "")
SMTP_PASS     = st.secrets.get("SMTP_PASS", "")
ALERT_EMAIL_TO= st.secrets.get("ALERT_EMAIL_TO", "")


# ========================= Session Init =============================================
for k, v in {
    "watchlist": [],
    "alerts_log": [],
    "cooldown": {},
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ========================= Sidebar ===================================================
st.sidebar.header("Navigation")
nav = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Crypto", "Stocks", "FX", "Scores", "Sentiment", "Signal Center", "Export"],
    index=0,
)

st.sidebar.subheader("Score Lens (Global)")
# NEW: Global Truth / Raw / Divergence filter
score_lens = st.sidebar.radio(
    "Choose how to rank & highlight",
    ["Fusion (default)", "Truth", "Raw", "Divergence"],
    index=0
)

st.sidebar.subheader("Appearance")
font_size = st.sidebar.slider("Font size", 14, 26, 18)
high_contrast = st.sidebar.toggle("High contrast mode", value=False)
apply_css(font_size, high_contrast, True)

st.sidebar.subheader("Watchlist")
wl_add = st.sidebar.text_input("Add symbol (BTC, ETH, AAPL...)", "")
if st.sidebar.button("Add"):
    t = wl_add.strip().upper()
    if t and t not in st.session_state.watchlist:
        st.session_state.watchlist.append(t)
if st.session_state.watchlist:
    st.sidebar.caption("Saved: " + ", ".join(st.session_state.watchlist))

st.sidebar.subheader("Refresh")
auto_refresh = st.sidebar.toggle("Auto refresh", value=False)
every_sec = st.sidebar.slider("Every (sec)", 10, 120, 30, 5)
if auto_refresh:
    time.sleep(every_sec)


# ========================= Utilities =================================================
USER_AGENT = {"User-Agent": "CHL-Phase13.7"}
CGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"

def chip(kind, text):
    klass = {"ok":"ok","warn":"warn","err":"err"}.get(kind, "warn")
    st.markdown(f'<span class="chip {klass}">{text}</span>', unsafe_allow_html=True)

def safe_get(url, params=None, timeout_sec=25):
    try:
        r = requests.get(url, params=params, headers=USER_AGENT, timeout=timeout_sec)
        if r.status_code == 200:
            return r
    except Exception:
        pass
    return None

def sig_pct_to_01(x):
    if pd.isna(x): return 0.5
    return 1.0 / (1.0 + math.exp(-float(x)/10.0))

def norm_weights(w):
    s = sum(max(0, v) for v in w.values()) or 1.0
    return {k: max(0, v) / s for k, v in w.items()}

def send_discord(webhook_url, message):
    try:
        r = requests.post(webhook_url, json={"content": message}, timeout=15)
        return r.status_code in (200, 201, 204)
    except Exception:
        return False

def send_email(to_email, subject, body):
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and to_email): return False
    try:
        msg = MIMEText(body); msg["Subject"]=subject; msg["From"]=SMTP_USER; msg["To"]=to_email
        ctx = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls(context=ctx); s.login(SMTP_USER, SMTP_PASS); s.sendmail(SMTP_USER,[to_email],msg.as_string())
        return True
    except Exception:
        return False


# ========================= Scoring Weights ==========================================
BASE_W   = dict(w_vol=0.30, w_m24=0.25, w_m7=0.20, w_liq=0.25)     # Truth Filter components
FUSION_W = dict(w_lipe=0.70, w_volatility=0.10, w_sent=0.10, w_fund=0.10)  # Fusion 2.0


# ========================= Data Functions ===========================================
@st.cache_data(ttl=60)
def cg_markets(vs="usd", limit=150):
    p = {
        "vs_currency": vs,
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": "1h,24h,7d",
    }
    r = safe_get(CGECKO_URL, p)
    return pd.DataFrame(r.json()) if r else pd.DataFrame()

def quick_volatility_01(df):
    p1 = df.get("price_change_percentage_24h_in_currency", pd.Series(0)).abs()
    p7 = df.get("price_change_percentage_7d_in_currency", pd.Series(0)).abs()
    v = (p1.clip(0, 20) + 0.5 * p7.clip(0, 40)) / 30.0
    return v.clip(0, 1).fillna(0.3)

@st.cache_data(ttl=300)
def rss_sentiment():
    if not FP_OK:
        return 0.5, []
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://news.bitcoin.com/feed/",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
    ]
    titles = []
    for u in feeds:
        try:
            d = feedparser.parse(u)
            for e in d.entries[:20]:
                titles.append(e.title)
        except Exception:
            continue
    if not titles:
        return 0.5, []
    if TB_OK:
        pol = np.mean([TextBlob(t).sentiment.polarity for t in titles])
        return float((pol + 1)/2), titles[:40]
    return 0.5, titles[:40]

NEWS_SENTIMENT, NEWS_TITLES = rss_sentiment()

def lipe_truth(df, w=None):
    w = norm_weights(w or BASE_W)
    if "liquidity01" not in df:
        df["liquidity01"] = 0.0
    if "vol_to_mc" not in df:
        vol = df.get("total_volume", pd.Series(0, index=df.index))
        v01 = (vol - vol.min()) / (vol.max() - vol.min() + 1e-9)
        df["vol_to_mc"] = 2.0 * v01
    return (
        w["w_vol"] * (df["vol_to_mc"]/2).clip(0,1) +
        w["w_m24"] * df.get("momo_24h01", 0.5) +
        w["w_m7"]  * df.get("momo_7d01", 0.5) +
        w["w_liq"] * df["liquidity01"]
    ).clip(0,1)

def fusion_score(df, kind="CRYPTO_OR_STOCK"):
    lipe = lipe_truth(df)
    vol01 = quick_volatility_01(df)
    vol_component = (1.0 - vol01).clip(0, 1)
    sent_component = float(NEWS_SENTIMENT)
    fund_component = pd.Series(0.5, index=df.index)  # placeholder (fundamentals optional)
    W = norm_weights(FUSION_W)
    fused = (
        W["w_lipe"] * lipe +
        W["w_volatility"] * vol_component +
        W["w_sent"] * sent_component +
        W["w_fund"] * fund_component
    ).clip(0,1)
    return fused

def build_crypto(vs="usd", limit=150):
    df = cg_markets(vs, limit)
    if df.empty: return df
    df["vol_to_mc"] = (df["total_volume"]/df["market_cap"]).replace([np.inf,-np.inf],np.nan).clip(0,2).fillna(0)
    df["momo_24h01"] = df["price_change_percentage_24h_in_currency"].apply(sig_pct_to_01)
    df["momo_7d01"]  = df["price_change_percentage_7d_in_currency"].apply(sig_pct_to_01)
    mc = df["market_cap"].fillna(0)
    df["liquidity01"] = 0 if mc.max()==0 else (mc-mc.min())/(mc.max()-mc.min()+1e-9)
    df["truth_full"] = lipe_truth(df, BASE_W)
    df["raw_heat"]   = (0.5*(df["vol_to_mc"]/2).clip(0,1) + 0.5*df["momo_24h01"]).clip(0,1)
    df["divergence"] = (df["raw_heat"] - df["truth_full"]).round(3)
    df["symbol"]     = df["symbol"].str.upper()
    df["asset_type"] = "CRYPTO"
    df["fusion20"]   = fusion_score(df, "CRYPTO")
    return df

def stocks_demo():
    rows = [("AAPL",227.4,0.85),("MSFT",424.1,0.65),("NVDA",114.7,-1.2),("AMZN",196.0,0.1),("GOOGL",174.8,-0.4)]
    df = pd.DataFrame(rows, columns=["symbol","current_price","price_change_percentage_24h_in_currency"])
    df["total_volume"]=1_000_000; df["market_cap"]=np.nan
    df["momo_24h01"]=df["price_change_percentage_24h_in_currency"].apply(sig_pct_to_01)
    df["momo_7d01"]=0.5; df["vol_to_mc"]=0.5; df["liquidity01"]=0.5
    df["truth_full"]=lipe_truth(df, BASE_W)
    df["raw_heat"]=(0.5*(df["vol_to_mc"]/2).clip(0,1) + 0.5*df["momo_24h01"]).clip(0,1)
    df["divergence"]=(df["raw_heat"]-df["truth_full"]).round(3)
    df["asset_type"]="STOCK"
    df["fusion20"]=fusion_score(df, "STOCK")
    return df

def build_stocks_or_fx(tickers_csv, period="6mo"):
    if not YF_OK: return stocks_demo(), "demo"
    try:
        tickers = [t.strip().upper() for t in tickers_csv.split(",") if t.strip()]
        data = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False, threads=True)
        if data is None or data.empty: return stocks_demo(), "demo"
        prices = data.get("Adj Close", data).ffill()
        if isinstance(prices, pd.Series): prices = prices.to_frame(tickers[0])
        last = prices.iloc[-1]; prev = prices.iloc[-2] if len(prices)>1 else prices.iloc[-1]
        chg24 = (last/prev - 1.0)*100.0
        rows = []
        for t in prices.columns:
            rows.append({"symbol":t.upper(),"current_price":float(last.get(t,np.nan)),
                         "price_change_percentage_24h_in_currency":float(chg24.get(t,np.nan))})
        df = pd.DataFrame(rows)
        for c in ("total_volume","market_cap"): df[c]=np.nan
        df["momo_24h01"]=df["price_change_percentage_24h_in_currency"].apply(sig_pct_to_01)
        df["momo_7d01"]=0.5; df["vol_to_mc"]=0.5; df["liquidity01"]=0.5
        df["truth_full"]=lipe_truth(df, BASE_W)
        df["raw_heat"]=(0.5*(df["vol_to_mc"]/2).clip(0,1) + 0.5*df["momo_24h01"]).clip(0,1)
        df["divergence"]=(df["raw_heat"]-df["truth_full"]).round(3)
        df["asset_type"]="STOCK/FX"
        df["fusion20"]=fusion_score(df, "STOCK")
        return df, "live"
    except Exception:
        return stocks_demo(), "demo"

def unify_frames(frames):
    cols = ["asset_type","symbol","name","current_price","price_change_percentage_24h_in_currency",
            "total_volume","market_cap","liquidity01","vol_to_mc","momo_24h01","momo_7d01",
            "truth_full","raw_heat","divergence","fusion20"]
    out=[]
    for f in frames:
        if f is not None and not f.empty:
            for c in cols:
                if c not in f.columns: f[c]=np.nan if c!="asset_type" else ""
            out.append(f[cols].copy())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


# ========================= Header ====================================================
ensure_periodic_clean()
st.markdown(
    f'<div class="phase-banner">🟢 {APP_TITLE}<br/><small>Powered by Jesse Ray Landingham Jr</small></div>',
    unsafe_allow_html=True
)


# ========================= Data Loads by Nav ========================================
vs_currency="usd"; topn=150
if nav in ("Dashboard","Crypto","Scores","Signal Center"):
    crypto_df = build_crypto(vs_currency, topn)
else:
    crypto_df = pd.DataFrame()

if nav in ("Dashboard","Stocks","Scores","Signal Center"):
    stocks_default="AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA"
    tickers = st.sidebar.text_input("Stocks (comma-separated)", stocks_default)
    stocks_df, stocks_mode = build_stocks_or_fx(tickers)
else:
    stocks_df, stocks_mode = pd.DataFrame(), "demo"

if nav in ("Dashboard","FX","Scores","Signal Center"):
    fx_default="EURUSD=X,USDJPY=X,GBPUSD=X,AUDUSD=X,USDCAD=X"
    fx_pairs = st.sidebar.text_input("FX pairs (Yahoo symbols)", fx_default)
    fx_df, fx_mode = build_stocks_or_fx(fx_pairs)
else:
    fx_df, fx_mode = pd.DataFrame(), "demo"


# ========================= Helpers: Lens ============================================
def lens_title():
    return {"Fusion (default)":"fusion20", "Truth":"truth_full", "Raw":"raw_heat", "Divergence":"divergence"}[score_lens]

def sort_with_lens(df):
    key = lens_title()
    asc = False if key != "divergence" else False  # still show largest positive divergence first
    if key not in df.columns:
        return df
    return df.sort_values(key, ascending=asc)

def kpi_row(df):
    # Big KPI cards for Truth/Raw/Divergence (averages)
    if df is None or df.empty:
        c1,c2,c3 = st.columns(3)
        for c in (c1,c2,c3):
            with c:
                st.markdown('<div class="kpi-card"><div class="kpi-title">No Data</div><p class="kpi-value">—</p><div class="kpi-sub">—</div></div>', unsafe_allow_html=True)
        return
    t = pd.to_numeric(df.get("truth_full", pd.Series()), errors="coerce")
    r = pd.to_numeric(df.get("raw_heat", pd.Series()), errors="coerce")
    d = pd.to_numeric(df.get("divergence", pd.Series()), errors="coerce")
    t_avg = np.nanmean(t) if len(t) else np.nan
    r_avg = np.nanmean(r) if len(r) else np.nan
    d_avg = np.nanmean(np.abs(d)) if len(d) else np.nan
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Truth (Avg)</div><p class="kpi-value">{t_avg:.2f}</p><div class="kpi-sub">Liquidity + stability</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Raw (Avg)</div><p class="kpi-value">{r_avg:.2f}</p><div class="kpi-sub">Momentum + volume</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">|Δ| (Avg)</div><p class="kpi-value">{d_avg:.3f}</p><div class="kpi-sub">Gap: Raw − Truth</div></div>', unsafe_allow_html=True)


# ========================= Pages =====================================================
if nav == "Dashboard":
    st.subheader("📊 Unified Market Snapshot")
    sent_kind = "ok" if NEWS_SENTIMENT>=0.6 else "warn" if NEWS_SENTIMENT>=0.4 else "err"
    chip("ok", f"Crypto: {'ok' if not crypto_df.empty else 'empty'}")
    chip("ok" if stocks_mode=="live" else "warn", f"Stocks: {stocks_mode}")
    chip("ok" if fx_mode=="live" else "warn", f"FX: {fx_mode}")
    chip(sent_kind, f"Sentiment: {NEWS_SENTIMENT:.2f}")

    uni = unify_frames([crypto_df, stocks_df, fx_df])
    kpi_row(uni)

    if uni.empty:
        st.error("No data loaded.")
    else:
        st.markdown(f"**Ranking by:** `{lens_title()}`")
        st.dataframe(sort_with_lens(uni)[
            ["asset_type","symbol","name","current_price","price_change_percentage_24h_in_currency",
             "truth_full","raw_heat","divergence","fusion20","liquidity01","vol_to_mc"]
        ], use_container_width=True, height=600)

elif nav == "Crypto":
    st.subheader("💎 Crypto Market")
    if crypto_df.empty:
        st.error("No crypto data.")
    else:
        kpi_row(crypto_df)
        st.markdown(f"**Ranking by:** `{lens_title()}`")
        b = sort_with_lens(crypto_df)
        st.dataframe(b[
            ["symbol","name","current_price","price_change_percentage_24h_in_currency",
             "truth_full","raw_heat","divergence","fusion20","liquidity01","vol_to_mc"]
        ], use_container_width=True, height=620)
        if PLOTLY_OK:
            fig = px.scatter(b, x="truth_full", y="raw_heat", text="symbol",
                             color="divergence", color_continuous_scale="Turbo")
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)

elif nav == "Stocks":
    st.subheader("📈 Stock Market")
    chip("ok" if stocks_mode=="live" else "warn", f"Source: {stocks_mode}")
    if stocks_df.empty:
        st.error("No stocks data.")
    else:
        kpi_row(stocks_df)
        st.markdown(f"**Ranking by:** `{lens_title()}`")
        b = sort_with_lens(stocks_df)
        st.dataframe(b[
            ["symbol","current_price","price_change_percentage_24h_in_currency",
             "truth_full","raw_heat","divergence","fusion20"]
        ], use_container_width=True, height=620)

elif nav == "FX":
    st.subheader("💱 Forex Market (Preview)")
    chip("ok" if fx_mode=="live" else "warn", f"Source: {fx_mode}")
    if fx_df.empty:
        st.error("No FX data.")
    else:
        kpi_row(fx_df)
        st.markdown(f"**Ranking by:** `{lens_title()}`")
        b = sort_with_lens(fx_df)
        st.dataframe(b[
            ["symbol","current_price","price_change_percentage_24h_in_currency",
             "truth_full","raw_heat","divergence","fusion20"]
        ], use_container_width=True, height=620)

elif nav == "Scores":
    st.subheader("🏁 Scores — Truth / Raw / Divergence (Global Lens Active)")
    src = st.selectbox("Source Table", ["Crypto","Stocks","FX","Unified"], index=0)
    if src == "Crypto": base = crypto_df
    elif src == "Stocks": base = stocks_df
    elif src == "FX": base = fx_df
    else: base = unify_frames([crypto_df, stocks_df, fx_df])

    kpi_row(base)
    if base is None or base.empty:
        st.warning("No data available for the selected source.")
    else:
        st.markdown(f"**Ranking by:** `{lens_title()}`")
        base = sort_with_lens(base)

        st.markdown("#### Truth (ranked)")
        cols_t = [c for c in ["asset_type","symbol","name","current_price",
                              "price_change_percentage_24h_in_currency","truth_full","liquidity01","vol_to_mc"] if c in base.columns]
        st.dataframe(base.sort_values("truth_full", ascending=False)[cols_t], use_container_width=True, height=280)

        st.markdown("#### Raw (ranked)")
        cols_r = [c for c in ["asset_type","symbol","current_price",
                              "price_change_percentage_24h_in_currency","raw_heat","vol_to_mc","momo_24h01"] if c in base.columns]
        st.dataframe(base.sort_values("raw_heat", ascending=False)[cols_r], use_container_width=True, height=280)

        st.markdown("#### Divergence (ranked)")
        cols_d = [c for c in ["asset_type","symbol","current_price","divergence",
                              "truth_full","raw_heat","price_change_percentage_24h_in_currency"] if c in base.columns]
        st.dataframe(base.sort_values("divergence", ascending=False)[cols_d], use_container_width=True, height=280)

        if PLOTLY_OK:
            fig = px.scatter(base, x="truth_full", y="raw_heat", text="symbol",
                             color="divergence", color_continuous_scale="Turbo")
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)

elif nav == "Sentiment":
    st.subheader("🧠 Market News & Sentiment")
    kind = "ok" if NEWS_SENTIMENT>=0.6 else "warn" if NEWS_SENTIMENT>=0.4 else "err"
    chip(kind, f"Aggregate Sentiment: {NEWS_SENTIMENT:.2f}")
    st.progress(min(max(NEWS_SENTIMENT, 0.0), 1.0))
    if NEWS_TITLES:
        st.markdown("Recent Headlines (sample):")
        for t in NEWS_TITLES[:25]:
            st.write("•", t)
    else:
        st.caption("Sentiment libs not installed; using neutral baseline.")

elif nav == "Signal Center":
    st.subheader("🚦 Signal Center — Rules & Alerts (Lens-aware)")
    src = st.selectbox("Source Table", ["Unified (Crypto+Stocks+FX)", "Crypto only", "Stocks only", "FX only"], index=0)
    if src == "Crypto only": base = crypto_df
    elif src == "Stocks only": base = stocks_df
    elif src == "FX only": base = fx_df
    else: base = unify_frames([crypto_df, stocks_df, fx_df])

    if base is None or base.empty:
        st.warning("No data available for selected source.")
    else:
        A,B,C = st.columns(3)
        with A:
            thr_fusion = st.slider("Fusion 2.0 >=", 0.0, 1.0, 0.82, 0.01)
            thr_truth  = st.slider("Truth >=", 0.0, 1.0, 0.75, 0.01)
        with B:
            thr_24h    = st.slider("24h % >=", -10.0, 15.0, 0.0, 0.1)
            thr_absdiv = st.slider("|Divergence| >=", 0.0, 1.0, 0.28, 0.01)
        with C:
            min_price  = st.number_input("Min price (opt.)", value=0.0, min_value=0.0, step=0.01, format="%.2f")
            only_watch = st.toggle("Filter to Watchlist only", value=False)

        q = base.copy()
        if "price_change_percentage_24h_in_currency" not in q:
            q["price_change_percentage_24h_in_currency"] = 0.0
        q["abs_div"] = q["divergence"].abs()
        mask = (
            (q["fusion20"] >= thr_fusion) &
            (q["truth_full"] >= thr_truth) &
            (q["price_change_percentage_24h_in_currency"] >= thr_24h) &
            (q["abs_div"] >= thr_absdiv) &
            (q["current_price"] >= min_price)
        )
        if only_watch and st.session_state.watchlist:
            mask = mask & (q["symbol"].isin([s.upper() for s in st.session_state.watchlist]))

        hits = sort_with_lens(q[mask]).sort_values(["fusion20","abs_div","truth_full"], ascending=False)
        st.success(f"Matches: {len(hits)}")
        if len(hits):
            st.dataframe(
                hits[["asset_type","symbol","current_price","fusion20","truth_full","raw_heat","divergence",
                      "price_change_percentage_24h_in_currency"]],
                use_container_width=True, height=420,
            )
        else:
            st.info("No matches. Adjust thresholds or watchlist.")

        st.markdown("---")
        st.subheader("Delivery")
        col1,col2 = st.columns(2)
        with col1:
            hook = st.text_input("Discord Webhook URL", value=DISCORD_HOOK, type="password")
            enable_discord = st.toggle("Enable Discord alerts", value=bool(hook))
        with col2:
            email_to = st.text_input("Alert Email To", value=ALERT_EMAIL_TO)
            enable_email = st.toggle("Enable Email alerts", value=bool(SMTP_HOST and SMTP_USER and SMTP_PASS and email_to))

        cooldown_min = st.slider("Cooldown per symbol (min)", 1, 240, 30, 1)
        if st.button("Send Alerts Now") and len(hits):
            sent = 0
            for _, r in hits.iterrows():
                key = f"{r['symbol']}"
                last = st.session_state.cooldown.get(key, None)
                now = datetime.now(timezone.utc)
                if last and (now - last) < timedelta(minutes=cooldown_min):
                    continue
                msg = (
                    f"CHL v13.7 Signal — {r['asset_type']} {r['symbol']}\n"
                    f"Price: {r['current_price']:.4g}\n"
                    f"Fusion20: {r['fusion20']:.3f} | Truth: {r['truth_full']:.3f} | Raw: {r['raw_heat']:.3f}\n"
                    f"Delta (Raw-Truth): {r['divergence']:+.3f} | 24h%: {r['price_change_percentage_24h_in_currency']:.2f}%\n"
                    f"Sentiment: {NEWS_SENTIMENT:.2f} | Time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                )
                ok = True
                if enable_discord and hook: ok = ok and send_discord(hook, msg)
                if enable_email and email_to: ok = ok and send_email(email_to, subject=f"CHL v13.7 — {r['symbol']}", body=msg)
                if ok:
                    st.session_state.cooldown[key] = now
                    st.session_state.alerts_log.append(
                        {"ts": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
                         "symbol": r["symbol"],
                         "fusion20": float(r["fusion20"]),
                         "truth": float(r["truth_full"]),
                         "raw": float(r["raw_heat"]),
                         "div": float(r["divergence"])}
                    )
                    sent += 1
            st.success(f"Alerts sent: {sent}")

        st.markdown("Recent Alerts Log")
        if st.session_state.alerts_log:
            log_df = pd.DataFrame(st.session_state.alerts_log)
            st.dataframe(log_df.sort_values("ts", ascending=False), use_container_width=True, height=240)
        else:
            st.caption("No alert logs yet.")

elif nav == "Export":
    st.subheader("📤 Export Data")
    choice = st.selectbox("Table", ["Crypto","Stocks","FX","Unified"])
    if choice == "Crypto": data = crypto_df
    elif choice == "Stocks": data = stocks_df
    elif choice == "FX": data = fx_df
    else: data = unify_frames([crypto_df, stocks_df, fx_df])

    if data is None or data.empty:
        st.warning("Nothing to export.")
    else:
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name=f"chl_{choice.lower()}.csv", mime="text/csv")


# ========================= Watchlist Pills ==========================================
st.markdown("### Watchlist Signals")
wl = [s.upper() for s in st.session_state.watchlist]
cur = unify_frames([crypto_df, stocks_df, fx_df])
if wl and cur is not None and not cur.empty:
    sub = cur[cur["symbol"].isin(wl)][["symbol","current_price","truth_full","raw_heat","divergence","fusion20"]].copy()
    if sub.empty:
        st.caption("Your watchlist symbols are not in the current tables.")
    else:
        html=[]
        for _, r in sub.iterrows():
            sign = "ok" if r["divergence"] >= 0 else "err"
            html.append(
                f'<span class="chip {sign}"><b>{r["symbol"]}</b> · ${r["current_price"]:.4g} · '
                f'F:{r["fusion20"]:.2f} · T:{r["truth_full"]:.2f} · R:{r["raw_heat"]:.2f} · Δ {r["divergence"]:+.2f}</span>'
            )
        st.markdown(" ".join(html), unsafe_allow_html=True)
else:
    st.caption("Add symbols in the sidebar (BTC, ETH, AAPL, etc.)")


# ========================= Footer ====================================================
st.divider()
st.markdown(
    "<p style='text-align:center; font-size:14px;'>© 2025 Jesse Ray Landingham Jr | Phase 13.7 | Global Lens: Truth / Raw / Divergence</p>",
    unsafe_allow_html=True
) 
