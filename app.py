# ================================== PHASE 13 — FUSION (SIGNALS + ALERTS) ==================================
# Crypto Hybrid Live — Phase 13 Fusion
# - Signal Center (live rules → alerts)
# - Discord + Email alerts (webhooks/SMTP via st.secrets)
# - Fusion Score 2.0 (Momentum + Liquidity + Volatility + Sentiment [+ Fundamentals if available])
# - Enhanced Stocks (yfinance live, demo fallback; optional Alpha Vantage fundamentals)
# - Retains: Crypto + Stocks + FX + Sentiment + Watchlist + Export + Branding
# ==========================================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests, math, time, smtplib, ssl
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText

# ------------- Optional libs (never crash if missing) -------------
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

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

# ------------- App config -------------
APP_TITLE = "Crypto Hybrid Live — Phase 13 (Fusion Signals + Alerts)"
st.set_page_config(page_title=APP_TITLE, layout="wide")
USER_AGENT = {"User-Agent": "Mozilla/5.0 (CHL-Phase13)"}
CG_API = "https://api.coingecko.com/api/v3/coins/markets"

# Secrets (optional)
ALPHA_V_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", None)
DISCORD_WEBHOOK = st.secrets.get("DISCORD_WEBHOOK_URL", "")
SMTP_HOST = st.secrets.get("SMTP_HOST", "")
SMTP_PORT = int(st.secrets.get("SMTP_PORT", "587")) if st.secrets.get("SMTP_PORT") else 587
SMTP_USER = st.secrets.get("SMTP_USER", "")
SMTP_PASS = st.secrets.get("SMTP_PASS", "")
ALERT_EMAIL_TO = st.secrets.get("ALERT_EMAIL_TO", "")

# State
if "watchlist" not in st.session_state: st.session_state.watchlist = []
if "alerts_log" not in st.session_state: st.session_state.alerts_log = []
if "cooldown" not in st.session_state: st.session_state.cooldown = {}  # key -> last_ts

# ------------- CSS / Theme -------------
def _apply_css(font_px=18, high_contrast=False, dark=True):
    base_bg = "#0d1117" if dark else "#ffffff"
    base_fg = "#e6edf3" if dark else "#111111"
    accent  = "#22c55e" if not high_contrast else "#ffdd00"
    ring    = "#16a34a" if not high_contrast else "#ffaa00"

    st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
        background:{base_bg};
        color:{base_fg};
        font-size:{font_px}px !important;
    }}
    .phase-banner {{
        font-size:{round(font_px*1.1)}px;
        font-weight:900;
        text-align:center;
        background:linear-gradient(90deg, {accent}, #15803d);
        color:white;
        border-radius:14px;
        padding:.6rem .8rem;
        margin-bottom:1rem;
    }}
    .byline {{ display:block; font-size:.9em; font-weight:700; letter-spacing:.3px; opacity:.9; margin-top:.25rem; }}
    .chip {{ display:inline-block; padding:.25rem .6rem; border-radius:999px; font-weight:700; margin-right:.4rem; }}
    .ok {{ background:#86efac; color:#111; }}
    .warn {{ background:#fde047; color:#111; }}
    .err {{ background:#fda4af; color:#111; }}
    .wl-card {{
        border:1px solid #334155; border-radius:10px; padding:.55rem .8rem; margin:.3rem .3rem; display:inline-block;
        background:rgba(148,163,184,.08)
    }}
    .wl-badge {{ font-weight:800; padding:.1rem .45rem; border-radius:6px; }}
    .wl-pos {{ background:#86efac; color:#111; }}
    .wl-neg {{ background:#fca5a5; color:#111; }}
    .card {{
        border:1px solid #334155; border-radius:10px; padding:.75rem 1rem; margin:.4rem 0;
        background:rgba(148,163,184,.08)
    }}
    </style>
    """, unsafe_allow_html=True)

_apply_css()

# ------------- Sidebar -------------
st.sidebar.header("🧭 Navigation")
nav = st.sidebar.radio("Go to", ["Dashboard", "Crypto", "Stocks", "FX", "Sentiment", "Signal Center", "Export"], index=0)

st.sidebar.subheader("⚙️ Appearance")
font_size = st.sidebar.slider("Font size", 14, 24, 18)
contrast = st.sidebar.toggle("High contrast mode", value=False)
_apply_css(font_size, contrast, True)

st.sidebar.subheader("⭐ Watchlist")
wl_add = st.sidebar.text_input("Add symbol (BTC, ETH, AAPL…)", "")
if st.sidebar.button("Add"):
    s = wl_add.strip().upper()
    if s and s not in st.session_state.watchlist:
        st.session_state.watchlist.append(s)
if st.session_state.watchlist:
    st.sidebar.caption("Saved:")
    st.sidebar.write(", ".join(st.session_state.watchlist))

st.sidebar.subheader("🔁 Refresh")
auto_refresh = st.sidebar.toggle("Auto Refresh", value=False)
every = st.sidebar.slider("Every (sec)", 10, 120, 30, 5)
if auto_refresh: time.sleep(every)

# ------------- Weights / Scores (Fusion 2.0) -------------
# Base LIPE weights:
BASE_W = dict(w_vol=0.30, w_m24=0.25, w_m7=0.20, w_liq=0.15)
# Fusion 2.0 adds volatility + sentiment (+ fundamentals if present)
FUSION_W = dict(w_lipe=0.70, w_volatility=0.10, w_sent=0.10, w_fund=0.10)

def _sig(x):
    if pd.isna(x): return 0.5
    return 1/(1+math.exp(-float(x)/10.0))

def _norm(w):
    s = sum(max(0,v) for v in w.values()) or 1.0
    return {k:max(0,v)/s for k,v in w.items()}

def safe_get(url, params=None, t=25):
    try:
        r = requests.get(url, params=params, headers=USER_AGENT, timeout=t)
        if r.status_code == 200: return r
    except Exception:
        pass
    return None

@st.cache_data(ttl=60)
def cg_markets(vs="usd", limit=150):
    p = {"vs_currency":vs,"order":"market_cap_desc","per_page":limit,"page":1,
         "sparkline":"false","price_change_percentage":"1h,24h,7d"}
    r = safe_get(CG_API, p)
    return pd.DataFrame(r.json()) if r else pd.DataFrame()

# Volatility proxy (no heavy history call): combine |1d %| and |7d %|
def quick_volatility_01(df):
    p1 = df.get("price_change_percentage_24h_in_currency", pd.Series(0)).abs()
    p7 = df.get("price_change_percentage_7d_in_currency", pd.Series(0)).abs()
    # Simple cap at 20% to avoid outliers, then normalize 0..1
    v = (p1.clip(0, 20) + 0.5*p7.clip(0, 40)) / 30.0
    return v.clip(0, 1).fillna(0.3)

# Fundamentals (optional Alpha Vantage Overview: PE, ProfitMargin) → normalize to 0..1
@st.cache_data(ttl=900)
def alpha_overview(symbol):
    if not ALPHA_V_KEY: return {}
    try:
        url = "https://www.alphavantage.co/query"
        p = {"function":"OVERVIEW", "symbol":symbol, "apikey":ALPHA_V_KEY}
        r = requests.get(url, params=p, timeout=20)
        if r.status_code != 200: return {}
        js = r.json()
        return js if isinstance(js, dict) else {}
    except Exception:
        return {}

def fund_score(symbol):
    """Earnings yield proxy from PE + profit margin signal. 0..1, fallback 0.5."""
    try:
        if not ALPHA_V_KEY: return 0.5
        ov = alpha_overview(symbol)
        pe  = float(ov.get("PERatio", "0") or 0)
        pm  = float(ov.get("ProfitMargin", "0") or 0)  # fraction
        # Earnings yield ~ 1/PE, clamp for sanity
        earn_yield = (1/pe) if pe>0 else 0.0
        earn_yield = min(max(earn_yield, 0.0), 0.2) / 0.2  # 0..1 assuming 5%+ is strong
        pm01 = min(max(pm, -0.1), 0.3)  # clamp -10%..30%
        pm01 = (pm01 + 0.1) / 0.4      # map to 0..1 baseline
        return float(0.6*earn_yield + 0.4*pm01)
    except Exception:
        return 0.5

def lipe_truth(df, w=None):
    w = _norm(w or BASE_W)
    if "liquidity01" not in df: df["liquidity01"] = 0.0
    if "vol_to_mc" not in df:
        vol = df.get("total_volume", pd.Series(0, index=df.index))
        v01 = (vol - vol.min()) / (vol.max() - vol.min() + 1e-9)
        df["vol_to_mc"] = 2*v01
    return (
        w["w_vol"]*(df["vol_to_mc"]/2).clip(0,1) +
        w["w_m24"]*df.get("momo_24h01",0.5) +
        w["w_m7"] *df.get("momo_7d01",0.5) +
        w["w_liq"]*df["liquidity01"]
    ).clip(0,1)

# Sentiment (RSS → TextBlob polarity)
@st.cache_data(ttl=300, show_spinner=False)
def rss_sentiment():
    if not FP_OK:
        return 0.5, []
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://news.bitcoin.com/feed/",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
    ]
    titles=[]
    for url in feeds:
        try:
            d = feedparser.parse(url)
            for e in d.entries[:20]:
                titles.append(e.title)
        except Exception:
            continue
    if not titles:
        return 0.5, []
    if TB_OK:
        pol = np.mean([TextBlob(t).sentiment.polarity for t in titles])
        return float((pol+1)/2), titles[:40]
    return 0.5, titles[:40]

NEWS_SENTIMENT, NEWS_TITLES = rss_sentiment()

# Fusion Score 2.0
def fusion_score(df, kind="CRYPTO_OR_STOCK"):
    # 1) Base LIPE truth
    lipe = lipe_truth(df)
    # 2) Volatility (lower vol slightly preferred; invert)
    vol01 = quick_volatility_01(df)
    vol_component = (1.0 - vol01).clip(0,1)
    # 3) Sentiment (shared market mood for now)
    sent_component = float(NEWS_SENTIMENT)
    # 4) Fundamentals (only for stocks symbols we can query)
    if kind.startswith("STOCK"):
        f = []
        for s in df["symbol"].astype(str).str.upper().tolist():
            f.append(fund_score(s))
        fund_component = pd.Series(f, index=df.index).clip(0,1)
    else:
        fund_component = pd.Series(0.5, index=df.index)

    W = _norm(FUSION_W)
    fused = (
        W["w_lipe"]*lipe +
        W["w_volatility"]*vol_component +
        W["w_sent"]*sent_component +
        W["w_fund"]*fund_component
    ).clip(0,1)
    return fused

# ------------- Builders -------------
def build_crypto(vs="usd", limit=150):
    df = cg_markets(vs, limit)
    if df.empty: return df
    df["vol_to_mc"] = (df["total_volume"]/df["market_cap"]).replace([np.inf,-np.inf],np.nan).clip(0,2).fillna(0)
    df["momo_24h01"] = df["price_change_percentage_24h_in_currency"].apply(_sig)
    df["momo_7d01"]  = df["price_change_percentage_7d_in_currency"].apply(_sig)
    mc = df["market_cap"].fillna(0)
    df["liquidity01"] = 0 if mc.max()==0 else (mc-mc.min())/(mc.max()-mc.min()+1e-9)
    df["truth_full"]  = lipe_truth(df, BASE_W)
    df["raw_heat"]    = (0.5*(df["vol_to_mc"]/2).clip(0,1) + 0.5*df["momo_24h01"]).clip(0,1)
    df["divergence"]  = (df["raw_heat"] - df["truth_full"]).round(3)
    df["symbol"]      = df["symbol"].str.upper()
    df["asset_type"]  = "CRYPTO"
    df["fusion20"]    = fusion_score(df, "CRYPTO")
    return df

def _stocks_demo():
    rows = [
        ("AAPL", 227.4,  0.85),
        ("MSFT", 424.1,  0.65),
        ("NVDA", 114.7, -1.20),
        ("AMZN", 196.0,  0.10),
        ("GOOGL",174.8, -0.40),
    ]
    df = pd.DataFrame(rows, columns=["symbol","current_price","price_change_percentage_24h_in_currency"])
    df["total_volume"]=1_000_000
    df["market_cap"]=np.nan
    df["momo_24h01"]=df["price_change_percentage_24h_in_currency"].apply(_sig)
    df["momo_7d01"]=0.5
    df["vol_to_mc"]=0.5
    df["liquidity01"]=0.5
    df["truth_full"]=lipe_truth(df, BASE_W)
    df["raw_heat"]=(0.5*(df["vol_to_mc"]/2).clip(0,1)+0.5*df["momo_24h01"]).clip(0,1)
    df["divergence"]=(df["raw_heat"]-df["truth_full"]).round(3)
    df["asset_type"]="STOCK"
    df["fusion20"] = fusion_score(df, "STOCK")
    return df

def build_stocks_or_fx(tickers_csv, period="6mo"):
    if not YF_OK:
        return _stocks_demo(), "demo"
    try:
        tickers=[t.strip().upper() for t in tickers_csv.split(",") if t.strip()]
        data = yf.download(tickers, period=period, interval="1d",
                           auto_adjust=True, progress=False, threads=True)
        if data is None or data.empty:
            return _stocks_demo(), "demo"
        prices = data.get("Adj Close", data).ffill()
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(tickers[0])
        last = prices.iloc[-1]
        prev = prices.iloc[-2] if len(prices)>1 else prices.iloc[-1]
        chg24 = (last/prev - 1.0)*100.0
        rows=[]
        for t in prices.columns:
            rows.append({"symbol":t.upper(),
                         "current_price":float(last.get(t,np.nan)),
                         "price_change_percentage_24h_in_currency":float(chg24.get(t,np.nan))})
        df = pd.DataFrame(rows)
        # Lightweight features
        df["total_volume"]=np.nan
        df["market_cap"]=np.nan
        df["momo_24h01"]=df["price_change_percentage_24h_in_currency"].apply(_sig)
        df["momo_7d01"]=0.5
        df["vol_to_mc"]=0.5
        df["liquidity01"]=0.5
        df["truth_full"]=lipe_truth(df, BASE_W)
        df["raw_heat"]=(0.5*(df["vol_to_mc"]/2).clip(0,1)+0.5*df["momo_24h01"]).clip(0,1)
        df["divergence"]=(df["raw_heat"]-df["truth_full"]).round(3)
        df["asset_type"]="STOCK/FX"
        df["fusion20"]=fusion_score(df, "STOCK")
        return df, "live"
    except Exception:
        return _stocks_demo(), "demo"

# ------------- Branding -------------
st.markdown(
    f'<div class="phase-banner">🟢 {APP_TITLE}'
    '<span class="byline">Powered by Jesse Ray Landingham Jr</span></div>',
    unsafe_allow_html=True
)

# ------------- Load per section -------------
vs_currency = "usd"
topn = 150

if nav in ("Dashboard", "Crypto", "Signal Center"):
    crypto_df = build_crypto(vs_currency, topn)
else:
    crypto_df = pd.DataFrame()

if nav in ("Dashboard", "Stocks", "Signal Center"):
    tick_def = "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA"
    tickers = st.sidebar.text_input("Stocks (comma-separated)", tick_def)
    stocks_df, stocks_mode = build_stocks_or_fx(tickers)
else:
    stocks_df, stocks_mode = pd.DataFrame(), "demo"

if nav in ("Dashboard", "FX", "Signal Center"):
    fx_def = "EURUSD=X,USDJPY=X,GBPUSD=X,AUDUSD=X,USDCAD=X"
    fx_pairs = st.sidebar.text_input("FX pairs (Yahoo symbols)", fx_def)
    fx_df, fx_mode = build_stocks_or_fx(fx_pairs)
else:
    fx_df, fx_mode = pd.DataFrame(), "demo"

# ------------- Helpers -------------
def unify_frames(frames):
    cols = ["asset_type","symbol","name","current_price","price_change_percentage_24h_in_currency",
            "total_volume","market_cap","liquidity01","vol_to_mc","momo_24h01","momo_7d01",
            "truth_full","raw_heat","divergence","fusion20"]
    out=[]
    for f in frames:
        if f is not None and not f.empty:
            for c in cols:
                if c not in f.columns: f[c] = np.nan if c!="asset_type" else ""
            out.append(f[cols].copy())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)

def chip(kind, text):
    cls = {"ok":"ok","warn":"warn","err":"err"}.get(kind, "warn")
    st.markdown(f'<span class="chip {cls}">{text}</span>', unsafe_allow_html=True)

# ------------- UI Sections -------------
if nav == "Dashboard":
    st.subheader("📊 Unified Market Snapshot")
    chip("ok", f"Crypto: {'live' if not crypto_df.empty else 'error'}")
    chip("ok" if stocks_mode=="live" else "warn", f"Stocks: {stocks_mode}")
    chip("ok" if fx_mode=="live" else "warn", f"FX: {fx_mode}")
    chip("ok" if NEWS_SENTIMENT>=0.6 else "warn" if NEWS_SENTIMENT>=0.4 else "err", f"Sentiment: {NEWS_SENTIMENT:.2f}")
    uni = unify_frames([crypto_df, stocks_df, fx_df])
    if uni.empty:
        st.error("No data loaded.")
    else:
        st.dataframe(uni.sort_values(["asset_type","fusion20"], ascending=[True,False]),
                     use_container_width=True, height=620)

elif nav == "Crypto":
    st.subheader("🪙 Crypto")
    if crypto_df.empty:
        st.error("No crypto data.")
    else:
        st.dataframe(crypto_df.sort_values("fusion20", ascending=False)[
            ["symbol","name","current_price","price_change_percentage_24h_in_currency",
             "truth_full","raw_heat","divergence","fusion20","liquidity01","vol_to_mc"]],
            use_container_width=True, height=620
        )
        if PLOTLY_OK:
            st.write("Truth vs Raw (size=fusion20)")
            fig = px.scatter(crypto_df, x="truth_full", y="raw_heat", text="symbol",
                             color="fusion20", size="fusion20", color_continuous_scale="Turbo")
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)

elif nav == "Stocks":
    st.subheader("📈 Stocks")
    chip("ok" if stocks_mode=="live" else "warn", f"Source: {stocks_mode}")
    if stocks_df.empty:
        st.error("No stocks data.")
    else:
        st.dataframe(stocks_df.sort_values("fusion20", ascending=False)[
            ["symbol","current_price","price_change_percentage_24h_in_currency",
             "truth_full","raw_heat","divergence","fusion20"]],
            use_container_width=True, height=620
        )

elif nav == "FX":
    st.subheader("💱 FX")
    chip("ok" if fx_mode=="live" else "warn", f"Source: {fx_mode}")
    if fx_df.empty:
        st.error("No FX data.")
    else:
        st.dataframe(fx_df.sort_values("fusion20", ascending=False)[
            ["symbol","current_price","price_change_percentage_24h_in_currency",
             "truth_full","raw_heat","divergence","fusion20"]],
            use_container_width=True, height=620
        )

elif nav == "Sentiment":
    st.subheader("📰 News & Sentiment")
    chip("ok" if NEWS_SENTIMENT>=0.6 else "warn" if NEWS_SENTIMENT>=0.4 else "err",
         f"Aggregate Sentiment: {NEWS_SENTIMENT:.2f}")
    st.progress(min(max(NEWS_SENTIMENT,0.0),1.0))
    if NEWS_TITLES:
        st.markdown("**Recent Headlines (sample):**")
        for t in NEWS_TITLES[:25]:
            st.write("•", t)
    else:
        st.info("Sentiment libraries not installed; showing neutral baseline.")

# ================================== SIGNAL CENTER ==================================
elif nav == "Signal Center":
    st.subheader("🔔 Signal Center — Rules, Alerts, Delivery")
    # Choose source
    src = st.selectbox("Source Table", ["Unified (Crypto+Stocks+FX)","Crypto only","Stocks only","FX only"], index=0)
    # Build base df
    if src == "Crypto only":
        base = crypto_df
    elif src == "Stocks only":
        base = stocks_df
    elif src == "FX only":
        base = fx_df
    else:
        base = unify_frames([crypto_df, stocks_df, fx_df])

    if base is None or base.empty:
        st.warning("No data available for selected source.")
    else:
        cA, cB, cC = st.columns(3)
        with cA:
            thr_fusion = st.slider("Fusion 2.0 ≥", 0.0, 1.0, 0.82, 0.01)
            thr_truth  = st.slider("Truth ≥", 0.0, 1.0, 0.75, 0.01)
        with cB:
            thr_24h    = st.slider("24h % ≥", -10.0, 15.0, 0.0, 0.1)
            thr_absdiv = st.slider("|Divergence| ≥", 0.0, 1.0, 0.28, 0.01)
        with cC:
            min_price  = st.number_input("Min price (optional)", value=0.0, min_value=0.0, step=0.01, format="%.2f")
            only_watch = st.toggle("Filter to Watchlist only", value=False)

        q = base.copy()
        if "price_change_percentage_24h_in_currency" not in q:
            q["price_change_percentage_24h_in_currency"] = 0.0
        q["abs_div"] = q["divergence"].abs()

        mask = (
            (q["fusion20"]>=thr_fusion) &
            (q["truth_full"]>=thr_truth) &
            (q["price_change_percentage_24h_in_currency"]>=thr_24h) &
            (q["abs_div"]>=thr_absdiv) &
            (q["current_price"]>=min_price)
        )
        if only_watch and st.session_state.watchlist:
            mask = mask & (q["symbol"].isin([s.upper() for s in st.session_state.watchlist]))

        hits = q[mask].sort_values(["fusion20","abs_div","truth_full"], ascending=False)
        st.success(f"Matches: {len(hits)}")
        if len(hits):
            st.dataframe(hits[
                ["asset_type","symbol","current_price","fusion20","truth_full","raw_heat","divergence",
                 "price_change_percentage_24h_in_currency"]],
                use_container_width=True, height=420
            )
        else:
            st.info("No matches — tweak thresholds or watchlist filter.")

        st.markdown("---")
        st.subheader("📡 Delivery")
        col1, col2 = st.columns(2)
        with col1:
            discord_url = st.text_input("Discord Webhook URL", value=DISCORD_WEBHOOK, type="password")
            enable_discord = st.toggle("Enable Discord alerts", value=bool(discord_url))
        with col2:
            email_to = st.text_input("Alert Email To", value=ALERT_EMAIL_TO)
            enable_email = st.toggle("Enable Email alerts", value=bool(SMTP_HOST and SMTP_USER and SMTP_PASS and email_to))

        cooldown_min = st.slider("Cooldown (minutes) per symbol", 1, 240, 30, 1)
        send_btn = st.button("Send Alerts Now")

        if send_btn and len(hits):
            sent = 0
            for _, r in hits.iterrows():
                key = f"{r['symbol']}"
                last = st.session_state.cooldown.get(key, None)
                now = datetime.now(timezone.utc)
                if last and (now - last) < timedelta(minutes=cooldown_min):
                    continue  # skip due to cooldown

                msg = (f"CHL v13 Signal — {r['asset_type']} {r['symbol']}\n"
                       f"Price: {r['current_price']:.4g}\n"
                       f"Fusion20: {r['fusion20']:.3f} | Truth: {r['truth_full']:.3f} | Raw: {r['raw_heat']:.3f}\n"
                       f"Δ (Raw-Truth): {r['divergence']:+.3f} | 24h%: {r['price_change_percentage_24h_in_currency']:.2f}%\n"
                       f"Sentiment: {NEWS_SENTIMENT:.2f} | Time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                ok = True
                if enable_discord and discord_url:
                    ok = ok and send_discord(discord_url, msg)
                if enable_email and email_to:
                    ok = ok and send_email(email_to, subject=f"CHL v13 Signal — {r['symbol']}", body=msg)
                if ok:
                    st.session_state.cooldown[key] = now
                    st.session_state.alerts_log.append({"ts": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
                                                        "symbol": r["symbol"],
                                                        "fusion20": float(r["fusion20"]),
                                                        "truth": float(r["truth_full"]),
                                                        "raw": float(r["raw_heat"]),
                                                        "div": float(r["divergence"])})
                    sent += 1
            st.success(f"Alerts sent: {sent}")

        st.markdown("#### 📜 Recent Alerts Log")
        if st.session_state.alerts_log:
            log_df = pd.DataFrame(st.session_state.alerts_log)
            st.dataframe(log_df.sort_values("ts", ascending=False), use_container_width=True, height=240)
        else:
            st.caption("No alert logs yet.")

elif nav == "Export":
    st.subheader("📤 Export Data")
    choice = st.selectbox("Table to export", ["Crypto","Stocks","FX","Unified"])
    if choice == "Crypto":
        data = crypto_df
    elif choice == "Stocks":
        data = stocks_df
    elif choice == "FX":
        data = fx_df
    else:
        data = unify_frames([crypto_df, stocks_df, fx_df])

    if data is None or data.empty:
        st.warning("Nothing to export.")
    else:
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, file_name=f"chl_{choice.lower()}.csv", mime="text/csv")

# ------------- Watchlist Cards -------------
st.markdown("### ⭐ Watchlist Signals")
wl = [s.upper() for s in st.session_state.watchlist]
current_table = unify_frames([crypto_df, stocks_df, fx_df])
if wl and current_table is not None and not current_table.empty:
    sub = current_table[current_table["symbol"].isin(wl)][["symbol","current_price","truth_full","raw_heat","divergence","fusion20"]].copy()
    if sub.empty:
        st.caption("Your watchlist symbols aren’t in the current table yet.")
    else:
        cards=[]
        for _,r in sub.iterrows():
            sign = "wl-pos" if r["divergence"]>=0 else "wl-neg"
            cards.append(
                f'<div class="wl-card"><b>{r["symbol"]}</b> · ${r["current_price"]:.4g} · '
                f'F:{r["fusion20"]:.2f} · T:{r["truth_full"]:.2f} · R:{r["raw_heat"]:.2f} · '
                f'<span class="wl-badge {sign}">Δ {r["divergence"]:+.2f}</span></div>'
            )
        st.markdown("".join(cards), unsafe_allow_html=True)
else:
    st.caption("Add symbols in the sidebar (BTC, ETH, AAPL, etc.)")

# ------------- Footer -------------
st.caption(
    "Sources: CoinGecko (Crypto). Yahoo Finance or demo fallback (Stocks/FX). "
    "RSS: CoinDesk / CoinTelegraph / Bitcoin.com / Yahoo Finance. • "
    "Powered by Jesse Ray Landingham Jr • © 2025"
)

# ================================== DELIVERY HELPERS ==================================
def send_discord(webhook_url: str, message: str) -> bool:
    try:
        r = requests.post(webhook_url, json={"content": message}, timeout=15)
        return r.status_code in (200, 204, 201)
    except Exception:
        return False

def send_email(to_email: str, subject: str, body: str) -> bool:
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and to_email):
        return False
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = to_email
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, [to_email], msg.as_string())
        return True
    except Exception:
        return False
