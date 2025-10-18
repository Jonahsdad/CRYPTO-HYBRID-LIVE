# ============================ PHASE 12 — MULTI-MARKET FUSION (ROBUST) ============================
# Crypto Hybrid Live — Phase 12
# - Unified Crypto + Stocks + FX schema
# - Sidebar navigation
# - Truth / Raw / Fusion scoring
# - Sentiment (RSS + TextBlob when available)
# - Alerts (rule-based) + Watchlist
# - yfinance fallback so Stocks/FX never break
# - Branding: Powered by Jesse Ray Landingham Jr
# ================================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests, math, time
from datetime import datetime, timezone

# Optional libs (never crash)
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


# ---------------------------- APP CONFIG ----------------------------
APP_TITLE = "Crypto Hybrid Live — Phase 12 (Multi-Market Fusion)"
st.set_page_config(page_title=APP_TITLE, layout="wide")
USER_AGENT = {"User-Agent": "Mozilla/5.0 (CHL-Phase12)"}
CG_API = "https://api.coingecko.com/api/v3/coins/markets"

if "watchlist" not in st.session_state: st.session_state.watchlist = []
if "alerts_log" not in st.session_state: st.session_state.alerts_log = []

# ---------------------------- CSS THEME -----------------------------
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
    .byline {{
        display:block; font-size:.9em; font-weight:700; letter-spacing:.3px;
        opacity:.9; margin-top:.25rem;
    }}
    .chip {{ display:inline-block; padding:.25rem .6rem; border-radius:999px; font-weight:700; margin-right:.4rem; }}
    .ok {{ background:#86efac; color:#111; }}
    .warn {{ background:#fde047; color:#111; }}
    .err {{ background:#fda4af; color:#111; }}
    .card {{
        border:1px solid #334155; border-radius:10px; padding:.75rem 1rem; margin:.4rem 0;
        background:rgba(148,163,184,.08)
    }}
    .wl-card {{
        border:1px solid #334155; border-radius:10px; padding:.55rem .8rem; margin:.3rem .3rem; display:inline-block;
        background:rgba(148,163,184,.08)
    }}
    .wl-badge {{ font-weight:800; padding:.1rem .45rem; border-radius:6px; }}
    .wl-pos {{ background:#86efac; color:#111; }}
    .wl-neg {{ background:#fca5a5; color:#111; }}
    </style>
    """, unsafe_allow_html=True)

_apply_css()

# ---------------------------- SIDEBAR NAV ---------------------------
st.sidebar.header("🧭 Navigation")
nav = st.sidebar.radio("Go to", ["Dashboard", "Crypto", "Stocks", "FX", "Sentiment", "Alerts", "Export"], index=0)

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

# ---------------------------- WEIGHTS / SCORING ---------------------
DEFAULT_WEIGHTS = dict(w_vol=0.30, w_m24=0.25, w_m7=0.25, w_liq=0.20)

def _sig(x):
    if pd.isna(x): return 0.5
    return 1/(1+math.exp(-float(x)/10.0))

def _norm(w):
    s = sum(max(0,v) for v in w.values()) or 1.0
    return {k:max(0,v)/s for k,v in w.items()}

def lipe_truth(df, w=None):
    w = _norm(w or DEFAULT_WEIGHTS)
    if "liquidity01" not in df: df["liquidity01"] = 0.0
    if "vol_to_mc" not in df:
        vol = df.get("total_volume", pd.Series(0, index=df.index))
        v01 = (vol - vol.min()) / (vol.max() - vol.min() + 1e-9)
        df["vol_to_mc"] = 2 * v01
    return (
        w["w_vol"] * (df["vol_to_mc"]/2).clip(0,1) +
        w["w_m24"] * df.get("momo_24h01", 0.5) +
        w["w_m7"]  * df.get("momo_7d01", 0.5) +
        w["w_liq"] * df["liquidity01"]
    ).clip(0,1)

# ---------------------------- HELPERS -------------------------------
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

def build_crypto(vs="usd", limit=150):
    df = cg_markets(vs, limit)
    if df.empty: return df
    df["vol_to_mc"] = (df["total_volume"]/df["market_cap"]).replace([np.inf,-np.inf],np.nan).clip(0,2).fillna(0)
    df["momo_24h01"] = df["price_change_percentage_24h_in_currency"].apply(_sig)
    df["momo_7d01"]  = df["price_change_percentage_7d_in_currency"].apply(_sig)
    mc = df["market_cap"].fillna(0)
    df["liquidity01"] = 0 if mc.max()==0 else (mc-mc.min())/(mc.max()-mc.min()+1e-9)
    df["truth_full"]  = lipe_truth(df, DEFAULT_WEIGHTS)
    df["raw_heat"]    = (0.5*(df["vol_to_mc"]/2).clip(0,1) + 0.5*df["momo_24h01"]).clip(0,1)
    df["divergence"]  = (df["raw_heat"] - df["truth_full"]).round(3)
    df["symbol"]      = df["symbol"].str.upper()
    df["asset_type"]  = "CRYPTO"
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
    df["truth_full"]=lipe_truth(df, DEFAULT_WEIGHTS)
    df["raw_heat"]=(0.5*(df["vol_to_mc"]/2).clip(0,1)+0.5*df["momo_24h01"]).clip(0,1)
    df["divergence"]=(df["raw_heat"]-df["truth_full"]).round(3)
    df["asset_type"]="STOCK"
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
        df["total_volume"]=np.nan
        df["market_cap"]=np.nan
        df["momo_24h01"]=df["price_change_percentage_24h_in_currency"].apply(_sig)
        df["momo_7d01"]=0.5
        df["vol_to_mc"]=0.5
        df["liquidity01"]=0.5
        df["truth_full"]=lipe_truth(df, DEFAULT_WEIGHTS)
        df["raw_heat"]=(0.5*(df["vol_to_mc"]/2).clip(0,1)+0.5*df["momo_24h01"]).clip(0,1)
        df["divergence"]=(df["raw_heat"]-df["truth_full"]).round(3)
        df["asset_type"]="STOCK/FX"
        return df, "live"
    except Exception:
        return _stocks_demo(), "demo"

# ---------------------------- SENTIMENT -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def rss_sentiment():
    if not FP_OK:
        return 0.5, []
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://news.bitcoin.com/feed/",
        # Add general markets:
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

# ---------------------------- BRANDING ------------------------------
st.markdown(
    f'<div class="phase-banner">🟢 {APP_TITLE}'
    '<span class="byline">Powered by Jesse Ray Landingham Jr</span></div>',
    unsafe_allow_html=True
)

# ---------------------------- DATA LOAD PER SECTION -----------------
vs_currency = "usd"
topn = 150

if nav in ("Dashboard", "Crypto"):
    crypto_df = build_crypto(vs_currency, topn)
else:
    crypto_df = pd.DataFrame()

if nav in ("Dashboard", "Stocks"):
    tickers_default = "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA"
    tickers = st.sidebar.text_input("Stocks (comma-separated)", tickers_default)
    stocks_df, stocks_mode = build_stocks_or_fx(tickers)
else:
    stocks_df, stocks_mode = pd.DataFrame(), "demo"

if nav == "FX" or nav == "Dashboard":
    fx_default = "EURUSD=X,USDJPY=X,GBPUSD=X,AUDUSD=X,USDCAD=X"
    fx_pairs = st.sidebar.text_input("FX pairs (Yahoo symbols)", fx_default)
    fx_df, fx_mode = build_stocks_or_fx(fx_pairs)
else:
    fx_df, fx_mode = pd.DataFrame(), "demo"

news_s, titles = rss_sentiment()

# ---------------------------- UNIFIED VIEW --------------------------
def unify_frames(frames):
    cols = ["asset_type","symbol","name","current_price","price_change_percentage_24h_in_currency",
            "total_volume","market_cap","liquidity01","vol_to_mc","momo_24h01","momo_7d01",
            "truth_full","raw_heat","divergence"]
    out=[]
    for f in frames:
        if f is not None and not f.empty:
            for c in cols:
                if c not in f.columns: f[c] = np.nan if c!="asset_type" else ""
            out.append(f[cols].copy())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)

if nav == "Dashboard":
    uni = unify_frames([crypto_df, stocks_df, fx_df])
else:
    uni = pd.DataFrame()

# ---------------------------- STATUS CHIPS --------------------------
def chip(kind, text):
    cls = {"ok":"ok","warn":"warn","err":"err"}.get(kind, "warn")
    st.markdown(f'<span class="chip {cls}">{text}</span>', unsafe_allow_html=True)

# ---------------------------- LAYOUT -------------------------------
if nav == "Dashboard":
    st.subheader("📊 Unified Market Snapshot")
    chip("ok", f"Crypto: {'live' if not crypto_df.empty else 'error'}")
    chip("ok" if stocks_mode=="live" else "warn", f"Stocks: {stocks_mode}")
    chip("ok" if fx_mode=="live" else "warn", f"FX: {fx_mode}")
    chip("ok" if news_s>=0.6 else "warn" if news_s>=0.4 else "err", f"Sentiment: {news_s:.2f}")
    st.write("")
    if uni.empty:
        st.error("No data loaded.")
    else:
        st.dataframe(uni.sort_values(["asset_type","truth_full"], ascending=[True,False]),
                     use_container_width=True, height=600)

elif nav == "Crypto":
    st.subheader("🪙 Crypto (Truth / Raw / Fusion-ready)")
    if crypto_df.empty:
        st.error("No crypto data.")
    else:
        st.dataframe(crypto_df.sort_values("truth_full", ascending=False)[
            ["symbol","name","current_price","price_change_percentage_24h_in_currency",
             "truth_full","raw_heat","divergence","liquidity01","vol_to_mc"]],
            use_container_width=True, height=620
        )
        if PLOTLY_OK:
            st.write("Truth vs Raw")
            fig = px.scatter(crypto_df, x="truth_full", y="raw_heat", text="symbol",
                             color="divergence", color_continuous_scale="Turbo")
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)

elif nav == "Stocks":
    st.subheader("📈 Stocks (with fallback)")
    chip("ok" if stocks_mode=="live" else "warn", f"Source: {stocks_mode}")
    if stocks_df.empty:
        st.error("No stocks data.")
    else:
        st.dataframe(stocks_df.sort_values("truth_full", ascending=False)[
            ["symbol","current_price","price_change_percentage_24h_in_currency",
             "truth_full","raw_heat","divergence"]],
            use_container_width=True, height=620
        )

elif nav == "FX":
    st.subheader("💱 FX (via Yahoo symbols)")
    chip("ok" if fx_mode=="live" else "warn", f"Source: {fx_mode}")
    if fx_df.empty:
        st.error("No FX data.")
    else:
        st.dataframe(fx_df.sort_values("truth_full", ascending=False)[
            ["symbol","current_price","price_change_percentage_24h_in_currency",
             "truth_full","raw_heat","divergence"]],
            use_container_width=True, height=620
        )

elif nav == "Sentiment":
    st.subheader("📰 News & Sentiment")
    chip("ok" if news_s>=0.6 else "warn" if news_s>=0.4 else "err", f"Aggregate Sentiment: {news_s:.2f}")
    st.progress(min(max(news_s,0.0),1.0))
    st.write("")
    if titles:
        st.markdown("**Recent Headlines (sample):**")
        for t in titles[:25]:
            st.write("•", t)
    else:
        st.info("Sentiment libraries not installed; showing neutral baseline.")

elif nav == "Alerts":
    st.subheader("🔔 Rules & Alerts (AND conditions)")
    colA, colB = st.columns(2)
    with colA:
        source_pick = st.selectbox("Source Table", ["Unified","Crypto only","Stocks only","FX only"], index=0)
        thr_truth = st.slider("Truth ≥", 0.0, 1.0, 0.85, 0.01)
        thr_24h   = st.slider("24h % ≥", -10.0, 10.0, 0.0, 0.1)
    with colB:
        thr_absdiv = st.slider("|Divergence| ≥", 0.0, 1.0, 0.30, 0.01)
        min_price  = st.number_input("Min price (optional)", value=0.0, min_value=0.0, step=0.01, format="%.2f")
        only_watch = st.toggle("Filter to Watchlist only", value=False)

    # choose df
    if source_pick == "Crypto only":
        base = crypto_df
    elif source_pick == "Stocks only":
        base = stocks_df
    elif source_pick == "FX only":
        base = fx_df
    else:
        base = uni if not uni.empty else unify_frames([crypto_df, stocks_df, fx_df])

    if base is None or base.empty:
        st.warning("No data available for selected source.")
    else:
        q = base.copy()
        if "price_change_percentage_24h_in_currency" not in q:
            q["price_change_percentage_24h_in_currency"] = 0.0
        q["abs_div"] = q["divergence"].abs()
        mask = (
            (q["truth_full"]>=thr_truth) &
            (q["price_change_percentage_24h_in_currency"]>=thr_24h) &
            (q["abs_div"]>=thr_absdiv) &
            (q["current_price"]>=min_price)
        )
        if only_watch and st.session_state.watchlist:
            mask = mask & (q["symbol"].isin([s.upper() for s in st.session_state.watchlist]))

        hits = q[mask].sort_values(["truth_full","abs_div"], ascending=False)
        st.success(f"Matches: {len(hits)}")
        if len(hits):
            st.dataframe(hits[
                ["asset_type","symbol","current_price","truth_full","raw_heat","divergence",
                 "price_change_percentage_24h_in_currency"]],
                use_container_width=True, height=440
            )
            if st.button("Log Alert Batch"):
                stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                st.session_state.alerts_log.append({"ts":stamp, "count":int(len(hits))})
                st.success(f"Logged {len(hits)} alerts at {stamp}")
        else:
            st.info("No matches yet—tweak thresholds.")

    st.markdown("#### Recent Alert Batches")
    if st.session_state.alerts_log:
        st.dataframe(pd.DataFrame(st.session_state.alerts_log), use_container_width=True, height=200)
    else:
        st.caption("No alert batches logged yet.")

elif nav == "Export":
    st.subheader("📤 Export Data")
    choice = st.selectbox("Table to export", ["Unified","Crypto","Stocks","FX"])
    if choice == "Unified":
        data = uni if not uni.empty else unify_frames([crypto_df, stocks_df, fx_df])
    elif choice == "Crypto":
        data = crypto_df
    elif choice == "Stocks":
        data = stocks_df
    else:
        data = fx_df
    if data is None or data.empty:
        st.warning("Nothing to export.")
    else:
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, file_name=f"chl_{choice.lower()}.csv", mime="text/csv")

# ---------------------------- WATCHLIST CARDS -----------------------
st.markdown("### ⭐ Watchlist Signals")
wl = [s.upper() for s in st.session_state.watchlist]
current_table = uni if not uni.empty else pd.concat([df for df in [crypto_df, stocks_df, fx_df] if df is not None and not df.empty], ignore_index=True) if any([(crypto_df is not None and not crypto_df.empty), (stocks_df is not None and not stocks_df.empty), (fx_df is not None and not fx_df.empty)]) else pd.DataFrame()
if wl and current_table is not None and not current_table.empty:
    sub = current_table[current_table["symbol"].isin(wl)][["symbol","current_price","truth_full","raw_heat","divergence"]].copy()
    if sub.empty:
        st.caption("Your watchlist symbols aren’t in the current table yet.")
    else:
        cards=[]
        for _,r in sub.iterrows():
            sign = "wl-pos" if r["divergence"]>=0 else "wl-neg"
            cards.append(
                f'<div class="wl-card"><b>{r["symbol"]}</b> · ${r["current_price"]:.4g} · '
                f'T:{r["truth_full"]:.2f} · R:{r["raw_heat"]:.2f} · '
                f'<span class="wl-badge {sign}">Δ {r["divergence"]:+.2f}</span></div>'
            )
        st.markdown("".join(cards), unsafe_allow_html=True)
else:
    st.caption("Add symbols in the sidebar (BTC, ETH, AAPL, etc.)")

# ---------------------------- FOOTER -------------------------------
st.caption(
    "Sources: CoinGecko (Crypto). Yahoo Finance or demo fallback (Stocks/FX). "
    "RSS: CoinDesk / CoinTelegraph / Bitcoin.com / Yahoo Finance. • "
    "Powered by Jesse Ray Landingham Jr • © 2025"
)
