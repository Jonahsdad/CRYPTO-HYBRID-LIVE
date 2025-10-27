# HYBRID INTELLIGENCE SYSTEMS - Neon Hero + Arena Tiles
# One-file Streamlit app. Restores your hero banner + tile home.
# Adds free-data integrations with caching, retries, and fallbacks.
# No blockchain; no paid vendors required.

import os, json, math, time, random
from time import monotonic, sleep
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import requests

# Optional plotting: use Plotly if available, else built-ins
try:
    import plotly.express as px
except Exception:
    px = None

# --------------- Page config ---------------
st.set_page_config(page_title="Hybrid Intelligence Systems",
                   page_icon="🧠", layout="wide")

# Keep all UI state in session
if "route" not in st.session_state:
    st.session_state.route = "home"    # "home" or one of: lottery/crypto/stocks/options/sports/real_estate/commodities/behavior/astrology
if "ledger" not in st.session_state:
    st.session_state.ledger = []       # run history
if "truth_filter" not in st.session_state:
    st.session_state.truth_filter = 55
if "astro_on" not in st.session_state:
    st.session_state.astro_on = False

# --------------- API client (cached + retries) ---------------
@st.cache_resource(show_spinner=False)
def _shared_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "HIS/1.0"})
    return s

def _call(method, url, *, params=None, json_body=None, headers=None, timeout=12, tries=3):
    sess = _shared_session()
    last = None
    for k in range(tries):
        t0 = monotonic()
        try:
            if method == "GET":
                r = sess.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
            else:
                r = sess.post(url, params=params or {}, json=json_body, headers=headers or {}, timeout=timeout)
            r.raise_for_status()
            dt = int((monotonic() - t0) * 1000)
            try:
                return r.json(), dt
            except Exception:
                return {"_raw": r.text}, dt
        except Exception as e:
            last = e
            sleep(0.2 + 0.1*random.random())
    raise last

@st.cache_data(ttl=60, show_spinner=False)
def http_get(url, params=None, headers=None, timeout=12):
    return _call("GET", url, params=params, headers=headers, timeout=timeout)

@st.cache_data(ttl=60, show_spinner=False)
def http_post(url, params=None, json_body=None, headers=None, timeout=12):
    return _call("POST", url, params=params, json_body=json_body, headers=headers, timeout=timeout)

def fresh(ms):
    if ms is None: return "fresh: n/a"
    if ms < 300: return f"{ms} ms • fresh"
    if ms < 1200: return f"{ms} ms"
    return f"{ms} ms (slow)"

# --------------- Global dials sidebar (minimal) ---------------
with st.sidebar:
    st.markdown("**System**")
    st.toggle("Enable Astrology Influence", key="astro_on", value=st.session_state.astro_on)
    st.slider("Truth Filter", 0, 100, key="truth_filter")
    st.divider()
    st.caption("All data via free public APIs • secure mode")

# --------------- Hero banner ---------------
def hero():
    st.markdown(
        """
        <div style="padding:24px;border-radius:14px;background:radial-gradient(80% 120% at 50% 0%, rgba(255,0,150,0.15), rgba(0,150,255,0.08));border:1px solid rgba(255,255,255,0.08)">
          <div style="text-align:center;">
            <div style="font-size:36px;font-weight:800;letter-spacing:1px;">HYBRID INTELLIGENCE SYSTEMS</div>
            <div style="margin-top:6px;font-size:14px;opacity:.85;">
              Powered by <b>JESSE RAY LANDINGHAM JR</b> • Hybrid Local/Remote • LIPE-Core
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------- Arena tiles (home) ---------------
def home_tiles():
    st.write("")
    st.markdown("#### Choose your arena")
    def tile(name, sub, icon, route):
        with st.container(border=True):
            cols = st.columns([1,7,2])
            cols[0].markdown(icon)
            cols[1].markdown(f"**{name}**  \n{sub}")
            if cols[2].button("Open", use_container_width=True, key=f"go_{route}"):
                st.session_state.route = route
                st.experimental_rerun()

    c1, c2, c3 = st.columns(3)
    with c1:
        tile("Lottery", "Daily numbers, picks, risk modes", "🎲", "lottery")
        tile("Options", "Chains and quick IV views", "🧾", "options")
        tile("Sports", "Moneylines and overlays", "🏈", "sports")
    with c2:
        tile("Crypto", "Live prices and pulse", "💰", "crypto")
        tile("Real Estate", "Rates and tilt metrics", "🏡", "real_estate")
        tile("Human Behavior", "Crowd pulse and topics", "🧠", "behavior")
    with c3:
        tile("Stocks", "Charts and headlines", "📈", "stocks")
        tile("Commodities", "Energy and metals", "🛢️", "commodities")
        tile("Astrology", "Symbolic meta-layer", "🌌", "astrology")

# --------------- Back button + header helper ---------------
def page_header(title, emoji):
    cols = st.columns([6,1])
    with cols[0]:
        st.markdown(f"### {emoji} {title}")
    with cols[1]:
        if st.button("Back", use_container_width=True):
            st.session_state.route = "home"
            st.experimental_rerun()

# --------------- Ledger helper ---------------
def log_run(module, **fields):
    rec = {"ts": datetime.now().isoformat(timespec="seconds"),
           "module": module, **fields}
    st.session_state.ledger.append(rec)

def show_ledger():
    if not st.session_state.ledger:
        st.caption("Run something to populate the ledger.")
        return
    df = pd.DataFrame(st.session_state.ledger)
    st.dataframe(df.tail(200), use_container_width=True)
    st.download_button("Download Run Ledger (CSV)",
                       data=df.to_csv(index=False).encode("utf-8"),
                       file_name="his_run_ledger.csv",
                       mime="text/csv")

# --------------- Astrology influence (simple) ---------------
def astro_adjust(conf, strength=0.25):
    if not st.session_state.astro_on:
        return float(conf), {"cosmic_index": 0.0}
    # lightweight synthetic index from time
    z = (math.sin(time.time()/3600.0) + 1.0) / 2.0  # 0..1 over the day
    adj = 0.5 + (conf - 0.5) * (1.0 + strength*(z - 0.5))
    return float(max(0.0, min(1.0, adj))), {"cosmic_index": round(z, 3)}

def truth_adjust(conf):
    tf = st.session_state.truth_filter / 100.0
    return 0.5 + (conf - 0.5) * (0.6 + 0.4 * tf)

# --------------- PAGES ---------------

# Lottery (free scrape + CSV API example)
def page_lottery():
    page_header("Lottery", "🎲")
    st.caption("Auto-fetch latest draws (examples). Upload your CSV to run full local analysis.")
    colA, colB = st.columns(2)

    with colA:
        st.write("**Illinois Pick-4 (LotteryUSA)**")
        try:
            html, ms = http_get("https://www.lotteryusa.com/illinois/pick-4/")
            # parse numbers if HTML came back
            if isinstance(html, dict): html = html.get("_raw","")
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            nums = soup.select_one(".c-results-card__numbers")
            draw = nums.get_text(strip=True) if nums else "unavailable"
            st.success(f"Latest: {draw}")
            st.caption("LotteryUSA • " + fresh(ms))
        except Exception as e:
            st.warning(f"IL fetch failed: {e}")

    with colB:
        st.write("**New York Take 5 (open data)**")
        try:
            rows, ms2 = http_get("https://data.ny.gov/resource/d6yy-54nr.json?$limit=1&$order=draw_date DESC")
            if isinstance(rows, list) and rows:
                row = rows[0]
                st.success(f"{row.get('draw_date','')}: {row.get('winning_numbers','')}")
                st.caption("NY open data • " + fresh(ms2))
        except Exception:
            st.info("NY feed unavailable.")

    st.divider()
    st.write("**Run Forecast (demo logic)**")
    pasted = st.text_input("Paste comma-separated prior draws (e.g., 1,2,3,4,5)", "")
    if st.button("Run Lottery Forecast", type="primary"):
        draws = [int(x.strip()) for x in pasted.split(",") if x.strip().isdigit()]
        if not draws:
            st.info("Provide some numbers first.")
        else:
            # toy model: mean-reversion toward 22 with noise
            pred = int(np.clip(round(np.mean(draws)*0.7 + 22*0.3 + random.random()*3), 0, 99))
            conf = 0.62
            conf, astro = astro_adjust(conf)
            conf = truth_adjust(conf)
            st.success(f"Top pick: {pred}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", f"{int(conf*100)}%")
            c2.metric("Entropy", "n/a")
            c3.metric("Astro", f"{astro['cosmic_index']}")
            log_run("lottery", pick=pred, conf=float(conf), astro=astro["cosmic_index"])

    st.divider(); st.write("**Vault / Ledger**"); show_ledger()

# Crypto (CoinGecko + Binance public)
def page_crypto():
    page_header("Crypto", "💰")
    ids = st.text_input("CoinGecko IDs", "bitcoin,ethereum,solana,dogecoin").replace(" ","")

    try:
        data, ms = http_get("https://api.coingecko.com/api/v3/coins/markets",
                            params={"vs_currency":"usd","ids":ids,"order":"market_cap_desc","per_page":50,"page":1,"sparkline":"false"})
        df = pd.DataFrame(data)[["name","symbol","current_price","price_change_percentage_24h","market_cap"]]
        st.dataframe(df, use_container_width=True)
        st.caption("CoinGecko • " + fresh(ms))
    except Exception as e:
        st.warning(f"CoinGecko error: {e}")

    # Optional Binance spot 24h
    try:
        raw, ms2 = http_get("https://api.binance.com/api/v3/ticker/24hr")
        if isinstance(raw, list):
            tickers = [f"{sym.upper()}USDT" for sym in [x for x in ids.split(",") if x][:5]]
            bd = pd.DataFrame(raw)
            sel = bd[bd["symbol"].isin(tickers)][["symbol","lastPrice","priceChangePercent","quoteVolume"]]
            if not sel.empty:
                st.write("**Binance 24h (spot)**")
                st.dataframe(sel, use_container_width=True)
                st.caption("Binance • " + fresh(ms2))
    except Exception:
        pass

# Stocks (yfinance + Yahoo RSS)
def page_stocks():
    page_header("Stocks", "📈")
    tickers = st.text_input("Tickers (comma)", "AAPL,MSFT,NVDA")
    period  = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y"], index=2)
    interval= st.selectbox("Interval", ["1d","1h","30m"], index=0)

    import yfinance as yf
    for t in [x.strip().upper() for x in tickers.split(",") if x.strip()]:
        try:
            with st.spinner(f"Loading {t}"):
                data = yf.download(t, period=period, interval=interval, progress=False)
            if data.empty:
                st.info(f"No data for {t}."); continue
            df = data.reset_index()
            if px:
                fig = px.line(df, x=df.columns[0], y="Close",
                              title=f"{t} Close — {period}/{interval}", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(df.set_index(df.columns[0])["Close"])
        except Exception as e:
            st.warning(f"{t} failed: {e}")

    # Simple Yahoo Finance RSS for AAPL
    try:
        rss, ms = http_get("https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US")
        if isinstance(rss, dict): rss = rss.get("_raw","")
        if isinstance(rss, str) and "<item>" in rss:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(rss, "xml")
            items = soup.select("item")[:5]
            st.write("**Latest headlines (Yahoo RSS)**")
            for it in items:
                st.markdown(f"- [{it.title.text}]({it.link.text})")
            st.caption("Yahoo RSS • " + fresh(ms))
    except Exception:
        pass

# Options (yfinance chains)
def page_options():
    page_header("Options", "🧾")
    import yfinance as yf
    ticker = st.text_input("Underlying", "AAPL")
    if st.button("Load Chain", type="primary"):
        try:
            tk = yf.Ticker(ticker)
            exps = tk.options
            if not exps: st.info("No expirations."); return
            sel = st.selectbox("Expiration", exps, index=0)
            chain = tk.option_chain(sel)
            calls, puts = chain.calls, chain.puts
            st.write("**Calls**"); st.dataframe(calls.head(30), use_container_width=True)
            st.write("**Puts**");  st.dataframe(puts.head(30), use_container_width=True)
            if "impliedVolatility" in calls.columns:
                ivp = float(calls["impliedVolatility"].dropna().mean()) * 100
                st.metric("Avg Call IV (proxy)", f"{ivp:.1f}%")
        except Exception as e:
            st.error(f"Chain error: {e}")

# Sports (Odds + heuristic pick)
def page_sports():
    page_header("Sports", "🏈")
    sport_key = st.selectbox("Sport", ["basketball_nba","americanfootball_nfl","baseball_mlb","icehockey_nhl","soccer_epl"])
    market = st.selectbox("Market", ["h2h","spreads","totals"])
    col1, col2 = st.columns(2)
    fetch_btn = col1.button("Fetch Odds", type="primary")
    run_btn   = col2.button("Run Forecast")

    if "sports_odds" not in st.session_state
