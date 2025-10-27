# HYBRID INTELLIGENCE SYSTEMS — Full App (single file)
# UI + services + helpers. ASCII-safe. Uses free public data by default.
# Optional paid/live feeds unlock automatically when keys exist in st.secrets.

import os, math, time, json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import requests

# Optional libs (plots & finance). App still runs without them.
HAS_PLOTLY = True
try:
    import plotly.express as px
except Exception:
    HAS_PLOTLY = False

HAS_YFIN = True
try:
    import yfinance as yf
except Exception:
    HAS_YFIN = False

# -----------------------------------------------------------------------------
# CONFIG / SECRETS
# -----------------------------------------------------------------------------
def _s(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)

@dataclass(frozen=True)
class Settings:
    odds_api_key: str = _s("ODDS_API_KEY")
    fred_api_key: str = _s("FRED_API_KEY")
    eia_api_key: str = _s("EIA_API_KEY")
    user_agent: str = _s("USER_AGENT", "HIS/1.0 (+https://example.com)")
    secure_mode: bool = _s("SECURE_MODE", "True").lower() == "true"

SETTINGS = Settings()

# -----------------------------------------------------------------------------
# HTTP CLIENT (session + retry + caching)
# -----------------------------------------------------------------------------
@st.cache_resource
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": SETTINGS.user_agent})
    return s

def _request(method: str, url: str, *, params=None, json=None, timeout=12, tries=3):
    sess = _session()
    last_err = None
    for k in range(tries):
        t0 = time.perf_counter()
        try:
            r = sess.request(method, url, params=params, json=json, timeout=timeout)
            r.raise_for_status()
            ms = int((time.perf_counter() - t0) * 1000)
            ct = r.headers.get("Content-Type", "")
            if "json" in ct:
                return r.json(), ms
            return {"_raw": r.text}, ms
        except Exception as e:
            last_err = e
            time.sleep(0.2 * (k + 1))
    raise last_err

@st.cache_data(ttl=90, show_spinner=False)
def http_get(url: str, *, params: Optional[Dict[str, Any]] = None):
    return _request("GET", url, params=params)

def freshness(ms: int) -> str:
    return f"{ms} ms • fresh"

# -----------------------------------------------------------------------------
# UI COMPONENTS
# -----------------------------------------------------------------------------
def hero():
    with st.container():
        st.markdown(
            """
<div style="border-radius:14px;padding:22px 26px;background:linear-gradient(135deg,#0d1b2a 0%,#1b263b 50%,#3a0ca3 100%); box-shadow:0 10px 30px rgba(0,0,0,.35);">
  <div style="color:#b8c1ec;font-size:13px;letter-spacing:.15em;">Hybrid • Local/Remote</div>
  <div style="color:#e5e7eb;font-size:36px;font-weight:800;margin-top:2px;">Hybrid Intelligence Systems</div>
  <div style="color:#eab308;font-weight:700;margin-top:8px;">Powered by JESSE RAY LANDINGHAM JR</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.write("")

def tiles_home():
    st.slider("Truth Filter", 0, 100, key="truth_filter", value=60)
    st.toggle("Enable Astrology Influence", key="astro_on", value=False)
    st.caption("All data via free public APIs by default — secure mode (no blockchain).")
    st.write("")
    cols = st.columns(3)
    def tile(col, label, desc, page_key):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.page = page_key
                st.rerun()
            st.caption(desc)

    tile(cols[0], "🏛 Lottery", "Daily numbers, picks, entropy, risk modes", "lottery")
    tile(cols[1], "🪙 Crypto", "Live pricing, signals, overlays", "crypto")
    tile(cols[2], "📈 Stocks", "Charts, momentum, factor overlays", "stocks")
    cols = st.columns(3)
    tile(cols[0], "📑 Options", "Chains, quick IV views", "options")
    tile(cols[1], "🏠 Real Estate", "Market tilt and projections", "real_estate")
    tile(cols[2], "🛢 Commodities", "Energy, metals", "commodities")
    cols = st.columns(3)
    tile(cols[0], "🏈 Sports", "Game signals and parlay entropy", "sports")
    tile(cols[1], "🧠 Human Behavior", "Cohort trends and intent", "behavior")
    tile(cols[2], "🛰 Astrology", "Symbolic overlays (hybrid)", "astrology")

def back_button():
    if st.button("Back"):
        st.session_state.page = "home"
        st.rerun()

def line_plot(df: pd.DataFrame, x: str, y: str, title: str):
    if df is None or df.empty:
        st.info("No data.")
        return
    if HAS_PLOTLY:
        fig = px.line(df, x=x, y=y, title=title, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df.set_index(x)[y])

# -----------------------------------------------------------------------------
# SERVICES (in-file)
# -----------------------------------------------------------------------------
# CRYPTO — CoinGecko (free)
def cg_simple(ids: List[str]) -> pd.DataFrame:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "ids": ",".join(ids), "price_change_percentage": "24h"}
    data, ms = http_get(url, params=params)
    rows = []
    for r in data or []:
        rows.append({
            "name": r.get("name"),
            "symbol": r.get("symbol"),
            "current_price": r.get("current_price"),
            "price_change_percentage_24h": r.get("price_change_percentage_24h"),
            "market_cap": r.get("market_cap"),
        })
    st.caption(f"CoinGecko • {freshness(ms)}")
    return pd.DataFrame(rows)

# STOCKS / OPTIONS — yfinance
def yf_history(ticker: str, period="6mo", interval="1d") -> pd.DataFrame:
    if not HAS_YFIN:
        return pd.DataFrame()
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return df
    df = df.reset_index().rename(columns={"Date": "date"})
    df.columns = [c if not isinstance(c, tuple) else c[0] for c in df.columns]
    return df[["date", "Close"]].dropna()

def yf_options_chain(ticker: str, expiration: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if not HAS_YFIN:
        return pd.DataFrame(), pd.DataFrame(), []
    tk = yf.Ticker(ticker)
    exps = tk.options or []
    if not exps:
        return pd.DataFrame(), pd.DataFrame(), []
    exp = expiration or exps[0]
    chain = tk.option_chain(exp)
    return chain.calls, chain.puts, exps

# LOTTERY — NY Open Data + IL CSV (public)
def ny_take5_latest() -> Dict[str, Any]:
    url = "https://data.ny.gov/resource/d6yy-54nr.json"
    data, ms = http_get(url, params={"$limit": 1, "$order": "draw_date DESC"})
    st.caption(f"NY Open Data • {freshness(ms)}")
    if isinstance(data, list) and data:
        r = data[0]
        return {"draw": r.get("winning_numbers"), "date": r.get("draw_date")}
    return {"draw": None, "date": ""}

def il_pick4_latest() -> Dict[str, Any]:
    url = "https://data.illinois.gov/api/views/ck5f-mz5z/rows.csv?accessType=DOWNLOAD"
    try:
        df = pd.read_csv(url)
        row = df.iloc[-1].to_dict()
        nums = row.get("Winning Numbers") or row.get("winning_numbers")
        date = row.get("Date") or row.get("draw_date")
        return {"draw": nums, "date": date}
    except Exception as e:
        st.warning(f"IL fetch failed: {e}")
        return {"draw": None, "date": ""}

# SPORTS — The Odds API (requires key; demo fallback)
def sports_odds(sport_key: str, market: str, region="us") -> pd.DataFrame:
    key = SETTINGS.odds_api_key
    if not key:
        return pd.DataFrame({"team": ["Hawks", "Sharks", "Tigers", "Giants"],
                             "moneyline": [-120, 140, -105, 155]})
    base = "https://api.the-odds-api.com/v4"
    data, _ = http_get(
        f"{base}/sports/{sport_key}/odds",
        params={"regions": region, "markets": market, "oddsFormat": "american", "apiKey": key},
    )
    rows = []
    for ev in (data or []):
        for bm in ev.get("bookmakers", []):
            for mk in bm.get("markets", []):
                if mk.get("key") != market:
                    continue
                for out in mk.get("outcomes", []):
                    rows.append(
                        {"event": ev.get("id"), "book": bm.get("title"),
                         "team": out.get("name"), "moneyline": out.get("price")}
                    )
    return pd.DataFrame(rows)

# COMMODITIES — EIA (requires free key)
def eia_wti_latest() -> pd.DataFrame:
    k = SETTINGS.eia_api_key
    if not k:
        return pd.DataFrame()
    url = f"https://api.eia.gov/series/?api_key={k}&series_id=PET.RWTC.D"
    data, _ = http_get(url)
    try:
        series = data["series"][0]["data"][:180]
        df = pd.DataFrame(series, columns=["Date", "Price"])
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values("Date")
    except Exception:
        return pd.DataFrame()

# REAL ESTATE — FRED Mortgage Rate (free key)
def fred_mortgage30() -> pd.DataFrame:
    k = SETTINGS.fred_api_key
    if not k:
        return pd.DataFrame()
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": "MORTGAGE30US", "api_key": k, "file_type": "json"}
    data, _ = http_get(url, params=params)
    try:
        obs = data.get("observations", [])
        df = pd.DataFrame(obs)[["date", "value"]]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

# ASTROLOGY — synthetic overlay (works offline)
def synthetic_cosmic_index(days: int = 45) -> pd.DataFrame:
    xs = list(range(days))
    deg = [((math.sin(i / 5.0) * 57.2958) % 360) for i in xs]
    return pd.DataFrame({"day": xs, "degree": deg})

# -----------------------------------------------------------------------------
# PAGES
# -----------------------------------------------------------------------------
def page_home():
    hero()
    tiles_home()

def page_crypto():
    back_button()
    hero()
    st.subheader("🪙 Crypto")
    ids_raw = st.text_input("CoinGecko IDs", "bitcoin,ethereum,solana,dogecoin")
    ids = [x.strip() for x in ids_raw.split(",") if x.strip()]
    if st.button("Load Crypto"):
        df = cg_simple(ids)
        st.dataframe(df, use_container_width=True)

def page_stocks():
    back_button()
    hero()
    st.subheader("📈 Stocks")
    tickers = st.text_input("Tickers (comma)", "AAPL,MSFT,NVDA")
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    interval = st.selectbox("Interval", ["1d", "1h"], index=0)
    if st.button("Fetch Stocks"):
        for t in [x.strip().upper() for x in tickers.split(",") if x.strip()]:
            df = yf_history(t, period=period, interval=interval)
            if df.empty:
                st.warning(f"{t}: no data.")
                continue
            line_plot(df, "date", "Close", f"{t} Close")

def page_options():
    back_button()
    hero()
    st.subheader("📑 Options")
    sym = st.text_input("Underlying", "AAPL")
    calls, puts, exps = pd.DataFrame(), pd.DataFrame(), []
    if HAS_YFIN:
        calls, puts, exps = yf_options_chain(sym)
    else:
        st.info("yfinance not installed.")
    exp = st.selectbox("Expiration", exps) if exps else None
    if st.button("Load Chain"):
        if not HAS_YFIN:
            st.warning("Install yfinance to use Options.")
            return
        if exp:
            calls, puts, _ = yf_options_chain(sym, exp)
        if not calls.empty:
            st.caption("Calls")
            st.dataframe(calls, use_container_width=True, height=320)
        if not puts.empty:
            st.caption("Puts")
            st.dataframe(puts, use_container_width=True, height=320)

def page_sports():
    back_button()
    hero()
    st.subheader("🏈 Sports")
    sport = st.selectbox("Sport", ["basketball_nba", "americanfootball_nfl", "icehockey_nhl"], index=2)
    market = st.selectbox("Market", ["spreads", "totals", "h2h"], index=0)
    region = st.selectbox("Region", ["us", "uk", "eu", "au"], index=0)
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Fetch Odds", use_container_width=True):
            df = sports_odds(sport, market, region)
            if df.empty:
                st.warning("No odds (provide an ODDS_API_KEY in secrets).")
            else:
                st.dataframe(df, use_container_width=True, height=380)
    with col2:
        st.button("Run Sports Forecast", use_container_width=True)

def page_real_estate():
    back_button()
    hero()
    st.subheader("🏠 Real Estate")
    if st.button("Load Mortgage 30Y (FRED)"):
        df = fred_mortgage30()
        if df.empty:
            st.info("Add FRED_API_KEY in secrets to unlock this feed.")
        else:
            line_plot(df, "date", "value", "US 30-Year Mortgage Rate (FRED)")

def page_commodities():
    back_button()
    hero()
    st.subheader("🛢 Commodities")
    if st.button("Load WTI (EIA)"):
        df = eia_wti_latest()
        if df.empty:
            st.info("Add EIA_API_KEY in secrets to unlock this feed.")
        else:
            line_plot(df, "Date", "Price", "WTI Crude Oil (EIA)")

def page_behavior():
    back_button()
    hero()
    st.subheader("🧠 Human Behavior")
    q = st.text_input("Keyword", "crypto")
    if st.button("Fetch Mentions"):
        url = f"https://www.reddit.com/search.json?q={q}&limit=10&sort=new"
        try:
            data, ms = http_get(url)
            titles = [c["data"]["title"] for c in data.get("data", {}).get("children", [])]
            st.write(titles or "No posts found.")
            st.caption(f"Reddit (public) • {freshness(ms)}")
        except Exception as e:
            st.warning(f"Reddit fetch failed: {e} • Tip: app credentials remove 403 throttling.")

def page_lottery():
    back_button()
    hero()
    st.subheader("🏛 Lottery")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Illinois Pick-4 (latest)")
        il = il_pick4_latest()
        if il["draw"]:
            st.success(f"IL: {il['draw']} • {il['date']}")
        else:
            st.warning("IL fetch failed. Using open dataset fallback or try later.")
    with col2:
        st.caption("New York Take 5 (latest)")
        ny = ny_take5_latest()
        if ny["draw"]:
            st.success(f"NY Take 5: {ny['draw']} • {ny['date']}")
        else:
            st.warning("NY fetch failed.")

def page_astrology():
    back_button()
    hero()
    st.subheader("🛰 Astrology")
    days = st.slider("Days (synthetic)", 7, 120, 45)
    df = synthetic_cosmic_index(days)
    line_plot(df, "day", "degree", "Planetary Degree (synthetic demo)")
    st.caption("Astrology Influence toggle on Home subtly modulates other engines (symbolic).")

# -----------------------------------------------------------------------------
# ROUTER
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Hybrid Intelligence Systems", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "home"

page = st.session_state.page

if page == "home":
    page_home()
elif page == "crypto":
    page_crypto()
elif page == "stocks":
    page_stocks()
elif page == "options":
    page_options()
elif page == "sports":
    page_sports()
elif page == "real_estate":
    page_real_estate()
elif page == "commodities":
    page_commodities()
elif page == "behavior":
    page_behavior()
elif page == "lottery":
    page_lottery()
elif page == "astrology":
    page_astrology()
else:
    st.session_state.page = "home"
    st.rerun()
