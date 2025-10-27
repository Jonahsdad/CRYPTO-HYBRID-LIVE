# app.py
import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime

# =========================
# CONFIG & GLOBAL SETTINGS
# =========================
st.set_page_config(
    page_title="Hybrid Intelligence Systems",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load secrets or fallback defaults
API_BASE = st.secrets.get("HIS_API", "http://localhost:8000")
GATEWAY_KEY = st.secrets.get("HIS_GATEWAY_KEY", "local_test_key")

# =========================
# API HELPERS (Gateway)
# =========================
@st.cache_data(ttl=30)
def api_get(path, **params):
    """Fetch data from LIPE Gateway"""
    try:
        r = requests.get(
            f"{API_BASE}{path}",
            params=params,
            headers={"X-HIS-KEY": GATEWAY_KEY},
            timeout=15
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=15)
def api_post(path, payload, **params):
    """Send data to LIPE Gateway"""
    try:
        r = requests.post(
            f"{API_BASE}{path}",
            json=payload,
            params=params,
            headers={"X-HIS-KEY": GATEWAY_KEY},
            timeout=15
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("Choose Your Arena")
pages = [
    "🏠 Home",
    "🎰 Lottery",
    "💰 Crypto",
    "📈 Stocks",
    "📊 Options",
    "🏈 Sports",
    "🏡 Real Estate",
    "🛢 Commodities",
    "🧠 Human Behavior",
    "🔮 Astrology",
]
page = st.sidebar.radio("Navigation", pages)

# =========================
# PAGE: HOME
# =========================
if page == "🏠 Home":
    st.title("Hybrid Intelligence Systems")
    st.caption("Powered by JESSE RAY LANDINGHAM JR")
    st.success("✅ Connected dashboard ready for live intelligence feeds.")
    st.divider()

    st.subheader("System Status")
    health = api_get("/health")
    st.json(health)

# =========================
# PAGE: CRYPTO
# =========================
elif page == "💰 Crypto":
    st.header("💰 Crypto Intelligence")
    ids = st.text_input("Enter CoinGecko IDs (comma separated):", "bitcoin,ethereum,solana,dogecoin")

    if st.button("Fetch Prices"):
        res = api_get("/v1/crypto/quotes", ids=ids)
        if "error" in res:
            st.error(res["error"])
        else:
            data = pd.DataFrame(eval(res["data"]))
            st.dataframe(data)
    st.info("Data via CoinGecko → cached via Gateway")

# =========================
# PAGE: LOTTERY
# =========================
elif page == "🎰 Lottery":
    st.header("🎰 Lottery Intelligence")
    st.caption("Illinois + New York draws")

    st.info("Fetching latest results via Gateway...")
    il = api_get("/v1/lottery/illinois")
    ny = api_get("/v1/lottery/newyork")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Illinois Pick 4 (latest)")
        st.json(il)
    with col2:
        st.subheader("New York Take 5 (latest)")
        st.json(ny)

# =========================
# PAGE: STOCKS
# =========================
elif page == "📈 Stocks":
    st.header("📈 Stock Data Feed")
    tickers = st.text_input("Tickers (comma separated):", "AAPL,MSFT,NVDA")
    if st.button("Fetch Stock Data"):
        st.info("🔄 Fetching mock stock data (extend via API later)...")
        df = pd.DataFrame({
            "Ticker": tickers.split(","),
            "Price": [123.45, 342.67, 488.12],
            "Change": [1.3, -0.5, 2.1],
        })
        st.dataframe(df)
        st.bar_chart(df.set_index("Ticker")["Price"])

# =========================
# PAGE: OPTIONS
# =========================
elif page == "📊 Options":
    st.header("📊 Options Chain")
    st.info("Placeholder — load from Yahoo or Tradier API later.")
    df = pd.DataFrame({
        "Strike": [120, 130, 140],
        "Bid": [2.3, 1.8, 1.1],
        "Ask": [2.5, 2.0, 1.3],
        "Volume": [100, 140, 180]
    })
    st.dataframe(df)

# =========================
# PAGE: SPORTS
# =========================
elif page == "🏈 Sports":
    st.header("🏈 Sports Forecasts")
    st.caption("Real-time odds + spread analysis")

    sport = st.selectbox("Sport", ["americanfootball_nfl", "basketball_nba", "icehockey_nhl"])
    market = st.selectbox("Market", ["spreads", "totals", "h2h"])
    region = st.selectbox("Region", ["us", "uk", "eu"])

    if st.button("Fetch Odds"):
        st.info("Connecting to sports odds API via Gateway...")
        res = api_get("/v1/sports/odds", sport=sport, market=market, region=region)
        st.json(res)

# =========================
# PAGE: REAL ESTATE
# =========================
elif page == "🏡 Real Estate":
    st.header("🏡 Real Estate Intelligence")
    st.info("Future integration: Zillow, Redfin, REIT analytics, and geospatial risk maps.")

# =========================
# PAGE: COMMODITIES
# =========================
elif page == "🛢 Commodities":
    st.header("🛢 Commodity Data")
    st.info("Fetching EIA/market data (future-ready).")
    data = api_get("/v1/commodities/oil")
    st.json(data)

# =========================
# PAGE: HUMAN BEHAVIOR
# =========================
elif page == "🧠 Human Behavior":
    st.header("🧠 Behavioral Pulse")
    kw = st.text_input("Keyword", "crypto")
    if st.button("Scan Reddit"):
        res = api_get("/v1/social/reddit", keyword=kw)
        st.json(res)

# =========================
# PAGE: ASTROLOGY
# =========================
elif page == "🔮 Astrology":
    st.header("🔮 Astrology Sync")
    st.info("Astro data and personality mapping (extend via astro_sync.py)")
    st.write("Run astrology analysis or LIPE harmonics here soon.")
