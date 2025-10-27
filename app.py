# HYBRID INTELLIGENCE SYSTEMS – Powered by Jesse Ray Landingham Jr
# Full-fix free-tier build (secure, no blockchain)

import streamlit as st
import yfinance as yf
import pandas as pd
import requests, datetime, json, time
from functools import lru_cache
from bs4 import BeautifulSoup

st.set_page_config(page_title="Hybrid Intelligence Systems",
                   page_icon="🌐", layout="wide")

# -------------------- Caching --------------------
@lru_cache(maxsize=64)
def cached_get(url):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json() if "application/json" in r.headers.get("Content-Type","") else r.text
    except Exception as e:
        st.error(f"Fetch error: {e}")
    return None

# -------------------- Lottery --------------------
def lottery_view():
    st.header("🎰 Lottery Intelligence")
    st.write("Auto-fetch latest Illinois & Hoosier draws.")
    try:
        url = "https://www.lotteryusa.com/illinois/pick-4/"
        html = requests.get(url, timeout=8).text
        soup = BeautifulSoup(html, "html.parser")
        draw = soup.find("span", {"class": "c-results-card__numbers"}).text.strip()
        st.success(f"Latest Illinois Pick-4: {draw}")
    except Exception:
        st.warning("Couldn’t fetch draws (site format may vary).")

# -------------------- Crypto --------------------
def crypto_view():
    st.header("💰 Crypto Intelligence")
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency":"usd","order":"market_cap_desc","per_page":10,"page":1}
    data = cached_get(f"{url}?vs_currency=usd&order=market_cap_desc&per_page=10&page=1")
    if data:
        df = pd.DataFrame(data)[["name","symbol","current_price","price_change_percentage_24h"]]
        st.dataframe(df)
    else:
        st.warning("No crypto data.")

# -------------------- Stocks --------------------
def stocks_view():
    st.header("📈 Stock Intelligence")
    ticker = st.text_input("Enter ticker (e.g., AAPL):", "AAPL")
    data = yf.download(ticker, period="1mo", interval="1d")
    st.line_chart(data["Close"])
    st.caption("Source: Yahoo Finance free endpoint")

# -------------------- Options --------------------
def options_view():
    st.header("🧾 Options Snapshot")
    ticker = st.text_input("Enter ticker:", "AAPL")
    tk = yf.Ticker(ticker)
    exps = tk.options
    if exps:
        date = exps[0]
        opt = tk.option_chain(date)
        st.subheader(f"Calls ({date})"); st.dataframe(opt.calls.head(10))
        st.subheader(f"Puts ({date})"); st.dataframe(opt.puts.head(10))
    else:
        st.info("No options found.")

# -------------------- Sports --------------------
def sports_view():
    st.header("🏈 Sports Odds (demo)")
    try:
        odds = cached_get("https://api.the-odds-api.com/v4/sports/upcoming/odds/?regions=us&oddsFormat=american&markets=h2h")
        if odds: st.json(odds[:3])
        else: st.info("Free tier limit or key required.")
    except Exception: st.warning("Odds feed unavailable.")

# -------------------- Real Estate --------------------
def realestate_view():
    st.header("🏡 Real Estate Metrics")
    fred = "https://api.stlouisfed.org/fred/series/observations"
    key = "api_key=guest"  # placeholder; optional
    url = f"{fred}?series_id=MORTGAGE30US&file_type=json&{key}"
    data = cached_get(url)
    if data:
        df = pd.DataFrame(data["observations"])
        df["value"] = df["value"].astype(float)
        st.line_chart(df.tail(100).set_index("date")["value"])
    else:
        st.info("FRED data unavailable.")
    st.caption("Source: FRED – 30-Year Mortgage Rate")

# -------------------- Commodities --------------------
def commodities_view():
    st.header("🛢️ Commodities")
    metals = cached_get("https://metals-api.com/api/latest?base=USD&symbols=XAU,XAG")
    if metals: st.json(metals)
    eia = cached_get("https://api.eia.gov/series/?api_key=DEMO_KEY&series_id=PET.RWTC.D")
    if eia: st.json(eia)
    st.caption("Sources: Metals-API demo, EIA API")

# -------------------- Human Behavior --------------------
def behavior_view():
    st.header("🧠 Human Behavior Signals")
    url = "https://api.pushshift.io/reddit/search/comment/?subreddit=wallstreetbets&size=20"
    data = cached_get(url)
    if data and "data" in data:
        texts = [d["body"] for d in data["data"]]
        st.write("\n\n".join(texts[:10]))
    else:
        st.info("Pushshift API temporarily offline.")

# -------------------- Astrology --------------------
def astrology_view():
    st.header("🌌 Astrology Ephemeris")
    now = datetime.datetime.utcnow().isoformat()
    nasa = cached_get(f"https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND='10'&OBJ_DATA='YES'")
    st.write("Sample NASA JPL ephemeris response:")
    st.json(nasa if nasa else {"info": "Limited demo data"})

# -------------------- Main UI --------------------
st.markdown(
    "<h1 style='text-align:center;color:#1E90FF;'>Hybrid Intelligence Systems</h1>"
    "<h4 style='text-align:center;'>Powered by Jesse Ray Landingham Jr</h4>",
    unsafe_allow_html=True
)

arena = st.sidebar.radio(
    "Choose Your Arena",
    ["🏠 Home","🎰 Lottery","💰 Crypto","📈 Stocks","🧾 Options",
     "🏈 Sports","🏡 Real Estate","🛢️ Commodities","🧠 Human Behavior","🌌 Astrology"]
)

if arena=="🏠 Home":
    st.success("Welcome to Hybrid Intelligence Systems. Select an arena from the sidebar to begin exploring live intelligence feeds.")
elif "Lottery" in arena: lottery_view()
elif "Crypto" in arena: crypto_view()
elif "Stocks" in arena: stocks_view()
elif "Options" in arena: options_view()
elif "Sports" in arena: sports_view()
elif "Real Estate" in arena: realestate_view()
elif "Commodities" in arena: commodities_view()
elif "Human Behavior" in arena: behavior_view()
elif "Astrology" in arena: astrology_view()

st.sidebar.caption("All data via free public APIs – secure mode (no blockchain)")
