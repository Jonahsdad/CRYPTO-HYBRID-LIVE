# app.py
import os
import json
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Hybrid Intelligence Systems",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Secrets (fallbacks for local dev)
API_BASE = st.secrets.get("HIS_API", "http://localhost:8000")
GATEWAY_KEY = st.secrets.get("HIS_GATEWAY_KEY", "local_test_key")

# =========================
# GATEWAY HELPERS
# =========================
def _headers():
    return {"X-HIS-KEY": GATEWAY_KEY}

@st.cache_data(ttl=30)
def api_get(path, **params):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, headers=_headers(), timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=15)
def api_post(path, payload, **params):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, params=params, headers=_headers(), timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# =========================
# SIDEBAR NAV
# =========================
st.sidebar.title("Choose Your Arena")
PAGES = [
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
page = st.sidebar.radio("Navigation", PAGES)

# Shared banner
st.markdown(
    """
    <div style="padding:18px 16px;border-radius:12px;background:linear-gradient(135deg,#0c0f14 0%,#101727 45%,#0e223a 100%);border:1px solid rgba(255,255,255,0.08);">
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="font-size:26px;">🧠</div>
        <div>
          <div style="font-size:22px;font-weight:700;color:#e8f3ff;letter-spacing:.3px;">Hybrid Intelligence Systems</div>
          <div style="font-size:13px;color:#9fb5cc;margin-top:2px;">Powered by <span style="color:#c6e2ff;font-weight:700;">JESSE RAY LANDINGHAM JR</span></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# =========================
# HOME
# =========================
if page == "🏠 Home":
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("System Status")
        health = api_get("/health")
        if "error" in health:
            st.error(health["error"])
        else:
            st.json(health)

        st.subheader("Gateway")
        st.code(API_BASE, language="text")

    with col2:
        st.subheader("Quick Start")
        st.markdown(
            """
            - Use the left sidebar to select an arena.
            - Each page reads data **via the Gateway** (secured by `X-HIS-KEY`).
            - LIPE endpoint is live at `/v1/lipe/forecast` (see Lottery & Crypto pages).
            """
        )
        st.info("Tip: If a page returns an error, the gateway route for that page may be a placeholder. The UI is wired; swap in your preferred backend route names at any time.")

# =========================
# LOTTERY
# =========================
elif page == "🎰 Lottery":
    st.header("🎰 Lottery Intelligence")

    # Live latest (placeholder endpoints – align with your gateway route names)
    st.subheader("Latest Results")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Illinois (example)")
        il = api_get("/v1/lottery/illinois")  # change to your exact route if different
        st.json(il)
    with c2:
        st.caption("New York (example)")
        ny = api_get("/v1/lottery/newyork")   # change to your exact route if different
        st.json(ny)

    st.divider()

    # LIPE Forecast (active)
    st.subheader("LIPE Forecast (Pick 4 example)")
    recent = st.text_area("Recent draws (one per line):", "4397\n2019\n7713")
    horizon = st.slider("How many picks to generate (horizon)", 1, 10, 3)

    colf1, colf2 = st.columns([1, 2])
    with colf1:
        if st.button("Run LIPE Forecast"):
            payload = {
                "game": "pick4",
                "recent_draws": [x.strip() for x in recent.splitlines() if x.strip()],
            }
            res = api_post("/v1/lipe/forecast", payload, arena="lottery", model="default", horizon=horizon)
            if "error" in res:
                st.error(res["error"])
            else:
                st.success("Forecast generated.")
                st.json(res)
    with colf2:
        st.info("This calls the Gateway ➜ `/v1/lipe/forecast?arena=lottery&model=default&horizon=N` and routes into your `lipe_core/lipe_engine.py`.")

# =========================
# CRYPTO
# =========================
elif page == "💰 Crypto":
    st.header("💰 Crypto Intelligence")

    st.subheader("Live Prices (via Gateway ➜ CoinGecko)")
    ids = st.text_input("Enter CoinGecko IDs (comma separated):", "bitcoin,ethereum,solana,dogecoin")
    if st.button("Fetch Prices"):
        res = api_get("/v1/crypto/quotes", ids=ids)
        if "error" in res:
            st.error(res["error"])
        else:
            # gateway returns {"cached":bool, "data": <json str or dict>}
            data = res.get("data")
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except Exception:
                    try:
                        data = eval(data)  # fallback
                    except Exception:
                        data = {}
            df = pd.DataFrame(data).T
            st.dataframe(df)
            if "usd" in df.columns:
                st.bar_chart(df["usd"])

    st.divider()

    st.subheader("LIPE Crypto Ranking")
    ids_rank = st.text_input("IDs to rank (comma separated):", "bitcoin,ethereum,solana")
    topn = st.slider("Top N", 1, 10, 5)
    if st.button("Rank with LIPE"):
        # Optionally fetch market snapshots first (here we only pass IDs)
        payload = {"symbols": [x.strip() for x in ids_rank.split(",") if x.strip()]}
        res = api_post("/v1/lipe/forecast", payload, arena="crypto", model="default", horizon=topn)
        if "error" in res:
            st.error(res["error"])
        else:
            st.json(res)
            # Pretty list
            forecast = res.get("forecast", {})
            ranks = forecast.get("ranks", [])
            if ranks:
                st.markdown("**Top Ranks:**")
                for i, row in enumerate(ranks, 1):
                    st.write(f"{i}. {row.get('id')} — score: {row.get('score')}")

# =========================
# STOCKS
# =========================
elif page == "📈 Stocks":
    st.header("📈 Stocks (scaffold)")
    tickers = st.text_input("Tickers (comma separated):", "AAPL,MSFT,NVDA")
    if st.button("Fetch Stock Data"):
        st.info("Demo data (wire to your gateway route when ready, e.g. `/v1/stocks/history`).")
        df = pd.DataFrame({
            "Ticker": [t.strip() for t in tickers.split(",") if t.strip()],
            "Price": [123.45, 342.67, 488.12][:len([t for t in tickers.split(',') if t.strip()])],
            "Change": [1.3, -0.5, 2.1][:len([t for t in tickers.split(',') if t.strip()])],
        })
        st.dataframe(df)
        if "Price" in df.columns and not df.empty:
            st.bar_chart(df.set_index("Ticker")["Price"])

# =========================
# OPTIONS
# =========================
elif page == "📊 Options":
    st.header("📊 Options Chain (scaffold)")
    st.info("Connect to your gateway → Tradier/Polygon route when ready.")
    df = pd.DataFrame({
        "Strike": [120, 130, 140],
        "Bid": [2.3, 1.8, 1.1],
        "Ask": [2.5, 2.0, 1.3],
        "Volume": [100, 140, 180]
    })
    st.dataframe(df)

# =========================
# SPORTS
# =========================
elif page == "🏈 Sports":
    st.header("🏈 Sports Forecasts")

    sport = st.selectbox("Sport", ["americanfootball_nfl", "basketball_nba", "icehockey_nhl"])
    market = st.selectbox("Market", ["spreads", "totals", "h2h"])
    region = st.selectbox("Region", ["us", "uk", "eu"])

    c1, c2 = st.columns([1,2])
    with c1:
        if st.button("Fetch Odds"):
            st.info("Connecting to sports odds API via Gateway...")
            res = api_get("/v1/sports/odds", sport=sport, market=market, region=region)
            if "error" in res:
                st.error(res["error"])
            else:
                st.json(res)
    with c2:
        st.info("Wire your gateway to TheOddsAPI (or similar). This UI is ready; just align the route.")

# =========================
# REAL ESTATE
# =========================
elif page == "🏡 Real Estate":
    st.header("🏡 Real Estate Intelligence (scaffold)")
    st.info("Future: Zillow/Redfin feeds, REIT analytics, geospatial overlays via your gateway.")

# =========================
# COMMODITIES
# =========================
elif page == "🛢 Commodities":
    st.header("🛢 Commodity Data")
    st.caption("Example oil endpoint via Gateway.")
    res = api_get("/v1/commodities/oil")  # align with your gateway naming
    if "error" in res:
        st.error(res["error"])
    else:
        st.json(res)

# =========================
# HUMAN BEHAVIOR
# =========================
elif page == "🧠 Human Behavior":
    st.header("🧠 Behavioral Pulse")
    kw = st.text_input("Keyword", "crypto")
    if st.button("Scan Reddit"):
        res = api_get("/v1/social/reddit", keyword=kw)  # change 'keyword' to 'q' if your gateway expects q
        if "error" in res:
            st.error(res["error"])
        else:
            st.json(res)

# =========================
# ASTROLOGY
# =========================
elif page == "🔮 Astrology":
    st.header("🔮 Astrology Sync (scaffold)")
    st.info("Connect to astro endpoints or your LIPE harmonics overlay via gateway when ready.")
    st.write("Design this as an engagement enhancer: personality overlays, timing windows, and journaling hooks.")
