# Author: Jesse Ray Landingham Jr
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

# Optional stock support (S&P 500) via yfinance
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

# ===================== PAGE CONFIG ====================
st.set_page_config(page_title="Crypto Hybrid Live", layout="wide", page_icon="🚀")

APP_NAME = "CRYPTO HYBRID LIVE"
st.title(f"🚀 {APP_NAME}")
st.caption("Powered by Jesse Ray Landingham Jr")
st.markdown("---")

# ===================== CRYPTO DATA ====================
def get_coin_data() -> pd.DataFrame:
    """Fetch crypto data from CoinGecko API with safe fallback."""
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 10,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h"
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data)[[
            "id", "symbol", "current_price", "market_cap", "price_change_percentage_24h"
        ]]
        df.columns = ["Name", "Symbol", "Price (USD)", "Market Cap", "24h Change (%)"]
        return df
    except Exception as e:
        st.warning(f"⚠️ CoinGecko error: {e}. Using fallback data.")
        return pd.DataFrame({
            "Name": ["Bitcoin", "Ethereum", "Solana", "XRP", "Cardano"],
            "Symbol": ["BTC", "ETH", "SOL", "XRP", "ADA"],
            "Price (USD)": [67000, 3200, 180, 0.52, 0.25],
            "Market Cap": [1.2e12, 390e9, 75e9, 28e9, 10e9],
            "24h Change (%)": [2.3, -1.2, 0.5, -0.8, 3.1],
        })

# ===================== S&P 500 DATA ====================

@st.cache_data(ttl=900, show_spinner="Loading S&P 500 constituents…")
def sp500_constituents() -> pd.DataFrame:
    """Load S&P 500 symbols from Wikipedia."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0].rename(columns={"Symbol": "Symbol", "Security": "Name", "GICS Sector": "Sector"})
        df["Symbol"] = df["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
        return df[["Symbol", "Name", "Sector"]]
    except Exception:
        return pd.DataFrame(columns=["Symbol", "Name", "Sector"])

@st.cache_data(ttl=300, show_spinner="Fetching S&P 500 prices…")
def sp500_snapshot(tickers: list[str]) -> pd.DataFrame:
    """Get last close and 1-day % change for a set of tickers using yfinance."""
    if not HAS_YF or not tickers:
        return pd.DataFrame()
    # yfinance can take a space-joined string of tickers
    data = yf.download(" ".join(tickers), period="5d", interval="1d", group_by="ticker", auto_adjust=True, progress=False)
    rows = []
    for t in tickers:
        try:
            s = data[t]
            last = float(s.iloc[-1]["Close"])
            prev = float(s.iloc[-2]["Close"]) if len(s) >= 2 else np.nan
            pct = (last / prev - 1.0) * 100.0 if pd.notna(prev) else np.nan
            rows.append({"Symbol": t, "Price (USD)": last, "1d Change (%)": pct})
        except Exception:
            rows.append({"Symbol": t, "Price (USD)": np.nan, "1d Change (%)": np.nan})
    return pd.DataFrame(rows)

# ===================== VIEWS ==========================
def view_dashboard():
    st.header("📊 Market Overview (Crypto)")
    df = get_coin_data()
    st.dataframe(df, use_container_width=True)
    fig = px.bar(
        df, x="Symbol", y="24h Change (%)", color="24h Change (%)",
        title="Top 10 Cryptos — 24h Change (%)", color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

def view_sp500():
    st.header("🏛️ S&P 500 — Live Snapshot")
    if not HAS_YF:
        st.error("`yfinance` not installed. Add `yfinance` to requirements.txt and redeploy.")
        return

    base = sp500_constituents()
    if base.empty:
        st.warning("Couldn’t load S&P 500 list (Wikipedia unreachable). Try again later.")
        return

    st.caption(f"Constituents loaded: {len(base)}")
    # Controls
    cols = st.columns(3)
    with cols[0]:
        sector = st.selectbox("Filter by sector", ["(All)"] + sorted(base["Sector"].dropna().unique().tolist()))
    with cols[1]:
        limit = st.slider("Tickers to snapshot", 25, min(500, len(base)), 150, step=25)
    with cols[2]:
        sort_metric = st.selectbox("Sort by", ["1d Change (%)", "Price (USD)"], index=0)

    df = base.copy()
    if sector != "(All)":
        df = df[df["Sector"] == sector]

    tickers = df["Symbol"].tolist()[:limit]
    snap = sp500_snapshot(tickers)

    if snap.empty:
        st.warning("No price data returned. Try fewer tickers and retry.")
        return

    merged = df.merge(snap, on="Symbol", how="right")
    merged = merged.sort_values(sort_metric, ascending=False)

    st.dataframe(
        merged[["Symbol", "Name", "Sector", "Price (USD)", "1d Change (%)"]],
        use_container_width=True, hide_index=True
    )

    # Chart top movers
    movers = merged.dropna(subset=["1d Change (%)"]).head(25)
    if not movers.empty:
        fig = px.bar(
            movers, x="Symbol", y="1d Change (%)", color="1d Change (%)",
            title="Top Movers (S&P 500 — 1d %)", color_continuous_scale=px.colors.sequential.Bluered
        )
        fig.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

def view_forecast():
    st.header("🧠 Forecast Module")
    st.info("Coming soon: LIPE predictive engine for crypto & equities.")

def view_about():
    st.header("ℹ️ About")
    st.write("""
**Crypto Hybrid Live** by **Jesse Ray Landingham Jr**  
This build includes:
- Crypto market view (CoinGecko)
- S&P 500 live snapshot (yfinance)
- Safe fallbacks so the app never crashes
""")

# ===================== ROUTER =========================
def main():
    menu = ["Dashboard (Crypto)", "S&P 500", "Forecast", "About"]
    choice = st.sidebar.radio("Navigate", menu, index=0)

    if choice == "Dashboard (Crypto)":
        view_dashboard()
    elif choice == "S&P 500":
        view_sp500()
    elif choice == "Forecast":
        view_forecast()
    else:
        view_about()

if __name__ == "__main__":
    main()
