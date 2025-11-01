import streamlit as st, pandas as pd
from ui.widgets import PrimaryButton, Loader, StatCard, Badge
from data_sources.crypto_feed import search_coins, get_simple_price, get_market_chart, CGError
from utils.log import audit_event
from utils.vault import insert_crypto_prices
from utils.engine import action

@st.cache_data(ttl=60, show_spinner=False)
def _cached_prices(ids: list[str]) -> pd.DataFrame:
    return get_simple_price(ids, vs="usd")

@st.cache_data(ttl=300, show_spinner=False)
def _cached_chart(cid: str, days: int) -> pd.DataFrame:
    return get_market_chart(cid, days=days)

DEFAULT_IDS = ["bitcoin","ethereum","solana"]

def view(theme):
    st.subheader("Crypto Arena")
    Badge("Live Data: CoinGecko (public API)", "success")

    with st.expander("Select Coins", expanded=True):
        st.caption("Type CoinGecko IDs (e.g., bitcoin, ethereum, solana)")
        ids_text = st.text_input("Coin IDs", ", ".join(DEFAULT_IDS), key="cg_ids")
        ids = [x.strip().lower() for x in ids_text.split(",") if x.strip()]
        with st.popover("Find IDs"):
            q = st.text_input("Search", "btc")
            if st.button("Search"):
                try: st.dataframe(search_coins(q), use_container_width=True)
                except CGError as e: st.error(str(e))

    colA,colB,colC = st.columns(3)

    with colA:
        def _do_refresh():
            try:
                with action("Crypto: Refresh Prices", "crypto.refresh_prices", ids=ids):
                    Loader("Contacting CoinGecko…", 0.25)
                    df=_cached_prices(ids)
                    st.dataframe(df, use_container_width=True)
                    if not df.empty:
                        insert_crypto_prices(df)
                        top=df.sort_values("price_usd", ascending=False).head(3)
                        StatCard("Top by Price (USD)", ", ".join(f"{r['id']} ${r['price_usd']:,}" for _,r in top.iterrows()))
            except CGError as e: st.error(str(e))
        PrimaryButton("Refresh Prices (Live)", key="c_refresh_live", run=_do_refresh)

    with colB:
        def _do_forecast():
            try:
                with action("Crypto: Forecast 7d slope", "crypto.forecast", ids=ids):
                    Loader("Computing 7d momentum…", 0.35)
                    rows=[]
                    for cid in ids[:5]:
                        s=_cached_chart(cid, days=7).reset_index(drop=True)
                        if s.empty or len(s)<5: continue
                        s["t"]=range(len(s))
                        cov=((s["t"]-s["t"].mean())*(s["price_usd"]-s["price_usd"].mean())).sum()
                        var=((s["t"]-s["t"].mean())**2).sum()
                        slope=cov/var if var!=0 else 0.0
                        rows.append({"id":cid,"slope_7d":slope,"last_price":float(s.iloc[-1]["price_usd"])})
                    out=pd.DataFrame(rows).sort_values("slope_7d", ascending=False)
                    if out.empty: st.info("Not enough data."); return
                    st.dataframe(out, use_container_width=True)
                    lead=out.iloc[0]; StatCard("Momentum Leader (7d)", f"{lead['id']} | slope={lead['slope_7d']:.6f}")
            except CGError as e: st.error(str(e))
        PrimaryButton("Run Forecast (7d Momentum POC)", key="c_forecast_live", run=_do_forecast)

    with colC:
        def _do_history():
            try:
                with action("Crypto: 24h history", "crypto.history", id=ids[0] if ids else None):
                    Loader("Loading history…", 0.25)
                    cid=ids[0]
                    df=_cached_chart(cid, days=1)
                    st.line_chart(df, x="timestamp", y="price_usd", height=220)
                    StatCard("History", f"{cid} — {len(df)} pts", "Source: CoinGecko market_chart")
            except CGError as e: st.error(str(e))
        PrimaryButton("View History (24h chart)", key="c_history_live", run=_do_history)
