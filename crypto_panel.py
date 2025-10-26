import streamlit as st, requests, pandas as pd

def crypto_panel():
    st.markdown("### 💰 Crypto Hybrid Forecasts")
    coins = ["bitcoin", "ethereum", "solana", "dogecoin"]
    data = []
    for c in coins:
        try:
            r = requests.get(f"https://api.coingecko.com/api/v3/simple/price",
                             params={"ids": c, "vs_currencies": "usd"})
            price = r.json()[c]["usd"]
            data.append({"Coin": c.title(), "Price": price})
        except Exception:
            pass
    if not data:
        st.warning("CoinGecko API unavailable.")
        return
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
