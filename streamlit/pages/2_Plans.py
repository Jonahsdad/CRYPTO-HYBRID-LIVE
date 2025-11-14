# streamlit/pages/2_Plans.py
import streamlit as st
import pandas as pd
st.set_page_config(page_title="Plans — HIS", page_icon="💳", layout="wide")

st.subheader("💳 Plans & Pricing")
st.caption("Simple starter plans. Wire Stripe later — this is the visible copy.")

df = pd.DataFrame([
    {"Plan": "Free", "Arenas": "Stocks (teaser) + RealEstate (teaser)", "Limit": "3 forecasts/arena/day", "Price": "$0"},
    {"Plan": "Crypto Pro", "Arenas": "Crypto", "Limit": "Unlimited", "Price": "$79/mo"},
    {"Plan": "All Markets", "Arenas": "Crypto + Sports + Lottery", "Limit": "Unlimited", "Price": "$199/mo"},
])
st.dataframe(df, use_container_width=True, hide_index=True)
st.info("Stripe checkout/portal can be added next. This page is here so users see pricing context.")
