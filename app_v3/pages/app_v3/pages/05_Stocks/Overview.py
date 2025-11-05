import streamlit as st

def render():
    st.title("Stocks — Overview")

    tabs = st.tabs(["Dashboard", "Screener", "Signals"])

    with tabs[0]:
        st.subheader("Dashboard")
        st.write("Add stock tickers, performance heatmaps, and top gainers/losers here.")

    with tabs[1]:
        st.subheader("Screener")
        st.write("Add P/E filters, momentum screens, and fundamentals here.")

    with tabs[2]:
        st.subheader("Signals")
        st.write("Show buy/sell/hold recommendations and AI forecast strength here.")
