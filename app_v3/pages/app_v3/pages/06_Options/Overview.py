import streamlit as st

def render():
    st.title("Options — Overview")

    tabs = st.tabs(["Dashboard", "Screener", "Signals"])

    with tabs[0]:
        st.subheader("Dashboard")
        st.write("Show volatility charts, open interest, and expiration calendars here.")

    with tabs[1]:
        st.subheader("Screener")
        st.write("Add implied volatility filters, gamma exposure, and strike analytics here.")

    with tabs[2]:
        st.subheader("Signals")
        st.write("Show call/put flow, sentiment data, and LIPE option bias signals here.")
