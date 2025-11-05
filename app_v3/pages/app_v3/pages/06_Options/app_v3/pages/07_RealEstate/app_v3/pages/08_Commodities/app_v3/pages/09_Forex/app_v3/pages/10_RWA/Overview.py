import streamlit as st

def render():
    st.title("RWA — Overview")

    tabs = st.tabs(["Dashboard", "Screener", "Signals"])

    with tabs[0]:
        st.subheader("Dashboard")
        st.write("Add real-world asset data such as tokenized property and private credit.")

    with tabs[1]:
        st.subheader("Screener")
        st.write("Add filters for yield, token liquidity, and sector exposure.")

    with tabs[2]:
        st.subheader("Signals")
        st.write("Show RWA market strength, adoption trends, and LIPE signals.")
