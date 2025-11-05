import streamlit as st

def render():
    st.title("Sports — Overview")

    tabs = st.tabs(["Dashboard", "Screener", "Signals"])

    with tabs[0]:
        st.subheader("Dashboard")
        st.write("Add sports analytics, performance data, and scoreboards here.")

    with tabs[1]:
        st.subheader("Screener")
        st.write("Add team filters, odds movement, and betting insights here.")

    with tabs[2]:
        st.subheader("Signals")
        st.write("Show sports forecasts, confidence ratings, and LIPE sports model outputs here.")
