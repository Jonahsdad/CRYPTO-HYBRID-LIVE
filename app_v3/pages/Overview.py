import streamlit as st

def render():
    st.title("Crypto — Overview")

    tabs = st.tabs(["Dashboard", "Screener", "Signals"])

    with tabs[0]:
        st.subheader("Dashboard")
        st.write("Add charts, KPIs, and key metrics here.")

    with tabs[1]:
        st.subheader("Screener")
        st.write("Add your filters, indicators, and ranking logic here.")

    with tabs[2]:
        st.subheader("Signals")
        st.write("Show LIPE/LINA outputs, confidence levels, or next forecasts here.")
