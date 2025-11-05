import streamlit as st

def render():
    st.title("Forex — Overview")

    tabs = st.tabs(["Dashboard", "Screener", "Signals"])

    with tabs[0]:
        st.subheader("Dashboard")
        st.write("Add currency pairs, DXY charts, and sentiment indexes here.")

    with tabs[1]:
        st.subheader("Screener")
        st.write("Add filters for strength, volatility, and macro trend alignment here.")

    with tabs[2]:
        st.subheader("Signals")
        st.write("Show FX forecast strength, bias levels, and confidence zones here.")
