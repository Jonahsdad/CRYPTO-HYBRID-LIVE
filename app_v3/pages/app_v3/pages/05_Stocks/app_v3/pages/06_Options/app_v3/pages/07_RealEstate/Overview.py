import streamlit as st

def render():
    st.title("Real Estate — Overview")

    tabs = st.tabs(["Dashboard", "Screener", "Signals"])

    with tabs[0]:
        st.subheader("Dashboard")
        st.write("Add property market trends, rental yields, and regional stats here.")

    with tabs[1]:
        st.subheader("Screener")
        st.write("Add filters for city, price range, and property type here.")

    with tabs[2]:
        st.subheader("Signals")
        st.write("Show LIPE real-estate forecasts, appreciation potential, and risk index here.")
