import streamlit as st

def render():
    st.title("Commodities — Overview")

    tabs = st.tabs(["Dashboard", "Screener", "Signals"])

    with tabs[0]:
        st.subheader("Dashboard")
        st.write("Add gold, silver, oil, and agricultural commodity charts here.")

    with tabs[1]:
        st.subheader("Screener")
        st.write("Add filters for price trends, futures data, and correlation metrics here.")

    with tabs[2]:
        st.subheader("Signals")
        st.write("Show commodity trend strength and next-move probabilities here.")
