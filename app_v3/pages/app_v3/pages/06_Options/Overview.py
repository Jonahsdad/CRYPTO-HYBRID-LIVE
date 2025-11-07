import streamlit as st

def show():
    st.title("🧩 Options — Overview")
    st.caption("HYBRID INTELLIGENCE SYSTEMS | Global Forecast OS")

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Underlying", "—")
    with col2: st.metric("Skew", "—")
    with col3: st.metric("IV Rank", "—")

    st.markdown("### Filters")
    c1, c2, c3 = st.columns(3)
    with c1: _ = st.text_input("Ticker", placeholder="e.g., AAPL")
    with c2: _ = st.selectbox("Strategy", ["All","Calls","Puts","Spreads","Iron Condor"])
    with c3: _ = st.selectbox("Expiry", ["This week","Next week","1m","3m","LEAPS"])

    st.markdown("### Actions")
    a1, a2, a3 = st.columns(3)
    a1.button("Unusual Flow")
    a2.button("Build Spread")
    a3.button("Risk Profile")

    st.markdown("---")
    st.warning("Hook your options API and greeks calc to populate this area.")
