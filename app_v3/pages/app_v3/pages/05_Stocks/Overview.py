import streamlit as st

def show():
    st.title("📈 Stocks — Overview")
    st.caption("HYBRID INTELLIGENCE SYSTEMS | Global Forecast OS")
st.session_state["arena_name"] = "Stocks"

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Universe", "S&P 500")
    with col2: st.metric("Horizon", "Daily")
    with col3: st.metric("Signal", "Momentum")

    st.markdown("### Filters")
    c1, c2, c3 = st.columns(3)
    with c1: _ = st.selectbox("Sector", ["All","Tech","Energy","Financials","Health Care"])
    with c2: _ = st.selectbox("Cap", ["All","Small","Mid","Large"])
    with c3: _ = st.selectbox("Window", ["1d","5d","1m","3m","6m"])

    st.markdown("### Actions")
    a1, a2, a3 = st.columns(3)
    a1.button("Scan Leaders")
    a2.button("Find Reversals")
    a3.button("Refresh Quotes")

    st.markdown("---")
    st.info("Leaders/laggards tables and charts render here once data is connected.")
