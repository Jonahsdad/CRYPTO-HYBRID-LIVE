import streamlit as st

def show():
    st.title("₿ Crypto — Overview")
    st.caption("HYBRID INTELLIGENCE SYSTEMS | Global Forecast OS")
st.session_state["arena_name"] = "Crypto"

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Watchlist", "Top 100")
    with col2: st.metric("Pulse Window", "24h")
    with col3: st.metric("Signal Mode", "Live")

    st.markdown("### Filters")
    c1, c2, c3 = st.columns(3)
    with c1: _ = st.selectbox("Segment", ["All", "Layer-1", "AI", "DeFi", "RWA", "Memes"])
    with c2: _ = st.selectbox("Market Cap", ["All", "<$100M", "$100M–$1B", ">$1B"])
    with c3: _ = st.selectbox("Timeframe", ["1h", "24h", "7d", "30d"])

    st.markdown("### Actions")
    a1, a2, a3 = st.columns(3)
    a1.button("Scan Momentum")
    a2.button("Find Breakouts")
    a3.button("Refresh Prices")

    st.markdown("---")
    st.success("Results will appear here (table / charts) once connected to live data.")
