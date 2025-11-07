import streamlit as st

def show():
    st.title("🏈 Sports — Overview")
    st.caption("HYBRID INTELLIGENCE SYSTEMS | Global Forecast OS")

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Leagues", "NFL / NBA / MLB / NHL")
    with col2: st.metric("Games Today", "—")
    with col3: st.metric("Signal Mode", "Pre-game")

    st.markdown("### Filters")
    c1, c2, c3 = st.columns(3)
    with c1: _ = st.selectbox("League", ["NFL", "NBA", "MLB", "NHL"])
    with c2: _ = st.selectbox("Market", ["Moneyline", "Spread", "Total"])
    with c3: _ = st.selectbox("Horizon", ["Today", "3 Days", "7 Days"])

    st.markdown("### Actions")
    a1, a2, a3 = st.columns(3)
    a1.button("Scan Match Edges")
    a2.button("Model Consensus")
    a3.button("Simulate Slate")

    st.markdown("---")
    st.info("Game models & edges will render here when the feeds are wired.")
