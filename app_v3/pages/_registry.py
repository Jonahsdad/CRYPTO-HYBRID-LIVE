import streamlit as st

MODULE_MAP = {
    "Crypto": "app_v3.pages.02_Crypto.Overview",
    "Sports": "app_v3.pages.03_Sports.Overview",
    "Lottery": "app_v3.pages.04_Lottery.Overview",
    "Stocks": "app_v3.pages.05_Stocks.Overview",
    "Options": "app_v3.pages.06_Options.Overview",
    "Real Estate": "app_v3.pages.07_RealEstate.Overview",
    "Commodities": "app_v3.pages.08_Commodities.Overview",
    "Forex": "app_v3.pages.09_Forex.Overview",
    "RWA": "app_v3.pages.10_RWA.Overview",
}

def HOME_RENDER():
    st.header("Home")
    st.caption("Status: UI Shell Ready — state is kept in session.")
    st.write("Arenas Installed: **Crypto, Sports, Lottery, Stocks, Options, Real Estate, Commodities, Forex, RWA**")

    cols = st.columns(3)
    arenas = list(MODULE_MAP.keys())
    for i, a in enumerate(arenas[:9]):
        with cols[i % 3]:
            if st.button(a, use_container_width=True, key=f"homegrid_{a}"):
                st.session_state["arena"] = a
