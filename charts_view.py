import streamlit as st, pandas as pd, matplotlib.pyplot as plt

def charts_view(forecast):
    if not forecast:
        st.info("Run a forecast to view metrics.")
        return
    st.markdown("### 📊 Forecast Analytics")
    df = pd.DataFrame({
        "Top Picks": forecast["top_picks"],
        "Alternates": forecast["alts"]
    })
    fig, ax = plt.subplots()
    ax.plot(df.index, df["Top Picks"], marker="o", label="Top Picks")
    ax.plot(df.index, df["Alternates"], marker="x", label="Alternates")
    ax.legend()
    st.pyplot(fig)
