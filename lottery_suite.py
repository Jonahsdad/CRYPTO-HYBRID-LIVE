import streamlit as st
from lipe_core import LIPE

def lottery_suite():
    st.markdown("### 🎲 Lottery Forecasts")
    engine = LIPE()
    draws = st.text_input("Enter draws (comma separated)")
    if st.button("Run Lottery Forecast"):
        try:
            draws = [int(x.strip()) for x in draws.split(",") if x.strip()]
            result = engine.run_forecast("Pick 4", draws, {"RollingMemory": 60})
            st.json(result)
        except Exception as e:
            st.error(str(e))
