import streamlit as st

def vault_view(logs):
    st.markdown("### 🧠 Vault / Logs")
    if not logs:
        st.info("No logs yet.")
        return
    for line in logs[-100:]:
        st.text(line)
