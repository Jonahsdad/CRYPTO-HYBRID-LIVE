import streamlit as st
from time import sleep

def PrimaryButton(label: str, key: str, run=None):
    if st.button(label, key=key, use_container_width=True):
        with st.spinner("Running…"):
            if run:
                run()

def StatCard(title: str, value: str, help: str = ""):
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.markdown(f"**{title}**")
    with col2:
        st.markdown(f"<div style='text-align:right; font-size:24px; font-weight:700;'>{value}</div>", unsafe_allow_html=True)
    if help:
        st.caption(help)
    st.divider()

def Loader(text="Working…", seconds=0.6):
    with st.spinner(text):
        sleep(seconds)

def Badge(text: str, tone: str = "success"):
    colors = {
        "success": "#22c55e",
        "warning": "#f59e0b",
        "danger": "#ef4444",
        "muted": "#94a3b8"
    }
    st.markdown(
        f"<span style='background:{colors.get(tone, '#94a3b8')}33; color:{colors.get(tone, '#94a3b8')}; padding:6px 10px; border-radius:999px;'>{text}</span>",
        unsafe_allow_html=True
    )
