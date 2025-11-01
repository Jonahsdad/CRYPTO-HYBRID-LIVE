import streamlit as st

def init_nav_state():
    if "arena" not in st.session_state:
        st.session_state["arena"] = "home"

def get_current_arena():
    return st.session_state.get("arena", "home")

def set_arena(key: str):
    st.session_state["arena"] = key

def SIDENAV(items, current, on_change, theme):
    st.markdown("### Choose Your Arena")
    for key, label in items:
        is_active = (key == current)
        classes = "sidenav-item active" if is_active else "sidenav-item"
        clicked = st.markdown(
            f"""
            <div class="{classes}" onclick="window.parent.postMessage({{'type': 'SIDENAV_CLICK', 'key':'{key}'}}, '*')">
              <span>{label}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        # Fallback clickable widget for Streamlit (works without JS)
        if st.button(f"{'• ' if is_active else ''}{label}", key=f"btn_{key}"):
            on_change(key)

    st.divider()
    st.caption("Tip: Navigation is instant. No full reloads — state is kept in session.")
