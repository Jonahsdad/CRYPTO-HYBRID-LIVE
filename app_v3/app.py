import streamlit as st
from ui.theme import THEME
from ui.nav import init_nav_state, get_current_arena, set_arena, SIDENAV
from ui.layout import render_shell
from pages import home, crypto, sports, lottery

# ---------- Page config ----------
st.set_page_config(page_title="PunchLogic Command", page_icon="🧭", layout="wide")

# ---------- Init state ----------
init_nav_state()

# ---------- App router ----------
ARENA = get_current_arena()

with render_shell(title="PunchLogic Command", theme=THEME) as shell:
    # Left: side navigation (never changes)
    SIDENAV(
        items=[
            ("home", "Home"),
            ("crypto", "Crypto"),
            ("sports", "Sports"),
            ("lottery", "Lottery"),
        ],
        current=ARENA,
        on_change=set_arena,  # updates session_state["arena"]
        theme=THEME,
    )

    # Right: main canvas (swaps by arena)
    if ARENA == "home":
        home.view(theme=THEME)
    elif ARENA == "crypto":
        crypto.view(theme=THEME)
    elif ARENA == "sports":
        sports.view(theme=THEME)
    elif ARENA == "lottery":
        lottery.view(theme=THEME)
    else:
        home.view(theme=THEME)
