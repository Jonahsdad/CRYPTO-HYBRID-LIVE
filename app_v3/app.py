import streamlit as st
from ui.theme import THEME
from ui.nav import init_nav_state, get_current_arena, set_arena, SIDENAV
from ui.layout import render_shell
from pages import home, crypto, sports, lottery
from utils.vault import ensure_schemas, migrate
from utils.settings import settings

# ---------- Page config ----------
st.set_page_config(page_title="PunchLogic Command — v3",
                   page_icon="🧭", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""
<style>
  [data-testid="stSidebar"], section[data-testid="stSidebar"],
  div[aria-label="sidebar"] {display:none !important;}
</style>
""", unsafe_allow_html=True)

# ---------- Boot ----------
ensure_schemas()
migrate()  # safe no-op if already current
init_nav_state()
ARENA = get_current_arena()

# ---------- Shell ----------
with render_shell(title="PunchLogic Command", theme=THEME) as shell:
    SIDENAV(items=[("home","Home"),("crypto","Crypto"),("sports","Sports"),("lottery","Lottery")],
            current=ARENA, on_change=set_arena, theme=THEME)
    if ARENA == "home":      home.view(theme=THEME)
    elif ARENA == "crypto":  crypto.view(theme=THEME)
    elif ARENA == "sports":  sports.view(theme=THEME)
    elif ARENA == "lottery": lottery.view(theme=THEME)
    else:                    home.view(theme=THEME)
