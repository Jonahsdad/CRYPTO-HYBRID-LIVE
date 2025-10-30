import streamlit as st
import requests
from datetime import datetime

# =========================
# Config & Boot
# =========================
st.set_page_config(
    page_title="Hybrid Intelligence Systems — Core Engine",
    layout="wide",
    page_icon="🧠",
)

# Default gateway from secrets; allow override from sidebar
DEFAULT_GATEWAY = st.secrets.get("GATEWAY_URL", "https://his-gateway.onrender.com")

if "arena" not in st.session_state:
    st.session_state.arena = "Home"

# =========================
# Helpers
# =========================
def ping(url: str, path: str = "/health", timeout: float = 6.0):
    try:
        r = requests.get(url.rstrip("/") + path, timeout=timeout)
        ok = r.status_code == 200
        data = r.json() if ok else {"error": f"HTTP {r.status_code}"}
        return ok, data
    except Exception as e:
        return False, {"error": str(e)}

def arena_card(title: str, emoji: str, subtitle: str, key: str):
    # Clickable card: sets session_state.arena
    card_css = """
    <style>
      div[data-testid="arena-card"]{
        border:1px solid rgba(255,255,255,0.08);
        border-radius:12px;
        padding:18px 16px;
        background:rgba(255,255,255,0.02);
        transition:all .2s ease;
        height:118px;
      }
      div[data-testid="arena-card"]:hover{
        border-color:rgba(255,255,255,0.22);
        background:rgba(255,255,255,0.04);
        cursor:pointer;
      }
      .arena-title{font-weight:700;margin:0;}
      .arena-sub{opacity:.75;font-size:0.85rem;margin-top:4px;}
    </style>
    """
    st.markdown(card_css, unsafe_allow_html=True)
    clicked = st.button(
        f"{emoji}  {title}\n\n{subtitle}",
        key=f"card-{key}",
        help=subtitle,
        use_container_width=True,
    )
    if clicked:
        st.session_state.arena = title

# =========================
# Sidebar (Control Panel)
# =========================
st.sidebar.markdown("### System")
compute = st.sidebar.radio(
    "Compute",
    ["Local (in-app)", "Remote API"],
    index=1,
    help="Local = demo stubs. Remote = use the HIS Gateway.",
)

gateway_url = st.sidebar.text_input(
    "API URL (Remote)",
    value=DEFAULT_GATEWAY,
    help="Your HIS Gateway base URL.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Truth Filter")
truth_filter = st.sidebar.slider(
    "Signal Strictness",
    min_value=0,
    max_value=100,
    value=55,
    help="Higher = stricter signals (fewer, higher confidence).",
)

st.sidebar.caption("Controls for each module appear on its page.")

# =========================
# Header
# =========================
st.markdown(
    """
    <div style="padding:8px 0 2px 0;">
      <h1 style="margin-bottom:4px;">🧬 Hybrid Intelligence Systems — Core Engine</h1>
      <div style="opacity:.75;">Powered by LIPE · Developed by Jesse Ray Landingham Jr</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# System Status + Actions
# =========================
cols = st.columns([1.2, 1.2, 1])
with cols[0]:
    if compute.startswith("Remote"):
        ok, data = ping(gateway_url)
        if ok:
            st.success("🟢 Gateway Online")
            st.json(data)
        else:
            st.error("🔴 Gateway Offline or Unreachable")
            st.json(data)
    else:
        st.info("🟡 Local demo mode (no external calls).")

with cols[1]:
    st.text_input("Gateway URL", gateway_url, disabled=True)

with cols[2]:
    st.text_input("Checked", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"), disabled=True)

st.markdown("### ⚡ Actions")
ac1, ac2, ac3 = st.columns([1, 1, 1])

with ac1:
    if st.button("🛰️  Ping Gateway", use_container_width=True):
        if compute.startswith("Remote"):
            ok, data = ping(gateway_url)
            st.toast("Gateway OK" if ok else "Gateway error", icon="✅" if ok else "❌")
        else:
            st.toast("Local mode — no ping", icon="ℹ️")

with ac2:
    if st.button(f"🔮  Get Forecast for {st.session_state.arena}", use_container_width=True):
        st.toast(f"Requested forecast for {st.session_state.arena}", icon="🧠")

with ac3:
    if st.button("🧷 Copy API URL", use_container_width=True):
        st.session_state["_copy"] = gateway_url
        st.toast("Copied (stored in session)", icon="📋")

st.divider()

# =========================
# Choose your arena (GRID)
# =========================
st.markdown("## Choose your arena")

# 9 cards → 3 rows x 3 cols
grid = [
    [
        ("Lottery", "🎲", "Daily numbers, picks, entropy, risk modes", "lottery"),
        ("Crypto", "💰", "Live pricing, signals, overlays", "crypto"),
        ("Stocks", "📈", "Charts, momentum, factor overlays", "stocks"),
    ],
    [
        ("Options", "🗂️", "Chains, skew & IV views", "options"),
        ("Real Estate", "🏡", "Market tilt & projections", "re"),
        ("Commodities", "🪙", "Energy, metals, ag", "commod"),
    ],
    [
        ("Sports", "🏆", "Game signals and parlay edges", "sports"),
        ("Human Behavior", "🧍", "Cognitive & sentiment lenses", "human"),
        ("Astrology", "🔮", "Playful probabilistic lens", "astro"),
    ],
]

for row in grid:
    c1, c2, c3 = st.columns(3)
    for c, (title, emoji, sub, key) in zip([c1, c2, c3], row):
        with c:
            arena_card(title, emoji, sub, key)

st.divider()

# =========================
# Arena content panel
# =========================
st.markdown(f"### 🏟️ {st.session_state.arena} Arena")

if st.session_state.arena == "Home":
    st.write("Welcome to Hybrid Intelligence Systems. Use the grid above to open an arena.")
else:
    # Placeholder controls per arena (you can expand these later)
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("**Controls**")
        st.selectbox("Horizon", ["Intraday", "Daily", "Weekly", "Monthly"])
        st.slider("Confidence floor", 0, 100, 60)
        st.checkbox("Use NBC strategy", value=True)
        st.checkbox("Echo alignment (FES)", value=True)
        if st.button("Run Forecast", type="primary"):
            if compute.startswith("Remote"):
                st.success(f"Queued forecast via {gateway_url} for **{st.session_state.arena}**")
            else:
                st.info("Local mode: returning demo output.")
            st.code(
                "{'forecast':'demo-output','entropy':0.37,'confidence':0.74,'tier':'T33','arena':'"
                + st.session_state.arena + "'}",
                language="json",
            )
    with right:
        st.markdown("**Notes**")
        st.write(f"- Truth Filter: **{truth_filter}%**")
        st.write("- Toggle *Remote API* in the left panel to use the live Gateway.")
        st.write("- Actions appear above; this panel will grow with arena-specific tools.")

st.caption("© 2025 Jesse Ray Landingham Jr · LIPE / HIS · All Rights Reserved")
