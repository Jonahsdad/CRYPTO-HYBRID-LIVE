# =========================
# FILE: streamlit_app.py
# HIS — Streamlit Flagship (Crypto v1) • Upgraded shell
# =========================
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import streamlit as st

# API bindings
from lib.api import (
    set_api_base, api_login, api_public_plans, api_public_slo
)

# ---------- page config ----------
st.set_page_config(
    page_title="HIS — Powered by LIPE",
    page_icon="⚡",
    layout="wide"
)

# ---------- theming / css ----------
def _load_css() -> None:
    try:
        css = Path("assets/style.css").read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception:
        # fail-safe minimal style
        st.markdown(
            """
            <style>
            .hero{padding:10px 0 8px}
            .hero h1{margin:0;font-size:28px;letter-spacing:.4px}
            .kicker{opacity:.85}
            .card{border:1px solid rgba(124,92,255,.18);background:#121a2b;
                  padding:12px;border-radius:12px;margin:6px 0}
            </style>
            """,
            unsafe_allow_html=True,
        )

_load_css()

# ---------- session defaults ----------
if "token" not in st.session_state:
    st.session_state.token = None
if "team" not in st.session_state:
    st.session_state.team = None
if "email" not in st.session_state:
    st.session_state.email = ""
if "HIS_API_BASE" not in st.session_state:
    # priority: secrets → env → fallback
    base = st.secrets.get("HIS_API_BASE", None) if hasattr(st, "secrets") else None
    st.session_state.HIS_API_BASE = base or os.getenv("HIS_API_BASE", "http://localhost:8000/v1")

# ---------- query params (deep links) ----------
def _read_query_params():
    try:
        qp = st.query_params  # new API
    except Exception:
        qp = st.experimental_get_query_params()  # legacy
    return {k: (v[0] if isinstance(v, list) else v) for k, v in dict(qp).items()}

def _set_query_params(**kwargs):
    try:
        st.query_params.update(kwargs)
    except Exception:
        st.experimental_set_query_params(**kwargs)

qp = _read_query_params()
# api_base override
if "api_base" in qp and qp["api_base"]:
    st.session_state.HIS_API_BASE = qp["api_base"]
# token deep-link
if "token" in qp and qp["token"]:
    st.session_state.token = qp["token"]
# embed mode (hide hero + account panel)
EMBED = qp.get("embed", "0") in ("1", "true", "yes")

# sync API base into client
set_api_base(st.session_state.HIS_API_BASE)

# ---------- small helpers ----------
def ping_backend() -> tuple[bool, str]:
    """Lightweight health using public SLO; no auth required."""
    try:
        slo = api_public_slo()
        # If it returns JSON, consider healthy.
        return True, "OK"
    except Exception as e:
        return False, f"{type(e).__name__}"

def plans_badge() -> Optional[str]:
    try:
        plans = api_public_plans()
        items = plans.get("plans", [])
        return f"<span class='badge'>plans: {len(items)}</span>"
    except Exception:
        return None

def deeplink(arena_slug: str = "") -> str:
    """Generate a link to reopen this app with current settings."""
    base = st.session_state.HIS_API_BASE
    tok  = st.session_state.token or ""
    qp   = {"api_base": base}
    if tok: qp["token"] = tok
    if EMBED: qp["embed"] = "1"
    if arena_slug: qp["arena"] = arena_slug
    # Build URL from current script location
    try:
        # Streamlit Cloud canonical; relative works too
        host = os.environ.get("STREAMLIT_SERVER_URL", "")
        path = ""  # root of app
        qstr = "&".join([f"{k}={v}" for k,v in qp.items()])
        return f"{host}{path}?{qstr}" if host else f"?{qstr}"
    except Exception:
        return "?"

# ---------- header / account ----------
if not EMBED:
    st.markdown(
        """
        <div class="hero">
          <h1>HYBRID INTELLIGENCE SYSTEMS</h1>
          <div class="kicker">All arenas. Hybrid live. <b>Powered by LIPE</b>.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Account")
        st.caption("Sign in to unlock Crypto forecasts & strategy backtests.")

        api_base_in = st.text_input(
            "API Base",
            value=st.session_state.HIS_API_BASE,
            help="FastAPI base URL ending with /v1"
        )

        # Apply base on change
        if api_base_in != st.session_state.HIS_API_BASE:
            st.session_state.HIS_API_BASE = api_base_in
            set_api_base(api_base_in)
            _set_query_params(api_base=api_base_in, embed=("1" if EMBED else "0"))

        # Live health
        ok, msg = ping_backend()
        st.write(("✅ Connected" if ok else "⚠️ Backend Unreachable") + ("" if ok else f" • {msg}"))

        email = st.text_input("Email", value=st.session_state.email or "")
        team  = st.text_input("Team", value=st.session_state.team or "")

        colA, colB = st.columns(2)
        with colA:
            if st.button("Sign in", use_container_width=True):
                try:
                    resp = api_login(email.strip(), team.strip())
                    st.session_state.token = resp.get("token")
                    st.session_state.team  = resp.get("team")
                    st.session_state.email = email.strip()
                    st.success(f"Signed in • {st.session_state.team}")
                    # reflect token in URL (handy for mobile resumes)
                    _set_query_params(api_base=st.session_state.HIS_API_BASE, token=st.session_state.token, embed=("1" if EMBED else "0"))
                except Exception as e:
                    st.error(f"Login failed — check email/team. {e}")
        with colB:
            if st.button("Sign out", use_container_width=True):
                st.session_state.token = None
                st.info("Signed out.")
                _set_query_params(api_base=st.session_state.HIS_API_BASE, embed=("1" if EMBED else "0"))

        # Quick deep link
        st.caption("Quick link (opens with your current settings):")
        st.code(deeplink(), language="text")

# ---------- arena launcher ----------
st.subheader("Choose your arena" if not EMBED else "Arenas")
cols = st.columns(3)

cards = [
    ("🔥 Crypto", "pages/1_Crypto_Flagship.py", "BTC/ETH • Bands • Strategy • Regime"),
    ("💳 Plans",  "pages/2_Plans.py",            "Pricing • Trials • Bundles"),
    ("📊 Status", "pages/3_Status.py",           "SLO • Accuracy • Health"),
    ("🏈 Sports", None,                          "Edges • Odds • Momentum"),
    ("🎰 Lottery",None,                          "GFW • Draws • Echo"),
    ("📈 Stocks", None,                          "Signals • Momentum • EOD"),
]

for i, (title, path, sub) in enumerate(cards):
    with cols[i % 3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f"### {title}")
        st.caption(sub)
        if path:
            st.page_link(path, label="Enter", icon="➡️")
        else:
            st.button("Coming Soon", disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- lightweight footer ----------
if not EMBED:
    pb = plans_badge()
    foot_left = "HYBRID INTELLIGENCE SYSTEM • Powered by LIPE"
    foot_right = f"v1.0 • Streamlit • {pb}" if pb else "v1.0 • Streamlit"
    st.caption(f"{foot_left}  |  {foot_right}")
