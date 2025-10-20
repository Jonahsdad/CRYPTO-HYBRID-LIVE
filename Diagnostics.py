# pages/Diagnostics.py
# Crypto Hybrid Live — Diagnostics
# Self-contained page. No edits to app.py required. It will appear in the sidebar automatically.

import os, sys, time, math, platform, importlib.util
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional

import streamlit as st

# ---- light, local helpers ----------------------------------------------------
def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

@st.cache_data(ttl=60)
def ping(url: str, timeout: float = 6.0) -> Tuple[bool, Optional[int], Optional[str]]:
    import requests
    try:
        r = requests.get(url, timeout=timeout)
        return True, r.status_code, None
    except Exception as e:
        return False, None, str(e)

def has_pkg(name: str) -> Tuple[bool, Optional[str]]:
    try:
        spec = importlib.util.find_spec(name)
        if spec is None:
            return False, None
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "unknown")
        return True, str(ver)
    except Exception:
        return False, None

def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""

# ---- page config & CSS -------------------------------------------------------
st.set_page_config(page_title="Diagnostics • Crypto Hybrid Live", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
.block-container { padding-top: 0.6rem; padding-bottom: 1.2rem; }
.hero {
  margin: 6px 0 12px 0; padding: 16px 18px;
  border-radius: 14px; border: 1px solid #2b3a4a;
  background: linear-gradient(90deg, #0b2a1d 0%, #0f172a 55%, #091a12 100%);
  color: #d8f3dc; font-weight: 800; letter-spacing: .3px;
  display:flex; align-items:center; justify-content:space-between;
}
.small { opacity: .85; font-weight: 600; font-size: 0.92rem; }
.card {
  border: 1px solid #ffffff22; background:#0e1117; padding: 14px; border-radius: 12px;
}
.ok    { background: #0f1f13; border-color:#1d4d2b; }
.warn  { background: #241a14; border-color:#5a3922; }
.bad   { background: #241418; border-color:#5a2231; }
.kv { display:flex; gap:10px; flex-wrap:wrap; }
.kv span { background:#ffffff14; padding:4px 10px; border-radius:999px; font-size:.85rem }
.hdr { font-weight:800; font-size:1.1rem; margin: 4px 0 10px 0;}
table td, table th { font-size: 0.92rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    "<div class='hero'><div>🛡️ DIAGNOSTICS</div>"
    f"<div class='small'>Updated {now_utc()}</div></div>",
    unsafe_allow_html=True,
)

# ---- system info -------------------------------------------------------------
colA, colB, colC = st.columns(3)
with colA:
    st.markdown("<div class='hdr'>Runtime</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='card kv'>"
        f"<span>Python {sys.version.split()[0]}</span>"
        f"<span>{platform.system()} {platform.release()}</span>"
        f"<span>Proc: {platform.machine()}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
with colB:
    st.markdown("<div class='hdr'>Packages</div>", unsafe_allow_html=True)
    checks = []
    for pkg in ["pandas", "numpy", "requests", "streamlit", "yfinance", "plotly", "altair"]:
        ok, ver = has_pkg(pkg)
        emoji = "✅" if ok else "❌"
        checks.append(f"{emoji} {pkg} {ver or ''}".strip())
    st.markdown("<div class='card'>" + "<br>".join(checks) + "</div>", unsafe_allow_html=True)

with colC:
    st.markdown("<div class='hdr'>Secrets (presence only)</div>", unsafe_allow_html=True)
    s_names = ["ALPHAVANTAGE_API_KEY", "FINNHUB_API_KEY"]
    rows = []
    for name in s_names:
        present = "✅ set" if secret(name) else "⚠️ missing"
        rows.append(f"{name}: {present}")
    st.markdown("<div class='card'>" + "<br>".join(rows) + "</div>", unsafe_allow_html=True)

# ---- external service pings --------------------------------------------------
st.markdown("<div class='hdr'>External Services</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    ok, code, err = ping("https://api.coingecko.com/api/v3/ping")
    klass = "ok" if ok else "bad"
    st.markdown(f"<div class='card {klass}'>🪙 CoinGecko<br>Status: {code or 'ERR'}<br>{'' if ok else err}</div>", unsafe_allow_html=True)

with c2:
    ok, code, err = ping("https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords=tesla")
    klass = "ok" if ok else "warn"
    st.markdown(f"<div class='card {klass}'>📈 Alpha Vantage<br>Status: {code or 'ERR'}<br>{'' if ok else err}</div>", unsafe_allow_html=True)

with c3:
    ok, code, err = ping("https://finnhub.io/api/v1/stock/symbol?exchange=US")
    klass = "ok" if ok else "warn"
    st.markdown(f"<div class='card {klass}'>📊 Finnhub<br>Status: {code or 'ERR'}<br>{'' if ok else err}</div>", unsafe_allow_html=True)

with c4:
    ok, code, err = ping("https://query1.finance.yahoo.com/v7/finance/options/AAPL")
    klass = "ok" if ok else "warn"
    st.markdown(f"<div class='card {klass}'>🟣 Yahoo (yfinance)<br>Status: {code or 'ERR'}<br>{'' if ok else err}</div>", unsafe_allow_html=True)

# ---- cache sanity -------------------------------------------------------------
st.markdown("<div class='hdr'>Cache & App Health</div>", unsafe_allow_html=True)
cL, cR = st.columns([2,1])
with cL:
    st.markdown("<div class='card'>If tables feel stale, click the button below to clear cached API pulls.</div>", unsafe_allow_html=True)
with cR:
    if st.button("♻️ Clear Cached Data"):
        st.cache_data.clear()
        st.success("Cache cleared. Reload the page.")

# ---- CI badge / status quick link --------------------------------------------
st.markdown("<div class='hdr'>CI Status</div>", unsafe_allow_html=True)
st.markdown(
    """
<div class='card'>
CI runs on every commit. Check the latest build here:<br>
<a href="https://github.com/Jonahsdad/CRYPTO-HYBRID-LIVE/actions" target="_blank">GitHub Actions → Jonahsdad/CRYPTO-HYBRID-LIVE</a>
</div>
""",
    unsafe_allow_html=True,
)

st.caption("Diagnostics ready • If any box is red, copy the text and send it to me.")
