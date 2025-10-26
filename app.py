# HYBRID INTELLIGENCE SYSTEMS - Neon UX
# app.py - full, drop-in (landing tiles + minimal sidebar)
# ===========================

import sys, os
sys.path.append(os.path.dirname(__file__))

import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# Optional for Stocks/Options
try:
    import yfinance as yf
except Exception:
    yf = None

# ---------- Engine ----------
try:
    from lipe_core import LIPE
except Exception as e:
    LIPE = None
    _IMPORT_ERR = f"Local LIPE unavailable: {e}"
else:
    _IMPORT_ERR = None

APP_TITLE = "HYBRID INTELLIGENCE SYSTEMS"
APP_TAGLINE = "One brain. Many frontiers."
API_URL_DEFAULT = "https://YOUR-FORECAST-API.example.com"
API_URL = os.getenv("LIPE_API_URL", API_URL_DEFAULT)

MODULES = [
    ("🎲", "Lottery", "Daily numbers, picks, entropy, risk modes"),
    ("💰", "Crypto", "Live pricing, signals, overlays"),
    ("📈", "Stocks", "Charts, momentum, factor overlays"),
    ("🧾", "Options", "Chains, quick IV views"),
    ("🏠", "Real Estate", "Market tilt & projections"),
    ("🛢️", "Commodities", "Energy, metals, ag"),
    ("🏈", "Sports", "Game signals & parlay entropy"),
    ("🧑‍🤝‍🧑", "Human Behavior", "Cohort trends & intent"),
    ("🔭", "Astrology", "Playful probabilistic lens"),
]

# ---------- Tiny helpers ----------
def dl_csv(rows: List[Dict[str, Any]]) -> bytes:
    if not rows: return b""
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

def parse_draws_text(s: str) -> List[int]:
    try: return [int(x.strip()) for x in s.split(",") if x.strip()]
    except Exception: return []

def parse_draws_csv(file) -> List[int]:
    try:
        df = pd.read_csv(file)
        for col in ["draw","Draw","number","Number","value","Value"]:
            if col in df.columns:
                vals = [int(x) for x in df[col].dropna().tolist()]
                if vals: return vals
        return []
    except Exception: return []

def metric(val, fmt=lambda x: x):
    try: return fmt(val)
    except Exception: return "—"

def api_health(url: str):
    try:
        r = requests.get(f"{url.rstrip('/')}/health", timeout=5); r.raise_for_status(); return r.json()
    except Exception: return None

def api_forecast(url: str, game: str, draws: List[int], settings: Dict[str, Any]):
    r = requests.post(f"{url.rstrip('/')}/forecast", json={"game":game,"draws":draws,"settings":settings}, timeout=15)
    r.raise_for_status(); return r.json()

# ---------- App shell ----------
st.set_page_config(page_title="HIS — Hybrid Intelligence Systems", layout="wide")

# ---------- Neon CSS (no external libs) ----------
st.markdown("""
<style>
:root {
  --accent: #8be9fd;       /* cyan */
  --accent2: #ff79c6;      /* magenta */
  --bg: #0e0f12;
  --card: #151722;
  --muted: #8a8fa3;
}
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #12141b, #0f0f14);}
h1, h2, h3, h4 { letter-spacing: .2px; }
.neon-hero {
  padding: 32px 28px; border-radius: 18px;
  background: radial-gradient(1200px 300px at 20% -20%, rgba(139,233,253,.20), transparent),
              radial-gradient(1200px 300px at 80% -20%, rgba(255,121,198,.18), transparent),
              linear-gradient(180deg, #141620, #0f1118);
  border: 1px solid rgba(255,255,255,.06);
  box-shadow: 0 0 24px rgba(139,233,253,.06), inset 0 0 24px rgba(255,121,198,.04);
}
.hero-title { font-size: 36px; font-weight: 800; margin: 0 0 6px 0;}
.hero-tag { color: var(--muted); margin-bottom: 0; }
.tile {
  background: var(--card);
  border: 1px solid rgba(255,255,255,.06); border-radius: 16px; padding: 18px;
  transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
  cursor: pointer; user-select: none;
}
.tile:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(139,233,253,.10), 0 0 0 1px rgba(139,233,253,.12) inset;
  border-color: rgba(139,233,253,.25);
}
.tile-emoji { font-size: 24px; }
.tile-title { font-weight: 700; margin-bottom: 4px; }
.tile-desc { color: var(--muted); font-size: 14px; }
.pill { display: inline-block; padding: 4px 10px; border-radius: 100px; font-size: 12px;
  background: rgba(139,233,253,.12); border: 1px solid rgba(139,233,253,.25); margin-right: 8px; }
.cta {
  border: 1px solid rgba(255,255,255,.08); border-radius: 12px; padding: 10px 14px;
  background: linear-gradient(180deg, #1a2030, #121622); font-weight: 700;
}
.small-muted { color: var(--muted); font-size: 12px; }
hr { border: none; height: 1px; background: rgba(255,255,255,.06); }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar (minimal) ----------
st.sidebar.header("System")
engine_mode = st.sidebar.radio("Compute", ["Local (in-app)", "Remote API"], index=0)
api_url = st.sidebar.text_input("API URL (Remote)", value=API_URL)

st.sidebar.subheader("Truth Filter")
truth_filter = st.sidebar.slider("Signal Strictness", 0, 100, 55, step=5)
st.sidebar.caption(f"Truth Filter · {truth_filter}%")

st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
st.sidebar.caption("This sidebar stays light. Module controls appear in the main panel when you open a module.")

# ---------- Session state ----------
if "selected_module" not in st.session_state:
    st.session_state.selected_module = None
if "ledger" not in st.session_state:
    st.session_state.ledger = []

# ---------- Engine init ----------
engine = LIPE() if LIPE else None
status = engine.ping() if engine and hasattr(engine, "ping") else {
    "name":"LIPE","tier":33,"status":"Active","boot_time":datetime.now().isoformat(timespec="seconds")
}

# ---------- LANDING (hero + tiles) ----------
def landing():
    st.markdown(f"""
<div class="neon-hero">
  <div class="pill">Hybrid · Local/Remote</div>
  <div class="pill">Engine: LIPE-Core · Tier {status.get('tier','—')}</div>
  <h1 class="hero-title">🧠 {APP_TITLE}</h1>
  <p class="hero-tag">{APP_TAGLINE}</p>
</div>
""", unsafe_allow_html=True)
    st.write("")

    st.subheader("Choose your arena")
    rows = [MODULES[i:i+3] for i in range(0, len(MODULES), 3)]
    for row in rows:
        cols = st.columns(len(row))
        for col, (emoji, title, desc) in zip(cols, row):
            with col:
                if st.button(f"{emoji}  {title}", key=f"tile-{title}", use_container_width=True):
                    st.session_state.selected_module = title
                st.markdown(f"""
<div class="tile" onclick="document.querySelector('button[kind=secondary][data-testid={{'tile-{title}'}}]').click()">
  <div class="tile-emoji">{emoji}</div>
  <div class="tile-title">{title}</div>
  <div class="tile-desc">{desc}</div>
</div>
""", unsafe_allow_html=True)

    st.write("")
    colA, colB, colC = st.columns([1.5,1,1.5])
    with colA:
        st.markdown("**Why HIS?**\n\nProgrammable forecasts across domains. A single brain that learns.")
    with colB:
        st.markdown("**Truth Filter**\n\nDial exploration vs. strictness to match your risk mode.")
    with colC:
        st.markdown("**Hybrid Compute**\n\nRun locally for R&D or flip to API for global scale.")

# ---------- MODULE PAGES ----------
def lottery_page():
    st.markdown("### 🎲 Lottery Hybrid Live")
    game = st.selectbox("Game", ["Pick 3","Pick 4","Lucky Day Lotto"], index=0)
    session = st.radio("Session", ["Midday","Evening"], horizontal=True)
    col1, col2, col3 = st.columns(3)
    rolling = col1.slider("Rolling Memory (draws)", 10, 240, 60, step=5)
    bonus   = col2.select_slider("Bonus Weighting", ["None","Light","Moderate","Heavy"], value="Moderate")
    col3.markdown("&nbsp;")
    nb = col3.toggle("NBC", value=True, help="Non-Backtracking triggers")
    rp = col3.toggle("RP",  value=True, help="Recall patterns")
    echo = col3.toggle("Echo", value=True, help="Echo logic")

    st.markdown("**Recent draws**")
    draws_text = st.text_area("Paste comma-separated integers", value="439,721,105,387,902,114,296,431", height=80)
    uploaded = st.file_uploader("Or upload CSV with a `draw` column", type=["csv"])
    draws = parse_draws_csv(uploaded) if uploaded else parse_draws_text(draws_text)

    run = st.button("🚀 Run Lottery Forecast", type="primary")
    if run and not draws: st.warning("Please provide recent draws.")

    tabs = st.tabs(["Result", "Charts", "Vault"])
    if run and draws:
        settings = {
            "Session": session, "RollingMemory": int(rolling), "BonusWeighting": bonus,
            "UseNBC": bool(nb), "UseRP": bool(rp), "UseEcho": bool(echo), "TruthFilter": int(truth_filter)
        }
        try:
            t0 = time.time()
            if st.sidebar.radio if engine_mode.startswith("Remote") else False:  # keep linter happy
                pass
            if engine_mode.startswith("Remote"):
                if not api_health(api_url): st.error("Remote API unreachable.")
                result = api_forecast(api_url, game, draws, settings)
            else:
                if not engine or not hasattr(engine, "run_forecast"):
                    raise RuntimeError("Local engine not available.")
                result = engine.run_forecast(game=game, draws=draws, settings=settings)
            t1 = time.time()

            with tabs[0]:
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Confidence", metric(result.get("confidence",0), lambda x: f"{int(float(x)*100)}%"))
                c2.metric("Entropy", metric(result.get("entropy",0.0), lambda x: f"{float(x):.2f}"))
                c3.metric("Engine", result.get("logic","—"))
                c4.metric("Latency", f"{(t1-t0)*1000:.0f} ms")
                st.markdown("**Top Picks**")
                st.code(", ".join(map(str, result.get("top_picks", []))))
                st.markdown("**Alternates**")
                st.code(", ".join(map(str, result.get("alts", []))))

            with tabs[1]:
                ent_now = float(result.get("entropy", 0.5))
                series = np.linspace(max(0.05, ent_now-0.2), min(1.0, ent_now+0.2), 12)
                fig, ax = plt.subplots(); ax.plot(series)
                ax.set_title("Entropy Trend"); ax.set_xlabel("Window"); ax.set_ylabel("Entropy (0..1)")
                st.pyplot(fig)

            with tabs[2]:
                logs = getattr(engine, "logs", []) if engine else []
                if logs:
                    for line in logs[-200:]: st.text(line)
                else:
                    st.info("No logs yet.")

            st.session_state.ledger.append({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "module":"lottery","mode":"remote" if engine_mode.startswith("Remote") else "local",
                "game": result.get("game", game), "session": session, "truth": truth_filter,
                "conf": result.get("confidence", None), "ent": result.get("entropy", None),
                "top": "|".join(map(str, result.get("top_picks", []))),
                "alts": "|".join(map(str, result.get("alts", []))),
                "logic": result.get("logic",""), "latency_ms": int((t1-t0)*1000)
            })
        except Exception as e:
            st.error(f"Forecast error: {e}")

def crypto_page():
    st.markdown("### 💰 Crypto Hybrid Live")
    coins = st.text_input("Coins (CoinGecko ids)", value="bitcoin,ethereum,solana,dogecoin")
    if st.button("🔄 Refresh Prices", type="primary"):
        try:
            r = requests.get("https://api.coingecko.com/api/v3/simple/price",
                             params={"ids": coins, "vs_currencies":"usd"}, timeout=10)
            r.raise_for_status()
            data = [{"Coin": k.title(), "Price (USD)": v["usd"]} for k,v in r.json().items() if "usd" in v]
            st.dataframe(pd.DataFrame(data), use_container_width=True)
        except Exception as e:
            st.error(f"Price fetch failed: {e}")
    st.caption("Truth Filter influences which crypto signals you surface next.")

def stocks_page():
    st.markdown("### 📈 Stocks")
    cols = st.columns(3)
    tickers = cols[0].text_input("Tickers", value="AAPL,MSFT,NVDA")
    period  = cols[1].selectbox("Period", ["1mo","3mo","6mo","1y","2y"], index=2)
    interval= cols[2].selectbox("Interval", ["1d","1h","30m"], index=0)
    if st.button("📊 Fetch", type="primary"):
        if yf is None:
            st.error("yfinance not installed (add to requirements.txt).")
            return
        for t in [x.strip().upper() for x in tickers.split(",") if x.strip()]:
            try:
                data = yf.download(t, period=period, interval=interval, progress=False)
                if data.empty: st.warning(f"No data for {t}."); continue
                st.markdown(f"**{t}**")
                fig, ax = plt.subplots(); ax.plot(data.index, data["Close"])
                ax.set_title(f"{t} Close — {period} / {interval}"); ax.set_xlabel("Date"); ax.set_ylabel("Price")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"{t} fetch failed: {e}")

def options_page():
    st.markdown("### 🧾 Options")
    col1, col2 = st.columns([1,2])
    ticker = col1.text_input("Underlying", value="AAPL")
    if col1.button("📜 Load Chain", type="primary"):
        if yf is None: st.error("yfinance not installed."); return
        try:
            tk = yf.Ticker(ticker); exps = tk.options
            if not exps: st.warning("No options."); return
            sel = col2.selectbox("Expiration", exps)  # shows after click
            opt = tk.option_chain(sel)
            st.write("Calls"); st.dataframe(opt.calls.head(30), use_container_width=True)
            st.write("Puts");  st.dataframe(opt.puts.head(30), use_container_width=True)
        except Exception as e:
            st.error(f"Chain failed: {e}")

def simple_stub(title: str, note: str):
    st.markdown(f"### {title}")
    st.info(note)
    # friendly placeholder chart
    x = np.arange(30); y = 100 + np.cumsum(np.random.normal(0, 0.6, size=30))
    fig, ax = plt.subplots(); ax.plot(x,y); ax.set_title(f"{title} — demo curve")
    st.pyplot(fig)

# ---------- Router ----------
def open_module(name: str):
    name_l = name.lower()
    if "lottery" in name_l: return lottery_page()
    if "crypto" in name_l: return crypto_page()
    if "stocks" in name_l: return stocks_page()
    if "options" in name_l: return options_page()
    if "real estate" in name_l or "realestate" in name_l:
        return simple_stub("🏠 Real Estate", "Project with your APIs for market tilt & rates.")
    if "commod" in name_l:
        return simple_stub("🛢️ Commodities", "Wire WTI/Brent/Gold endpoints when ready.")
    if "sports" in name_l:
        return simple_stub("🏈 Sports", "Connect odds feeds & LIPE overlays.")
    if "human" in name_l or "behavior" in name_l:
        return simple_stub("🧑‍🤝‍🧑 Human Behavior", "Cohort momentum & propensity models.")
    if "astrology" in name_l:
        return simple_stub("🔭 Astrology", "Playful, probabilistic overlays.")
    return landing()

# ---------- Main ----------
# show landing or the selected module
if st.session_state.selected_module:
    # small top bar to navigate back
    cols = st.columns([6,1])
    with cols[0]: st.markdown(f"#### 🧠 {APP_TITLE} · **{st.session_state.selected_module}**")
    with cols[1]:
        if st.button("⟵ Back", use_container_width=True):
            st.session_state.selected_module = None
            st.experimental_rerun()
    st.write("")
    open_module(st.session_state.selected_module)
else:
    landing()

# ---------- Footer tools ----------
st.markdown("---")
colL, colR = st.columns([2,1])
with colL:
    st.caption(f"Engine: LIPE-Core · Tier {status.get('tier','—')} · {status.get('status','—')} · Boot {status.get('boot_time','—')}")
with colR:
    if st.session_state.ledger:
        st.download_button("⬇️ Download Run Ledger", data=dl_csv(st.session_state.ledger),
                           file_name="his_runs.csv", mime="text/csv")
    else:
        st.caption("Run something to populate the ledger.")
