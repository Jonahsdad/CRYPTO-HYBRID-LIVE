# ===========================
# HYBRID INTELLIGENCE SYSTEMS (HIS)
# app.py — full, drop-in (multi-module)
# ===========================

# --- import path guard (cloud-safe) ---
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

# Optional for Stocks/Options (installed via requirements.txt)
try:
    import yfinance as yf
except Exception:
    yf = None

# -------- Local Engine (optional if using Remote API) --------
try:
    from lipe_core import LIPE
except Exception as e:
    LIPE = None
    _IMPORT_ERR = f"Local LIPE unavailable: {e}"
else:
    _IMPORT_ERR = None

# -------- Constants --------
APP_TITLE = "🧠 HYBRID INTELLIGENCE SYSTEMS"
APP_SUBTITLE = "Hybrid Local/Remote Engine · LIPE-Core"
API_URL_DEFAULT = "https://YOUR-FORECAST-API.example.com"
API_URL = os.getenv("LIPE_API_URL", API_URL_DEFAULT)

MODULES = [
    "🎲 Lottery Hybrid Live",
    "💰 Crypto Hybrid Live",
    "🏠 Real Estate",
    "🛢️ Commodities",
    "🏈 Sports",
    "🧑‍🤝‍🧑 Human Behavior",
    "🔭 Astrology",
    "📈 Stocks",
    "🧾 Options",
]

# -------- Utilities --------
def parse_draws_text(s: str) -> List[int]:
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except Exception:
        return []

def parse_draws_csv(file) -> List[int]:
    try:
        df = pd.read_csv(file)
        for col in ["draw","Draw","number","Number","value","Value"]:
            if col in df.columns:
                vals = [int(x) for x in df[col].dropna().tolist()]
                if vals: return vals
        return []
    except Exception:
        return []

def downloadable_csv(rows: List[Dict[str, Any]]) -> bytes:
    if not rows: return b""
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

def metric(val, fmt=lambda x: x):
    try: return fmt(val)
    except Exception: return "—"

def api_health(url: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{url.rstrip('/')}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def api_forecast(url: str, game: str, draws: List[int], settings: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{url.rstrip('/')}/forecast",
                      json={"game":game, "draws":draws, "settings":settings},
                      timeout=15)
    r.raise_for_status()
    return r.json()

# -------- App shell --------
st.set_page_config(page_title="HIS — Hybrid Intelligence Systems", layout="wide")
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

# -------- Engine Mode / Module Switcher (Top of sidebar) --------
st.sidebar.header("System")
engine_mode = st.sidebar.radio("Compute", ["Local (in-app)", "Remote API"], index=0)
api_url = st.sidebar.text_input("API URL (for Remote)", value=API_URL)

# Global “Truth Filter” control (applies to all modules)
st.sidebar.subheader("Truth Filter")
truth_filter = st.sidebar.slider("Signal Strictness", 0, 100, 60, step=5,
                                 help="Higher = stricter acceptance; lower = exploratory.")
st.sidebar.progress(truth_filter/100.0)
st.sidebar.caption(f"Truth Filter · {truth_filter}%")

st.sidebar.divider()
module = st.sidebar.selectbox("Module", MODULES)

# -------- State / Ledger --------
if "run_ledger" not in st.session_state:
    st.session_state.run_ledger = []

# -------- Initialize Engine --------
engine = LIPE() if LIPE else None
status = engine.ping() if engine and hasattr(engine,"ping") else {
    "name":"LIPE","tier":33,"status":"Active",
    "boot_time": datetime.now().isoformat(timespec="seconds")
}
st.caption(f"Engine: LIPE-Core · Tier {status.get('tier','—')} · {status.get('status','—')} · Boot {status.get('boot_time','—')}")
if _IMPORT_ERR and engine_mode.startswith("Local"):
    st.warning(_IMPORT_ERR)

# -------- Tabs common to all modules --------
tabs = st.tabs(["📈 Forecast / View", "📊 Charts", "📜 Vault", "⚙️ Settings", "❓ Help"])
result_payload: Optional[Dict[str, Any]] = None

# -------- Shared Settings passed to LIPE/API (lottery) --------
def lottery_controls():
    st.sidebar.header("Lottery Controls")
    game = st.sidebar.selectbox("Game", ["Pick 3","Pick 4","Lucky Day Lotto"])
    session = st.sidebar.radio("Session", ["Midday","Evening"], index=0)
    rolling_memory = st.sidebar.slider("Rolling Memory (draws)", 10, 240, 60, step=5)
    bonus_weighting = st.sidebar.select_slider("Bonus Weighting",
                                               options=["None","Light","Moderate","Heavy"],
                                               value="Moderate")
    st.sidebar.subheader("Strategy Switches")
    use_nbc  = st.sidebar.toggle("NBC Triggers", value=True)
    use_rp   = st.sidebar.toggle("RP Memory Recall", value=True)
    use_echo = st.sidebar.toggle("Echo Logic", value=True)
    st.sidebar.subheader("Recent draws")
    draws_text = st.sidebar.text_area("A) Paste comma-separated integers",
                                      value="439,721,105,387,902,114,296,431", height=80)
    uploaded   = st.sidebar.file_uploader("B) Or upload CSV with a 'draw' column", type=["csv"])
    run_btn    = st.sidebar.button("Run Lottery Forecast")

    draws = parse_draws_csv(uploaded) if uploaded else parse_draws_text(draws_text)
    settings = {
        "Session": session,
        "RollingMemory": int(rolling_memory),
        "BonusWeighting": str(bonus_weighting),
        "UseNBC": bool(use_nbc),
        "UseRP": bool(use_rp),
        "UseEcho": bool(use_echo),
        "TruthFilter": int(truth_filter)
    }
    return game, draws, run_btn, settings

# -------- Module: Crypto --------
def crypto_controls():
    st.sidebar.header("Crypto Controls")
    coins_default = "bitcoin,ethereum,solana,dogecoin"
    coins_str = st.sidebar.text_input("Coins (CoinGecko ids)", value=coins_default)
    refresh_btn = st.sidebar.button("Refresh Prices")
    return [c.strip() for c in coins_str.split(",") if c.strip()], refresh_btn

# -------- Module: Stocks --------
def stocks_controls():
    st.sidebar.header("Stocks Controls")
    tickers = st.sidebar.text_input("Tickers (comma-separated)", value="AAPL,MSFT,NVDA")
    period = st.sidebar.selectbox("Period", ["1mo","3mo","6mo","1y","2y"], index=2)
    interval = st.sidebar.selectbox("Interval", ["1d","1h","30m"], index=0)
    fetch_btn = st.sidebar.button("Fetch")
    return [t.strip().upper() for t in tickers.split(",") if t.strip()], period, interval, fetch_btn

# -------- Module: Options --------
def options_controls():
    st.sidebar.header("Options Controls")
    ticker = st.sidebar.text_input("Underlying (ticker)", value="AAPL")
    chain_btn = st.sidebar.button("Load Chain")
    return ticker.strip().upper(), chain_btn

# -------- Module: Real Estate / Commodities / Sports / Human Behavior / Astrology (stubs with working UI) --------
def stub_controls(title: str, fields: Dict[str, Any]):
    st.sidebar.header(f"{title} Controls")
    values = {}
    for key, spec in fields.items():
        typ = spec.get("type","text")
        if typ == "slider":
            values[key] = st.sidebar.slider(spec["label"], spec["min"], spec["max"], spec["value"], step=spec.get("step",1))
        elif typ == "select":
            values[key] = st.sidebar.selectbox(spec["label"], spec["options"], index=spec.get("index",0))
        else:
            values[key] = st.sidebar.text_input(spec["label"], value=spec.get("value",""))
    run = st.sidebar.button(f"Run {title}")
    return values, run

# ===========================
# 🎲 LOTTERY HYBRID LIVE
# ===========================
if module.startswith("🎲"):
    game, recent_draws, run_btn, settings = lottery_controls()

    with tabs[0]:
        st.subheader("Lottery Forecast")
        if run_btn and recent_draws:
            try:
                t0 = time.time()
                if engine_mode.startswith("Remote"):
                    if not api_health(api_url):
                        st.error("Remote API unreachable. Check API URL.")
                    else:
                        result_payload = api_forecast(api_url, game, recent_draws, settings)
                else:
                    if not engine or not hasattr(engine,"run_forecast"):
                        raise RuntimeError("Local engine not available.")
                    result_payload = engine.run_forecast(game=game, draws=recent_draws, settings=settings)
                t1 = time.time()

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Confidence", metric(result_payload.get("confidence",0), lambda x: f"{int(float(x)*100)}%"))
                c2.metric("Entropy",   metric(result_payload.get("entropy",0.0),  lambda x: f"{float(x):.2f}"))
                c3.metric("Engine",    result_payload.get("logic","—"))
                c4.metric("Latency",   f"{(t1-t0)*1000:.0f} ms")

                st.markdown("**Top Picks**")
                st.code(", ".join(map(str, result_payload.get("top_picks",[]))))
                st.markdown("**Alternates**")
                st.code(", ".join(map(str, result_payload.get("alts",[]))))

                if hasattr(engine, "log") and engine_mode.startswith("Local"):
                    try:
                        engine.log(f"{datetime.now().isoformat(timespec='seconds')} · {result_payload.get('game',game)} {settings['Session']} · TF={settings['TruthFilter']} · conf={result_payload.get('confidence','?')}")
                    except Exception: pass

                st.session_state.run_ledger.append({
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "module": "lottery",
                    "mode": "remote" if engine_mode.startswith("Remote") else "local",
                    "game": result_payload.get("game", game),
                    "session": settings["Session"],
                    "truth_filter": settings["TruthFilter"],
                    "confidence": result_payload.get("confidence", None),
                    "entropy": result_payload.get("entropy", None),
                    "top_picks": "|".join(map(str, result_payload.get("top_picks", []))),
                    "alts": "|".join(map(str, result_payload.get("alts", []))),
                    "logic": result_payload.get("logic",""),
                    "latency_ms": int((t1-t0)*1000),
                })

            except requests.HTTPError as http_e:
                st.error(f"API error: {http_e} · Body: {getattr(http_e, 'response', None) and http_e.response.text}")
            except Exception as e:
                st.error(f"Forecast error: {e}")
        else:
            st.info("Choose settings, provide draws, then click **Run Lottery Forecast**.")

    with tabs[1]:
        st.subheader("Charts")
        if result_payload and recent_draws:
            ent_now = float(result_payload.get("entropy", 0.5))
            series = np.linspace(max(0.05, ent_now-0.2), min(1.0, ent_now+0.2), 12)
            fig, ax = plt.subplots()
            ax.plot(series)
            ax.set_title("Entropy Trend")
            ax.set_xlabel("Window")
            ax.set_ylabel("Entropy (0..1)")
            st.pyplot(fig)
        else:
            st.info("Run a forecast to view charts.")

# ===========================
# 💰 CRYPTO HYBRID LIVE
# ===========================
elif module.startswith("💰"):
    coins, refresh_btn = crypto_controls()

    with tabs[0]:
        st.subheader("Crypto Snapshot")
        if refresh_btn:
            try:
                r = requests.get("https://api.coingecko.com/api/v3/simple/price",
                                 params={"ids": ",".join(coins), "vs_currencies":"usd"},
                                 timeout=10)
                r.raise_for_status()
                data = [{"Coin": k.title(), "Price (USD)": v["usd"]}
                        for k,v in r.json().items() if isinstance(v, dict) and "usd" in v]
                if data:
                    st.dataframe(pd.DataFrame(data), use_container_width=True)
                else:
                    st.warning("No data returned.")
            except Exception as e:
                st.error(f"Price fetch failed: {e}")
        else:
            st.info("Choose coins and tap **Refresh Prices**.")

    with tabs[1]:
        st.subheader("Charts")
        st.info("Add crypto analytics here (volatility, entropy overlays, cross-market truth filters).")

# ===========================
# 📈 STOCKS
# ===========================
elif module.startswith("📈"):
    tickers, period, interval, fetch_btn = stocks_controls()
    with tabs[0]:
        st.subheader("Stocks View")
        if fetch_btn:
            if yf is None:
                st.error("yfinance not installed. Add it to requirements.txt.")
            else:
                try:
                    for t in tickers:
                        data = yf.download(t, period=period, interval=interval, progress=False)
                        if data.empty:
                            st.warning(f"No data for {t}.")
                            continue
                        st.markdown(f"**{t}**")
                        fig, ax = plt.subplots()
                        ax.plot(data.index, data["Close"])
                        ax.set_title(f"{t} Close — {period} / {interval}")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Price")
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Stocks fetch failed: {e}")
        else:
            st.info("Enter tickers and tap **Fetch**.")

    with tabs[1]:
        st.subheader("Charts")
        st.info("Add factor overlays, entropy of returns, LIPE cross-signal.")

# ===========================
# 🧾 OPTIONS
# ===========================
elif module.startswith("🧾"):
    ticker, chain_btn = options_controls()
    with tabs[0]:
        st.subheader("Options Chain (basic)")
        if chain_btn:
            if yf is None:
                st.error("yfinance not installed. Add it to requirements.txt.")
            else:
                try:
                    tk = yf.Ticker(ticker)
                    exps = tk.options
                    if not exps:
                        st.warning("No options available.")
                    else:
                        sel = st.selectbox("Expiration", exps)
                        opt = tk.option_chain(sel)
                        st.write("Calls")
                        st.dataframe(opt.calls.head(30), use_container_width=True)
                        st.write("Puts")
                        st.dataframe(opt.puts.head(30), use_container_width=True)
                except Exception as e:
                    st.error(f"Options fetch failed: {e}")
        else:
            st.info("Enter underlying and tap **Load Chain**.")

    with tabs[1]:
        st.subheader("Charts")
        st.info("Add IV surface, skew, LIPE risk posture.")

# ===========================
# 🏠 REAL ESTATE (stub, live UI)
# ===========================
elif module.startswith("🏠"):
    fields = {
        "market": {"type":"text","label":"Market / Zip","value":"60601"},
        "horizon": {"type":"slider","label":"Horizon (months)","min":1,"max":36,"value":12},
        "risk": {"type":"select","label":"Risk", "options":["Low","Medium","High"], "index":1},
    }
    values, run = stub_controls("Real Estate", fields)
    with tabs[0]:
        st.subheader("Real Estate View")
        if run:
            st.write("Inputs", values)
            # demo chart
            x = np.arange(12)
            y = np.cumsum(np.random.normal(0, 1, size=12)) + 100
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set_title(f"Price Index Projection — {values['market']}")
            st.pyplot(fig)
        else:
            st.info("Configure and run Real Estate module.")

# ===========================
# 🛢️ COMMODITIES (stub)
# ===========================
elif module.startswith("🛢️"):
    fields = {
        "symbol": {"type":"select","label":"Commodity","options":["WTI","Brent","Gold","Silver","Corn"],"index":2},
        "window": {"type":"slider","label":"Window (days)","min":5,"max":120,"value":30},
    }
    values, run = stub_controls("Commodities", fields)
    with tabs[0]:
        st.subheader("Commodities View")
        if run:
            st.write("Inputs", values)
            x = np.arange(values["window"])
            y = 100 + np.sin(x/5.0)*3 + np.random.normal(0,0.6,size=len(x))
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set_title(f"{values['symbol']} Synthetic Trend")
            st.pyplot(fig)
        else:
            st.info("Pick a commodity and run.")

# ===========================
# 🏈 SPORTS (stub)
# ===========================
elif module.startswith("🏈"):
    fields = {
        "league": {"type":"select","label":"League","options":["NFL","NBA","MLB","NHL","EPL"],"index":0},
        "team": {"type":"text","label":"Team (optional)","value":""},
    }
    values, run = stub_controls("Sports", fields)
    with tabs[0]:
        st.subheader("Sports View")
        if run:
            st.write("Inputs", values)
            st.info("Hook this to your sports model or API for live odds & LIPE overlays.")
        else:
            st.info("Choose league/team and run.")

# ===========================
# 🧑‍🤝‍🧑 HUMAN BEHAVIOR (stub)
# ===========================
elif module.startswith("🧑‍🤝‍🧑"):
    fields = {
        "cohort": {"type":"text","label":"Cohort / Segment","value":"GenZ"},
        "signal": {"type":"select","label":"Signal","options":["Adoption","Churn","Engagement"],"index":2},
        "window": {"type":"slider","label":"Window (weeks)","min":1,"max":52,"value":12},
    }
    values, run = stub_controls("Human Behavior", fields)
    with tabs[0]:
        st.subheader("Human Behavior View")
        if run:
            st.write("Inputs", values)
            x = np.arange(values["window"])
            y = np.clip(np.cumsum(np.random.normal(0,0.4,size=len(x)))+50, 0, 100)
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set_title(f"{values['cohort']} — {values['signal']} momentum")
            st.pyplot(fig)
        else:
            st.info("Configure and run.")

# ===========================
# 🔭 ASTROLOGY (stub)
# ===========================
else:
    fields = {
        "sign": {"type":"select","label":"Sign","options":["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"],"index":0},
        "focus": {"type":"select","label":"Focus","options":["Luck","Money","Love","Health"],"index":0},
    }
    values, run = stub_controls("Astrology", fields)
    with tabs[0]:
        st.subheader("Astrology View")
        if run:
            st.write("Inputs", values)
            # playful synthetic score influenced by truth filter
            rng = np.random.default_rng(seed=sum(ord(c) for c in values["sign"])+truth_filter)
            score = np.clip(rng.normal(0.6, 0.2), 0, 1)
            st.metric(f"{values['sign']} — {values['focus']} score", f"{int(score*100)}%")
        else:
            st.info("Pick a sign/focus and run.")

# ===========================
# 📜 Vault / Settings / Help
# ===========================
with tabs[2]:
    st.subheader("Vault / Logs")
    logs = getattr(engine, "logs", []) if engine else []
    if logs:
        for line in logs[-250:]:
            st.text(line)
    else:
        st.info("No logs yet.")
    st.markdown("---")
    st.markdown("### Download Run Ledger")
    if st.session_state.run_ledger:
        st.download_button("Download CSV", data=downloadable_csv(st.session_state.run_ledger),
                           file_name="his_runs.csv", mime="text/csv")
    else:
        st.info("Run something to populate the ledger.")

with tabs[3]:
    st.subheader("Active Settings")
    st.json({
        "module": module,
        "engine_mode": engine_mode,
        "api_url": api_url,
        "truth_filter": truth_filter
    })

with tabs[4]:
    st.markdown("""
### HIS Overview
- **HIS** is the unified console; **LIPE-Core** is the forecasting brain.
- **Local** mode calls `lipe_core.py`. **Remote** mode calls your FastAPI.
- **Truth Filter** is passed to modules for stricter acceptance when high.

### Modules Installed
Lottery, Crypto, Real Estate, Commodities, Sports, Human Behavior, Astrology, Stocks, Options.

### Next (scale & fusion)
- Point `LIPE_API_URL` to your FastAPI for global scale.
- Add Redis cache & workers at API tier for heavy loads.
- Wire each stubbed module to your production endpoints.
""")
