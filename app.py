# ===========================
# HYBRID INTELLIGENCE SYSTEMS (HIS)
# app.py — full, drop-in
# ===========================

# --- import path guard (cloud-safe) ---
import sys, os
sys.path.append(os.path.dirname(__file__))

import time, json
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

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

def entropy_score(digits: List[int]) -> float:
    if not digits: return 0.0
    vals, counts = np.unique(digits, return_counts=True)
    p = counts / counts.sum()
    h = -np.sum(p * np.log2(p))
    return float(h / np.log2(max(2, len(vals))))

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

def metric(val, fmt=lambda x: x):
    try: return fmt(val)
    except Exception: return "—"

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
# simple visual: progress bar + label
st.sidebar.progress(truth_filter/100.0)
st.sidebar.caption(f"Truth Filter · {truth_filter}%")

st.sidebar.divider()
module = st.sidebar.selectbox("Module", ["🎲 Lottery Hybrid Live", "💰 Crypto Hybrid Live"])

# -------- Shared controls (shown/hidden per module below) --------
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
    run_btn    = st.sidebar.button("Run Forecast")

    draws = parse_draws_csv(uploaded) if uploaded else parse_draws_text(draws_text)
    if not draws:
        st.warning("Provide recent draws: paste comma-separated values OR upload a CSV with a `draw` column.", icon="⚠️")

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

def crypto_controls():
    st.sidebar.header("Crypto Controls")
    coins_default = "bitcoin,ethereum,solana,dogecoin"
    coins_str = st.sidebar.text_input("Coins (comma-separated ids)", value=coins_default,
                                      help="CoinGecko ids, e.g., bitcoin,ethereum,solana")
    refresh_btn = st.sidebar.button("Refresh Prices")
    return [c.strip() for c in coins_str.split(",") if c.strip()], refresh_btn

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

# -------- Tabs common to modules --------
tabs = st.tabs(["📈 Forecast", "📊 Charts", "📜 Vault", "⚙️ Settings", "❓ Help"])
forecast_result: Optional[Dict[str, Any]] = None

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
                        forecast_result = api_forecast(api_url, game, recent_draws, settings)
                else:
                    if not engine or not hasattr(engine,"run_forecast"):
                        raise RuntimeError("Local engine not available.")
                    forecast_result = engine.run_forecast(game=game, draws=recent_draws, settings=settings)
                t1 = time.time()

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Confidence", metric(forecast_result.get("confidence",0), lambda x: f"{int(float(x)*100)}%"))
                c2.metric("Entropy",   metric(forecast_result.get("entropy",0.0),  lambda x: f"{float(x):.2f}"))
                c3.metric("Engine",    forecast_result.get("logic","—"))
                c4.metric("Latency",   f"{(t1-t0)*1000:.0f} ms")

                st.markdown("**Top Picks**")
                st.code(", ".join(map(str, forecast_result.get("top_picks",[]))))
                st.markdown("**Alternates**")
                st.code(", ".join(map(str, forecast_result.get("alts",[]))))

                if hasattr(engine, "log") and engine_mode.startswith("Local"):
                    try:
                        engine.log(f"{datetime.now().isoformat(timespec='seconds')} · {forecast_result.get('game',game)} {settings['Session']} · TF={settings['TruthFilter']} · conf={forecast_result.get('confidence','?')}")
                    except Exception: pass

                st.session_state.run_ledger.append({
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "module": "lottery",
                    "mode": "remote" if engine_mode.startswith("Remote") else "local",
                    "game": forecast_result.get("game", game),
                    "session": settings["Session"],
                    "truth_filter": settings["TruthFilter"],
                    "confidence": forecast_result.get("confidence", None),
                    "entropy": forecast_result.get("entropy", None),
                    "top_picks": "|".join(map(str, forecast_result.get("top_picks", []))),
                    "alts": "|".join(map(str, forecast_result.get("alts", []))),
                    "logic": forecast_result.get("logic",""),
                    "latency_ms": int((t1-t0)*1000),
                })

            except requests.HTTPError as http_e:
                st.error(f"API error: {http_e} · Body: {getattr(http_e, 'response', None) and http_e.response.text}")
            except Exception as e:
                st.error(f"Forecast error: {e}")
        else:
            st.info("Choose settings, provide draws, then click **Run Forecast**.")

    with tabs[1]:
        st.subheader("Charts")
        if forecast_result and recent_draws:
            ent_now = float(forecast_result.get("entropy", 0.5))
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
else:
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
### How HIS works
- **HIS** is the unified console.  
- **LIPE-Core** is the brain running forecasts.  
- **Local** mode calls `lipe_core.py`. **Remote** mode calls your FastAPI (`LIPE_API_URL`).  
- **Truth Filter** tightens/loosens acceptance; it is passed to LIPE for stricter logic.

### Tips
- Lottery: paste draws or upload CSV with `draw` column.  
- Crypto: list CoinGecko ids (e.g., `bitcoin,ethereum`).  
- Vault: download your run ledger for audit.

### Next
- Point `LIPE_API_URL` to your API to scale globally.  
- Add Redis cache & workers in the API tier for heavy loads.  
""")
