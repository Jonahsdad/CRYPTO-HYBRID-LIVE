# ===========================
# LIPE DASHBOARD (HYBRID)
# One-file, drop-in app.py
# ===========================

# --- import path guard (cloud-safe) ---
import sys, os
sys.path.append(os.path.dirname(__file__))

import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# --- LOCAL ENGINE (expects lipe_core.py next to this file) ---
# lipe_core.py must define class LIPE with:
#   - ping() -> dict
#   - run_forecast(game:str, draws:List[int], settings:dict) -> dict
#   - log(msg:str)
#   - logs: list[str]
try:
    from lipe_core import LIPE
except Exception as e:
    LIPE = None  # Remote API mode can still run without local import
    st.session_state["_import_error"] = f"Local LIPE unavailable: {e}"

# ===========================
# CONFIG / CONSTANTS
# ===========================
API_URL_DEFAULT = "https://YOUR-FORECAST-API.example.com"  # set your FastAPI URL when ready
API_URL = os.getenv("LIPE_API_URL", API_URL_DEFAULT)

APP_TITLE = "🧠 LIPE — Living Intelligence Predictive Engine"
APP_SUBTITLE = "Hybrid dashboard (Local Engine or Remote API)"

# ===========================
# UTILITIES
# ===========================
def parse_draws_text(s: str) -> List[int]:
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except Exception:
        return []

def parse_draws_csv(file) -> List[int]:
    try:
        df = pd.read_csv(file)
        for col in ["draw", "Draw", "number", "Number", "value", "Value"]:
            if col in df.columns:
                vals = [int(x) for x in df[col].dropna().tolist()]
                if vals:
                    return vals
        return []
    except Exception:
        return []

def downloadable_csv(rows: List[Dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")

def entropy_score(digits: List[int]) -> float:
    if not digits:
        return 0.0
    vals, counts = np.unique(digits, return_counts=True)
    p = counts / counts.sum()
    h = -np.sum(p * np.log2(p))
    return float(h / np.log2(max(2, len(vals))))  # normalized 0..1

def safe_metric(val, fmt=lambda x: x):
    try:
        return fmt(val)
    except Exception:
        return "—"

# ===========================
# REMOTE API CLIENT (HYBRID)
# ===========================
def api_health(url: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{url.rstrip('/')}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def api_forecast(url: str, game: str, draws: List[int], settings: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"game": game, "draws": draws, "settings": settings}
    r = requests.post(f"{url.rstrip('/')}/forecast", json=payload, timeout=15)
    r.raise_for_status()
    return r.json()

# ===========================
# PANELS (Charts / Crypto / Lottery / Vault)
# ===========================
def charts_panel(forecast: Optional[Dict[str, Any]], recent_draws: List[int]):
    st.markdown("### 📊 Charts")
    if not forecast:
        st.info("Run a forecast to view charts.")
        return

    # Entropy trend demo (single matplotlib plot, no explicit colors)
    if recent_draws:
        ent_now = float(forecast.get("entropy", 0.5))
        series = np.linspace(max(0.05, ent_now - 0.2), min(1.0, ent_now + 0.2), 12)
        fig, ax = plt.subplots()
        ax.plot(series)
        ax.set_title("Entropy Trend")
        ax.set_xlabel("Window")
        ax.set_ylabel("Entropy (0..1)")
        st.pyplot(fig)
    else:
        st.info("Provide draws for entropy visualization.")

def crypto_panel():
    st.markdown("### 💰 Crypto Hybrid")
    st.caption("Live prices via CoinGecko (public endpoint).")
    coins = ["bitcoin", "ethereum", "solana", "dogecoin"]
    data = []
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": ",".join(coins), "vs_currencies": "usd"},
            timeout=10,
        )
        r.raise_for_status()
        js = r.json()
        for c in coins:
            if c in js and "usd" in js[c]:
                data.append({"Coin": c.title(), "Price (USD)": js[c]["usd"]})
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No data returned from CoinGecko.")
    except Exception as e:
        st.error(f"Crypto fetch failed: {e}")

def lottery_panel(engine_local: Optional[Any]):
    st.markdown("### 🎲 Lottery Tools")
    st.caption("Quick single-run utility using current LIPE settings.")
    txt = st.text_input("Enter draws (comma-separated)", value="439,721,105,387,902,114,296,431")
    col1, col2 = st.columns(2)
    g = col1.selectbox("Game", ["Pick 3", "Pick 4", "Lucky Day Lotto"])
    mem = col2.slider("Rolling Memory (local)", 10, 240, 60, step=5)

    if st.button("Run Lottery Forecast (Local)"):
        if not engine_local:
            st.error("Local engine not available.")
            return
        try:
            draws = parse_draws_text(txt)
            result = engine_local.run_forecast(
                game=g,
                draws=draws,
                settings={"RollingMemory": mem, "Session": "Evening", "BonusWeighting": "Moderate",
                          "UseNBC": True, "UseRP": True, "UseEcho": True},
            )
            st.json(result)
        except Exception as e:
            st.error(f"Local run failed: {e}")

def vault_panel(engine_logs: Optional[List[str]], run_ledger: List[Dict[str, Any]]):
    st.markdown("### 🧠 Vault / Logs")
    colL, colR = st.columns(2)

    with colL:
        st.markdown("#### Engine Logs")
        if engine_logs:
            for line in engine_logs[-250:]:
                st.text(line)
        else:
            st.info("No logs yet.")

    with colR:
        st.markdown("#### Download Run Ledger")
        if run_ledger:
            st.download_button(
                "Download CSV",
                data=downloadable_csv(run_ledger),
                file_name="lipe_runs.csv",
                mime="text/csv",
            )
        else:
            st.info("Run a forecast to populate the ledger.")

# ===========================
# STREAMLIT APP
# ===========================
st.set_page_config(page_title="LIPE Dashboard", layout="wide")
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

# Engine mode (Hybrid)
st.sidebar.header("Engine Mode")
engine_mode = st.sidebar.radio("Compute", ["Local (in-app)", "Remote API"], index=0)
custom_api = st.sidebar.text_input("API URL (optional)", value=API_URL)

# Controls
st.sidebar.header("Controls")
game = st.sidebar.selectbox("Game", ["Pick 3", "Pick 4", "Lucky Day Lotto"])
session = st.sidebar.radio("Session", ["Midday", "Evening"], index=0)
rolling_memory = st.sidebar.slider("Rolling Memory (draws)", 10, 240, 60, step=5)
bonus_weighting = st.sidebar.select_slider("Bonus Weighting", options=["None", "Light", "Moderate", "Heavy"], value="Moderate")

st.sidebar.subheader("Strategy")
use_nbc  = st.sidebar.toggle("NBC Triggers", value=True)
use_rp   = st.sidebar.toggle("RP Memory Recall", value=True)
use_echo = st.sidebar.toggle("Echo Logic", value=True)

st.sidebar.divider()
st.sidebar.subheader("Recent draws")
draws_text = st.sidebar.text_area("A) Paste as comma-separated integers", value="439,721,105,387,902,114,296,431", height=80)
uploaded   = st.sidebar.file_uploader("B) Or upload CSV with a 'draw' column", type=["csv"])
run_btn    = st.sidebar.button("Run Forecast")

# Build draws
recent_draws = parse_draws_csv(uploaded) if uploaded else parse_draws_text(draws_text)
if not recent_draws:
    st.warning("Provide recent draws: paste comma-separated values OR upload a CSV with a `draw` column.", icon="⚠️")

# Initialize local engine
engine = LIPE() if LIPE else None
status = engine.ping() if engine and hasattr(engine, "ping") else {
    "name": "LIPE", "tier": 33, "status": "Active", "boot_time": datetime.now().isoformat(timespec="seconds")
}
st.caption(f"{status.get('name','LIPE')} · Tier {status.get('tier','—')} · {status.get('status','—')} · Boot {status.get('boot_time','—')}")

# Ledger
if "run_ledger" not in st.session_state:
    st.session_state.run_ledger = []

# Settings object shared with engine/API
settings = {
    "Session": session,
    "RollingMemory": int(rolling_memory),
    "BonusWeighting": str(bonus_weighting),
    "UseNBC": bool(use_nbc),
    "UseRP": bool(use_rp),
    "UseEcho": bool(use_echo),
}

# Tabs
tabs = st.tabs(["📈 Forecast", "📊 Charts", "💰 Crypto", "🎲 Lottery", "📜 Vault", "⚙️ Settings", "❓ Help"])
forecast_result: Optional[Dict[str, Any]] = None

with tabs[0]:
    st.subheader("Forecast")
    if run_btn and recent_draws:
        try:
            if engine_mode.startswith("Remote"):
                # Remote API call
                health = api_health(custom_api)
                if not health:
                    st.error("Remote API unreachable. Check LIPE_API_URL or 'API URL' above.")
                else:
                    t0 = time.time()
                    forecast_result = api_forecast(custom_api, game, recent_draws, settings)
                    t1 = time.time()
                    # metrics row
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Confidence", safe_metric(forecast_result.get("confidence", 0), lambda x: f"{int(float(x)*100)}%"))
                    c2.metric("Entropy",   safe_metric(forecast_result.get("entropy", 0.0), lambda x: f"{float(x):.2f}"))
                    c3.metric("Engine",    forecast_result.get("logic", "—"))
                    c4.metric("Latency",   f"{(t1 - t0)*1000:.0f} ms")

                    st.markdown("**Top Picks**")
                    st.code(", ".join(map(str, forecast_result.get("top_picks", []))))

                    st.markdown("**Alternates**")
                    st.code(", ".join(map(str, forecast_result.get("alts", []))))

                    st.caption(f"Trace: {forecast_result.get('trace_id','—')} · Model: {forecast_result.get('model_version','—')}")

                    st.session_state.run_ledger.append({
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "mode": "remote",
                        "game": forecast_result.get("game", game),
                        "session": settings["Session"],
                        "confidence": forecast_result.get("confidence", None),
                        "entropy": forecast_result.get("entropy", None),
                        "top_picks": "|".join(map(str, forecast_result.get("top_picks", []))),
                        "alts": "|".join(map(str, forecast_result.get("alts", []))),
                        "logic": forecast_result.get("logic", ""),
                        "RollingMemory": settings["RollingMemory"],
                        "BonusWeighting": settings["BonusWeighting"],
                        "NBC": settings["UseNBC"],
                        "RP": settings["UseRP"],
                        "Echo": settings["UseEcho"],
                        "latency_ms": int((t1 - t0) * 1000),
                    })

            else:
                # Local engine call
                if not engine or not hasattr(engine, "run_forecast"):
                    raise RuntimeError("Local engine not available.")
                t0 = time.time()
                forecast_result = engine.run_forecast(game=game, draws=recent_draws, settings=settings)
                t1 = time.time()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Confidence", safe_metric(forecast_result.get("confidence", 0), lambda x: f"{int(float(x)*100)}%"))
                c2.metric("Entropy",   safe_metric(forecast_result.get("entropy", 0.0), lambda x: f"{float(x):.2f}"))
                c3.metric("Engine",    forecast_result.get("logic", "—"))
                c4.metric("Latency",   f"{(t1 - t0)*1000:.0f} ms")

                st.markdown("**Top Picks**")
                st.code(", ".join(map(str, forecast_result.get("top_picks", []))))

                st.markdown("**Alternates**")
                st.code(", ".join(map(str, forecast_result.get("alts", []))))

                if hasattr(engine, "log"):
                    try:
                        engine.log(f"{datetime.now().isoformat(timespec='seconds')} · {forecast_result.get('game', game)} {settings['Session']} · conf={forecast_result.get('confidence','?')}")
                    except Exception:
                        pass

                st.session_state.run_ledger.append({
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "mode": "local",
                    "game": forecast_result.get("game", game),
                    "session": settings["Session"],
                    "confidence": forecast_result.get("confidence", None),
                    "entropy": forecast_result.get("entropy", None),
                    "top_picks": "|".join(map(str, forecast_result.get("top_picks", []))),
                    "alts": "|".join(map(str, forecast_result.get("alts", []))),
                    "logic": forecast_result.get("logic", ""),
                    "RollingMemory": settings["RollingMemory"],
                    "BonusWeighting": settings["BonusWeighting"],
                    "NBC": settings["UseNBC"],
                    "RP": settings["UseRP"],
                    "Echo": settings["UseEcho"],
                    "latency_ms": int((t1 - t0) * 1000),
                })

        except requests.HTTPError as http_e:
            st.error(f"API error: {http_e} · Body: {getattr(http_e, 'response', None) and http_e.response.text}")
        except Exception as e:
            st.error(f"Forecast error: {e}")
    else:
        st.info("Choose settings, provide draws, then click **Run Forecast**.")

with tabs[1]:
    charts_panel(forecast_result, recent_draws)

with tabs[2]:
    crypto_panel()

with tabs[3]:
    lottery_panel(engine)

with tabs[4]:
    logs = getattr(engine, "logs", []) if engine else []
    vault_panel(logs, st.session_state.run_ledger)

with tabs[5]:
    st.subheader("Active Settings")
    st.json({
        "engine_mode": engine_mode,
        "api_url": custom_api,
        "game": game,
        "settings": settings,
        "draws_count": len(recent_draws),
    })
    if "_import_error" in st.session_state:
        st.warning(st.session_state["_import_error"])

with tabs[6]:
    st.markdown("""
**How to use**
1) Select **Engine Mode** (Local/Remote).  
2) Choose **Game**, set **Session**, memory and strategy toggles.  
3) Paste draws or upload CSV (with `draw` column).  
4) Click **Run Forecast** → review picks, confidence, entropy, charts.  
5) Use **Vault** to inspect logs and download the run ledger.

**Hybrid notes**
- Local mode calls `lipe_core.py` in this repo.  
- Remote mode calls your FastAPI at `LIPE_API_URL` or the URL above.  
- Switch anytime via the sidebar toggle.

**Next**
- Point `LIPE_API_URL` to your FastAPI service to scale globally.  
- Add caching/queues in the API for heavy traffic.  
- Keep Streamlit for internal analytics & premium dashboards.
""")
