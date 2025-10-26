import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any

# ========= LIPE CORE =========
# lipe_core.py must define class LIPE with:
# - ping() -> dict
# - run_forecast(game:str, draws:List[int], settings:dict) -> dict
# - log(msg:str)
# - logs: list[str]
from lipe_core import LIPE

# ---------- helpers ----------
def parse_draws_text(s: str) -> List[int]:
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except Exception:
        return []

def parse_draws_csv(file) -> List[int]:
    try:
        df = pd.read_csv(file)
        # common column names
        for col in ["draw","Draw","number","Number","value","Value"]:
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

# ========= UI =========
st.set_page_config(page_title="LIPE Dashboard", layout="wide")
st.title("🧠 LIPE — Living Intelligence Predictive Engine")

engine = LIPE()
status = engine.ping() if hasattr(engine, "ping") else {
    "name":"LIPE","tier":33,"status":"Active","boot_time":datetime.now().isoformat(timespec="seconds")
}
st.caption(f"{status.get('name','LIPE')} · Tier {status.get('tier','—')} · {status.get('status','—')} · Boot {status.get('boot_time','—')}")

# Sidebar — controls
st.sidebar.header("Controls")
game = st.sidebar.selectbox("Game", ["Pick 3", "Pick 4", "Lucky Day Lotto"])
session = st.sidebar.radio("Session", ["Midday", "Evening"], index=0)
rolling_memory = st.sidebar.slider("Rolling Memory (draws)", 10, 240, 60, step=5)
bonus_weighting = st.sidebar.select_slider("Bonus Weighting", options=["None","Light","Moderate","Heavy"], value="Moderate")

st.sidebar.subheader("Strategy Switches")
use_nbc  = st.sidebar.toggle("NBC Triggers", value=True)
use_rp   = st.sidebar.toggle("RP Memory Recall", value=True)
use_echo = st.sidebar.toggle("Echo Logic", value=True)

st.sidebar.divider()
st.sidebar.subheader("Recent draws input")
draws_text = st.sidebar.text_area("A) Paste comma-separated integers", value="439,721,105,387,902,114,296,431", height=80)
uploaded   = st.sidebar.file_uploader("B) Or upload CSV with a 'draw' column", type=["csv"])
run_btn    = st.sidebar.button("Run Forecast")

# Build draws list
recent_draws = parse_draws_csv(uploaded) if uploaded else parse_draws_text(draws_text)
if not recent_draws:
    st.warning("Provide recent draws: paste comma-separated values OR upload a CSV containing a `draw` column.", icon="⚠️")

# Tabs
tab_forecast, tab_logs, tab_settings, tab_help = st.tabs(["📈 Forecast", "📜 Vault / Logs", "⚙️ Settings Echo", "❓ Help"])

# Persist run ledger
if "run_ledger" not in st.session_state:
    st.session_state.run_ledger = []

# Settings to engine
settings = {
    "Session": session,
    "RollingMemory": int(rolling_memory),
    "BonusWeighting": str(bonus_weighting),
    "UseNBC": bool(use_nbc),
    "UseRP": bool(use_rp),
    "UseEcho": bool(use_echo),
}

with tab_forecast:
    st.subheader("Forecast")
    if run_btn and recent_draws:
        try:
            if not hasattr(engine, "run_forecast"):
                raise AttributeError("lipe_core.LIPE lacks run_forecast(). Add it per template.")
            result: Dict[str, Any] = engine.run_forecast(game=game, draws=recent_draws, settings=settings)

            required = {"game","top_picks","alts","confidence","entropy","logic"}
            if not required.issubset(result.keys()):
                raise ValueError(f"run_forecast() must return keys {required}. Got {list(result.keys())}")

            # Metrics row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Confidence", f"{int(float(result['confidence'])*100)}%")
            c2.metric("Entropy",   f"{float(result['entropy']):.2f}")
            c3.metric("Engine",    result.get("logic","—"))
            c4.metric("Session",   settings['Session'])

            # Picks
            st.markdown("**Top Picks**")
            st.code(", ".join(map(str, result["top_picks"])))
            st.markdown("**Alternates**")
            st.code(", ".join(map(str, result.get("alts", []))))

            # Notes (optional)
            if result.get("notes"):
                with st.expander("Engine Notes"):
                    st.write(result["notes"])

            # Entropy trend (matplotlib, single plot, no explicit colors)
            if len(recent_draws) >= 10:
                ent_now = float(result["entropy"])
                series = np.linspace(max(0.05, ent_now-0.2), min(1.0, ent_now+0.2), 12)
                fig, ax = plt.subplots()
                ax.plot(series)
                ax.set_title("Entropy Trend")
                ax.set_xlabel("Window")
                ax.set_ylabel("Entropy (0..1)")
                st.pyplot(fig)

            # Vault log + ledger
            try:
                engine.log(f"{datetime.now().isoformat(timespec='seconds')} · {result['game']} {settings['Session']} · conf={result['confidence']} · top={result['top_picks']}")
            except Exception:
                pass

            st.session_state.run_ledger.append({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "game": result["game"],
                "session": settings["Session"],
                "confidence": result["confidence"],
                "entropy": result["entropy"],
                "top_picks": "|".join(map(str, result["top_picks"])),
                "alts": "|".join(map(str, result.get("alts", []))),
                "logic": result.get("logic",""),
                "RollingMemory": settings["RollingMemory"],
                "BonusWeighting": settings["BonusWeighting"],
                "NBC": settings["UseNBC"],
                "RP": settings["UseRP"],
                "Echo": settings["UseEcho"]
            })

        except Exception as e:
            st.error(f"Forecast error: {e}")

with tab_logs:
    colL, colR = st.columns(2)
    with colL:
        st.markdown("### Vault Log")
        if hasattr(engine, "logs") and engine.logs:
            for line in engine.logs[-250:]:
                st.text(line)
        else:
            st.info("No logs yet.")
    with colR:
        st.markdown("### Download Run Ledger")
        if st.session_state.run_ledger:
            st.download_button(
                "Download CSV",
                data=downloadable_csv(st.session_state.run_ledger),
                file_name="lipe_runs.csv",
                mime="text/csv"
            )
        else:
            st.info("Run a forecast to populate the ledger.")

with tab_settings:
    st.write("**Active Settings**")
    st.json(settings)
    st.caption("Edit settings in the sidebar, then re-run.")

with tab_help:
    st.markdown("""
**How to use**
1. Choose game (Pick 3 / Pick 4 / Lucky Day).
2. Paste draws like `439,721,105,...` or upload CSV with a `draw` column.
3. Adjust strategy switches (NBC / RP / Echo) and memory/weighting.
4. Click **Run Forecast**. Review confidence, entropy, picks.
5. Check **Vault / Logs** for audit. Download the run ledger if needed.

**Install your latest LIPE logic**
- Put your real logic inside `LIPE.run_forecast()` in `lipe_core.py`.
- Keep the return keys: `game, top_picks, alts, confidence, entropy, logic`.
""")
