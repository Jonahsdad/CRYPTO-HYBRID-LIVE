# HYBRID INTELLIGENCE SYSTEMS - Neon UX (Hybrid Astrology + Arena Tiles) - FULL APP
# One-file drop-in. No external modules required.

import sys, os, time, math, random
from datetime import datetime, timedelta
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# Optional market lib
try:
    import yfinance as yf
except Exception:
    yf = None

# Optional local engine
try:
    from lipe_core import LIPE
except Exception as e:
    LIPE = None
    _IMPORT_ERR = f"Local LIPE unavailable: {e}"
else:
    _IMPORT_ERR = None

# ---------------------- App Config ----------------------
APP_TITLE = "HYBRID INTELLIGENCE SYSTEMS"
APP_TAGLINE = "One brain. Many frontiers."
API_URL_DEFAULT = "https://YOUR-FORECAST-API.example.com"
API_URL = os.getenv("LIPE_API_URL", API_URL_DEFAULT)

MODULES = [
    ("🎰", "Lottery", "Daily numbers, picks, entropy, risk modes"),
    ("💰", "Crypto", "Live pricing, signals, overlays"),
    ("📈", "Stocks", "Charts, momentum, factor overlays"),
    ("🧾", "Options", "Chains, quick IV views"),
    ("🏡", "Real Estate", "Market tilt and projections"),
    ("🛢️", "Commodities", "Energy, metals, ag"),
    ("🏈", "Sports", "Game signals and parlay entropy"),
    ("🧠", "Human Behavior", "Cohort trends and intent"),
    ("🌌", "Astrology", "Planetary cycles and symbolic overlays"),
]

st.set_page_config(page_title="HIS — Hybrid Intelligence Systems", layout="wide")

# ---------------------- CSS ----------------------
st.markdown("""
<style>
:root { --accent:#8be9fd; --accent2:#ff79c6; --gold:#f7d774; --bg:#0e0f12; --card:#151722; --muted:#8a8fa3; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#12141b,#0f0f14); }
.neon-hero{ padding:22px 24px;border-radius:16px;
  background: radial-gradient(900px 250px at 15% -20%, rgba(139,233,253,.16), transparent),
              radial-gradient(900px 250px at 85% -20%, rgba(255,121,198,.14), transparent),
              linear-gradient(180deg,#141620,#0f1118);
  border:1px solid rgba(255,255,255,.06);
  box-shadow:0 0 14px rgba(139,233,253,.06), inset 0 0 18px rgba(255,121,198,.04);
  text-align:center; }
.pill{ display:inline-block;padding:4px 10px;border-radius:100px;font-size:12px;
  background:rgba(139,233,253,.12);border:1px solid rgba(139,233,253,.25); margin:0 6px 6px 0;}
.hero-title{ font-size:32px;font-weight:800;margin:4px 0 4px 0;color:#fff; }
.signature{ font-size:16px;color:var(--muted);margin-top:4px; }
.signature span{ color:var(--gold);font-weight:800; }
.hero-tag{ color:var(--muted);margin-top:8px;font-size:14px; }

.tile{ background:var(--card);border:1px solid rgba(255,255,255,.06);
  border-radius:16px;padding:16px;cursor:pointer;
  transition:transform .12s, box-shadow .12s, border-color .12s;}
.tile:hover{ transform:translateY(-2px);
  box-shadow:0 8px 20px rgba(139,233,253,.10), 0 0 0 1px rgba(139,233,253,.12) inset;
  border-color:rgba(139,233,253,.25); }
.tile-emoji{ font-size:24px; } .tile-title{ font-weight:700;margin-bottom:4px;}
.tile-desc{ color:var(--muted);font-size:14px;}

hr{ border:none;height:1px;background:rgba(255,255,255,.06); }
</style>
""", unsafe_allow_html=True)

# ---------------------- Helpers ----------------------
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

def api_health(url: str):
    try:
        r = requests.get(f"{url.rstrip('/')}/health", timeout=5); r.raise_for_status(); return r.json()
    except Exception: return None

def api_forecast(url: str, game: str, draws: List[int], settings: Dict[str, Any]):
    r = requests.post(f"{url.rstrip('/')}/forecast",
                      json={"game":game,"draws":draws,"settings":settings},
                      timeout=20)
    r.raise_for_status(); return r.json()

# ---------------------- Astro Sync (hidden meta-layer) ----------------------
def astro_get_influence() -> Dict[str, float]:
    day = datetime.utcnow().timetuple().tm_yday
    mars = abs(math.sin(day / 58.6))
    venus = abs(math.sin(day / 224.7))
    mercury = abs(math.sin(day / 87.97))
    jupiter = abs(math.sin(day / 433.0))
    saturn = abs(math.cos(day / 10759.0))
    cosmic = (mars + venus + mercury + jupiter + saturn) / 5.0
    return {
        "mars": round(mars,3),
        "venus": round(venus,3),
        "mercury": round(mercury,3),
        "jupiter": round(jupiter,3),
        "saturn": round(saturn,3),
        "cosmic_index": round(cosmic,3)
    }

def astro_adjust(value: float, strength: float = 0.5):
    """Blend numeric value with synthetic planetary influence."""
    data = astro_get_influence()
    modulation = 1.0 + ((data["cosmic_index"] - 0.5) * strength)
    return max(0.0, value * modulation), data

# ---------------------- Astrology Arena helpers (visible layer) ----------------------
def astro_positions(planet: str = "Mars", days_back: int = 30):
    base = datetime.utcnow()
    out = []
    for i in range(days_back):
        d = base - timedelta(days=i)
        degree = (math.sin(i/5.0) * 180.0 / math.pi) % 360.0
        retro = random.choice([True, False])
        out.append({"date": d.date(), "degree": round(degree,2), "retrograde": retro})
    return out

def astro_interpret(deg: float) -> str:
    if 0 <= deg < 90: return "New beginnings"
    if 90 <= deg < 180: return "Growth and tension"
    if 180 <= deg < 270: return "Reflection and correction"
    return "Completion and harvest"

# ---------------------- Sidebar ----------------------
st.sidebar.header("System")
engine_mode = st.sidebar.radio("Compute", ["Local (in-app)", "Remote API"], index=0)
api_url = st.sidebar.text_input("API URL (Remote)", value=API_URL)

if "truth_filter" not in st.session_state: st.session_state.truth_filter = 55
truth_filter = st.sidebar.slider("Truth Filter", 0, 100, st.session_state.truth_filter, step=5)
st.session_state.truth_filter = truth_filter

astro_on = st.sidebar.checkbox("Enable Astrology Influence", value=True)
st.sidebar.caption(f"Truth Filter: {truth_filter}%")
st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
st.sidebar.caption("Module-specific controls live on each page.")

# ---------------------- Session State ----------------------
if "selected_module" not in st.session_state:
    st.session_state.selected_module = None
if "ledger" not in st.session_state:
    st.session_state.ledger = []

# ---------------------- Engine ----------------------
engine = LIPE() if LIPE else None
status = engine.ping() if engine and hasattr(engine, "ping") else {
    "name":"LIPE", "tier":33, "status":"Active",
    "boot_time": datetime.now().isoformat(timespec="seconds")
}

# ---------------------- Landing ----------------------
def landing():
    st.markdown(f"""
<div class="neon-hero">
  <div class="pill">Hybrid - Local/Remote</div>
  <div class="pill">Engine: LIPE-Core - Tier {status.get('tier','—')} - {status.get('status','—')}</div>
  <h1 class="hero-title">🧠 {APP_TITLE}</h1>
  <div class="signature">Powered by <span>JESSE RAY LANDINGHAM JR</span></div>
  <div class="hero-tag">{APP_TAGLINE}</div>
</div>
""", unsafe_allow_html=True)

    st.write("")
    st.subheader("Choose your arena")
    rows = [MODULES[i:i+3] for i in range(0, len(MODULES), 3)]
    for row in rows:
        cols = st.columns(len(row))
        for col, (emoji, title, desc) in zip(cols, row):
            with col:
                if st.button(f"{emoji}  {title}", key=f"btn-{title}", use_container_width=True):
                    st.session_state.selected_module = title
                    st.rerun()
                st.markdown(f"""
<div class="tile" onclick="document.querySelector('button[data-testid=&quot;btn-{title}&quot;]')?.click()">
  <div class="tile-emoji">{emoji}</div>
  <div class="tile-title">{title}</div>
  <div class="tile-desc">{desc}</div>
</div>
""", unsafe_allow_html=True)

# ---------------------- Topbar with Back (fixed) ----------------------
def topbar():
    cols = st.columns([6,1])
    with cols[0]:
        label = f"🧠 {APP_TITLE}"
        if st.session_state.selected_module:
            label += f" — {st.session_state.selected_module}"
        st.markdown(f"#### {label}")
    with cols[1]:
        if st.session_state.selected_module:
            if st.button("Back", use_container_width=True):
                st.session_state.selected_module = None
                st.rerun()

# ---------------------- Modules ----------------------
def page_lottery():
    st.markdown("### 🎰 Lottery")
    if not engine and engine_mode.startswith("Local"):
        st.info("Local engine unavailable; switch to Remote API or add lipe_core.py.")
    game = st.selectbox("Game", ["Pick 3","Pick 4","Lucky Day Lotto"])
    session = st.radio("Session", ["Midday","Evening"], horizontal=True)
    c1,c2,c3 = st.columns(3)
    rolling = c1.slider("Rolling Memory (draws)", 10, 240, 60, step=5)
    bonus   = c2.select_slider("Bonus Weighting", ["None","Light","Moderate","Heavy"], value="Moderate")
    c3.markdown("&nbsp;")
    nb = c3.toggle("NBC", True); rp = c3.toggle("RP", True); echo = c3.toggle("Echo", True)

    st.markdown("**Recent draws**")
    txt = st.text_area("Comma-separated integers", value="439,721,105,387,902,114,296,431", height=80)
    up = st.file_uploader("Or upload CSV with a 'draw' column", type=["csv"])
    draws = parse_draws_csv(up) if up else parse_draws_text(txt)
    run = st.button("Run Lottery Forecast", type="primary")

    if run and not draws:
        st.warning("Please provide recent draws.")

    if run and draws:
        settings = {
            "Session": session, "RollingMemory": int(rolling), "BonusWeighting": bonus,
            "UseNBC": bool(nb), "UseRP": bool(rp), "UseEcho": bool(echo),
            "TruthFilter": int(st.session_state.truth_filter)
        }
        try:
            t0 = time.time()
            if engine_mode.startswith("Remote"):
                if not api_health(api_url): st.error("Remote API unreachable."); return
                result = api_forecast(api_url, game, draws, settings)
            else:
                if not engine or not hasattr(engine, "run_forecast"):
                    raise RuntimeError("Local engine not available.")
                result = engine.run_forecast(game=game, draws=draws, settings=settings)
            t1 = time.time()

            # Metrics
            conf = float(result.get("confidence", 0.0))
            ent  = float(result.get("entropy", 0.0))
            if astro_on:
                conf, astro_data = astro_adjust(conf, strength=0.35)
                st.caption(f"Astrology Influence ON — Cosmic Index {astro_data['cosmic_index']}")

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Confidence", f"{int(conf*100)}%")
            c2.metric("Entropy", f"{ent:.2f}")
            c3.metric("Engine", result.get("logic","—"))
            c4.metric("Latency", f"{(t1-t0)*1000:.0f} ms")

            st.markdown("**Top Picks**"); st.code(", ".join(map(str, result.get("top_picks",[]))))
            st.markdown("**Alternates**"); st.code(", ".join(map(str, result.get("alts",[]))))

            st.session_state.ledger.append({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "module":"lottery", "mode":"remote" if engine_mode.startswith("Remote") else "local",
                "game": result.get("game", game), "session": session,
                "truth": st.session_state.truth_filter,
                "conf": conf, "ent": ent,
                "top": "|".join(map(str, result.get("top_picks", []))),
                "alts": "|".join(map(str, result.get("alts", []))),
                "logic": result.get("logic",""), "latency_ms": int((t1-t0)*1000)
            })
        except Exception as e:
            st.error(f"Forecast error: {e}")

def page_crypto():
    st.markdown("### 💰 Crypto")
    coins = st.text_input("Coins (CoinGecko ids)", value="bitcoin,ethereum,solana,dogecoin")
    if st.button("Refresh Prices", type="primary"):
        try:
            r = requests.get("https://api.coingecko.com/api/v3/simple/price",
                             params={"ids": coins, "vs_currencies":"usd"}, timeout=10)
            r.raise_for_status()
            data = [{"Coin": k.title(), "Price (USD)": v.get("usd")} for k,v in r.json().items()]
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            if astro_on:
                _, astro_data = astro_adjust(1.0, strength=0.2)
                st.caption(f"Astrology Influence ON — Cosmic Index {astro_data['cosmic_index']} (informing risk overlays)")
        except Exception as e:
            st.error(f"Price fetch failed: {e}")

def page_stocks():
    st.markdown("### 📈 Stocks")
    c1,c2,c3 = st.columns(3)
    tickers = c1.text_input("Tickers", value="AAPL,MSFT,NVDA")
    period  = c2.selectbox("Period", ["1mo","3mo","6mo","1y","2y"], index=2)
    interval= c3.selectbox("Interval", ["1d","1h","30m"], index=0)
    if st.button("Fetch", type="primary"):
        if yf is None:
            st.error("yfinance not installed."); return
        for t in [x.strip().upper() for x in tickers.split(",") if x.strip()]:
            try:
                data = yf.download(t, period=period, interval=interval, progress=False)
                if data.empty: st.warning(f"No data for {t}."); continue
                st.markdown(f"**{t}**")
                fig, ax = plt.subplots(); ax.plot(data.index, data["Close"])
                ax.set_title(f"{t} Close - {period} / {interval}")
                ax.set_xlabel("Date"); ax.set_ylabel("Price")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"{t} fetch failed: {e}")

def page_options():
    st.markdown("### 🧾 Options")
    c1,c2 = st.columns([1,2])
    ticker = c1.text_input("Underlying", value="AAPL")
    if c1.button("Load Chain", type="primary"):
        if yf is None: st.error("yfinance not installed."); return
        try:
            tk = yf.Ticker(ticker); exps = tk.options
            if not exps: st.warning("No options."); return
            sel = c2.selectbox("Expiration", exps)
            opt = tk.option_chain(sel)
            st.write("Calls"); st.dataframe(opt.calls.head(30), use_container_width=True)
            st.write("Puts");  st.dataframe(opt.puts.head(30), use_container_width=True)
        except Exception as e:
            st.error(f"Chain failed: {e}")

def page_realestate():
    st.markdown("### 🏡 Real Estate")
    st.info("Plug mortgage rates, zip-level indices, and supply metrics here.")
    # Demo curve
    x = np.arange(60); y = 100 + np.cumsum(np.random.normal(0, 0.3, size=60))
    fig, ax = plt.subplots(); ax.plot(x,y); ax.set_title("Market Tilt Demo"); st.pyplot(fig)

def page_commodities():
    st.markdown("### 🛢️ Commodities")
    x = list(range(30)); y = [100 + (i*0.2) + random.uniform(-0.5,0.5) for i in x]
    fig, ax = plt.subplots(); ax.plot(x,y); ax.set_title("Gold demo curve"); st.pyplot(fig)

def page_sports():
    st.markdown("### 🏈 Sports")
    st.info("Wire odds feeds (The Odds API / Sportradar) and LIPE overlays.")
    teams = ["Hawks","Sharks","Tigers"]; odds = [random.randint(-150, 250) for _ in teams]
    df = pd.DataFrame({"Team":teams,"Moneyline":odds})
    st.dataframe(df, use_container_width=True)
    if astro_on:
        _, astro_data = astro_adjust(1.0, strength=0.25)
        st.caption(f"Astrology Influence ON — Cosmic Index {astro_data['cosmic_index']}")

def page_behavior():
    st.markdown("### 🧠 Human Behavior")
    kw = st.text_input("Keyword", value="crypto")
    if st.button("Run Sentiment Demo"):
        xs = list(range(60))
        ys = [50 + 10*math.sin(i/8.0) + random.uniform(-2,2) for i in xs]
        fig, ax = plt.subplots(); ax.plot(xs,ys); ax.set_title(f"Sentiment proxy — {kw}")
        st.pyplot(fig)

def page_astrology():
    st.markdown("### 🌌 Astrology")
    st.write("Explore planetary motion and symbolic cycles. Toggle in sidebar to influence other arenas.")
    planet = st.selectbox("Planet", ["Mercury","Venus","Mars","Jupiter","Saturn"])
    days = st.slider("Days back", 7, 120, 30)
    data = astro_positions(planet, days)
    df = pd.DataFrame(data)
    if not df.empty:
        fig, ax = plt.subplots()
        ax.plot(df["date"], df["degree"]); ax.set_title(f"{planet} degrees over time")
        ax.set_xlabel("Date"); ax.set_ylabel("Degree")
        st.pyplot(fig)
        st.success(f"Current symbolic phase: {astro_interpret(float(df.iloc[0]['degree']))}")
    st.markdown("**Recent Retrograde Status (last 10)**")
    st.dataframe(df[["date","retrograde"]].head(10), use_container_width=True)

# ---------------------- Router ----------------------
def open_module(name: str):
    n = (name or "").lower()
    if "lottery" in n: return page_lottery()
    if "crypto" in n: return page_crypto()
    if "stocks" in n: return page_stocks()
    if "options" in n: return page_options()
    if "real" in n: return page_realestate()
    if "commod" in n: return page_commodities()
    if "sports" in n: return page_sports()
    if "human" in n: return page_behavior()
    if "astrology" in n: return page_astrology()
    return landing()

# ---------------------- Main ----------------------
topbar()
if st.session_state.selected_module:
    open_module(st.session_state.selected_module)
else:
    landing()

# ---------------------- Footer ----------------------
st.markdown("---")
cL, cR = st.columns([2,1])
with cL:
    st.caption(f"Engine: LIPE-Core — Tier {status.get('tier','—')} — {status.get('status','—')} — Boot {status.get('boot_time','—')}")
with cR:
    if st.session_state.ledger:
        st.download_button("Download Run Ledger", data=dl_csv(st.session_state.ledger),
                           file_name="his_runs.csv", mime="text/csv")
    else:
        st.caption("Run something to populate the ledger.")
