# HIS — Streamable Crypto Flagship (API-or-Fallback)
from __future__ import annotations
import os, math, time, requests, numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime, timedelta, timezone

API_BASE = os.getenv("HIS_API_BASE", "").rstrip("/")
TENANT   = os.getenv("HIS_TENANT_ID", "punch-dev")
UEMAIL   = os.getenv("HIS_USER_EMAIL", "you@punch.dev")
TOKEN    = os.getenv("HIS_BEARER", "")

st.set_page_config(page_title="HIS — Powered by LIPE", page_icon="⚡", layout="wide")
st.markdown("""
<style>
:root{ --ink:#e7ecff; --muted:#9bb0ff; }
h1,h2,h3,h4{ letter-spacing:.3px }
.kpi{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
.card{padding:10px;border-radius:12px;border:1px solid rgba(124,92,255,.25);background:rgba(16,22,35,.85)}
.label{color:var(--muted);font-size:12px}
.val{font-size:22px;font-weight:800}
.small{color:var(--muted);font-size:12px}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def _hdr():
    h = {"Accept":"application/json","x-tenant-id":TENANT,"x-user-email":UEMAIL}
    if TOKEN: h["Authorization"] = f"Bearer {TOKEN}"
    return h

def try_lipe_forecast(symbol: str, horizon: int):
    """Call your LIPE Core if HIS_API_BASE is set, else return None to use fallback."""
    if not API_BASE:
        return None
    try:
        r = requests.post(f"{API_BASE}/forecast",
                          headers=_hdr(),
                          json={"arena":"crypto","symbol":symbol,"horizon":horizon},
                          timeout=(3,20))
        r.raise_for_status()
        return r.json()    # expected: {"event":{...}}
    except Exception as e:
        st.toast("API fallback (LIPE unreachable)", icon="⚠️")
        return None

def coingecko_series(coin_id: str, days: int = 180):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    r = requests.get(url, params={"vs_currency":"usd","days":days,"interval":"daily"}, timeout=20)
    r.raise_for_status()
    data = r.json().get("prices", [])
    df = pd.DataFrame(data, columns=["ts_ms","price"])
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df[["ts","price"]]

def simple_forecast(df: pd.DataFrame, horizon: int):
    """Gaussian return model → yhat / q10 / q90 + entropy/regime."""
    y = df["price"].astype(float).values
    rets = np.diff(np.log(y))
    mu, sigma = float(np.mean(rets)), float(np.std(rets) + 1e-9)
    last = float(y[-1])
    steps = horizon
    ts0 = df["ts"].iloc[-1]
    out = []
    # Use geometric BM style extrapolation
    for i in range(1, steps+1):
        t = ts0 + pd.Timedelta(days=i)
        mean_ret = mu * i
        var_ret  = (sigma**2) * i
        yhat = last * math.exp(mean_ret + 0.5*var_ret*0)  # median-ish
        q10  = last * math.exp(np.quantile(rets,0.10)*i)
        q90  = last * math.exp(np.quantile(rets,0.90)*i)
        out.append({"ts": t.isoformat(), "yhat": yhat, "q10": q10, "q90": q90})
    # crude entropy proxy: normalized sigma
    entropy = max(0.0, min(1.0, sigma / 0.05))
    regime  = "EXPANSION" if mu>0 and entropy<0.5 else ("COMPRESSION" if entropy<0.25 else "CHAOTIC")
    edge    = float(np.clip(mu/sigma if sigma>0 else 0.0, -1.5, 1.5))
    return {"forecast": out, "metrics": {"entropy": entropy, "regime": regime, "edge": edge}}

def plot_price_and_forecast(hist_df: pd.DataFrame, fc: dict):
    tr = []
    tr.append(go.Scatter(x=hist_df["ts"], y=hist_df["price"], name="Actual", mode="lines"))
    if fc and fc.get("forecast"):
        f = pd.DataFrame(fc["forecast"])
        f["ts"] = pd.to_datetime(f["ts"], utc=True)
        tr.append(go.Scatter(x=f["ts"], y=f["q90"], name="q90", mode="lines", line=dict(width=0.1), showlegend=False))
        tr.append(go.Scatter(x=f["ts"], y=f["q10"], name="q10", mode="lines",
                             fill="tonexty", line=dict(width=0.1), fillcolor="rgba(124,92,255,.18)", showlegend=False))
        tr.append(go.Scatter(x=f["ts"], y=f["yhat"], name="Forecast", mode="lines", line=dict(dash="dash", width=2)))
    fig = go.Figure(tr)
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), hovermode="x unified")
    return fig

# ---------- UI ----------
st.title("HYBRID INTELLIGENCE SYSTEMS")
st.caption("All arenas. Hybrid live. **Powered by LIPE**.")

left, right = st.columns([2,1])

with right:
    st.subheader("Controls")
    coin = st.selectbox("Symbol", ["BTCUSDT","ETHUSDT","SOLUSDT"], index=0)
    horizon = st.slider("Horizon (days)", 1, 30, 5)
    use_api = st.toggle("Use LIPE API if available", value=bool(API_BASE))
    if st.button("Run Forecast", use_container_width=True):
        st.session_state["_run"] = time.time()

with left:
    st.subheader("Chart")

# Run once on load
if "_run" not in st.session_state:
    st.session_state["_run"] = time.time()

# Map ticker→coingecko id for fallback
cg_map = {"BTCUSDT":"bitcoin", "ETHUSDT":"ethereum", "SOLUSDT":"solana"}

# 1) Try LIPE Core
lipe_res = try_lipe_forecast(coin, horizon) if use_api else None

# 2) Fetch history + forecast (API or fallback)
if lipe_res:
    evt = lipe_res.get("event", {})
    hist = evt.get("context", {}).get("history", [])
    if hist:
        hdf = pd.DataFrame(hist)
        hdf["ts"] = pd.to_datetime(hdf["ts"], utc=True)
        hdf.rename(columns={"close":"price"}, inplace=True)
    else:
        hdf = coingecko_series(cg_map[coin], days=200)
    fc = {"forecast": evt.get("forecast", {}).get("points", []),
          "metrics": evt.get("metrics", {})}
else:
    hdf = coingecko_series(cg_map[coin], days=200)
    fc = simple_forecast(hdf, horizon)

fig = plot_price_and_forecast(hdf, fc)
st.plotly_chart(fig, use_container_width=True, height=520)

# KPIs
m = fc.get("metrics", {})
entropy = float(m.get("entropy", 0.0))
regime  = str(m.get("regime", "—"))
edge    = float(m.get("edge", 0.0))

st.markdown('<div class="kpi">', unsafe_allow_html=True)
for label, val in [
    ("Regime", regime),
    ("Entropy", f"{entropy:.2f}"),
    ("Edge", f"{edge:+.2f}"),
    ("Last Price (USD)", f"{hdf['price'].iloc[-1]:,.2f}")
]:
    st.markdown(f'<div class="card"><div class="label">{label}</div><div class="val">{val}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="small">Tip: set <code>HIS_API_BASE</code> in Streamlit Secrets to use LIPE Core. Without it, the app uses a public fallback feed with a statistical band model.</div>', unsafe_allow_html=True)
