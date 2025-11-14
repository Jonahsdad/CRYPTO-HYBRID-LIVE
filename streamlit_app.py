# =========================
# FILE: streamlit_app.py
# HIS — Streamlit Flagship (Crypto v1, single-file)
# =========================
from __future__ import annotations

import os, math, time, random, json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objs as go
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import streamlit as st

# ---------- basic setup ----------
st.set_page_config(page_title="HIS — Powered by LIPE", page_icon="⚡", layout="wide")

# inline CSS (award-winning vibes, minimal deps)
st.markdown("""
<style>
:root { --ink:#e7ecff; --muted:#9bb0ff; --brand:#7c5cff; --brand2:#00e5ff; --good:#1dd75b; --bad:#ff4d6d; }
.block { padding:10px 12px; border:1px solid rgba(124,92,255,.18); border-radius:12px; background:rgba(124,92,255,.05); }
.kpi { display:inline-block; margin:4px 8px 4px 0; padding:6px 10px; border-radius:999px;
       background:rgba(124,92,255,.08); border:1px solid rgba(124,92,255,.18); font-size:13px; color:#9bb0ff; }
.hero h1 { margin:0; font-weight:900; }
.hero .sub { color:var(--muted); }
.btn { display:inline-block; padding:10px 14px; border-radius:10px; background:linear-gradient(135deg,var(--brand),var(--brand2));
       color:white; text-decoration:none; font-weight:700; }
.badge { font-size:12px; padding:4px 8px; border-radius:999px; }
.badge-ok { background:rgba(29,215,91,.12); border:1px solid #1dd75b55; }
.badge-warn { background:rgba(255,179,0,.12); border:1px solid #ffb30055; }
.badge-bad { background:rgba(255,77,109,.12); border:1px solid #ff4d6d55; }
</style>
""", unsafe_allow_html=True)

# ---------- config / identity ----------
API_BASE_DEFAULT = os.getenv("HIS_API_BASE", "").rstrip("/")
TENANT_DEFAULT   = os.getenv("HIS_TENANT_ID", "punch-dev")
EMAIL_DEFAULT    = os.getenv("HIS_USER_EMAIL", "you@punch.dev")
BEARER_DEFAULT   = os.getenv("HIS_BEARER", "")

# keep a small state
if "api_base" not in st.session_state: st.session_state.api_base = API_BASE_DEFAULT
if "tenant"   not in st.session_state: st.session_state.tenant   = TENANT_DEFAULT
if "email"    not in st.session_state: st.session_state.email    = EMAIL_DEFAULT
if "bearer"   not in st.session_state: st.session_state.bearer   = BEARER_DEFAULT

# ---------- http client with retries ----------
def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=5, connect=3, read=3, backoff_factor=0.25,
                    status_forcelist=[429,500,502,503,504],
                    allowed_methods={"GET","POST"}, respect_retry_after_header=True)
    ad = HTTPAdapter(max_retries=retries)
    s.mount("http://", ad); s.mount("https://", ad)
    return s

S = _session()
TIMEOUT: Tuple[float, float] = (3.0, 25.0)

def _hdr() -> Dict[str,str]:
    h = {"Accept":"application/json",
         "x-tenant-id": st.session_state.tenant or TENANT_DEFAULT,
         "x-user-email": st.session_state.email or EMAIL_DEFAULT}
    if st.session_state.bearer:
        h["Authorization"] = f"Bearer {st.session_state.bearer}"
    return h

# ---------- demo fallback (works offline) ----------
def _demo_forecast(symbol:str, horizon:int) -> Dict[str,Any]:
    now = datetime.now(timezone.utc)
    pts = []
    base = 50000.0 if symbol.upper().startswith("BTC") else 3000.0
    # recent history stub
    for i in range(120):
        t = now - timedelta(hours=(120 - i))
        val = base * (1 + 0.03*math.sin(i/12.0)) * (1 + 0.01*random.random())
        pts.append({"ts": t.isoformat(), "yhat": val, "q10": val*0.98, "q90": val*1.02})
    # horizon points
    for j in range(horizon):
        t = now + timedelta(hours=j+1)
        drift = 1 + 0.001*j
        val = pts[-1]["yhat"] * drift * (1 + 0.01*math.sin(j/3.0))
        pts.append({"ts": t.isoformat(), "yhat": val, "q10": val*0.975, "q90": val*1.025})
    return {
        "event": {
            "symbol": symbol,
            "forecast": {"points": pts[-(horizon+40):]},  # recent + horizon
            "metrics": {"entropy": 0.33, "edge": 0.27, "regime":"Compression→Expansion"},
        }
    }

def _demo_signals(symbol:str, limit:int=50) -> List[Dict[str,Any]]:
    now = datetime.now(timezone.utc)
    out = []
    for i in range(limit):
        out.append({
            "ts": (now - timedelta(minutes=5*i)).isoformat(),
            "symbol": symbol,
            "kind": random.choice(["breakout","entropy","volatility"]),
            "score": round(random.uniform(0.1, 0.9), 2),
            "message": random.choice([
                "Edge rising vs baseline",
                "Entropy compression detected",
                "q90 proximity breach"]),
            "edge": round(random.uniform(0.05, 0.35), 2),
            "entropy": round(random.uniform(0.2, 0.8), 2)
        })
    return out

def _demo_wins(symbol:str, limit:int=50) -> List[Dict[str,Any]]:
    now = datetime.now(timezone.utc)
    out = []
    for i in range(limit):
        out.append({
            "ts": (now - timedelta(hours=2*i)).isoformat(),
            "symbol": symbol,
            "label": random.choice(["Green Flag","Echo Win","Band Ride"]),
            "score": round(random.uniform(0.2, 0.95), 2),
            "meta": {"note":"demo"}
        })
    return out

def _demo_strategy(symbol:str, lookback:int) -> Dict[str,Any]:
    now = datetime.now(timezone.utc)
    eq = 100.0
    curve=[]
    for i in range(lookback):
        eq *= (1 + random.uniform(-0.01, 0.015))
        curve.append({"ts": (now - timedelta(days=lookback-i)).isoformat(), "equity": eq})
    return {"metrics":{"HitRate":0.58,"ROI":(eq/100.0-1.0),"MaxDrawdown":0.22,"Trades":48}, "equity_curve":curve}

# ---------- core API calls (with graceful fallback) ----------
def api_ping() -> Dict[str,Any]:
    base = st.session_state.api_base
    if not base: return {"ok": True, "version":"demo"}
    r = S.get(f"{base}/ping", headers=_hdr(), timeout=TIMEOUT)
    r.raise_for_status(); return r.json()

def api_forecast(arena:str, symbol:str, horizon:int) -> Dict[str,Any]:
    base = st.session_state.api_base
    if not base: return _demo_forecast(symbol,horizon)
    try:
        r = S.post(f"{base}/forecast", headers=_hdr(),
                   json={"arena": arena, "symbol": symbol, "horizon": int(horizon)},
                   timeout=TIMEOUT)
        if r.status_code == 402:  # paywall
            raise requests.HTTPError("402", response=r)
        r.raise_for_status()
        res = r.json()
        return res if "event" in res else {"event": res}
    except requests.HTTPError as e:
        if getattr(e, "response", None) and e.response.status_code == 402:
            return {"__402__": True}
        # fallback so you can still demo UI
        return _demo_forecast(symbol,horizon)

def api_signals(arena:str, symbol:str|None, limit:int) -> List[Dict[str,Any]]:
    base = st.session_state.api_base
    if not base: return _demo_signals(symbol or "BTCUSDT", limit)
    try:
        path = f"{base}/signals/current?arena={arena}&limit={int(limit)}"
        if symbol: path += f"&symbol={symbol}"
        r = S.get(path, headers=_hdr(), timeout=TIMEOUT)
        r.raise_for_status()
        res = r.json()
        return res if isinstance(res, list) else res.get("signals", [])
    except Exception:
        return _demo_signals(symbol or "BTCUSDT", limit)

def api_wins(arena:str, symbol:str|None, limit:int) -> List[Dict[str,Any]]:
    base = st.session_state.api_base
    if not base: return _demo_wins(symbol or "BTCUSDT", limit)
    try:
        path = f"{base}/wins?arena={arena}&limit={int(limit)}"
        if symbol: path += f"&symbol={symbol}"
        r = S.get(path, headers=_hdr(), timeout=TIMEOUT)
        r.raise_for_status()
        res = r.json()
        return res if isinstance(res, list) else res.get("wins", [])
    except Exception:
        return _demo_wins(symbol or "BTCUSDT", limit)

def api_strategy_eval(arena:str, symbol:str, spec:Dict[str,Any], lookback:int=180) -> Dict[str,Any]:
    base = st.session_state.api_base
    if not base: return _demo_strategy(symbol, lookback)
    try:
        r = S.post(f"{base}/strategy/eval", headers=_hdr(),
                   json={"arena": arena, "symbol": symbol, "spec": spec, "lookback_days": int(lookback)},
                   timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return _demo_strategy(symbol, lookback)

def api_checkout(arena:str, plan:str="pro-monthly", trial_days:int=7) -> Dict[str,Any]:
    base = st.session_state.api_base
    if not base:
        return {"url": "https://example.com/checkout-demo"}
    r = S.post(f"{base}/billing/checkout", headers=_hdr(),
               json={"arena": arena, "plan": plan, "trial_days": int(trial_days)}, timeout=TIMEOUT)
    r.raise_for_status(); return r.json()

def api_portal() -> Dict[str,Any]:
    base = st.session_state.api_base
    if not base: return {"url":"https://example.com/portal-demo"}
    r = S.post(f"{base}/billing/portal", headers=_hdr(), json={}, timeout=TIMEOUT)
    r.raise_for_status(); return r.json()

def api_webhook_status() -> Dict[str,Any]:
    base = st.session_state.api_base
    if not base: return {"ok": True}
    try:
        r = S.get(f"{base}/billing/webhook/status", headers=_hdr(), timeout=TIMEOUT)
        if r.status_code == 404:
            r = S.get(f"{base}/status", headers=_hdr(), timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"ok": False}

# ---------- header ----------
st.markdown("""
<div class="hero">
  <h1>HYBRID INTELLIGENCE SYSTEMS</h1>
  <div class="sub">All arenas. Hybrid live. <b>Powered by LIPE</b>.</div>
</div>
""", unsafe_allow_html=True)

# ---------- sidebar: connection & billing ----------
with st.sidebar:
    st.subheader("Connection")
    st.session_state.api_base = st.text_input("API Base (/v1)", st.session_state.api_base)
    st.session_state.email    = st.text_input("Email", st.session_state.email)
    st.session_state.tenant   = st.text_input("Team / Tenant", st.session_state.tenant)
    st.session_state.bearer   = st.text_input("Bearer (optional)", st.session_state.bearer, type="password")

    colA, colB = st.columns(2)
    with colA:
        if st.button("Ping", use_container_width=True):
            try:
                pong = api_ping()
                st.success(f"Core online • v{pong.get('version','?')}")
            except Exception as e:
                st.error(f"Core unreachable: {e}")
    with colB:
        ws = api_webhook_status()
        healthy = bool(ws.get("ok") or ws.get("webhook_ok"))
        klass = "badge-ok" if healthy else "badge-warn"
        st.markdown(f"<span class='badge {klass}'>Webhook: {'healthy' if healthy else 'check'}</span>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Billing")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Subscribe Crypto", use_container_width=True):
            try:
                ck = api_checkout("crypto", "pro-monthly", 7)
                url = ck.get("url") or ck.get("checkout_url")
                if url: st.markdown(f"[Open checkout ↗]({url})")
                else: st.info("No checkout link returned.")
            except Exception as e:
                st.error(f"Checkout error: {e}")
    with c2:
        if st.button("Customer Portal", use_container_width=True):
            try:
                pr = api_portal()
                url = pr.get("url") or pr.get("portal_url")
                if url: st.markdown(f"[Open portal ↗]({url})")
                else: st.info("No portal link returned.")
            except Exception as e:
                st.error(f"Portal error: {e}")

# ---------- control row ----------
row = st.columns([1.5, 1.0, 1.0, 1.0])
with row[0]:
    symbol = st.text_input("Symbol", value="BTCUSDT")
with row[1]:
    horizon = st.slider("Horizon (steps)", 1, 30, 7)
with row[2]:
    refresh = st.button("Run Forecast", use_container_width=True)
with row[3]:
    st.text("")  # spacer

# ---------- helpers ----------
def plot_forecast(evt: Dict[str,Any]) -> go.Figure:
    fc = evt.get("forecast") or {}
    pts = fc.get("points", [])
    fig = go.Figure()
    if pts:
        xs = [p.get("ts") for p in pts]
        yhat = [p.get("yhat") for p in pts]
        q10  = [p.get("q10", p.get("yhat")) for p in pts]
        q90  = [p.get("q90", p.get("yhat")) for p in pts]
        fig.add_scatter(x=xs, y=q90, mode="lines", name="q90")
        fig.add_scatter(x=xs, y=q10, mode="lines", name="q10", fill="tonexty")
        fig.add_scatter(x=xs, y=yhat, mode="lines", name="yhat", line=dict(dash="dash"))
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), hovermode="x unified")
    return fig

def kpi(label:str, value:str, tone:str="neutral"):
    color = {"good":"#1dd75b","warn":"#ffb300","bad":"#ff4d6d","neutral":"#9bb0ff"}.get(tone,"#9bb0ff")
    st.markdown(f"<span class='kpi' style='color:{color}'><b>{label}</b>: {value}</span>", unsafe_allow_html=True)

# ---------- tabs ----------
tab_overview, tab_signals, tab_wins, tab_backtest = st.tabs(["Overview","Signals","Wins","Backtest"])

# ---- Overview ----
with tab_overview:
    st.subheader("Forecast")
    if refresh:
        res = api_forecast("crypto", symbol, horizon)
        if res.get("__402__"):
            st.warning("Crypto is paywalled. Use **Subscribe** in the sidebar, then re-run.")
        evt = res.get("event", res)
        st.plotly_chart(plot_forecast(evt), use_container_width=True)
        m = evt.get("metrics", {})
        regime  = (m.get("regime") or evt.get("regime") or "—")
        entropy = float(m.get("entropy", evt.get("entropy", float("nan"))))
        edge    = float(m.get("edge", evt.get("edge", float("nan"))))
        tone_e  = "good" if entropy <= 0.35 else ("warn" if entropy <= 0.6 else "bad")
        tone_g  = "good" if edge >= 0.25 else ("warn" if edge >= 0.1 else "neutral")
        kpi("Regime", regime, "neutral")
        kpi("Entropy", f"{entropy:.2f}", tone_e)
        kpi("LIPE Edge", f"{edge:.2f}", tone_g)
    else:
        st.info("Pick a symbol and horizon, then press **Run Forecast**.")

# ---- Signals ----
with tab_signals:
    st.subheader("Current Signals")
    lim = st.slider("Rows", 10, 200, 50, 10)
    if st.button("Load Signals"):
        sigs = api_signals("crypto", symbol, lim)
        if not sigs: st.info("No signals currently.")
        else:
            df = pd.DataFrame(sigs)
            pri = [c for c in ["ts","symbol","kind","score","message","edge","entropy"] if c in df.columns]
            df = df[pri + [c for c in df.columns if c not in pri]]
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                               file_name=f"signals_{symbol}.csv", mime="text/csv")

# ---- Wins ----
with tab_wins:
    st.subheader("Green Flag Wins")
    limw = st.slider("Rows ", 10, 200, 50, 10, key="wins_lim")
    if st.button("Load Wins"):
        wins = api_wins("crypto", symbol, limw)
        if not wins: st.info("No wins yet.")
        else:
            dfw = pd.DataFrame(wins)
            if "ts" in dfw.columns: dfw = dfw.sort_values("ts")
            fig = go.Figure()
            if "ts" in dfw.columns and "score" in dfw.columns:
                fig.add_scatter(x=dfw["ts"], y=dfw["score"], mode="lines+markers", name="GFW score")
                fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), hovermode="x")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(dfw, use_container_width=True, hide_index=True)
            st.download_button("Download CSV", dfw.to_csv(index=False).encode("utf-8"),
                               file_name=f"wins_{symbol}.csv", mime="text/csv")

# ---- Backtest ----
with tab_backtest:
    st.subheader("Quick Backtest (Strategy Eval)")
    c1,c2,c3,c4 = st.columns(4)
    with c1: enter_edge    = st.slider("Enter: LIPE Edge ≥", 0.0, 1.0, 0.30, 0.01)
    with c2: enter_entropy = st.slider("Enter: Entropy ≤", 0.0, 1.0, 0.40, 0.01)
    with c3: exit_entropy  = st.slider("Exit: Entropy ≥",  0.0, 1.0, 0.75, 0.01)
    with c4: lookback      = st.slider("Lookback (days)", 30, 365, 180, 5)

    if st.button("Run Backtest"):
        spec = {
            "horizon": horizon,
            "enter": [{"field":"edge","op":">=","value":enter_edge},
                      {"field":"entropy","op":"<=","value":enter_entropy}],
            "exit":  [{"field":"entropy","op":">=","value":exit_entropy}],
        }
        bt = api_strategy_eval("crypto", symbol, spec, lookback)
        m = bt.get("metrics", {})
        eq = bt.get("equity_curve", [])
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Hit Rate",      f"{m.get('HitRate', float('nan')):.2f}")
        k2.metric("ROI",           f"{m.get('ROI',     float('nan')):.2f}")
        k3.metric("Max Drawdown",  f"{m.get('MaxDrawdown', float('nan')):.2f}")
        k4.metric("Trades",        f"{int(m.get('Trades', 0))}")
        if eq:
            xs = [p.get("ts") for p in eq]
            ys = [float(p.get("equity", 0)) for p in eq]
            fig_eq = go.Figure([go.Scatter(x=xs, y=ys, mode="lines", name="Equity")])
            fig_eq.update_layout(margin=dict(l=10,r=10,t=10,b=10), hovermode="x unified")
            st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.info("No equity curve returned.")

st.caption("HYBRID INTELLIGENCE SYSTEMS • Powered by LIPE • Streamlit Flagship v1")
