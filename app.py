# app.py — LIPE / HIS Dashboard (Production UI)
# - 9 arenas (3x3)
# - NBC buttons per arena
# - Status + latency badges
# - JSON hidden by default (Dev Mode toggle)
# - Graceful fallbacks if an endpoint is missing

import time
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st

APP_VERSION = "v1.1.0"
DEFAULT_API = "https://his-gateway.onrender.com"  # change if you moved the gateway
DEFAULT_STRICTNESS = 55

# ---------- Session Defaults ----------
if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API
if "arena" not in st.session_state:
    st.session_state.arena = "Lottery"
if "dev_mode" not in st.session_state:
    st.session_state.dev_mode = False
if "last_resp" not in st.session_state:
    st.session_state.last_resp = None
if "status_cache" not in st.session_state:
    st.session_state.status_cache = {"ok": False, "service": "HIS_Gateway", "env": "production", "ts": None}

# ---------- Helpers ----------
def _pretty_utc(ts: Optional[float]) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

def _json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)

def get_root_or_status(api: str) -> Tuple[bool, Dict[str, Any], float]:
    """Ping root '/', fall back to '/status'. Returns (ok, payload, latency_sec)."""
    t0 = time.time()
    try:
        r = requests.get(api, timeout=10)
        r.raise_for_status()
        payload = r.json()
        t1 = time.time()
        return True, payload, max(t1 - t0, 0.0)
    except Exception:
        pass

    try:
        r = requests.get(api.rstrip("/") + "/status", timeout=10)
        r.raise_for_status()
        payload = r.json()
        t1 = time.time()
        return True, payload, max(t1 - t0, 0.0)
    except Exception as e:
        return False, {"error": str(e)}, max(time.time() - t0, 0.0)

def call_scan(api: str, arena: str) -> Tuple[bool, Dict[str, Any], float]:
    """
    Try common patterns. We never break the UI:
    1) GET /scan?arena=...
    2) POST /scan {arena}
    3) GET /{arena.lower()}/scan
    Fallback -> stub payload.
    """
    t0 = time.time()
    routes = [
        ("GET", f"{api.rstrip('/')}/scan", {"params": {"arena": arena}}),
        ("POST", f"{api.rstrip('/')}/scan", {"json": {"arena": arena}}),
        ("GET", f"{api.rstrip('/')}/{arena.lower()}/scan", {}),
    ]
    for method, url, kwargs in routes:
        try:
            r = requests.request(method, url, timeout=15, **kwargs)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            return True, r.json(), max(time.time() - t0, 0.0)
        except Exception:
            continue

    # Fallback stub (so UI always renders)
    return False, {
        "provider": "stub",
        "arena": arena,
        "note": "No scan endpoint found on Gateway; showing placeholder.",
        "signals": [],
    }, max(time.time() - t0, 0.0)

def call_nbc(api: str, arena: str) -> Tuple[bool, Dict[str, Any], float]:
    """
    NBC trigger:
    1) POST /nbc/scan {arena}
    2) POST /nbc {arena, mode:'scan'}
    3) GET  /{arena.lower()}/nbc
    Fallback -> stub.
    """
    t0 = time.time()
    routes = [
        ("POST", f"{api.rstrip('/')}/nbc/scan", {"json": {"arena": arena}}),
        ("POST", f"{api.rstrip('/')}/nbc", {"json": {"arena": arena, "mode": "scan"}}),
        ("GET", f"{api.rstrip('/')}/{arena.lower()}/nbc", {}),
    ]
    for method, url, kwargs in routes:
        try:
            r = requests.request(method, url, timeout=20, **kwargs)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            return True, r.json(), max(time.time() - t0, 0.0)
        except Exception:
            continue

    return False, {
        "provider": "stub-nbc",
        "arena": arena,
        "nbc": {"decision": "HOLD", "reason": "No NBC endpoint on Gateway."},
    }, max(time.time() - t0, 0.0)

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("System", divider="grey")
    compute = st.radio("Compute", ["Local (in-app)", "Remote API"], index=1, help="UI runs local; data from Gateway.")
    st.markdown("**API URL (Remote)**")
    st.session_state.api_url = st.text_input(
        "Gateway URL",
        value=st.session_state.api_url,
        label_visibility="collapsed",
        placeholder="https://<your-gateway>.onrender.com",
    )
    strict = st.slider("Signal Strictness", 0, 100, DEFAULT_STRICTNESS, help="Higher = stricter NBC thresholds.")

    st.divider()
    st.subheader("Actions")
    colA, colB = st.columns(2)
    if colA.button("⚡ Ping Gateway", use_container_width=True):
        ok, payload, lat = get_root_or_status(st.session_state.api_url)
        st.session_state.status_cache = {"ok": ok, "payload": payload, "lat": lat, "ts": time.time()}
        if ok:
            st.toast(f"Gateway OK ({lat:.0f} ms)", icon="✅")
        else:
            st.toast("Gateway offline (see Dev Mode for details)", icon="❌")
    if colB.button("🧠 Run NBC", use_container_width=True):
        ok, payload, lat = call_nbc(st.session_state.api_url, st.session_state.arena)
        st.session_state.last_resp = {"nbc": payload, "ok": ok, "lat": lat, "ts": time.time()}
        if ok:
            st.toast(f"NBC updated for {st.session_state.arena} ({lat:.0f} ms)", icon="🧠")
        else:
            st.toast("NBC fallback (no endpoint).", icon="⚠️")

    st.divider()
    st.caption("API URL")
    st.code(st.session_state.api_url, language="text")

# ---------- Header ----------
st.title("🧠 Hybrid Intelligence Systems — Core Engine")
st.caption(f"Powered by LIPE • Production UI • {APP_VERSION}")

# Ensure we have fresh status on initial render
if st.session_state.status_cache.get("ts") is None:
    ok, payload, lat = get_root_or_status(st.session_state.api_url)
    st.session_state.status_cache = {"ok": ok, "payload": payload, "lat": lat, "ts": time.time()}

status = st.session_state.status_cache
ok = status.get("ok", False)
lat = status.get("lat", None)
checked = _pretty_utc(status.get("ts"))

# Status banner
if ok:
    st.success(f"🟢 Gateway Online • {_pretty_utc(status.get('ts'))} • {lat*1000:.0f} ms", icon="✅")
else:
    st.error("🔴 Gateway Offline • check URL / service • Dev Mode for details", icon="🚨")

# ---------- Choose Your Arena (9) ----------
ARENAS = [
    "Lottery", "Crypto", "Stocks",
    "Options", "Commodities", "Real Estate",
    "Sports", "Human Behavior", "Meta Timing",
]

st.subheader("Choose your arena")
rows = [ARENAS[i:i+3] for i in range(0, len(ARENAS), 3)]
for row in rows:
    cols = st.columns(len(row))
    for i, label in enumerate(row):
        active = (st.session_state.arena == label)
        if cols[i].button(f"{'✅ ' if active else ''}{label}", use_container_width=True, key=f"arena_{label}"):
            st.session_state.arena = label
            st.toast(f"Selected arena: {label}", icon="🎯")

# ---------- Home Arena Controls ----------
st.subheader("Home Arena")
st.caption(f"Selected: **{st.session_state.arena}**  •  Strictness: **{strict}**")

c1, c2, c3 = st.columns(3)
if c1.button("🔎 Scan", use_container_width=True):
    ok2, payload2, lat2 = call_scan(st.session_state.api_url, st.session_state.arena)
    st.session_state.last_resp = {"scan": payload2, "ok": ok2, "lat": lat2, "ts": time.time()}
    st.toast(f"Scan { 'OK' if ok2 else 'fallback' } ({lat2:.0f}s)" if lat2 >= 1 else f"Scan done ({lat2*1000:.0f} ms)", icon="🔎")

if c2.button("🧠 NBC Mode", use_container_width=True):
    ok3, payload3, lat3 = call_nbc(st.session_state.api_url, st.session_state.arena)
    st.session_state.last_resp = {"nbc": payload3, "ok": ok3, "lat": lat3, "ts": time.time()}
    decision = payload3.get("nbc", {}).get("decision", "—") if isinstance(payload3, dict) else "—"
    st.toast(f"NBC: {decision}", icon="🧠")

if c3.toggle("👨‍💻 Developer Mode (show payloads)", value=st.session_state.dev_mode, key="dev_mode", help="Toggle visibility of raw JSON."):
    pass  # state handled by Streamlit

# ---------- NBC Summary Banner ----------
nbc_decision = "—"
nbc_note = ""
if st.session_state.last_resp and isinstance(st.session_state.last_resp, dict):
    data = st.session_state.last_resp.get("nbc") or {}
    if isinstance(data, dict) and "nbc" in data:
        nbc_decision = data["nbc"].get("decision", "—")
        nbc_note = data["nbc"].get("reason", "")

nbc_col = st.container()
with nbc_col:
    if nbc_decision == "GO":
        st.success(f"✅ NBC = GO — Execute window open for **{st.session_state.arena}**", icon="🚀")
    elif nbc_decision in ("WAIT", "HOLD"):
        st.warning(f"⏳ NBC = {nbc_decision} — {nbc_note or 'entropy unstable'}", icon="⏳")
    elif nbc_decision == "PAUSE":
        st.error("🛑 NBC = PAUSE — conflicts detected", icon="🧯")
    else:
        st.info("ℹ️ NBC standing by — run Scan or NBC Mode to update.", icon="🧠")

# ---------- Dev Mode Payload (hidden by default) ----------
if st.session_state.dev_mode:
    st.divider()
    st.write("### Developer Payload")
    st.caption("For diagnostics only. Hidden in production.")
    st.code(_json(st.session_state.last_resp or {"note": "no calls yet"}), language="json")

# ---------- Footer ----------
st.divider()
st.caption(f"Running {APP_VERSION} • Gateway: {st.session_state.api_url} • Checked: {checked}")
