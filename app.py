# app.py — HIS Dashboard (Production UI) for live2 gateway
import os, json, time
from datetime import datetime, timezone
import requests
import streamlit as st

# ========= Config & Session Defaults =========
st.set_page_config(page_title="HIS — Core Engine", layout="wide")
SS = st.session_state

DEFAULT_API = os.getenv("HIS_API_URL", "https://crypto-hybrid-live-2.onrender.com")
SS.setdefault("api_url", DEFAULT_API)
SS.setdefault("compute", "Remote API")          # "Local (in-app)" | "Remote API"
SS.setdefault("strictness", 55)                 # 0—100
SS.setdefault("arena", "Lottery")
SS.setdefault("developer", False)               # Developer Mode toggle
SS.setdefault("last_payload", None)             # last request/response for Dev Mode panel

ARENAS = [
    ("Lottery", "🏛️"),
    ("Crypto", "🟩"),
    ("Stocks", "📈"),
    ("Options", "🧮"),
    ("Commodities", "⛽"),
    ("Real Estate", "🏠"),
    ("Sports", "🏟️"),
    ("Human Behavior", "🧠"),
    ("Meta Timing", "⏳"),
]

# ========= Small helpers =========
def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")

def api_get(path, timeout=12):
    url = SS["api_url"].rstrip("/") + path
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, {"error": str(e), "url": url}

def api_post(path, payload, timeout=18):
    url = SS["api_url"].rstrip("/") + path
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, {"error": str(e), "url": url, "payload": payload}

def gateway_ok():
    ok, data = api_get("/status")
    return ok, data

def nbc_endpoint_available():
    # Treat 200 at /nbc as available, anything else: fallback
    ok, data = api_post("/nbc", {"ping": True})
    if ok and isinstance(data, dict):
        # If the gateway handles the ping test, it will echo something; that's enough
        return True
    return False

# ========= Sidebar (system controls) =========
with st.sidebar:
    st.subheader("System")
    compute = st.radio("Compute", ["Local (in-app)", "Remote API"], index=1 if SS["compute"]=="Remote API" else 0)
    SS["compute"] = compute

    api_input = st.text_input("API URL (Remote)", SS["api_url"])
    SS["api_url"] = api_input.strip()

    SS["strictness"] = st.slider("Signal Strictness", 0, 100, SS["strictness"])

    st.subheader("Actions")
    colA, colB = st.columns(2)
    if colA.button("⚡ Ping Gateway", use_container_width=True):
        ok, data = gateway_ok()
        SS["last_payload"] = {"action":"ping","ok":ok,"data":data,"ts":now_iso()}
        st.toast("Gateway OK ✅" if ok else "Gateway offline ❌", icon="✅" if ok else "❌")

    run_nbc_clicked = colB.button("🧠 Run NBC", use_container_width=True)

    st.subheader("Modes")
    m1, m2 = st.columns(2)
    SS["developer"] = bool(m2.toggle("🔐 Developer Mode (show payloads)", value=SS["developer"]))

# ========= Header & Gateway status =========
st.markdown("## 🧠 Hybrid Intelligence Systems — **Core Engine**")
st.caption("Powered by LIPE • Production UI • v1.1.0")

ok, gdata = gateway_ok()
status_box = st.empty()
if ok:
    status_box.success(f"✅ Gateway Online • {gdata.get('ts','')}", icon="✅")
else:
    status_box.error("❌ Gateway Offline — check API URL or Render service.", icon="❌")

# ========= Arena chooser =========
st.markdown("### Choose your arena")
rows = [ARENAS[i:i+3] for i in range(0, len(ARENAS), 3)]
for row in rows:
    c = st.columns(3)
    for (label, emoji), slot in zip(row, c):
        active = (SS["arena"] == label)
        btn = slot.button(f"{emoji} {label}", use_container_width=True)
        if btn:
            SS["arena"] = label

st.markdown("### Home Arena")
st.write(f"Selected: **{SS['arena']}** • Strictness: **{SS['strictness']}**")

# Primary actions for the selected arena
pc1, pc2, pc3 = st.columns([1,1,1])
scan_clicked = pc1.button("🔎 Scan", use_container_width=True)
nbc_toggle   = pc2.toggle("🧠 NBC Mode", value=False, help="When on, we’ll prefer NBC forecast over simple scan.")
# Developer Mode toggle already in sidebar; repeat here only as a label:
pc3.write("")

# ========= Run logic (Scan / NBC) =========
def ui_show_result(title, payload, resp, ok_flag):
    if ok_flag:
        st.success(title, icon="✅")
    else:
        st.warning(title, icon="⚠️")
    if SS["developer"]:
        with st.expander("Developer payloads"):
            st.code(json.dumps({"request": payload, "response": resp}, indent=2), language="json")

def run_scan():
    if SS["compute"].startswith("Local"):
        # Placeholder local compute; in production we always use gateway
        data = {"provider":"local", "arena":SS["arena"], "ts":now_iso()}
        ui_show_result("Scan (local fallback)", {"mode":"scan-local"}, data, True)
        return

    payload = {
        "mode": "scan",
        "arena": SS["arena"],
        "strictness": SS["strictness"],
        "ts": now_iso(),
    }
    ok2, resp = api_post("/scan", payload)
    SS["last_payload"] = {"action":"scan", "ok":ok2, "request":payload, "response":resp}
    title = f"Scan {'OK' if ok2 else 'FAILED'} — {SS['arena']}"
    ui_show_result(title, payload, resp, ok2)

def run_nbc():
    if SS["compute"].startswith("Local"):
        data = {"note":"NBC local fallback", "arena":SS["arena"], "ts":now_iso()}
        ui_show_result("NBC (local fallback)", {"mode":"nbc-local"}, data, True)
        return

    # Try NBC endpoint; if missing, tell the user and do a scan fallback
    # First, make a lightweight probe to avoid long error delays
    ok_probe = False
    try:
        ok_probe = nbc_endpoint_available()
    except Exception:
        ok_probe = False

    if not ok_probe:
        st.info("🧠 NBC fallback (no endpoint on Gateway). Running standard Scan.", icon="ℹ️")
        run_scan()
        return

    payload = {
        "mode": "nbc",
        "arena": SS["arena"],
        "strictness": SS["strictness"],
        "ts": now_iso(),
    }
    ok3, resp = api_post("/nbc", payload)
    SS["last_payload"] = {"action":"nbc", "ok":ok3, "request":payload, "response":resp}
    title = f"NBC {'OK' if ok3 else 'FAILED'} — {SS['arena']}"
    ui_show_result(title, payload, resp, ok3)

# Trigger actions
if run_nbc_clicked or nbc_toggle:
    run_nbc()
elif scan_clicked:
    run_scan()

# Footer / diagnostics (minimal, clean)
st.markdown("---")
st.caption(f"Running v1.1.0 • Gateway: {SS['api_url']} • Checked: {now_iso()}")
