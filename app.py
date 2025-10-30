import streamlit as st, requests, toml, os

st.set_page_config(page_title="Hybrid Intelligence Systems — Core Engine", layout="wide")

st.sidebar.header("🧠 Choose Your Arena")
arena = st.sidebar.radio("Navigation", [
    "🏠 Home", "🎲 Lottery", "💰 Crypto", "📈 Stocks", "🏆 Sports",
    "🏡 Real Estate", "🪙 Commodities", "🧍 Human Behavior", "🔮 Astrology"
])

# ---- Config from Streamlit Secrets (cloud) or local file (dev) ----
def load_cfg():
    cfg = {"GATEWAY_URL": "", "HIS_KEY": ""}
    try:
        cfg.update(st.secrets)  # Streamlit cloud
    except Exception:
        pass
    if not cfg.get("GATEWAY_URL"):
        try:
            cfg.update(toml.load(".streamlit/secrets.toml"))
        except Exception:
            pass
    return cfg

CFG = load_cfg()
GATEWAY_URL = CFG.get("GATEWAY_URL", "").rstrip("/")
HIS_KEY = CFG.get("HIS_KEY", "")

st.title("🧬 Hybrid Intelligence Systems — Core Engine")
st.caption("Powered by LIPE · Developed by Jesse Ray Landingham Jr")

def gw_get(path: str, timeout=10):
    url = f"{GATEWAY_URL}{path}"
    return requests.get(url, headers={"X-HIS-KEY": HIS_KEY} if HIS_KEY else {}, timeout=timeout)

def check_gateway():
    if not GATEWAY_URL:
        return False, {"status": "offline", "error": "GATEWAY_URL not set"}
    try:
        r = gw_get("/health", timeout=10)
        if r.status_code == 200:
            return True, r.json()
        return False, {"status": "offline", "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return False, {"status": "offline", "error": str(e)}

ok, info = check_gateway()
col1, col2 = st.columns([1,2])
with col1:
    if ok:
        st.success("🟢 Gateway Online")
    else:
        st.error("🚫 Gateway Offline or Unreachable")
    st.json(info)
with col2:
    st.markdown("### Quick Start")
    st.write("- Use the left sidebar to select an arena.")
    st.write("- Each page reads data via the Gateway.")
    st.code(GATEWAY_URL or "(not set)", language="text")

if arena == "🏠 Home":
    st.subheader("🏠 Home Arena")
    st.write("Welcome to Hybrid Intelligence Systems.")
    st.image("https://i.imgur.com/0V9H9qk.png", use_column_width=True)
    st.info("HIS Core Engine powered by LIPE — Tier 33 Intelligence Layer.")
    st.divider()

elif arena == "💰 Crypto":
    st.subheader("💰 Crypto — Demo Forecast")
    payload = {"domain": "crypto", "pair": "BTC-USD", "horizon": "1d"}
    if st.button("Request Forecast"):
        try:
            r = requests.post(f"{GATEWAY_URL}/v1/forecast",
                              json=payload,
                              headers={"X-HIS-KEY": HIS_KEY} if HIS_KEY else {},
                              timeout=20)
            st.json(r.json())
        except Exception as e:
            st.error(str(e))
