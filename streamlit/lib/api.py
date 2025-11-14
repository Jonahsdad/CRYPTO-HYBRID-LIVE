import os, requests
TIMEOUT = (3, 20)
API_BASE = os.getenv("API_BASE_URL", "").rstrip("/")

def set_api_base(url: str):  # optional, if you want a text box to override
    global API_BASE
    API_BASE = (url or "").rstrip("/")

def forecast(arena: str, symbol: str, horizon: int, token: str | None = None):
    if not API_BASE:
        raise RuntimeError("API_BASE_URL not set")
    h = {"Accept":"application/json"}
    if token: h["Authorization"] = f"Bearer {token}"
    r = requests.post(f"{API_BASE}/v1/forecast",
                      json={"arena":arena, "symbol":symbol, "horizon":horizon},
                      headers=h, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()
