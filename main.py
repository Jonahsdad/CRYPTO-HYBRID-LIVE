# main.py  — HIS Gateway (FastAPI)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Dict, Any

app = FastAPI(title="HIS Gateway", version="v1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

def nowz() -> str:
    return datetime.utcnow().isoformat() + "Z"

@app.get("/")
def root():
    return {
        "message": "Welcome to the HIS Gateway ⚡ powered by LIPE",
        "status": "running",
        "docs": "/docs",
        "service": "HIS_Gateway",
        "env": "production",
        "ts": nowz(),
    }

@app.get("/status")
def status():
    return {
        "ok": True,
        "service": "HIS_Gateway",
        "env": "production",
        "status": "running",
        "checked": nowz()
    }

# ---------- Forecast/Scan placeholders (safe no-404) ----------
@app.post("/v1/lipe/forecast")
def lipe_forecast(payload: Dict[str, Any]):
    # Echo request; stub a predictable shape
    game = payload.get("game", "pick4")
    window = payload.get("window", "last_30")
    mode = payload.get("mode", "standard")
    strict = payload.get("strictness", 55)
    return {
        "provider": "lipe-forecast",
        "requested": {"game": game, "window": window, "mode": mode, "strictness": strict},
        "note": "Add real LIPE logic here to replace this stub."
    }

@app.post("/v1/crypto/scan")
def crypto_scan(payload: Dict[str, Any]):
    uni: List[str] = payload.get("universe", [])
    return {
        "provider": "crypto-scan",
        "universe": uni,
        "signals": [{"symbol": s, "score": 0.0, "note": "stub"} for s in uni],
        "ts": nowz()
    }

@app.post("/v1/stocks/scan")
def stocks_scan(payload: Dict[str, Any]):
    wl: List[str] = payload.get("watchlist", [])
    return {
        "provider": "stocks-scan",
        "watchlist": wl,
        "signals": [{"ticker": t, "score": 0.0, "note": "stub"} for t in wl],
        "ts": nowz()
    }
