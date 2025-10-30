# services/gateway_main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import Literal, Optional
from datetime import datetime, timezone

SERVICE_NAME = os.getenv("SERVICE", "HIS Gateway")
ENV = os.getenv("ENV", "production")

# If you later want to lock this down, set HIS_KEY in Render env and uncomment check_key()
HIS_KEY = os.getenv("HIS_KEY", "")  # optional

app = FastAPI(
    title="HIS Gateway",
    description="Hybrid Intelligence Systems Gateway — handles requests from LIPE / Streamlit UI.",
    version="1.0.0",
)

# Allow Streamlit UI to call this service
origins = [
    os.getenv("UI_ORIGIN", "*"),   # set to https://his-ui.onrender.com in Render env later for tighter security
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models -------------------------------------------------------------------
class HealthOut(BaseModel):
    status: Literal["ok"]
    service: str
    env: str
    ts: str

class TestOut(BaseModel):
    ok: bool
    echo: str
    service: str
    env: str
    ts: str

class ForecastOut(BaseModel):
    domain: Literal["home", "lottery", "crypto", "stocks", "sports", "real_estate", "commodities", "human", "astrology"]
    status: str
    note: str
    sample: dict
    service: str
    env: str
    ts: str

# --- Helpers ------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

ARENA_MAP = {
    "home": "home",
    "lottery": "lottery",
    "crypto": "crypto",
    "stocks": "stocks",
    "sports": "sports",
    "real_estate": "real_estate",
    "commodities": "commodities",
    "human": "human",          # Human Behavior
    "astrology": "astrology",
}

SAMPLE_PAYLOADS = {
    "home": {"message": "Welcome to LIPE / HIS"},
    "lottery": {"next_draw": "TBD", "stub_pick3": [1, 2, 3], "stub_pick4": [1, 2, 3, 4]},
    "crypto": {"tickers": ["BTC", "ETH", "SOL"], "signal": "neutral"},
    "stocks": {"tickers": ["SPY", "QQQ", "NVDA"], "signal": "neutral"},
    "sports": {"league": "NFL", "game_count": 0},
    "real_estate": {"mortgage_rate": "stub", "region": "US"},
    "commodities": {"WTI": "stub", "Gold": "stub"},
    "human": {"insight": "behavioral stub"},
    "astrology": {"sign": "Libra", "note": "stub"},
}

# def check_key(key: Optional[str]) -> None:
#     if HIS_KEY and key != HIS_KEY:
#         raise HTTPException(status_code=401, detail="Invalid HIS-KEY")

# --- Routes -------------------------------------------------------------------
@app.get("/", summary="Root")
def root():
    return {
        "message": "Welcome to the HIS Gateway — powered by LIPE.",
        "status": "running",
        "docs": "/docs",
        "service": SERVICE_NAME,
        "env": ENV,
    }

@app.get("/health", response_model=HealthOut, summary="Health")
def health():
    return HealthOut(status="ok", service=SERVICE_NAME, env=ENV, ts=now_iso())

@app.get("/api/test", response_model=TestOut, summary="Ping / Echo")
def api_test(msg: str = Query("pong", description="Message to echo")):
    return TestOut(ok=True, echo=msg, service=SERVICE_NAME, env=ENV, ts=now_iso())

@app.get(
    "/api/forecast",
    response_model=ForecastOut,
    summary="Forecast stub for each arena"
)
def api_forecast(
    domain: Literal[
        "home", "lottery", "crypto", "stocks", "sports",
        "real_estate", "commodities", "human", "astrology"
    ] = Query(..., description="Arena/domain to forecast")
    # , his_key: Optional[str] = Header(None)  # enable if you want key auth
):
    # check_key(his_key)
    d = ARENA_MAP[domain]
    sample = SAMPLE_PAYLOADS[d]
    return ForecastOut(
        domain=d,
        status="ok",
        note="Stub data — plug your real model here.",
        sample=sample,
        service=SERVICE_NAME,
        env=ENV,
        ts=now_iso(),
    )
