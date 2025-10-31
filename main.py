FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any
from datetime import datetime
import random

app = FastAPI(title="HIS Gateway", version="1.0")

# Allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────────────
# HEALTH & STATUS ROUTES
# ───────────────────────────────
@app.get("/")
def root():
    return {"service": "HIS_Gateway", "ok": True, "ts": datetime.utcnow().isoformat() + "Z"}

@app.get("/ping")
def ping():
    return {"status": "ok", "service": "HIS_Gateway", "ts": datetime.utcnow().isoformat() + "Z"}

@app.get("/status")
def status():
    return {
        "status": "ok",
        "service": "HIS_Gateway",
        "env": "production",
        "ts": datetime.utcnow().isoformat() + "Z"
    }

# ───────────────────────────────
# FORECAST ENDPOINT
# ───────────────────────────────
class ForecastIn(BaseModel):
    game: Literal["pick3", "pick4", "ldl"]
    window: Literal["last_30", "midday_30", "evening_30"] = "last_30"
    mode: Literal["standard", "nbc", "rp"] = "standard"
    strictness: int = Field(default=55, ge=0, le=100)

def make_picks(game: str) -> List[Dict[str, Any]]:
    rnd = random.Random()
    if game == "pick3":
        return [{"pick": "".join(str(rnd.randint(0, 9)) for _ in range(3)), "confidence": round(rnd.uniform(0.55, 0.85), 2)} for _ in range(5)]
    if game == "pick4":
        return [{"pick": "".join(str(rnd.randint(0, 9)) for _ in range(4)), "bonus": rnd.randint(0, 9), "confidence": round(rnd.uniform(0.52, 0.82), 2)} for _ in range(5)]
    return [{"pick": sorted(rnd.sample(range(1, 46), 5)), "confidence": round(rnd.uniform(0.48, 0.78), 2)} for _ in range(5)]

@app.post("/v1/lipe/forecast")
def lipe_forecast(inp: ForecastIn) -> Dict[str, Any]:
    return {
        "provider": "lipe-forecast",
        "request": inp.model_dump(),
        "summary": {
            "class": inp.mode.upper(),
            "entropy_signature": round(random.uniform(0.25, 0.75), 3),
            "echo_match": random.randint(0, 3),
            "tier_validation": "T33",
            "ts": datetime.utcnow().isoformat() + "Z"
        },
        "picks": make_picks(inp.game)
    }
