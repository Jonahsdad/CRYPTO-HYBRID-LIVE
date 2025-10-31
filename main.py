from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any
from datetime import datetime
import random
import os

app = FastAPI(title="HIS Gateway", version=os.getenv("VERSION", "v1.0.0"))

# CORS: keep open for now; later lock to your Streamlit domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("ALLOW_ORIGINS", "*").split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Welcome to the HIS Gateway ⚡ powered by LIPE",
        "status": "running",
        "docs": "/docs",
        "service": os.getenv("SERVICE_NAME", "HIS Gateway"),
        "env": os.getenv("ENV", "production"),
        "ts": datetime.utcnow().isoformat() + "Z"
    }

@app.get("/ping")
def ping():
    return {"status": "ok", "service": os.getenv("SERVICE_NAME", "HIS_Gateway"),
            "ts": datetime.utcnow().isoformat()+"Z"}

@app.get("/status")
def status():
    return {"status": "ok",
            "service": os.getenv("SERVICE_NAME", "HIS_Gateway"),
            "env": os.getenv("ENV", "production"),
            "version": os.getenv("VERSION", "v1.0.0"),
            "checked": datetime.utcnow().isoformat()+"Z"}

# ---------- Simple forecast model (wire test) ----------
class ForecastIn(BaseModel):
    game: Literal["pick3","pick4","ldl"] = "pick4"
    window: Literal["last_30","midday_30","evening_30"] = "last_30"
    mode: Literal["standard","nbc","rp"] = "standard"
    strictness: int = Field(default=55, ge=0, le=100)

def sample_picks(game: str):
    rnd = random.Random()
    if game == "pick3":
        return [{"pick":"".join(str(rnd.randint(0,9)) for _ in range(3)),
                 "confidence": round(rnd.uniform(0.55,0.85),2)} for _ in range(5)]
    if game == "pick4":
        return [{"pick":"".join(str(rnd.randint(0,9)) for _ in range(4)),
                 "bonus": rnd.randint(0,9),
                 "confidence": round(rnd.uniform(0.52,0.82),2)} for _ in range(5)]
    # Lucky Day Lotto demo
    return [{"pick": sorted(rnd.sample(range(1,46),5)),
             "confidence": round(rnd.uniform(0.48,0.78),2)} for _ in range(5)]

def build_response(inp: ForecastIn):
    return {
        "provider": "lipe-forecast",
        "request": inp.model_dump(),
        "summary": {
            "class": inp.mode.upper(),
            "entropy_signature": round(random.uniform(0.25,0.75),3),
            "echo_match": random.randint(0,3),
            "tier_validation": "T33",
            "ts": datetime.utcnow().isoformat() + "Z"
        },
        "picks": sample_picks(inp.game)
    }

@app.post("/forecast")
def forecast_simple(inp: ForecastIn):
    # simple endpoint used by your Streamlit test button
    return build_response(inp)

@app.post("/v1/lipe/forecast")
def forecast_v1(inp: ForecastIn):
    # compatible with your “wire test” panel
    return build_response(inp)
