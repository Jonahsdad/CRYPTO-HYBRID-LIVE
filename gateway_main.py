# gateway_main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os, httpx, time, redis

app = FastAPI(title="Hybrid Intelligence Gateway", version="1.0")

# === CONFIG ===
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL_DEFAULT", 60))
GATEWAY_SECRET = os.getenv("GATEWAY_SECRET", "local_test_key")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === REDIS CACHE ===
try:
    redis_client = redis.from_url(REDIS_URL)
except Exception:
    redis_client = None

def cache_get(key):
    if not redis_client: return None
    val = redis_client.get(key)
    return val.decode() if val else None

def cache_set(key, val, ttl=CACHE_TTL):
    if redis_client:
        redis_client.setex(key, ttl, val)

# === ROUTES ===
@app.get("/health")
async def health():
    return {"ok": True, "ts": int(time.time()*1000)}

@app.get("/v1/crypto/quotes")
async def crypto_quotes(ids: str):
    key = f"quotes:{ids}"
    if cache_get(key): 
        return {"cached": True, "data": cache_get(key)}
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"
    async with httpx.AsyncClient() as client:
        res = await client.get(url)
        res.raise_for_status()
        data = res.text
        cache_set(key, data)
        return {"cached": False, "data": data}

@app.post("/v1/lipe/forecast")
async def lipe_forecast(req: Request):
    body = await req.json()
    return {"received": body, "forecast": "This is a placeholder forecast."}

# === ERROR HANDLER ===
@app.exception_handler(Exception)
async def on_error(request: Request, exc: Exception):
    return JSONResponse({"error": str(exc)}, status_code=500)
