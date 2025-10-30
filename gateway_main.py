import os, time, json, hashlib
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# ----- Observability (optional but recommended) -----
SENTRY_DSN = os.getenv("SENTRY_DSN", "")
if SENTRY_DSN:
    import sentry_sdk
    sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=0.1)

# Prometheus metrics
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    USE_PROM = True
except Exception:
    USE_PROM = False

# ----- Redis (cache + rate limit) -----
REDIS_URL = os.getenv("REDIS_URL", "")
redis = None
if REDIS_URL:
    import redis as _redis
    redis = _redis.from_url(REDIS_URL, decode_responses=True)

# ----- Security -----
HIS_KEY = os.getenv("HIS_KEY", "")

def require_key(req: Request):
    if not HIS_KEY:
        return  # key not enforced
    supplied = req.headers.get("x-his-key") or req.headers.get("X-HIS-KEY")
    if supplied != HIS_KEY:
        raise HTTPException(status_code=401, detail="Invalid HIS key")

# ----- App -----
app = FastAPI(
    title="HIS Gateway",
    version="1.0.0",
    description="Hybrid Intelligence Systems Gateway — routes LIPE/Streamlit requests."
)

# CORS: allow your UI + any future domains (add more as needed)
ALLOWED_ORIGINS = [
    "https://crypto-hybrid-live-1.onrender.com",
    "https://share.streamlit.io",
    "https://streamlit.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach metrics
if USE_PROM:
    Instrumentator().instrument(app).expose(app, include_in_schema=False)

# ----- Helpers -----
def rl_key(req: Request) -> str:
    ip = req.client.host if req.client else "unknown"
    k = req.headers.get("x-his-key","anon")
    return f"rl:{k}:{ip}"

def rate_limit(req: Request, limit_per_min: int = 60):
    if not redis:
        return
    k = rl_key(req)
    now_min = int(time.time() // 60)
    bucket = f"{k}:{now_min}"
    current = redis.incr(bucket)
    if current == 1:
        redis.expire(bucket, 70)  # 1 min + buffer
    if current > limit_per_min:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

def cache_get(key: str) -> Optional[Dict[str, Any]]:
    if not redis:
        return None
    raw = redis.get(key)
    return json.loads(raw) if raw else None

def cache_set(key: str, value: Dict[str, Any], ttl: int = 30):
    if not redis:
        return
    redis.setex(key, ttl, json.dumps(value))

def hash_payload(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

# ----- Routes -----
@app.get("/", tags=["default"])
def root():
    return {
        "message": "Welcome to the HIS Gateway — powered by LIPE.",
        "status": "running",
        "docs": "/docs",
        "service": os.getenv("RENDER_SERVICE_NAME", "gateway")
    }

@app.get("/health", tags=["default"])
def health():
    status = {"status": "ok", "service": "HIS Gateway"}
    if redis:
        try:
            redis.ping()
            status["cache"] = "ok"
        except Exception:
            status["cache"] = "down"
    return status

@app.get("/api/test", tags=["default"])
def api_test(req: Request):
    require_key(req); rate_limit(req)
    return {"ok": True, "ts": time.time()}

# Example forecast stub with caching
@app.post("/v1/forecast", tags=["forecast"])
async def forecast(req: Request):
    require_key(req); rate_limit(req)
    body = await req.json()
    cache_key = f"fc:{hash_payload(body)}"
    cached = cache_get(cache_key)
    if cached:
        return {"cached": True, **cached}

    # ---- PLACEHOLDER: call LIPE here ----
    # result = lipe_engine.run(body)  # integrate later
    result = {"provider": "demo", "received": body, "forecast": ["123","456","789"]}

    cache_set(cache_key, result, ttl=30)
    return {"cached": False, **result}

# Global error handler
@app.exception_handler(Exception)
async def on_error(request: Request, exc: Exception):
    payload = {"path": request.url.path, "error": str(exc)}
    return JSONResponse(payload, status_code=500)
