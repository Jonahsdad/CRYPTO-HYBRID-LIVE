# backend/app/pro_main.py
from __future__ import annotations
import uuid
import time
import redis as _redis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import CORS_ORIGINS, X_API_KEY_SEED, REDIS_URL
from .db import Base, engine, SessionLocal
from .models import ApiKey
from .metrics import init_metrics
from .health import router as health_router
from .api import router as api_router
from .alerts import router as alerts_router
from .integrations import router as integrations_router
from .webhooks import router as webhooks_router
from .scenario import router as scenario_router

app = FastAPI(title="LIPE Multi-Arena API", version="1.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- metrics middleware + /metrics endpoint ----
init_metrics(
    app,
    version="1.4",
    exclude_prefixes=("/metrics", "/healthz", "/readyz"),
    bind_endpoint=True,
)

# ---- simple correlation-id + lightweight latency logging to Redis (optional) ----
_RPUB = _redis.from_url(REDIS_URL, decode_responses=True)


@app.middleware("http")
async def add_cid(request: Request, call_next):
    cid = request.headers.get("x-correlation-id", str(uuid.uuid4()))
    t0 = time.time()
    response = await call_next(request)
    dt = time.time() - t0
    # coarse per-endpoint timing for quick SLO snapshots (optional)
    ep = request.url.path.split("?")[0]
    try:
        _RPUB.lpush(f"pub:lat:{ep}", f"{dt:.6f}")
        _RPUB.ltrim(f"pub:lat:{ep}", 0, 999)
    except Exception:
        pass
    response.headers["x-correlation-id"] = cid
    return response


# ---- routers ----
app.include_router(health_router)
app.include_router(api_router, prefix="/v1")
app.include_router(alerts_router)
app.include_router(integrations_router)
app.include_router(webhooks_router)
app.include_router(scenario_router)


# ---- startup: create tables + seed dev API key ----
@app.on_event("startup")
def startup():
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        seed = X_API_KEY_SEED
        plan = seed.split(":", 1)[1] if ":" in seed else "free"
        if not db.get(ApiKey, seed):
            db.add(ApiKey(api_key=seed, tenant="demo", role="admin", plan=plan))
            db.commit()
