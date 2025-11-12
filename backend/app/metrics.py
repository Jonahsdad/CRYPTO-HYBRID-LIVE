# backend/app/metrics.py
from __future__ import annotations
import os
import time
from typing import Iterable
from fastapi import APIRouter, Request
from starlette.responses import Response
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    CONTENT_TYPE_LATEST,
    generate_latest,
)
try:
    # optional multiprocess (gunicorn/uvicorn workers)
    from prometheus_client import multiprocess
except Exception:  # pragma: no cover
    multiprocess = None  # type: ignore

# ---- registry (works in single or multiprocess mode) ----
def _make_registry() -> CollectorRegistry:
    if multiprocess and os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        reg = CollectorRegistry()
        multiprocess.MultiProcessCollector(reg)
        return reg
    return CollectorRegistry(auto_describe=True)

REGISTRY: CollectorRegistry = _make_registry()

# ---- metrics (LOW cardinality; avoid user IDs, raw paths, query params) ----
REQUEST_LATENCY = Histogram(
    "his_request_latency_seconds",
    "Latency per request",
    labelnames=["endpoint", "method", "status"],
    buckets=[0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
    registry=REGISTRY,
)

REQUESTS_TOTAL = Counter(
    "his_requests_total",
    "Total requests",
    labelnames=["endpoint", "method", "status"],
    registry=REGISTRY,
)

REQUESTS_IN_PROGRESS = Gauge(
    "his_requests_in_progress",
    "In-progress requests",
    labelnames=["endpoint", "method"],
    registry=REGISTRY,
)

RATE_LIMITED = Counter(
    "his_rate_limited_total",
    "429 responses",
    labelnames=["endpoint"],
    registry=REGISTRY,
)

EXCEPTIONS_TOTAL = Counter(
    "his_exceptions_total",
    "Unhandled exceptions",
    labelnames=["endpoint", "type"],
    registry=REGISTRY,
)

APP_INFO = Gauge(
    "his_app_info",
    "Static app info (labels only)",
    labelnames=["version"],
    registry=REGISTRY,
)

router = APIRouter()

@router.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

# ---- helpers ----
_EXCLUDE_PREFIXES: tuple[str, ...] = ("/metrics", "/healthz", "/readyz")
def _should_exclude(path: str) -> bool:
    return any(path.startswith(p) for p in _EXCLUDE_PREFIXES)

def _endpoint_name(req: Request) -> str:
    # use templated route to prevent label explosion; fall back to first segment
    route = req.scope.get("route")
    if route and getattr(route, "path", None):
        return str(route.path)
    # fallback: /v1/predict/{source}/{symbol} -> /v1/predict
    seg = req.url.path.split("/")
    return "/".join(seg[:3]) if len(seg) > 2 else req.url.path

# ---- public init for app.py / pro_main.py ----
def init_metrics(app, version: str = "dev", exclude_prefixes: Iterable[str] | None = None, bind_endpoint: bool = True) -> None:
    """Attach middleware + expose /metrics (optional)."""
    global _EXCLUDE_PREFIXES
    if exclude_prefixes:
        _EXCLUDE_PREFIXES = tuple(exclude_prefixes)
    APP_INFO.labels(version=version).set(1)

    @app.middleware("http")
    async def _metrics_mw(request: Request, call_next):
        path = request.url.path
        if _should_exclude(path):
            return await call_next(request)

        endpoint = _endpoint_name(request)
        method = request.method.upper()

        REQUESTS_IN_PROGRESS.labels(endpoint, method).inc()
        t0 = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        except Exception as exc:  # 5xx path
            EXCEPTIONS_TOTAL.labels(endpoint, type=exc.__class__.__name__).inc()
            status = "500"
            raise
        finally:
            dt = time.perf_counter() - t0
            REQUEST_LATENCY.labels(endpoint, method, status).observe(dt)
            REQUESTS_TOTAL.labels(endpoint, method, status).inc()
            REQUESTS_IN_PROGRESS.labels(endpoint, method).dec()

    if bind_endpoint:
        app.include_router(router)

# ---- convenience for external modules ----
def mark_rate_limited(endpoint: str) -> None:
    RATE_LIMITED.labels(endpoint).inc()
