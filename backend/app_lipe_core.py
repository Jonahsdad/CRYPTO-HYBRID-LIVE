# backend/app_lipe_core.py
from __future__ import annotations
import os, math, time, json, random
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ------------- models -------------
class ForecastReq(BaseModel):
    arena: str = "crypto"
    symbol: str = "BTCUSDT"
    horizon: int = 5  # days

class ForecastPoint(BaseModel):
    ts: str
    yhat: float
    q10: float
    q90: float

class ForecastEvent(BaseModel):
    id: str
    arena: str
    symbol: str
    horizon: int
    made_at: str
    series: List[Dict[str, Any]]  # {"ts": iso, "close": float}
    forecast: List[ForecastPoint]
    metrics: Dict[str, Any]

# ------------- app -------------
app = FastAPI(title="HIS — LIPE Core (lite)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory store (share tokens)
_SHARE: Dict[str, Dict[str, Any]] = {}
_LAST_EVENT: Optional[Dict[str, Any]] = None

# ------------- data fetch -------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _synthetic_series(n: int = 300, start: float = 30000.0) -> List[Dict[str, Any]]:
    xs, y = [], start
    base = _now_utc() - timedelta(minutes=5*n)
    for i in range(n):
        drift = 0.0002
        vol = 0.01
        y *= (1.0 + np.random.normal(drift, vol))
        xs.append({
            "ts": (base + timedelta(minutes=5*i)).isoformat(),
            "close": float(y)
        })
    return xs

def _ccxt_series(symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
    """Try to fetch OHLCV via CCXT. Fall back to synthetic if any error."""
    try:
        import ccxt
        ex_name = os.getenv("CRYPTO_EXCHANGE", "binance")
        ex = getattr(ccxt, ex_name)({"enableRateLimit": True})
        m = ex.load_markets()
        sym = symbol if symbol in m else "BTC/USDT"
        ohlcv = ex.fetch_ohlcv(sym, timeframe="5m", limit=limit)
        xs = []
        for ts, _o, _h, _l, c, _v in ohlcv:
            xs.append({"ts": datetime.fromtimestamp(ts/1000, tz=timezone.utc).isoformat(),
                       "close": float(c)})
        if not xs:
            return _synthetic_series(limit)
        return xs
    except Exception:
        return _synthetic_series(limit)

# ------------- simple “LIPE-like” forecaster -------------
def _ewma_forecast(series: List[Dict[str, Any]], horizon: int) -> Dict[str, Any]:
    closes = np.array([p["close"] for p in series], dtype=float)
    if len(closes) < 50:
        closes = np.pad(closes, (50-len(closes), 0), mode="edge")

    # log-return based drift/vol
    r = np.diff(np.log(closes))
    mu = np.mean(r)
    sigma = np.std(r) + 1e-9

    last = closes[-1]
    # entropy: 0 (calm) .. 1 (chaos) normalized by typical crypto range
    entropy = float(np.clip(sigma / 0.03, 0.0, 1.0))
    # edge: sign & magnitude of recent drift
    edge = float(np.tanh(mu / (sigma + 1e-9)))

    # regime (toy)
    regime = "EXPANSION" if entropy < 0.35 and edge > 0 else \
             "COMPRESSION" if entropy < 0.35 else \
             "TURBULENT"

    step = 60*60*24  # daily seconds
    base_ts = datetime.fromisoformat(series[-1]["ts"])
    pts: List[ForecastPoint] = []
    for i in range(1, horizon+1):
        # geometric brownian like projection
        exp_ret = (mu - 0.5*sigma**2) * i
        yhat = last * math.exp(exp_ret)
        band = last * math.exp(exp_ret + np.array([-1.28, 1.28]) * sigma * math.sqrt(i))
        pts.append(ForecastPoint(
            ts=(base_ts + timedelta(seconds=step*i)).isoformat(),
            yhat=float(yhat),
            q10=float(band[0]),
            q90=float(band[1]),
        ))

    metrics = {
        "entropy": round(entropy, 4),
        "edge": round(edge, 4),
        "regime": regime,
        "rp": round(abs(edge) * (1.0 - entropy), 4),      # rough “pattern strength”
        "sfh": max(1, int((1.0 - entropy) * 14)),         # suggested forecast horizon
    }
    return {"points": [p.dict() for p in pts], "metrics": metrics}

# ------------- routes -------------
@app.get("/healthz")
def healthz():
    return {"ok": True, "ts": _now_utc().isoformat()}

@app.post("/v1/forecast", response_model=ForecastEvent)
def forecast(body: ForecastReq):
    arena = (body.arena or "crypto").lower()
    if arena != "crypto":
        raise HTTPException(400, "Only 'crypto' arena is enabled in this lite build")

    series = _ccxt_series(body.symbol, limit=400)
    fc = _ewma_forecast(series, horizon=int(max(1, min(body.horizon, 30))))
    evt = ForecastEvent(
        id=f"evt_{int(time.time()*1000)}",
        arena=arena,
        symbol=body.symbol,
        horizon=body.horizon,
        made_at=_now_utc().isoformat(),
        series=series[-300:],                 # last 300 for chart
        forecast=[ForecastPoint(**p) for p in fc["points"]],
        metrics=fc["metrics"],
    ).dict()
    global _LAST_EVENT
    _LAST_EVENT = evt
    return evt

# simple share tokens (in-memory)
class ShareReq(BaseModel):
    ttl_minutes: int = 60

@app.post("/v1/share/create")
def share_create(body: ShareReq):
    if not _LAST_EVENT:
        raise HTTPException(400, "no forecast yet")
    token = f"s_{random.randrange(10**9, 10**10-1)}"
    _SHARE[token] = {
        "exp": _now_utc() + timedelta(minutes=max(1, body.ttl_minutes)),
        "event": _LAST_EVENT,
    }
    return {"token": token, "url": f"/v1/share/{token}"}

@app.get("/v1/share/{token}")
def share_get(token: str):
    row = _SHARE.get(token)
    if not row:
        raise HTTPException(404, "not found")
    if _now_utc() > row["exp"]:
        raise HTTPException(410, "expired")
    return row["event"]
