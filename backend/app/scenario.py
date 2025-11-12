# backend/app/scenario.py
from __future__ import annotations
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
import json, random, statistics, redis
from .api import require_key
from .config import REDIS_URL

router = APIRouter(prefix="/v1/scenario")
R = redis.from_url(REDIS_URL, decode_responses=True)


def _returns(pts: List[Dict[str, Any]]) -> List[float]:
    out = []
    prev = None
    for p in pts:
        v = p.get("close") or p.get("value")
        if v is None:
            continue
        if prev is None:
            prev = float(v)
            continue
        out.append((float(v) - prev) / max(1e-9, abs(prev)))
        prev = float(v)
    return out


@router.get("/simulate")
def simulate(
    source: str,
    symbol: str,
    horizon: int = 20,
    trials: int = 2000,
    _=Depends(require_key),
):
    # why: quick "what-if" distribution for decisions; no heavy libs
    s = R.get(":".join(("series", source, symbol, "", "")))
    if not s:
        raise HTTPException(404, "series_not_cached_call_timeseries_first")
    pts = json.loads(s)["points"]
    if len(pts) < 60:
        raise HTTPException(400, "insufficient_history")
    rets = _returns(pts)[-250:]  # last 250 steps
    if not rets:
        raise HTTPException(400, "no_valid_returns")
    last = pts[-1].get("close") or pts[-1].get("value") or 0.0
    paths = []
    for _i in range(trials):
        x = float(last)
        for _h in range(horizon):
            r = random.choice(rets)
            x *= (1 + r)
        paths.append(x)
    mean = statistics.fmean(paths)
    paths_sorted = sorted(paths)
    p10 = paths_sorted[int(0.10 * len(paths_sorted))]
    p50 = paths_sorted[int(0.50 * len(paths_sorted))]
    p90 = paths_sorted[int(0.90 * len(paths_sorted))]
    return {
        "asof": pts[-1]["ts"],
        "last": last,
        "horizon": horizon,
        "trials": trials,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "mean": mean,
    }
