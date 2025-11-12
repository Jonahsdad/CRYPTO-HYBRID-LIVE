# backend/app/webhooks.py
from __future__ import annotations
import time, json, hmac, hashlib, os
from fastapi import APIRouter, Header, HTTPException, Depends, Request
from pydantic import BaseModel
import redis
from .config import REDIS_URL
from .api import require_key

R = redis.from_url(REDIS_URL, decode_responses=True)
router = APIRouter(prefix="/v1/webhooks")

WH_SECRET = os.getenv("WEBHOOK_SECRET", "")


class HookIn(BaseModel):
    event: str
    payload: dict


def _verify(sig: str | None, body: bytes) -> bool:
    if not WH_SECRET:
        # if no secret set, accept all (dev mode)
        return True
    if not sig:
        return False
    mac = hmac.new(WH_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(mac, sig)


@router.post("/ingest")
async def ingest(
    hook: HookIn,
    request: Request,
    x_his_signature: str | None = Header(default=None),
    _=Depends(require_key),
):
    # why: idempotent, auditable ingress for partners
    raw = await request.body()
    if not _verify(x_his_signature, raw):
        raise HTTPException(401, "bad signature")
    key = f"hook:{hook.event}:{int(time.time())}"
    R.setex(key, 86400, json.dumps(hook.payload))
    return {"ok": True, "key": key}
