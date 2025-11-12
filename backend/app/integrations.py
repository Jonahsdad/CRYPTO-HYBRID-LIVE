# backend/app/integrations.py
from __future__ import annotations
import os, httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from .api import require_key

# why: single notify endpoint fans out to Slack/Discord/SMS without UI changes
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK", "")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")

router = APIRouter(prefix="/v1/integrations")


class NotifyIn(BaseModel):
    channel: str = Field(..., description="slack|discord|sms|telegram|email (sms/telegram already exist)")
    text: str
    title: str | None = None


async def _post_json(url: str, payload: dict):
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(url, json=payload)
        r.raise_for_status()


@router.post("/notify")
async def notify(payload: NotifyIn, _=Depends(require_key)):
    ch = payload.channel.lower()
    if ch == "slack":
        if not SLACK_WEBHOOK:
            raise HTTPException(400, "slack not configured")
        body = {"text": f"*{payload.title or 'HIS/LIPE'}*\n{payload.text}"}
        await _post_json(SLACK_WEBHOOK, body)
        return {"ok": True}
    if ch == "discord":
        if not DISCORD_WEBHOOK:
            raise HTTPException(400, "discord not configured")
        body = {"content": f"**{payload.title or 'HIS/LIPE'}**\n{payload.text}"}
        await _post_json(DISCORD_WEBHOOK, body)
        return {"ok": True}
    raise HTTPException(400, f"unsupported channel: {payload.channel}")
