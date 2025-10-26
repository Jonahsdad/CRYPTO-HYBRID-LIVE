# services/cache.py
import os, json, hashlib, time
from typing import Callable

# Optional Redis; falls back to in-memory dict on Streamlit Cloud
REDIS_URL = None
try:
    import redis  # type: ignore
    REDIS_URL = os.environ.get("REDIS_URL") or ""
    if not REDIS_URL:
        # allow secrets.toml too
        from streamlit.runtime.secrets import secrets
        REDIS_URL = secrets.get("api", {}).get("REDIS_URL", "")
    r = redis.from_url(REDIS_URL, decode_responses=True) if REDIS_URL else None
except Exception:
    r = None

_mem = {}

def _mem_get(k):
    v = _mem.get(k)
    if not v: return None
    data, exp = v
    if exp and exp < time.time():
        _mem.pop(k, None)
        return None
    return data

def _mem_setex(k, ttl, v):
    _mem[k] = (v, time.time() + ttl if ttl else None)

def cached(key_prefix: str, ttl: int = 60):
    """Cache decorator: uses Redis if available; otherwise in-memory dict."""
    def deco(fn: Callable):
        def wrap(*a, **kw):
            raw = json.dumps({"a": a, "kw": kw}, sort_keys=True, default=str)
            key = f"{key_prefix}:{hashlib.sha256(raw.encode()).hexdigest()}"
            if r:
                v = r.get(key)
                if v: return json.loads(v)
                out = fn(*a, **kw)
                r.setex(key, ttl, json.dumps(out, default=str))
                return out
            else:
                v = _mem_get(key)
                if v is not None: return v
                out = fn(*a, **kw)
                _mem_setex(key, ttl, out)
                return out
        return wrap
    return deco
