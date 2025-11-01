import time
from collections import defaultdict
from utils.settings import settings

_buckets = defaultdict(lambda: {"tokens": settings.rate_capacity, "ts": time.monotonic()})

def allow(host: str) -> bool:
    b = _buckets[host]
    now = time.monotonic()
    # refill
    delta = now - b["ts"]
    b["ts"] = now
    b["tokens"] = min(settings.rate_capacity, b["tokens"] + delta * settings.rate_refill_per_sec)
    if b["tokens"] >= 1:
        b["tokens"] -= 1
        return True
    return False
