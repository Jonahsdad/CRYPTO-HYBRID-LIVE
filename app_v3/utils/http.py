import time, random, requests
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from utils.settings import settings

class HttpError(Exception): ...

@retry(
    stop=stop_after_attempt(settings.http_retries + 1),
    wait=wait_exponential_jitter(initial=settings.http_backoff_min, max=settings.http_backoff_max),
    retry=retry_if_exception_type(HttpError),
    reraise=True
)
def fetch_json(url: str, params=None, headers=None, timeout=None):
    timeout = timeout or settings.http_timeout
    hdrs = {"User-Agent": "PunchLogic/1.0 (+dash)"} | (headers or {})
    try:
        r = requests.get(url, params=params or {}, headers=hdrs, timeout=timeout)
        if r.status_code == 429:
            # gentle pause so we don’t hammer on free tiers
            time.sleep(random.uniform(0.2, 0.6))
            raise HttpError("HTTP 429 Rate limited")
        if r.status_code >= 400:
            raise HttpError(f"HTTP {r.status_code}: {r.text[:160]}")
        return r.json()
    except requests.RequestException as e:
        raise HttpError(str(e))
