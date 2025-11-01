import time, random, requests
from urllib.parse import urlparse
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from utils.errors import HttpError
from utils.settings import settings
from utils.rate import allow

# simple circuit breaker per host (opens after N failures, closes after cooldown)
_CB = {}  # host -> {"fails":int,"opened_at":float}
_CB_FAILS = 4
_CB_COOLDOWN = 20.0

def _cb_allows(host: str) -> bool:
    st = _CB.get(host)
    if not st: return True
    if st["fails"] < _CB_FAILS: return True
    if (time.monotonic() - st["opened_at"]) > _CB_COOLDOWN:
        _CB[host] = {"fails":0,"opened_at":0.0}
        return True
    return False

def _cb_note(host: str, ok: bool):
    st = _CB.setdefault(host, {"fails":0, "opened_at":0.0})
    if ok:
        st["fails"] = 0
        return
    st["fails"] += 1
    if st["fails"] >= _CB_FAILS: st["opened_at"] = time.monotonic()

@retry(
    stop=stop_after_attempt(settings.http_retries + 1),
    wait=wait_exponential_jitter(initial=settings.http_backoff_min, max=settings.http_backoff_max),
    retry=retry_if_exception_type(HttpError),
    reraise=True
)
def fetch_json(url: str, params=None, headers=None, timeout=None):
    timeout = timeout or settings.http_timeout
    host = urlparse(url).netloc

    if not _cb_allows(host):
        raise HttpError(f"Circuit open for {host}; cooling down")

    if not allow(host):
        # gentle local backoff to respect our own budget
        time.sleep(random.uniform(0.05, 0.2))

    hdrs = {"User-Agent": "PunchLogic/1.0 (+dash)"} | (headers or {})
    try:
        r = requests.get(url, params=params or {}, headers=hdrs, timeout=timeout)
        if r.status_code == 429:
            raise HttpError("HTTP 429 Rate limited")
        if r.status_code >= 400:
            raise HttpError(f"HTTP {r.status_code}: {r.text[:160]}")
        _cb_note(host, ok=True)
        return r.json()
    except requests.RequestException as e:
        _cb_note(host, ok=False)
        raise HttpError(str(e))
