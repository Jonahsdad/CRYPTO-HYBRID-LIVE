# services/http.py
import time, requests

def _try_json(resp):
    try:
        return resp.json()
    except Exception:
        return {"_raw": resp.text}

def get(url, *, params=None, headers=None, timeout=12, retries=3, backoff=0.75):
    params = params or {}
    headers = headers or {}
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return _try_json(r)
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(backoff * (2 ** i))

def post(url, *, json=None, headers=None, timeout=15, retries=3, backoff=0.75):
    headers = headers or {}
    for i in range(retries):
        try:
            r = requests.post(url, json=json, headers=headers, timeout=timeout)
            r.raise_for_status()
            return _try_json(r)
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(backoff * (2 ** i))
