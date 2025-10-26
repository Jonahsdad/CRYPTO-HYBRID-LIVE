# services/realestate.py
from services.http import get
from services.cache import cached
import os

FRED = os.environ.get("FRED") or ""

@cached("fred:mortgage", ttl=3600)
def mortgage_30y():
    if not FRED:
        return {"error":"FRED key missing"}
    return get("https://api.stlouisfed.org/fred/series/observations",
               params={"series_id": "MORTGAGE30US", "api_key": FRED, "file_type": "json"})
