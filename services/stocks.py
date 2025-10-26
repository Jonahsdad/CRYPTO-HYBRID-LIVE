# services/stocks.py
import os
from services.http import get
from services.cache import cached

POLY = os.environ.get("POLYGON") or ""

@cached("poly:aggs", ttl=120)
def aggregates(ticker: str, start="2024-01-01", end="2025-12-31", timespan="day"):
    if not POLY:
        # free fallback (Alpha Vantage demo) – limited
        return get("https://www.alphavantage.co/query",
                   params={"function":"TIME_SERIES_DAILY_ADJUSTED","symbol":ticker,"apikey":"demo"})
    return get(f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/1/{timespan}/{start}/{end}",
               params={"apiKey": POLY, "limit": 5000})

@cached("fmp:profile", ttl=3600)
def fundamentals(ticker: str, key: str):
    return get(f"https://financialmodelingprep.com/api/v3/profile/{ticker.upper()}",
               params={"apikey": key})
