# services/options.py
import os
from services.http import get
from services.cache import cached

POLY = os.environ.get("POLYGON") or ""

@cached("poly:optionsSearch", ttl=300)
def contracts_for_underlying(ticker: str, expiration: str = None):
    if not POLY:
        return {"error":"Polygon API key missing"}
    params = {"underlying_ticker": ticker.upper(), "apiKey": POLY}
    if expiration: params["expiration_date"] = expiration
    return get("https://api.polygon.io/v3/reference/options/contracts", params=params)

@cached("poly:optionChain", ttl=120)
def snapshot_chain(ticker: str, date: str):
    if not POLY:
        return {"error":"Polygon API key missing"}
    # You can refine this with more endpoints per your plan.
    return contracts_for_underlying(ticker, expiration=date)
