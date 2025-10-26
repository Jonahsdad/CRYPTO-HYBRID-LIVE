# services/crypto.py
import os
from services.http import get
from services.cache import cached

@cached("cg:price", ttl=30)
def prices(ids: str, vs="usd"):
    return get("https://api.coingecko.com/api/v3/simple/price",
               params={"ids": ids, "vs_currencies": vs})

@cached("cc:ohlcv", ttl=300)
def ohlcv_minute(symbol: str, aggregate=30):
    key = os.environ.get("CRYPTOCOMPARE") or ""
    return get("https://min-api.cryptocompare.com/data/v2/histominute",
               params={"fsym": symbol.upper(), "tsym":"USD", "limit": 1440,
                       "aggregate": aggregate, "api_key": key})
