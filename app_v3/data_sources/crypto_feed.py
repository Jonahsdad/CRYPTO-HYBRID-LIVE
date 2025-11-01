import pandas as pd
from datetime import datetime, timezone
from utils.http import fetch_json
from utils.errors import CGError
from utils.settings import settings

BASE = settings.coingecko_base

def _get(url, params=None):
    try:
        return fetch_json(url, params=params)
    except Exception as e:
        raise CGError(str(e))

def search_coins(query: str) -> pd.DataFrame:
    data = _get(f"{BASE}/search", {"query": query})
    rows = [{"id": c["id"], "symbol": c["symbol"].upper(), "name": c["name"]} for c in data.get("coins", [])[:25]]
    return pd.DataFrame(rows)

def get_simple_price(coin_ids: list[str], vs="usd") -> pd.DataFrame:
    if not coin_ids: return pd.DataFrame(columns=["timestamp","id","price_usd","change_24h_pct"])
    ids = ",".join(coin_ids[:200])  # guardrail
    data = _get(f"{BASE}/simple/price", {"ids": ids, "vs_currencies": vs, "include_24hr_change": "true"})
    now = datetime.now(timezone.utc).isoformat()
    rows = [{"timestamp": now, "id": cid, "price_usd": v.get(vs), "change_24h_pct": v.get(f"{vs}_24h_change")}
            for cid, v in data.items()]
    return pd.DataFrame(rows).sort_values("id")

def get_market_chart(coin_id: str, vs="usd", days=7) -> pd.DataFrame:
    data = _get(f"{BASE}/coins/{coin_id}/market_chart", {"vs_currency": vs, "days": days})
    prices = data.get("prices", [])
    df = pd.DataFrame(prices, columns=["ms", "price_usd"])
    df["timestamp"] = pd.to_datetime(df["ms"], unit="ms", utc=True)
    return df.drop(columns=["ms"])[["timestamp","price_usd"]]
