import requests
import pandas as pd
from datetime import datetime, timezone

BASE = "https://api.coingecko.com/api/v3"

class CGError(Exception):
    pass

def _get(url, params=None, timeout=15):
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        if r.status_code == 429:
            raise CGError("Rate limited by CoinGecko (HTTP 429). Try again shortly.")
        if r.status_code >= 400:
            raise CGError(f"CoinGecko error {r.status_code}: {r.text[:120]}")
        return r.json()
    except requests.RequestException as e:
        raise CGError(f"Network error contacting CoinGecko: {e}")

def search_coins(query: str) -> pd.DataFrame:
    """Search coin IDs by free text (useful if user types 'btc' or 'bitcoin')."""
    data = _get(f"{BASE}/search", {"query": query})
    coins = data.get("coins", [])
    rows = []
    for c in coins[:25]:
        rows.append({"id": c["id"], "symbol": c["symbol"].upper(), "name": c["name"]})
    return pd.DataFrame(rows)

def get_simple_price(coin_ids: list[str], vs="usd") -> pd.DataFrame:
    """Current prices + 24h change for a list of coin ids."""
    ids = ",".join(coin_ids)
    data = _get(f"{BASE}/simple/price",
                {"ids": ids, "vs_currencies": vs, "include_24hr_change": "true"})
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for cid, v in data.items():
        rows.append({
            "timestamp": now,
            "id": cid,
            "price_usd": v.get(vs, None),
            "change_24h_pct": v.get(f"{vs}_24h_change", None)
        })
    df = pd.DataFrame(rows).sort_values("id")
    return df

def get_market_chart(coin_id: str, vs="usd", days=7) -> pd.DataFrame:
    """OHLC-like time series (prices) for last N days (granularity 5m-1d)."""
    data = _get(f"{BASE}/coins/{coin_id}/market_chart", {"vs_currency": vs, "days": days})
    prices = data.get("prices", [])
    # Convert to DataFrame with datetime
    df = pd.DataFrame(prices, columns=["ms", "price_usd"])
    df["timestamp"] = pd.to_datetime(df["ms"], unit="ms", utc=True)
    df = df.drop(columns=["ms"])[["timestamp", "price_usd"]]
    return df
