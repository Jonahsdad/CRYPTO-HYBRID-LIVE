# stocks_fetcher.py
import os
import time
import requests
import pandas as pd
from typing import Optional, Dict, Any
import yfinance as yf

# Retry/backoff helper
def retry(func, attempts=3, backoff=1.0):
    for i in range(attempts):
        try:
            return func()
        except Exception as e:
            if i == attempts - 1:
                raise
            time.sleep(backoff * (2 ** i))

# ------------- PROVIDER IMPLEMENTATIONS -------------
def fetch_from_iex(symbol: str) -> Dict[str, Any]:
    # requires IEX_API_KEY in env / secrets
    key = os.getenv("IEX_API_KEY", "")
    if not key:
        raise RuntimeError("No IEX key set")
    url = f"https://cloud.iexapis.com/stable/stock/{symbol}/quote"
    params = {"token": key}
    r = requests.get(url, params=params, timeout=8)
    r.raise_for_status()
    j = r.json()
    return {
        "symbol": j.get("symbol"),
        "price": j.get("latestPrice"),
        "change": j.get("changePercent") or j.get("changePercent24h") or None,
        "source": "iex"
    }

def fetch_from_finnhub(symbol: str) -> Dict[str, Any]:
    key = os.getenv("FINNHUB_API_KEY", "")
    if not key:
        raise RuntimeError("No Finnhub key set")
    url = "https://finnhub.io/api/v1/quote"
    params = {"symbol": symbol, "token": key}
    r = requests.get(url, params=params, timeout=8)
    r.raise_for_status()
    j = r.json()
    # finnhub returns c (current), pc (prev close)
    return {
        "symbol": symbol,
        "price": j.get("c"),
        "change": None if j.get("c") is None or j.get("pc") is None else (j.get("c") - j.get("pc")) / (j.get("pc") or 1),
        "source": "finnhub"
    }

def fetch_from_alpha_vantage(symbol: str) -> Dict[str, Any]:
    key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    if not key:
        raise RuntimeError("No AlphaVantage key set")
    url = "https://www.alphavantage.co/query"
    params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": key}
    r = requests.get(url, params=params, timeout=8)
    r.raise_for_status()
    j = r.json()
    gq = j.get("Global Quote", {})
    if not gq:
        raise RuntimeError("AlphaVantage returned no quote")
    price = float(gq.get("05. price", "nan"))
    prev_close = float(gq.get("08. previous close", "nan") or 0)
    # change = None
    if prev_close:
        change = (price - prev_close) / prev_close
    return {"symbol": symbol, "price": price, "change": change, "source": "alphavantage"}

def fetch_from_yfinance(symbol: str) -> Dict[str, Any]:
    t = yf.Ticker(symbol)
    info = t.history(period="1d")
    if info.empty:
        raise RuntimeError("yfinance returned empty for symbol")
    latest = info.iloc[-1]
    price = float(latest["Close"])
    # We'll get previous close if available:
    if len(info) >= 2:
        prev = float(info.iloc[-2]["Close"])
        change = (price - prev) / prev if prev else None
    else:
        change = None
    return {"symbol": symbol, "price": price, "change": change, "source": "yfinance"}

# ------------- UNIFIED FETCHER -------------
def fetch_price(symbol: str, providers=None, debug=False) -> Dict[str, Any]:
    """
    Try providers in order; return first successful quote dict:
    {'symbol':..., 'price':float, 'change':float or None, 'source': '<provider>'}
    """
    if providers is None:
        # order = primary, secondary, tertiary
        providers = ["iex", "finnhub", "alphavantage", "yfinance"]

    last_exc = None
    for p in providers:
        try:
            if p == "iex":
                result = retry(lambda: fetch_from_iex(symbol), attempts=2, backoff=0.8)
            elif p == "finnhub":
                result = retry(lambda: fetch_from_finnhub(symbol), attempts=2, backoff=0.8)
            elif p == "alphavantage":
                result = retry(lambda: fetch_from_alpha_vantage(symbol), attempts=2, backoff=1.2)
            elif p == "yfinance":
                result = retry(lambda: fetch_from_yfinance(symbol), attempts=2, backoff=1.0)
            else:
                raise RuntimeError(f"Unknown provider {p}")
            if debug:
                result["_debug"] = f"provider={p}"
            return result
        except Exception as e:
            last_exc = e
            if debug:
                print(f"[stocks_fetcher] provider {p} failed for {symbol}: {e}")
            continue

    # final fallback: return a sentinel so caller can show placeholder
    raise RuntimeError(f"All providers failed for {symbol}. last_err={last_exc}")
