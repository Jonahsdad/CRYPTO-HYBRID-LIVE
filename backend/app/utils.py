from __future__ import annotations
from datetime import datetime
import pandas as pd
from collections import defaultdict

CACHE_SERIES = {}
CACHE_FORECAST = {}
CACHE_EXPLAIN = {}

def parse_dt(x):
    if not x: return None
    try:
        return datetime.fromisoformat(x)
    except Exception:
        return None

def validate_timeseries(df: pd.DataFrame):
    if df.empty: return ["empty"]
    warns = []
    if df["close"].isna().any(): warns.append("missing_values")
    if len(df) < 3: warns.append("short_series")
    return warns

def psi(a: pd.Series, b: pd.Series) -> float:
    import numpy as np
    bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 10)
    c1, _ = np.histogram(a, bins=bins)
    c2, _ = np.histogram(b, bins=bins)
    c1 = c1 / max(c1.sum(), 1)
    c2 = c2 / max(c2.sum(), 1)
    return float(((c1 - c2) * np.log((c1 + 1e-9)/(c2 + 1e-9))).sum())

def quantiles_latency(samples):
    if not samples: return (0, 0)
    s = sorted(samples)
    n = len(s)
    p50 = s[int(0.5*n)]
    p95 = s[int(0.95*n) - 1]
    return (p50, p95)

def load_plugins():
    from .plugins.csv_plugins import load_all
    plugs = {p.id: p for p in load_all()}
    return plugs
