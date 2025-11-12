from __future__ import annotations

AUTH = {"require_api_key": True}
RATES = {"rpm": 60, "rpd": 5000}
CACHE = {"series": 300, "forecast": 300}

DATA_AUTHORITY = {
    "Stocks": {"poll_s": 300},
    "Crypto": {"poll_s": 300},
    "Sports": {"poll_s": 300},
    "Lottery": {"poll_s": 300}
}

LATENCY_SLO = {"forecast_ms": 1000}
HORIZON_CAP = {"Stocks": {"D": 30}, "Crypto": {"D": 30}}
FALLBACK = {"enabled": True}
UNCERTAINTY = {"default": 0.05}
EXPLAIN = {"enabled": True}
BACKTEST = {"enabled": True}
SIGNALS = {"edge_threshold": 0.01}
WATCHLISTS = {"max_per_user": 20}

GOV = {"persist": True, "data_version": "v1", "code_hash": "demo-001"}
DRIFT = {"psi_alert": 0.3}
BILLING = {"enabled": False}
EXPORT = {"enabled": True}
SCHED = {"enabled": False}
OBS = {}
