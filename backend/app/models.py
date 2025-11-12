from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel

class SourceMeta(BaseModel):
    id: str
    kind: str
    name: str

class SymbolInfo(BaseModel):
    source: str
    symbol: str
    name: Optional[str] = None

class Candle(BaseModel):
    ts: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    value: Optional[float] = None

class TimeSeriesResponse(BaseModel):
    source: str
    symbol: str
    points: List[Candle]
    warnings: List[str]

class ForecastPoint(BaseModel):
    ts: str
    yhat: float
    q10: Optional[float] = None
    q50: Optional[float] = None
    q90: Optional[float] = None

class ForecastResponse(BaseModel):
    source: str
    symbol: str
    forecast: List[ForecastPoint]
    horizon: int
    model: dict
    explain: dict
    warnings: List[str]

class BacktestRequest(BaseModel):
    source: str
    symbol: str
    horizon: int
    window: int

class BacktestResponse(BaseModel):
    metrics: dict
    equity_curve: list
    n_trades: int

class WatchlistItem(BaseModel):
    id: Optional[int] = None
    name: str
    symbols: List[str]

class UsageResponse(BaseModel):
    rpm_used: int
    rpd_used: int
