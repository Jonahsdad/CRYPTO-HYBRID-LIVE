import pandas as pd
from datetime import datetime
from utils.settings import settings
import duckdb, os

os.makedirs(os.path.dirname(settings.vault_path) or ".", exist_ok=True)

def _con():
    return duckdb.connect(settings.vault_path)

def _to_naive_utc(series: pd.Series) -> pd.Series:
    """
    Converts any string or datetime Series into naive UTC datetime.
    Works for both tz-aware and naive inputs.
    """
    s = pd.to_datetime(series, errors="coerce", utc=True)
    # Convert each element safely
    return s.dt.tz_localize(None)

def insert_crypto_prices(df: pd.DataFrame):
    """Safely insert crypto price data into DuckDB with clean timestamps."""
    if df.empty:
        return
    df2 = df.copy()
    if "timestamp" not in df2.columns:
        df2["timestamp"] = datetime.utcnow()
    df2["ts"] = _to_naive_utc(df2["timestamp"])
    df2 = df2[["ts", "id", "price_usd", "change_24h_pct"]]
    with _con() as con:
        con.register("df2", df2)
        con.execute("INSERT INTO crypto_prices SELECT * FROM df2")
