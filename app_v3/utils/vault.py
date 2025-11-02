import os
from datetime import datetime
import duckdb
import pandas as pd

from utils.settings import settings

# ensure vault dir exists
os.makedirs(os.path.dirname(settings.vault_path) or ".", exist_ok=True)

def _con():
    return duckdb.connect(settings.vault_path)

# ---------- schema & migrations ----------
def ensure_schemas():
    with _con() as con:
        con.execute("PRAGMA threads=4;")
        con.execute("""
        CREATE TABLE IF NOT EXISTS __migrations__(
            id INTEGER PRIMARY KEY,
            applied_at TIMESTAMP
        );
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS crypto_prices(
            ts TIMESTAMP,
            id VARCHAR,
            price_usd DOUBLE,
            change_24h_pct DOUBLE
        );
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS sports_events(
            ts TIMESTAMP,
            scoreboard_date VARCHAR,
            league VARCHAR,
            start_time TIMESTAMP,
            status VARCHAR,
            status_detail VARCHAR,
            away VARCHAR, away_score INTEGER,
            home VARCHAR, home_score INTEGER,
            odds VARCHAR
        );
        """)

def _applied(con, mid: int) -> bool:
    return con.execute("SELECT 1 FROM __migrations__ WHERE id=?", [mid]).fetchone() is not None

def migrate():
    with _con() as con:
        if not _applied(con, 1):
            con.execute("CREATE INDEX IF NOT EXISTS idx_crypto_ts ON crypto_prices(ts);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_sports_date ON sports_events(scoreboard_date);")
            con.execute("INSERT INTO __migrations__ VALUES (?, ?)", [1, datetime.utcnow()])

# ---------- datetime helper ----------
def _to_naive_utc(series: pd.Series) -> pd.Series:
    """
    Accept strings or datetimes; return naive UTC datetime (no tzinfo),
    safe for DuckDB TIMESTAMP.
    """
    s = pd.to_datetime(series, errors="coerce", utc=True)   # tz-aware UTC
    return s.dt.tz_localize(None)                           # drop tz on Series

# ---------- inserts ----------
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

def insert_sports_events(df: pd.DataFrame, league: str):
    """Persist ESPN scoreboard rows with normalized datetimes."""
    if df.empty:
        return
    df2 = df.copy()
    if "start_time" in df2.columns:
        df2["start_time"] = _to_naive_utc(df2["start_time"])
    df2["ts"] = datetime.utcnow()
    if "scoreboard_date" not in df2:
        df2["scoreboard_date"] = None
    # add league column at position 2 for readability
    if "league" not in df2.columns:
        df2.insert(2, "league", league)
    df2 = df2[
        ["ts","scoreboard_date","league","start_time","status","status_detail",
         "away","away_score","home","home_score","odds"]
    ]
    with _con() as con:
        con.register("df2", df2)
        con.execute("INSERT INTO sports_events SELECT * FROM df2")
