import duckdb, os, pandas as pd
from datetime import datetime
from utils.settings import settings

os.makedirs(os.path.dirname(settings.vault_path), exist_ok=True)

def _con(): return duckdb.connect(settings.vault_path)

def ensure_schemas():
    with _con() as con:
        con.execute("PRAGMA threads=4;")
        con.execute("""
        CREATE TABLE IF NOT EXISTS __migrations__(id INTEGER PRIMARY KEY, applied_at TIMESTAMP);
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

def _applied(con, mid): 
    return con.execute("SELECT 1 FROM __migrations__ WHERE id=?", [mid]).fetchone() is not None

def migrate():
    # example future migration slot
    with _con() as con:
        if not _applied(con, 1):
            # add index for faster queries
            con.execute("CREATE INDEX IF NOT EXISTS idx_crypto_ts ON crypto_prices(ts);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_sports_date ON sports_events(scoreboard_date);")
            con.execute("INSERT INTO __migrations__ VALUES (1, ?)", [datetime.utcnow()])

def insert_crypto_prices(df: pd.DataFrame):
    if df.empty: return
    df2 = df.copy()
    df2["ts"] = pd.to_datetime(df2["timestamp"], utc=True).tz_convert(None)
    df2 = df2[["ts","id","price_usd","change_24h_pct"]]
    with _con() as con:
        con.register("df2", df2)
        con.execute("INSERT INTO crypto_prices SELECT * FROM df2")

def insert_sports_events(df: pd.DataFrame, league: str):
    if df.empty: return
    df2 = df.copy()
    df2["ts"] = datetime.utcnow()
    if "scoreboard_date" not in df2: df2["scoreboard_date"] = None
    df2.insert(2, "league", league)
    df2 = df2[["ts","scoreboard_date","league","start_time","status","status_detail",
               "away","away_score","home","home_score","odds"]]
    with _con() as con:
        con.register("df2", df2)
        con.execute("INSERT INTO sports_events SELECT * FROM df2")
