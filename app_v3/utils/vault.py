def insert_crypto_prices(df: pd.DataFrame):
    """
    Safely insert crypto price data into vault.duckdb
    Accepts 'timestamp' column as string or datetime, handles UTC conversion cleanly.
    """
    if df.empty:
        return

    df2 = df.copy()

    # --- Normalize timestamp to naive UTC datetime (no tzinfo)
    try:
        # Ensure timestamp column exists and is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df2["timestamp"]):
            df2["timestamp"] = pd.to_datetime(df2["timestamp"], errors="coerce", utc=True)

        # Strip tz if present
        df2["ts"] = df2["timestamp"].dt.tz_convert(None)
    except Exception:
        # fallback: if tz_convert fails (no tz), just strip it
        df2["ts"] = pd.to_datetime(df2["timestamp"], errors="coerce").fillna(pd.Timestamp.utcnow())

    # --- Keep only needed fields
    df2 = df2[["ts", "id", "price_usd", "change_24h_pct"]]

    # --- Insert to DuckDB
    with _con() as con:
        con.register("df2", df2)
        con.execute("INSERT INTO crypto_prices SELECT * FROM df2")
