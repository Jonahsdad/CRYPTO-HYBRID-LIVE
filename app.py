# =========================
# CRYPTO-HYBRID-LIVE — APP
# Clean full-file fix (Streamlit)
# =========================
from __future__ import annotations
import time
from typing import List, Dict

import pandas as pd
import numpy as np
import streamlit as st

# ---------- GLOBAL CONFIG ----------
st.set_page_config(
    page_title="Hybrid Intelligence Systems — Control Panel",
    page_icon="⚡",
    layout="wide",
)

# ---------- UTIL / MOCK DATA (safe defaults so app always runs) ----------
def mock_symbols(n: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    symbols = [f"SYMB{i:02d}" for i in range(1, n + 1)]
    return pd.DataFrame(
        {
            "symbol": symbols,
            "price": rng.uniform(1, 500, n).round(2),
            "score": rng.uniform(0, 1, n).round(3),
            "vol24h": rng.uniform(1e3, 1e6, n).round(0),
            "entropy": rng.uniform(0.1, 1.0, n).round(3),
            "momentum": rng.normal(0, 1, n).round(3),
        }
    )

DATA_DEFAULT = mock_symbols(50)

# ---------- CHART HELPERS (FIX) ----------
def chart_bar(df: pd.DataFrame, metric: str, top: int = 20, title: str | None = None):
    """
    Render a bar chart of the top N rows by <metric>.
    Requires columns: ["symbol", <metric>].
    Fails safe to table if plotly is missing/erroneous.
    """
    try:
        import plotly.express as px
        d = df.nlargest(top, metric)[["symbol", metric]].copy()
        d = d.rename(columns={metric: str(metric)})
        fig = px.bar(d, x="symbol", y=str(metric), title=title or f"Top {top} by {metric}")
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Chart fallback (plotly error: {e})")
        st.dataframe(df.nlargest(top, metric)[["symbol", metric]])

def kpi_row(items: List[Dict[str, str | float]]):
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        c.metric(it.get("label", ""), it.get("value", ""), it.get("delta", ""))

# ---------- AUDIT DIAGNOSTICS ----------
def diagnose_issue(row: pd.Series) -> str:
    """
    Simple example:
    - high entropy -> suggest rescan
    - negative momentum -> suggest model refresh
    Extend with your real rules later.
    """
    notes = []
    if row.get("entropy", 0) > 0.8:
        notes.append("Entropy spike → run entropy scan.")
    if row.get("momentum", 0) < -0.8:
        notes.append("Momentum drop → refresh model weights.")
    if row.get("score", 0) < 0.25:
        notes.append("Low score → quarantine from top picks.")
    return " | ".join(notes) if notes else "OK"

def audit_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["diagnosis"] = d.apply(diagnose_issue, axis=1)
    return d

# ---------- ARENA PAGES ----------
def page_overview():
    st.subheader("System Overview")
    kpi_row(
        [
            {"label": "Models", "value": "9", "delta": "+1 new"},
            {"label": "Latency", "value": "78 ms", "delta": "-12 ms"},
            {"label": "Uptime", "value": "99.9%", "delta": "+0.1%"},
            {"label": "Audit Status", "value": "Green", "delta": "stable"},
        ]
    )
    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.caption("Forecast Score")
        chart_bar(DATA_DEFAULT, "score", 20, "Top 20 by Forecast Score")
    with right:
        st.caption("Entropy")
        chart_bar(DATA_DEFAULT, "entropy", 20, "Top 20 by Entropy")

    st.divider()
    st.caption("Audit Diagnostics")
    st.dataframe(audit_table(DATA_DEFAULT).head(25), use_container_width=True)

def page_crypto():
    st.subheader("Crypto Arena")
    top_n = st.slider("Top N", 5, 50, 20, 5)
    metric = st.selectbox("Metric", ["score", "momentum", "vol24h", "entropy"], index=0)
    chart_bar(DATA_DEFAULT, metric, top_n, f"Top {top_n} by {metric}")
    st.dataframe(DATA_DEFAULT.sort_values(metric, ascending=False).head(top_n), use_container_width=True)

def page_sports():
    st.subheader("Sports Arena")
    st.info("Stubbed demo — plug your model feed here.")
    chart_bar(DATA_DEFAULT, "score", 15, "Signal Strength (Demo)")

def page_lottery():
    st.subheader("Lottery Arena")
    st.info("Stubbed demo — integrate LIPE Pick 3/4 here.")
    chart_bar(DATA_DEFAULT, "entropy", 15, "Entropy Pressure (Demo)")

def page_stocks():
    st.subheader("Stocks Arena")
    st.info("Stubbed demo — connect to your equities feed.")
    chart_bar(DATA_DEFAULT, "momentum", 15, "Momentum (Demo)")

def page_options():
    st.subheader("Options Arena")
    st.info("Stubbed demo — greeks & IV surfaces forthcoming.")
    chart_bar(DATA_DEFAULT, "score", 15, "Edge Score (Demo)")

def page_real_estate():
    st.subheader("Real Estate Arena")
    st.info("Stubbed demo — add MLS/price-index adapters.")
    chart_bar(DATA_DEFAULT, "vol24h", 15, "Liquidity Proxy (Demo)")

def page_commodities():
    st.subheader("Commodities Arena")
    st.info("Stubbed demo — add futures/spot adapters.")
    chart_bar(DATA_DEFAULT, "momentum", 15, "Momentum (Demo)")

def page_forex():
    st.subheader("Forex Arena")
    st.info("Stubbed demo — add FX pairs feed.")
    chart_bar(DATA_DEFAULT, "score", 15, "Signal Strength (Demo)")

def page_rwa():
    st.subheader("RWA Arena")
    st.info("Stubbed demo — tokenize & risk map data here.")
    chart_bar(DATA_DEFAULT, "entropy", 15, "Entropy (Demo)")

# ---------- ROUTER ----------
ARENAS = {
    "Overview": page_overview,
    "Crypto": page_crypto,
    "Sports": page_sports,
    "Lottery": page_lottery,
    "Stocks": page_stocks,
    "Options": page_options,
    "Real Estate": page_real_estate,
    "Commodities": page_commodities,
    "Forex": page_forex,
    "RWA": page_rwa,
}

# ---------- SIDEBAR: CHOOSE YOUR ARENA (9) ----------
with st.sidebar:
    st.markdown("## CHOOSE YOUR ARENA")
    arena = st.radio(
        "Select",
        list(ARENAS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Audit Shortcuts**")
    if st.button("Run Quick Audit"):
        with st.spinner("Running audit…"):
            time.sleep(1.2)
        st.success("Audit finished (demo). See Overview → Audit Diagnostics.")
    if st.button("Export Report"):
        st.info("Demo: hook to report generator. (PDF/HTML)")

# ---------- MAIN ----------
st.title("Hybrid Intelligence Systems — Control Panel")
st.caption("Powered by LIPE — Living Intelligence Predictive Engine")

# Route to selected page
ARENAS.get(arena, page_overview)()
