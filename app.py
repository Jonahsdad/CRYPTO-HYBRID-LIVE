# ============================================================
# HYBRID INTELLIGENCE SYSTEMS — CONTROL PANEL (Full Replace)
# Streamlit 1.37+ | Full working app with 9 arenas, subpages,
# charts, diagnostics, audit helpers, and export stubs.
# This file is intentionally long (400+ lines) to exceed prior
# line offsets and avoid partial fixes.
# ============================================================

from __future__ import annotations

import io
import os
import sys
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(
    page_title="Hybrid Intelligence Systems — Control Panel",
    page_icon="⚡",
    layout="wide",
)

APP_TITLE = "Hybrid Intelligence Systems — Control Panel"
POWERED_BY = "Powered by LIPE — Living Intelligence Predictive Engine"
VERSION = "v1.2.0 (full-file replacement)"

# ------------------------- GLOBAL STATE -------------------------
@dataclass
class AppState:
    theme_dark: bool = True
    last_audit_ts: float = 0.0
    data_seed: int = 42

if "state" not in st.session_state:
    st.session_state.state = AppState()

STATE: AppState = st.session_state.state

# ------------------------- UTILS -------------------------
def load_csv_if_exists(path: str, required_cols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Try to load a CSV if it exists and (optionally) validate required columns.
    """
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if required_cols:
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    st.warning(f"CSV found at {path} but missing columns: {missing}")
                    return None
            return df
    except Exception as e:
        st.warning(f"Failed to read {path}: {e}")
    return None

def mock_symbols(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """
    Mock dataset so the app always boots cleanly.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "symbol": [f"SYMB{i:02d}" for i in range(1, n + 1)],
            "price": rng.uniform(1, 500, n).round(2),
            "score": rng.uniform(0, 1, n).round(3),
            "vol24h": rng.uniform(1e3, 1e6, n).round(0),
            "entropy": rng.uniform(0.1, 1.0, n).round(3),
            "momentum": rng.normal(0, 1, n).round(3),
        }
    )

def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Ensure required columns exist; if missing, create safe defaults.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0
    return out

# Try to load real data, fallback to mock.
DATA = (
    load_csv_if_exists("data/crypto_sample.csv", ["symbol", "score", "momentum", "vol24h", "entropy"])
    or mock_symbols(60, seed=STATE.data_seed)
)

# ------------------------- CHART HELPERS (FIXED) -------------------------
def chart_bar(
    df: pd.DataFrame,
    metric: str,
    top: int = 20,
    title: Optional[str] = None,
    sort_ascending: bool = False,
) -> None:
    """
    Render a bar chart of the top N rows by <metric>.
    Falls back to a table if Plotly is missing or errors.
    """
    try:
        import plotly.express as px
        m = str(metric)
        if m not in df.columns:
            st.error(f"Metric '{m}' not in dataframe.")
            st.dataframe(df.head(20))
            return
        d = df.nsmallest(top, m) if sort_ascending else df.nlargest(top, m)
        d = d[["symbol", m]].copy()
        fig = px.bar(d, x="symbol", y=m, title=title or f"Top {top} by {m}")
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Chart fallback (plotly error: {e})")
        m = str(metric)
        if m in df.columns:
            st.dataframe(df.sort_values(m, ascending=sort_ascending).head(top)[["symbol", m]])
        else:
            st.dataframe(df.head(top))

def chart_line(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    """
    Simple line chart with safe fallback.
    """
    try:
        import plotly.express as px
        if x not in df.columns or y not in df.columns:
            st.warning(f"Line chart missing columns: {x} or {y}")
            st.dataframe(df.head(20))
            return
        fig = px.line(df, x=x, y=y, title=title)
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Line chart fallback (plotly error: {e})")
        st.dataframe(df[[x, y]].head(20) if x in df.columns and y in df.columns else df.head(20))

def kpi_row(items: List[Dict[str, str]]) -> None:
    """
    Draw a row of Streamlit KPIs.
    """
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.metric(it.get("label", ""), it.get("value", ""), it.get("delta", ""))

# ------------------------- AUDIT & DIAGNOSTICS -------------------------
def diagnose_issue(row: pd.Series) -> str:
    """
    Rudimentary rules. Extend with your LIPE ruleset later.
    """
    notes = []
    if row.get("entropy", 0) > 0.85:
        notes.append("Entropy spike → run entropy scan")
    if row.get("momentum", 0) < -0.9:
        notes.append("Momentum break → refresh model")
    if row.get("score", 0) < 0.2:
        notes.append("Low score → quarantine")
    return " | ".join(notes) if notes else "OK"

def audit_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["diagnosis"] = d.apply(diagnose_issue, axis=1)
    return d

def export_html_report(df: pd.DataFrame, title: str = "HIS_Report") -> bytes:
    """
    Simple HTML export. Replace with PDF export later if needed.
    """
    html = io.StringIO()
    html.write(f"<h2>{title}</h2>")
    html.write(f"<p>{POWERED_BY} — {VERSION}</p>")
    html.write(df.to_html(index=False))
    return html.getvalue().encode("utf-8")

# ------------------------- ARENA SUBPAGES -------------------------
def subpage_recent_table(df: pd.DataFrame, sort_key: str, k: int = 20):
    st.caption("Recent Signals")
    st.dataframe(df.sort_values(sort_key, ascending=False).head(k), use_container_width=True)

def subpage_diagnostics(df: pd.DataFrame):
    st.caption("Diagnostics")
    st.dataframe(audit_table(df).head(30), use_container_width=True)

def subpage_charts(df: pd.DataFrame, primary_metric: str):
    c1, c2 = st.columns([1, 1])
    with c1:
        chart_bar(df, primary_metric, 20, f"Top 20 by {primary_metric}")
    with c2:
        alt = "entropy" if primary_metric != "entropy" else "score"
        chart_bar(df, alt, 20, f"Top 20 by {alt}")

# ------------------------- ARENA PAGES -------------------------
def page_overview():
    st.subheader("System Overview")
    kpi_row(
        [
            {"label": "Models", "value": "9", "delta": "+1"},
            {"label": "Latency", "value": "78 ms", "delta": "-12 ms"},
            {"label": "Uptime", "value": "99.9%", "delta": "+0.1%"},
            {"label": "Audit", "value": "Green", "delta": "stable"},
        ]
    )
    st.divider()
    subpage_charts(DATA, "score")
    st.divider()
    subpage_diagnostics(DATA)

def page_crypto():
    st.subheader("Crypto Arena")
    top_n = st.slider("Top N", 5, 60, 20, 5, key="crypto_topn")
    metric = st.selectbox("Metric", ["score", "momentum", "vol24h", "entropy"], index=0, key="crypto_metric")
    chart_bar(DATA, metric, top_n, f"Top {top_n} by {metric}")
    subpage_recent_table(DATA, metric, top_n)

def page_sports():
    st.subheader("Sports Arena")
    st.info("Stub — connect sports model feed here (odds, player props, simulations).")
    subpage_charts(DATA, "score")
    subpage_diagnostics(DATA)

def page_lottery():
    st.subheader("Lottery Arena")
    st.info("Stub — integrate LIPE Pick 3/4 pipelines and echo-mapping.")
    subpage_charts(DATA, "entropy")
    subpage_recent_table(DATA, "entropy", 15)

def page_stocks():
    st.subheader("Stocks Arena")
    st.info("Stub — plug equities feed (Yahoo/Polygon/Alpaca) + factors.")
    subpage_charts(DATA, "momentum")
    subpage_recent_table(DATA, "momentum", 15)

def page_options():
    st.subheader("Options Arena")
    st.info("Stub — greeks & IV surfaces forthcoming.")
    subpage_charts(DATA, "score")
    subpage_recent_table(DATA, "score", 15)

def page_real_estate():
    st.subheader("Real Estate Arena")
    st.info("Stub — MLS/index connectors and geospatial overlays.")
    subpage_charts(DATA, "vol24h")
    subpage_recent_table(DATA, "vol24h", 15)

def page_commodities():
    st.subheader("Commodities Arena")
    st.info("Stub — futures/spot connectors.")
    subpage_charts(DATA, "momentum")
    subpage_recent_table(DATA, "momentum", 15)

def page_forex():
    st.subheader("Forex Arena")
    st.info("Stub — FX pairs, DXY correlation, carry.")
    subpage_charts(DATA, "score")
    subpage_recent_table(DATA, "score", 15)

def page_rwa():
    st.subheader("RWA Arena")
    st.info("Stub — tokenize & risk map sources.")
    subpage_charts(DATA, "entropy")
    subpage_recent_table(DATA, "entropy", 15)

# ------------------------- EXPLAINER / TOOLS PAGES -------------------------
def page_scores():
    st.subheader("Scores — Explainer")
    st.markdown(
        """
**RAW (0..1)** = crowd heat now (volume/market-cap + Δ momentum)  
**TRUTH (0..1)** = stability × you/MCAP × 24h + 7d  
**CONFLUENCE (0..1)** = RAW × TRUTH × agree, consistent (24h→7d), energetic & liquid  
**ABSOLUTE GAP (0..1)** = absolute gap |RAW–TRUTH| (higher = mismatch)
        """
    )
    st.divider()
    subpage_charts(DATA, "score")

def page_export():
    st.subheader("Export")
    st.write("Generate a quick HTML report of the current audit table.")
    df = audit_table(DATA).sort_values("score", ascending=False).head(50)
    if st.button("Export HTML"):
        buf = export_html_report(df, "HIS_Report")
        st.download_button("Download report.html", data=buf, file_name="report.html", mime="text/html")

def page_settings():
    st.subheader("Settings")
    c1, c2 = st.columns([1, 1])
    with c1:
        STATE.theme_dark = st.toggle("Dark theme", value=STATE.theme_dark)
        STATE.data_seed = st.number_input("Data seed", value=STATE.data_seed, step=1, min_value=0)
        if st.button("Regenerate sample data"):
            global DATA
            DATA = mock_symbols(60, seed=int(STATE.data_seed))
            st.success("Sample data regenerated.")
    with c2:
        st.markdown("**About**")
        st.write(POWERED_BY)
        st.write(VERSION)
        st.caption("This is a full replacement file crafted to avoid partial-line errors.")

# ------------------------- ROUTER MAP -------------------------
ARENAS: Dict[str, Callable[[], None]] = {
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
    "Scores": page_scores,
    "Export": page_export,
    "Settings": page_settings,
}

# ------------------------- SIDEBAR & SHORTCUTS -------------------------
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
            STATE.last_audit_ts = time.time()
        st.success("Audit complete (demo). See Overview → Diagnostics.")

    if st.button("Export Report"):
        df = audit_table(DATA).sort_values("score", ascending=False).head(50)
        buf = export_html_report(df, "HIS_Report")
        st.download_button("Download report.html", data=buf, file_name="report.html", mime="text/html")

    st.markdown("---")
    st.caption(f"Last audit: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(STATE.last_audit_ts)) if STATE.last_audit_ts else '—'}")

# ------------------------- MAIN -------------------------
st.title(APP_TITLE)
st.caption(f"{POWERED_BY} — {VERSION}")

# Route to selected page
ARENAS.get(arena, page_overview)()

# ------------------------- FOOTER / SPACER -------------------------
st.write("")
st.write("")
st.caption("© Hybrid Intelligence Systems — Full-file replacement for stability.")
