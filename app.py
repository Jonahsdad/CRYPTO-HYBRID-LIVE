# -----------------------------------------------------------------------------
# Crypto Hybrid Live — Phase 15 (FULL, Color-Coded TRUTH/RAW/DELTA)
# Powered by Jesse Ray Landingham Jr
# -----------------------------------------------------------------------------
# This file is complete & standalone. Replace your repo's app.py with this file.
# ASCII-only header (safe for iPad). No partials. No compact mode.
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Optional: robust stocks via yfinance (app still runs if missing)
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

PHASE_TAG = "PHASE 15 — FULL COLOR"

# ============================== PAGE CONFIG / THEME ===========================

st.set_page_config(
    page_title="Crypto Hybrid Live — Phase 15 (Full Color)",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
/* Layout & Banner */
.block-container { padding-top: 0.6rem; padding-bottom: 2.0rem; }
.phase-badge {
  padding: 10px; border-radius: 10px; background: #0f172a; border: 1px solid #334155;
  color: #7dfca3; font-weight: 800; text-align: center; margin-bottom: 8px;
}
/* Section titles & metric cards */
.section-title { font-size: 24px; font-weight: 800; margin: 6px 0 6px 0; }
.metric-box { border: 1px solid #ffffff22; border-radius: 12px; padding: 0.8rem; background: #0e1117; }
/* Mini badges row */
.badge { display:inline-block; padding: 6px 10px; border-radius: 999px; font-weight:700; margin-right:6px; }
.badge-raw   { background:#241c14; color:#ff9b63; border:1px solid #ff9b6333; }
.badge-truth { background:#172017; color:#7dff96; border:1px solid #7dff9633; }
.badge-div   { background:#161a22; color:#8ecbff; border:1px solid #8ecbff33; }
.badge-hot   { background:#231616; color:#ff7a7a; border:1px solid #ff7a7a33; }
/* Color bars inside HTML tables */
.tbl { width: 100%; border-collapse: collapse; }
.tbl th, .tbl td { padding: 6px 8px; border-bottom: 1px solid #222; font-size: 0.95em; }
.tbl th { text-align: left; color: #ddd; font-weight: 800; }
.tbl td { color: #eee; }
.barcell { width: 220px; }
.badgecell { width: 70px; text-align: center; }
.num { text-align: right; white-space:nowrap; }
.sym { color:#9ad; font-weight:800; }
.name { color:#fff; font-weight:700; }
.note { opacity: 0.75; font-size: 0.9rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(f"<div class='phase-badge'>✅ {PHASE_TAG}</div>", unsafe_allow_html=True)

# ============================== SIDEBAR ======================================

with st.sidebar:
    st.header("Navigation")
    nav = st.radio(
        "Go to",
        ["Dashboard", "Crypto", "Stocks", "Fusion", "Scores", "Export"],
        index=0,
        key="nav_radio",
    )

    st.header("Appearance")
    font_size = st.slider("Font size", 14, 24, 18, key="font_size")
    st.markdown(
        f"<style>html, body, [class*='css'] {{ font-size: {font_size}px; }}</style>",
        unsafe_allow_html=True,
    )
    high_contrast = st.toggle("High contrast mode", value=False, key="hc")
    if high_contrast:
        st.markdown("<style>.metric-box{background:#0b0d12;border-color:#8ecbff44}</style>", unsafe_allow_html=True)

    st.header("Watchlist (Stocks)")
    wl = st.text_input(
        "Tickers (comma-separated)",
        value=st.session_state.get("wl", "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA"),
        help="Stocks only here. Crypto handled on Crypto page.",
        key="watchlist",
    )
    st.session_state["wl"] = wl

    st.header("Refresh")
    auto = st.toggle("Auto refresh", value=False, help="Re-run periodically", key="auto")
    every = st.slider("Every (sec)", 10, 120, 30, key="every")

# ============================== CORE / SCORING ================================

def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def pct_sigmoid(pct) -> float:
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return 0.5
    try:
        x = float(pct) / 10.0
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5

@st.cache_data(ttl=60, show_spinner="Loading CoinGecko…")
def fetch_cg_markets(vs: str = "usd", per_page: int = 250) -> pd.DataFrame:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    p = {
        "vs_currency": vs, "order": "market_cap_desc",
        "per_page": int(max(1, min(per_page, 250))), "page": 1,
        "sparkline": "false", "price_change_percentage": "1h,24h,7d", "locale": "en"
    }
    r = requests.get(url, params=p, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    for k in [
        "name","symbol","current_price","market_cap","total_volume",
        "price_change_percentage_1h_in_currency",
        "price_change_percentage_24h_in_currency",
        "price_change_percentage_7d_in_currency",
    ]:
        if k not in df.columns:
            df[k] = np.nan
    return df

def score_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df.copy()
    t = df.copy()
    t["vol_to_mc"] = ((t.get("total_volume", 0) / t.get("market_cap", np.nan))
                      .replace([np.inf, -np.inf], np.nan)).clip(0, 2).fillna(0)
    t["m1h"] = t.get("price_change_percentage_1h_in_currency", pd.Series(np.nan, index=t.index)).apply(pct_sigmoid)
    t["m24"] = t.get("price_change_percentage_24h_in_currency", pd.Series(np.nan, index=t.index)).apply(pct_sigmoid)
    t["m7d"] = t.get("price_change_percentage_7d_in_currency", pd.Series(np.nan, index=t.index)).apply(pct_sigmoid)
    mc = t.get("market_cap", pd.Series(0, index=t.index)).fillna(0)
    t["liq01"] = 0 if mc.max() == 0 else (mc - mc.min()) / (mc.max() - mc.min() + 1e-9)

    # Scores
    t["raw_heat"] = (0.5 * (t["vol_to_mc"] / 2).clip(0, 1) + 0.5 * t["m1h"].fillna(0.5)).clip(0, 1)
    t["truth_full"] = (
        0.30 * (t["vol_to_mc"] / 2).clip(0, 1) +
        0.25 * t["m24"].fillna(0.5) +
        0.25 * t["m7d"].fillna(0.5) +
        0.20 * t["liq01"].fillna(0.0)
    ).clip(0, 1)
    t["divergence"] = (t["raw_heat"] - t["truth_full"]).abs()
    return t

# ============================== STOCKS (yfinance) =============================

@st.cache_data(ttl=120, show_spinner="Loading stocks…")
def yf_snapshot(tickers: List[str]) -> pd.DataFrame:
    if not HAS_YF or not tickers:
        return pd.DataFrame()
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(
            tickers=" ".join(tickers),
            period="5d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
    except Exception:
        return pd.DataFrame()

    rows = []
    for t in tickers:
        try:
            s = data[t]
            last = float(s.iloc[-1]["Close"])
            prev = float(s.iloc[-2]["Close"]) if len(s) >= 2 else np.nan
            pct24 = (last / prev - 1.0) * 100.0 if pd.notna(prev) else np.nan
            rows.append({
                "name": t, "symbol": t, "current_price": last,
                "price_change_percentage_24h_in_currency": pct24,
                "market_cap": np.nan, "total_volume": np.nan,
                "price_change_percentage_1h_in_currency": np.nan,
                "price_change_percentage_7d_in_currency": np.nan
            })
        except Exception:
            rows.append({
                "name": t, "symbol": t,
                "current_price": np.nan,
                "price_change_percentage_24h_in_currency": np.nan,
                "market_cap": np.nan, "total_volume": np.nan,
                "price_change_percentage_1h_in_currency": np.nan,
                "price_change_percentage_7d_in_currency": np.nan
            })
    return pd.DataFrame(rows)

# ============================== COLOR HELPERS (HTML) ==========================

def _clamp01(x: float) -> float:
    try:
        x = float(x)
        if np.isnan(x): return 0.0
        return 0.0 if x < 0 else (1.0 if x > 1 else x)
    except Exception:
        return 0.0

def _grad_hex(v: float, palette: str) -> str:
    v = _clamp01(v)
    if palette == "raw":      # orange/red
        # 0 -> #3a261a  ... 1 -> #ff8a4c
        r = int(58 + v * (255-58)); g = int(38 + v * (138-38)); b = int(26 + v * (76-26))
    elif palette == "truth":  # green/teal
        # 0 -> #1a2b1f  ... 1 -> #6cff9a
        r = int(26 + v * (108-26)); g = int(43 + v * (255-43)); b = int(31 + v * (154-31))
    else:                     # delta (blue)
        # 0 -> #172030  ... 1 -> #8ecbff
        r = int(23 + v * (142-23)); g = int(32 + v * (203-32)); b = int(48 + v * (255-48))
    return f"#{r:02x}{g:02x}{b:02x}"

def _bar_html(v: float, palette: str, width: int = 200, height: int = 12) -> str:
    v = _clamp01(v)
    pct = int(v * 100)
    bg = "#111419"
    fg = _grad_hex(v, palette)
    return (
        f"<div style='background:{bg}; width:{width}px; height:{height}px; border-radius:6px; overflow:hidden;'>"
        f"<div style='background:{fg}; width:{pct}%; height:{height}px;'></div></div>"
    )

def _badge_html(v: float, kind: str) -> str:
    v = _clamp01(v)
    if kind == "raw":
        emoji = "🔥" if v >= 0.45 else "·"
        color = _grad_hex(v, "raw")
    elif kind == "truth":
        emoji = "💧" if v >= 0.45 else "·"
        color = _grad_hex(v, "truth")
    else:
        emoji = "◆"
        color = _grad_hex(v, "delta")
    return f"<span style='display:inline-block;padding:2px 6px;border-radius:8px;background:{color}22;border:1px solid {color}55;color:{color};font-weight:800;'>{emoji}</span>"

def render_table_html(df: pd.DataFrame, columns: List[str], bars: List[tuple], topn: int = 25) -> str:
    # bars: list of tuples: (column_name, palette, show_badge_bool)
    head = "".join([f"<th>{c.upper()}</th>" for c in columns])
    # append bar columns
    for c,palette,badge in bars:
        head += f"<th class='barcell'>{c.upper()}</th>"
        head += "<th class='badgecell'> </th>"
    rows_html = ""
    for _, r in df.head(topn).iterrows():
        cells = []
        for c in columns:
            if c in ("current_price","market_cap","total_volume","price_change_percentage_24h_in_currency"):
                val = r.get(c, np.nan)
                txt = "-" if pd.isna(val) else (f"${val:,.2f}" if c!="price_change_percentage_24h_in_currency" else f"{val:+.2f}%")
                cells.append(f"<td class='num'>{txt}</td>")
            elif c == "symbol":
                cells.append(f"<td class='sym'>{str(r.get(c,''))}</td>")
            elif c == "name":
                cells.append(f"<td class='name'>{str(r.get(c,''))}</td>")
            else:
                cells.append(f"<td>{str(r.get(c,''))}</td>")
        # bars
        for bcol, palette, badge in bars:
            v = r.get(bcol, 0.0)
            cells.append(f"<td class='barcell'>{_bar_html(v, palette)}</td>")
            cells.append(f"<td class='badgecell'>{_badge_html(v, palette if palette!='delta' else 'delta')}</td>")
        rows_html += "<tr>" + "".join(cells) + "</tr>"
    html = f"<table class='tbl'><thead><tr>{head}</tr></thead><tbody>{rows_html}</tbody></table>"
    return html

# ============================== UI SECTIONS ===================================

def section_header(title: str, caption: str = "") -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if caption: st.caption(caption)
    st.write("")

def kpi_row(df_scored: pd.DataFrame, label: str) -> None:
    n = len(df_scored)
    p24 = float(df_scored.get("price_change_percentage_24h_in_currency", pd.Series(dtype=float)).mean())
    tavg = float(df_scored.get("truth_full", pd.Series(dtype=float)).mean())
    ravg = float(df_scored.get("raw_heat", pd.Series(dtype=float)).mean())
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown("<div class='metric-box'><b>Assets</b><br>{}</div>".format(n), unsafe_allow_html=True)
    with c2: st.markdown("<div class='metric-box'><b>Avg 24h %</b><br>{:.2f}%</div>".format(0 if np.isnan(p24) else p24), unsafe_allow_html=True)
    with c3: st.markdown("<div class='metric-box'><b>Avg TRUTH</b><br>{:.2f}</div>".format(0 if np.isnan(tavg) else tavg), unsafe_allow_html=True)
    with c4: st.markdown("<div class='metric-box'><b>Avg RAW</b><br>{:.2f}</div>".format(0 if np.isnan(ravg) else ravg), unsafe_allow_html=True)
    st.caption(f"{PHASE_TAG} • Updated {now_utc_str()} • Mode: {label}")

def truth_raw_panels_color(df_scored: pd.DataFrame, topn: int = 25) -> None:
    st.markdown(
        "<span class='badge badge-raw'>RAW</span>"
        "<span class='badge badge-truth'>TRUTH</span>"
        "<span class='badge badge-div'>DELTA</span>"
        "<span class='badge badge-hot'>MOVERS</span>",
        unsafe_allow_html=True,
    )
    st.write("")
    c1,c2,c3 = st.columns(3)

    with c1:
        st.subheader("RAW — Heat (color)")
        cols = ["name","symbol","current_price","market_cap","total_volume"]
        bars = [("raw_heat","raw",True)]
        html = render_table_html(df_scored.sort_values("raw_heat", ascending=False), cols, bars, topn=topn)
        st.markdown(html, unsafe_allow_html=True)

    with c2:
        st.subheader("TRUTH — Stability (color)")
        cols = ["name","symbol","current_price","market_cap"]
        bars = [("truth_full","truth",True)]
        html = render_table_html(df_scored.sort_values("truth_full", ascending=False), cols, bars, topn=topn)
        st.markdown(html, unsafe_allow_html=True)

    with c3:
        st.subheader("MOVERS — 24h (with Δ color)")
        if "price_change_percentage_24h_in_currency" in df_scored.columns:
            g = df_scored.sort_values("price_change_percentage_24h_in_currency", ascending=False)
            l = df_scored.sort_values("price_change_percentage_24h_in_currency", ascending=True)
            cols = ["name","symbol","current_price","price_change_percentage_24h_in_currency"]
            bars = [("divergence","delta",True)]
            st.markdown("Top Gainers")
            st.markdown(render_table_html(g, cols, bars, topn=10), unsafe_allow_html=True)
            st.markdown("Top Losers")
            st.markdown(render_table_html(l, cols, bars, topn=10), unsafe_allow_html=True)
        else:
            st.info("No 24h % column available.")

# ============================== PAGES =========================================

def page_dashboard() -> None:
    section_header("Crypto Hybrid Live — Dashboard", "Glance across markets using TRUTH vs RAW (color).")
    dfc = score_table(fetch_cg_markets("usd", 200))
    kpi_row(dfc, "Crypto")
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Top TRUTH (Crypto)")
        html = render_table_html(dfc.sort_values("truth_full", ascending=False), ["name","symbol","current_price"], [("truth_full","truth",True)], topn=15)
        st.markdown(html, unsafe_allow_html=True)
    with c2:
        st.subheader("Top RAW (Crypto)")
        html = render_table_html(dfc.sort_values("raw_heat", ascending=False), ["name","symbol","current_price"], [("raw_heat","raw",True)], topn=15)
        st.markdown(html, unsafe_allow_html=True)

def page_crypto() -> None:
    section_header("Crypto", "Live CoinGecko with TRUTH vs RAW vs DELTA — color coded.")
    topn = st.slider("Show top N (Crypto)", 50, 250, 150, key="crypto_topn")
    df = score_table(fetch_cg_markets("usd", topn))
    if df.empty:
        st.warning("No data received from CoinGecko.")
        return
    kpi_row(df, "Crypto")
    truth_raw_panels_color(df, topn=25)

def page_stocks() -> None:
    section_header("Stocks", "Robust yfinance snapshot scored by the same lens (color).")
    default = "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA"
    raw = st.text_input("Tickers (comma-separated)", value=st.session_state.get("stock_input", default), key="stock_input")
    if not HAS_YF:
        st.error("yfinance not installed on this deployment. Add `yfinance` to requirements.txt and reboot the app.")
        return
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    if not tickers:
        st.info("Enter at least one ticker.")
        return
    df0 = yf_snapshot(tickers)
    if df0.empty:
        st.warning("No stock data returned. Check tickers.")
        return
    df = score_table(df0)
    kpi_row(df, "Stocks")
    truth_raw_panels_color(df, topn=min(25, len(df)))

def page_fusion() -> None:
    section_header("Fusion", "Compare Crypto vs Stocks in one color-coded view.")
    # Crypto side
    dfc = score_table(fetch_cg_markets("usd", 120))
    dfc["universe"] = "CRYPTO"
    # Stocks side
    if HAS_YF:
        wl = st.session_state.get("wl", "")
        tick = [x.strip().upper() for x in wl.split(",") if x.strip()]
        dfs = score_table(yf_snapshot(tick)) if tick else pd.DataFrame()
        if dfs.empty:
            st.info("Stocks snapshot empty. Add tickers in the sidebar Watchlist.")
            dfs = pd.DataFrame(columns=dfc.columns)
        dfs["universe"] = "STOCKS"
    else:
        dfs = pd.DataFrame(columns=dfc.columns)
    try:
        both = pd.concat([dfc, dfs], ignore_index=True)
    except Exception:
        both = dfc.copy()
    kpi_row(both, "Fusion")
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Crypto — Top TRUTH")
        html = render_table_html(both[both.get("universe","")== "CRYPTO"].sort_values("truth_full", ascending=False),
                                 ["name","symbol","current_price"], [("truth_full","truth",True), ("raw_heat","raw",True)], topn=20)
        st.markdown(html, unsafe_allow_html=True)
    with c2:
        st.subheader("Stocks — Top TRUTH")
        html = render_table_html(both[both.get("universe","")== "STOCKS"].sort_values("truth_full", ascending=False),
                                 ["name","symbol","current_price"], [("truth_full","truth",True), ("raw_heat","raw",True)], topn=20)
        st.markdown(html, unsafe_allow_html=True)

def page_scores() -> None:
    section_header("Scores — Explainer")
    st.markdown("""
**RAW**: crowd heat right now (volume/marketcap + 1h momentum), scaled 0..1.  
**TRUTH**: steady heartbeat (volume/MC, 24h momentum, 7d momentum, liquidity), scaled 0..1.  
**DELTA**: gap |RAW − TRUTH|. Large = possible overextension or mean reversion setup.
""")
    st.info("Weights fixed in Phase 15. Next phase adds presets and a weight editor.")

def page_export() -> None:
    section_header("Export", "Download CSV snapshots.")
    dfc = score_table(fetch_cg_markets("usd", 200))
    st.download_button("Download Crypto CSV", data=dfc.to_csv(index=False).encode("utf-8"),
                       file_name="crypto_truth_raw.csv", mime="text/csv")
    if HAS_YF:
        wl = st.session_state.get("wl","AAPL,MSFT,NVDA,TSLA")
        tick = [x.strip().upper() for x in wl.split(",") if x.strip()]
        dfs = score_table(yf_snapshot(tick)) if tick else pd.DataFrame()
        if not dfs.empty:
            st.download_button("Download Stocks CSV", data=dfs.to_csv(index=False).encode("utf-8"),
                               file_name="stocks_truth_raw.csv", mime="text/csv")
        else:
            st.info("No stocks in snapshot. Add tickers in Watchlist on the sidebar.")
    else:
        st.info("yfinance not installed; add it to requirements to enable Stocks export.")

# ============================== ROUTER / REFRESH ==============================

if nav == "Dashboard":
    page_dashboard()
elif nav == "Crypto":
    page_crypto()
elif nav == "Stocks":
    page_stocks()
elif nav == "Fusion":
    page_fusion()
elif nav == "Scores":
    page_scores()
else:
    page_export()

if auto:
    st.caption(f"{PHASE_TAG} • Auto refresh every {int(every)}s")
    time.sleep(max(5, int(every)))
    st.rerun()
