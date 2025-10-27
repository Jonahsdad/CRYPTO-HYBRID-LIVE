# ==============================
# HYBRID INTELLIGENCE SYSTEMS
# Neon UX + Arena Tiles + Free APIs (secure mode, no blockchain)
# Powered by JESSE RAY LANDINGHAM JR
# ==============================

import os, csv, json, math, random
from datetime import datetime
from time import monotonic, sleep
from typing import Any, Dict, List

import streamlit as st
import pandas as pd
import numpy as np
import requests

# Optional plotting
try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# Optional libs for data
try:
    import yfinance as yf
except Exception:
    yf = None
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# -------------------- App chrome --------------------
st.set_page_config(page_title="Hybrid Intelligence Systems", page_icon="🧠", layout="wide")

st.markdown("""
<style>
/* Hero */
.his-hero {
  background: radial-gradient(1200px 400px at 50% -10%, rgba(255,50,180,.18), rgba(30,144,255,.12), transparent 70%);
  border:1px solid rgba(255,255,255,.08);
  box-shadow: inset 0 0 40px rgba(255,255,255,.04), 0 0 30px rgba(30,144,255,.15);
  border-radius:14px; padding:28px; text-align:center; margin: 8px 0 18px 0;
}
.his-title {font-size:38px; font-weight:800; letter-spacing:.5px}
.his-sub {opacity:.85}

/* Tiles */
.tile {border:1px solid rgba(255,255,255,.10); border-radius:12px;
       padding:16px; margin-bottom:14px; background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
       transition: all .15s ease; cursor:pointer;}
.tile:hover {transform: translateY(-2px); border-color: rgba(30,144,255,.35); box-shadow:0 6px 20px rgba(0,0,0,.25)}
.tile h4 {margin:0 0 6px 0; font-weight:700}
.small {font-size:13px; opacity:.85}
a, a:visited { color:#7dc1ff; text-decoration:none; }
</style>
""", unsafe_allow_html=True)

# -------------------- Session defaults --------------------
def _init_state():
    s = st.session_state
    s.setdefault("page", "home")
    s.setdefault("ledger", [])
    s.setdefault("truth_filter", 60)
    s.setdefault("astro_on", False)
    s.setdefault("sports_odds", None)
_init_state()

# -------------------- Resilient, cached HTTP --------------------
@st.cache_resource(show_spinner=False)
def _shared_session():
    sess = requests.Session()
    sess.headers.update({"User-Agent": "HIS/1.0 (Streamlit)"})
    return sess

def _call(method, url, *, params=None, json_body=None, headers=None, timeout=12, tries=3):
    sess = _shared_session()
    last = None
    for k in range(tries):
        t0 = monotonic()
        try:
            if method == "GET":
                r = sess.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
            else:
                r = sess.post(url, params=params or {}, json=json_body, headers=headers or {}, timeout=timeout)
            r.raise_for_status()
            ms = int((monotonic() - t0) * 1000)
            try:
                return r.json(), ms
            except Exception:
                return {"_raw": r.text}, ms
        except Exception as e:
            last = e
            sleep(0.25 + 0.1*k)
    raise last

@st.cache_data(ttl=60, show_spinner=False)
def http_get(url, params=None, headers=None, timeout=12):
    return _call("GET", url, params=params, headers=headers, timeout=timeout)

def freshness(ms: int|None) -> str:
    if ms is None: return "n/a"
    if ms < 300: return f"{ms} ms · fresh"
    if ms < 1200: return f"{ms} ms"
    return f"{ms} ms (slow)"

# -------------------- Storage helpers (Lottery history) --------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def _append_csv(name: str, row: Dict[str, Any]) -> None:
    path = os.path.join(DATA_DIR, name)
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)

def _load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

# -------------------- Banner --------------------
def hero():
    st.markdown("""
<div class="his-hero">
  <div class="his-sub">Hybrid • Local/Remote</div>
  <div class="his-title">Hybrid Intelligence Systems</div>
  <div class="his-sub"><b>Powered by JESSE RAY LANDINGHAM JR</b></div>
</div>
""", unsafe_allow_html=True)

# -------------------- Arena home --------------------
def arena_home():
    hero()
    c1, c2, c3 = st.columns([1,1,1.8])
    with c1: st.slider("Truth Filter", 0, 100, key="truth_filter")
    with c2: st.toggle("Enable Astrology Influence", key="astro_on")
    with c3: st.caption("All data via free public APIs — secure mode (no blockchain)")

    st.markdown("#### Choose your arena")
    rows = [
        [("🎰", "Lottery", "Daily numbers, picks, entropy, risk modes"),
         ("💰", "Crypto", "Live pricing, signals, overlays"),
         ("📈", "Stocks", "Charts, momentum, factor overlays")],
        [("🧾", "Options", "Chains, quick IV views"),
         ("🏡", "Real Estate", "Market tilt and projections"),
         ("🛢️", "Commodities", "Energy, metals, ag")],
        [("🏈", "Sports", "Game signals and parlay entropy"),
         ("🧠", "Human Behavior", "Cohort trends and intent"),
         ("🌌", "Astrology", "Symbolic overlays (hybrid)")]
    ]
    for row in rows:
        cols = st.columns(3)
        for i,(icon, name, desc) in enumerate(row):
            with cols[i]:
                if st.button(f"{icon}  {name}", use_container_width=True):
                    st.session_state.page = name.lower().replace(" ","_")
                    st.rerun()
                st.markdown(f"<div class='tile'><div class='small'>{desc}</div></div>", unsafe_allow_html=True)

# -------------------- Ledger --------------------
def log_ledger(module: str, payload: Dict[str, Any]):
    st.session_state.ledger.append({
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "module": module,
        **payload
    })

def ledger_panel():
    if not st.session_state.ledger:
        st.caption("Run something to populate the ledger.")
        return
    df = pd.DataFrame(st.session_state.ledger)
    st.download_button("Download Run Ledger (CSV)",
                       data=df.to_csv(index=False).encode("utf-8"),
                       file_name="his_run_ledger.csv",
                       mime="text/csv")
    with st.expander("View ledger"):
        st.dataframe(df, use_container_width=True)

# -------------------- Pages --------------------
# Lottery (Illinois + New York, with CSV history)
def page_lottery():
    hero()
    st.markdown("### 🎰 Lottery")

    c1, c2 = st.columns(2)

    # Illinois Pick-4 via LotteryUSA (scrape)
    with c1:
        st.caption("Illinois Pick-4 (latest)")
        if BeautifulSoup is None:
            st.info("Install beautifulsoup4 to enable this fetch.")
            il_draw, il_date = None, ""
        else:
            try:
                html, ms = http_get("https://www.lotteryusa.com/illinois/pick-4/")
                html = html.get("_raw","") if isinstance(html, dict) else html
                soup = BeautifulSoup(html, "html.parser")
                nums = soup.select_one(".c-results-card__numbers")
                title = soup.select_one(".c-results-card__title")
                il_draw = nums.get_text(strip=True) if nums else None
                il_date = title.get_text(strip=True) if title else ""
                if il_draw:
                    st.success(f"Illinois Pick-4: {il_draw}  •  {il_date}")
                    st.caption("LotteryUSA • " + freshness(ms))
                else:
                    st.warning("Unavailable (site layout may have changed).")
            except Exception as e:
                il_draw, il_date = None, ""
                st.warning(f"IL fetch failed: {e}")
        if il_draw:
            _append_csv("il_pick4.csv", {"timestamp": datetime.utcnow().isoformat(), "draw": il_draw, "draw_date": il_date})

    # New York Take 5 via NY Open Data JSON
    with c2:
        st.caption("New York Take 5 (latest)")
        try:
            data, ms2 = http_get("https://data.ny.gov/resource/d6yy-54nr.json",
                                 params={"$limit": 1, "$order": "draw_date DESC"})
            if isinstance(data, list) and data:
                row = data[0]
                ny_draw = row.get("winning_numbers")
                ny_date = row.get("draw_date","")
                st.success(f"NY Take 5: {ny_draw}  •  {ny_date}")
                st.caption("NY Open Data • " + freshness(ms2))
                if ny_draw:
                    _append_csv("ny_take5.csv", {"timestamp": datetime.utcnow().isoformat(), "draw": ny_draw, "draw_date": ny_date})
            else:
                st.info("No rows returned.")
        except Exception as e:
            st.warning(f"NY fetch failed: {e}")

    st.divider()
    with st.expander("View historical draws"):
        t1, t2 = st.tabs(["Illinois", "New York"])
        with t1:
            st.dataframe(_load_csv("il_pick4.csv"), use_container_width=True)
        with t2:
            st.dataframe(_load_csv("ny_take5.csv"), use_container_width=True)

    if st.button("Log to Ledger (Lottery)"):
        log_ledger("lottery", {"IL": _load_csv("il_pick4.csv").tail(1).to_dict("records"),
                               "NY": _load_csv("ny_take5.csv").tail(1).to_dict("records")})
        st.success("Logged to ledger.")
    st.divider()
    ledger_panel()

# Crypto
def page_crypto():
    hero()
    st.markdown("### 💰 Crypto")
    ids = st.text_input("CoinGecko IDs", "bitcoin,ethereum,solana,dogecoin").replace(" ","")

    # CoinGecko markets
    try:
        data, ms = http_get("https://api.coingecko.com/api/v3/coins/markets",
                            params={"vs_currency":"usd","ids":ids,"order":"market_cap_desc","per_page":50,"page":1,"sparkline":"false"})
        df = pd.DataFrame(data)[["name","symbol","current_price","price_change_percentage_24h","market_cap"]]
        st.dataframe(df, use_container_width=True)
        st.caption("CoinGecko • " + freshness(ms))
    except Exception as e:
        st.warning(f"CoinGecko error: {e}")

    # Binance 24h (optional)
    try:
        raw, ms2 = http_get("https://api.binance.com/api/v3/ticker/24hr")
        if isinstance(raw, list):
            wanted = [f"{s.upper()}USDT" for s in [x.strip() for x in ids.split(",") if x.strip()][:5]]
            bd = pd.DataFrame(raw)
            sel = bd[bd["symbol"].isin(wanted)][["symbol","lastPrice","priceChangePercent","quoteVolume"]]
            if not sel.empty:
                st.write("Binance 24h (spot)")
                st.dataframe(sel, use_container_width=True)
                st.caption("Binance • " + freshness(ms2))
    except Exception:
        pass

    # Reddit public JSON pulse
    q = st.text_input("Reddit keyword", "bitcoin")
    try:
        posts, ms3 = http_get("https://www.reddit.com/search.json", params={"q": q, "limit": 10, "sort":"new"})
        if isinstance(posts, dict) and posts.get("data",{}).get("children"):
            titles = [c["data"]["title"] for c in posts["data"]["children"]]
            st.write("\n\n".join("• " + t for t in titles[:8]))
            st.caption("Reddit (public JSON) • " + freshness(ms3))
    except Exception:
        pass

    st.divider()
    ledger_panel()

# Stocks
def page_stocks():
    hero()
    st.markdown("### 📈 Stocks")
    tickers = st.text_input("Tickers (comma)", "AAPL,MSFT,NVDA")
    period  = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y"], index=2)
    interval= st.selectbox("Interval", ["1d","1h","30m"], index=0)

    if yf is None:
        st.warning("Install yfinance to enable stocks.")
    else:
        for t in [x.strip().upper() for x in tickers.split(",") if x.strip()]:
            try:
                data = yf.download(t, period=period, interval=interval, progress=False)
                if data.empty:
                    st.info(f"No data for {t}."); continue
                df = data.reset_index()
                if HAS_PLOTLY:
                    fig = px.line(df, x="Date", y="Close", title=f"{t} Close — {period}/{interval}", template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.line_chart(df.set_index("Date")["Close"])
            except Exception as e:
                st.warning(f"{t} failed: {e}")

    # Yahoo RSS headlines (optional)
    if BeautifulSoup:
        try:
            rss, ms = http_get("https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US")
            raw = rss.get("_raw","") if isinstance(rss, dict) else rss
            if isinstance(raw, str) and "<item>" in raw:
                soup = BeautifulSoup(raw, "xml")
                its = soup.select("item")[:5]
                if its:
                    st.write("Latest headlines (Yahoo RSS):")
                    for it in its:
                        st.markdown(f"- [{it.title.text}]({it.link.text})")
                    st.caption("Yahoo RSS • " + freshness(ms))
        except Exception:
            pass

    st.divider()
    ledger_panel()

# Options
def page_options():
    hero()
    st.markdown("### 🧾 Options")
    if yf is None:
        st.warning("Install yfinance to enable options.")
        return
    ticker = st.text_input("Underlying", "AAPL")
    if st.button("Load Chain", type="primary"):
        try:
            tk = yf.Ticker(ticker)
            exps = tk.options
            if not exps:
                st.info("No expirations."); return
            sel = st.selectbox("Expiration", exps, index=0)
            oc = tk.option_chain(sel)
            st.write("Calls"); st.dataframe(oc.calls.head(30), use_container_width=True)
            st.write("Puts");  st.dataframe(oc.puts.head(30), use_container_width=True)
            if "impliedVolatility" in oc.calls.columns:
                ivp = float(oc.calls["impliedVolatility"].dropna().mean())*100
                st.metric("Avg Call IV (proxy)", f"{ivp:.1f}%")
        except Exception as e:
            st.error(f"Chain error: {e}")
    st.divider()
    ledger_panel()

# Sports (fixed)
def page_sports():
    hero()
    st.markdown("### 🏈 Sports")
    c1, c2, c3 = st.columns([1.5,1.2,1.2])
    with c1: sport_key = st.selectbox("Sport", ["basketball_nba","americanfootball_nfl","baseball_mlb","icehockey_nhl","soccer_epl"], index=0)
    with c2: market = st.selectbox("Market", ["h2h","spreads","totals"], index=0)
    with c3: region = st.selectbox("Region", ["us","us2","eu","uk"], index=0)

    a1, a2 = st.columns(2)
    fetch_btn = a1.button("Fetch Odds", type="primary", use_container_width=True)
    run_btn   = a2.button("Run Sports Forecast", use_container_width=True)

    if "sports_odds" not in st.session_state:
        st.session_state.sports_odds = None

    if fetch_btn:
        key = st.secrets.get("ODDS_API_KEY","")
        try:
            if key:
                url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
                data, ms = http_get(url, params={"regions":region,"markets":market,"oddsFormat":"american","apiKey":key})
                rows = []
                if isinstance(data, list):
                    for ev in data:
                        eid = ev.get("id")
                        for bm in ev.get("bookmakers", []):
                            for mk in bm.get("markets", []):
                                if mk.get("key") != market: continue
                                for out in mk.get("outcomes", []):
                                    rows.append({"event": eid, "team": out.get("name"), "moneyline": out.get("price", None)})
                df = pd.DataFrame(rows)
                if df.empty:
                    df = pd.DataFrame({"team":["Hawks","Sharks","Tigers","Giants"], "moneyline":[-120,140,-105,155]})
                    st.info("No live odds; demo shown.")
                st.session_state.sports_odds = {"df": df, "ms": ms}
            else:
                df = pd.DataFrame({"team":["Hawks","Sharks","Tigers","Giants"], "moneyline":[-120,140,-105,155]})
                st.session_state.sports_odds = {"df": df, "ms": None}
        except Exception as e:
            st.error(f"Odds fetch failed: {e}")

    blob = st.session_state.sports_odds
    if blob and "df" in blob:
        df = blob["df"].copy()
        st.subheader("Current Odds")
        st.dataframe(df, use_container_width=True)
        if "moneyline" in df.columns:
            if HAS_PLOTLY:
                fig = px.bar(df, x="team", y="moneyline", title=f"{sport_key} — {market}", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df.set_index("team")["moneyline"])
        st.caption("Odds • " + freshness(blob["ms"]))

    if run_btn:
        df = (st.session_state.sports_odds or {}).get("df")
        if df is None or df.empty:
            st.info("Fetch odds first."); return

        def prob_from_ml(ml: float) -> float:
            ml = float(ml)
            return (100.0/(ml+100.0)) if ml > 0 else (abs(ml)/(abs(ml)+100.0))

        if "moneyline" in df.columns:
            df_calc = df.copy()
            df_calc["p"] = df_calc["moneyline"].apply(prob_from_ml)
            def b_from_ml(ml):
                ml = float(ml)
                dec = (ml/100.0 + 1.0) if ml > 0 else (1.0 + 100.0/abs(ml))
                return max(dec-1.0, 1e-9)
            df_calc["b"] = df_calc["moneyline"].apply(b_from_ml)
            def kelly(p,b):
                q = 1-p; return float(min(max((b*p - q)/b,0.0),1.0))
            df_calc["kelly"] = df_calc.apply(lambda r: kelly(r["p"], r["b"]), axis=1)
            best = df_calc.sort_values("kelly", ascending=False).iloc[0]
            pick = best["team"]; raw_conf = 0.5 + 0.5*float(best["kelly"])
        else:
            pick, raw_conf = df.iloc[0].get("team","Team A"), 0.55

        conf = raw_conf
        if st.session_state.astro_on: conf = 0.5 + (conf - 0.5) * 1.10
        tf = st.session_state.truth_filter / 100.0
        conf = 0.5 + (conf - 0.5) * (0.6 + 0.4 * tf)

        st.success(f"Top pick: {pick}")
        cA, cB, cC = st.columns(3)
        cA.metric("Confidence", f"{int(conf*100)}%")
        cB.metric("Market", market.upper())
        cC.metric("Truth Filter", f"{st.session_state.truth_filter}%")

        log_ledger("sports", {"pick": pick, "conf": float(conf), "sport": sport_key, "market": market, "truth": st.session_state.truth_filter})

    st.divider()
    ledger_panel()

# Real Estate
def page_real_estate():
    hero(); st.markdown("### 🏡 Real Estate")
    fred_key = st.secrets.get("FRED_API_KEY","")
    def fred_series(series_id):
        try:
            data, ms = http_get("https://api.stlouisfed.org/fred/series/observations",
                                params={"series_id":series_id,"file_type":"json","api_key":fred_key})
            if isinstance(data, dict) and "observations" in data:
                df = pd.DataFrame(data["observations"])[["date","value"]]
                df["value"] = pd.to_numeric(df["value"], errors="coerce").dropna()
                return df, ms
        except Exception:
            pass
        return pd.DataFrame(), None

    m30, ms1 = fred_series("MORTGAGE30US")
    cpi, ms2 = fred_series("CPIAUCSL")

    if not m30.empty:
        (st.plotly_chart(px.line(m30.tail(240), x="date", y="value", title="30Y Mortgage Rate", template="plotly_dark"), use_container_width=True)
         if HAS_PLOTLY else st.line_chart(m30.set_index("date")["value"].tail(240)))
        st.caption("FRED MORTGAGE30US • " + freshness(ms1))
    if not cpi.empty:
        (st.plotly_chart(px.line(cpi.tail(240), x="date", y="value", title="CPI (Index)", template="plotly_dark"), use_container_width=True)
         if HAS_PLOTLY else st.line_chart(cpi.set_index("date")["value"].tail(240)))
        st.caption("FRED CPIAUCSL • " + freshness(ms2))

    st.divider(); ledger_panel()

# Commodities
def page_commodities():
    hero(); st.markdown("### 🛢️ Commodities")
    eia_key = st.secrets.get("EIA_API_KEY","DEMO_KEY")
    try:
        obj, ms = http_get("https://api.eia.gov/series/", params={"api_key": eia_key, "series_id":"PET.RWTC.D"})
        series = obj.get("series",[{}])[0].get("data", []) if isinstance(obj, dict) else []
        if series:
            df = pd.DataFrame(series, columns=["date","price"]).sort_values("date")
            (st.plotly_chart(px.line(df.tail(365), x="date", y="price", title="WTI Spot (EIA)", template="plotly_dark"), use_container_width=True)
             if HAS_PLOTLY else st.line_chart(df.set_index("date")["price"].tail(365)))
            st.caption("EIA PET.RWTC.D • " + freshness(ms))
    except Exception:
        st.info("EIA fetch failed.")
    try:
        demo, ms2 = http_get("https://metals-api.com/api/latest?base=USD&symbols=XAU,XAG")
        if isinstance(demo, dict) and "rates" in demo:
            st.json({"XAU_USD": demo["rates"].get("XAU"), "XAG_USD": demo["rates"].get("XAG")})
            st.caption("Metals-API demo • " + freshness(ms2))
    except Exception:
        pass
    st.divider(); ledger_panel()

# Human Behavior
def page_behavior():
    hero(); st.markdown("### 🧠 Human Behavior")
    kw = st.text_input("Keyword", "crypto")
    try:
        posts, ms = http_get("https://www.reddit.com/search.json", params={"q": kw, "limit": 15, "sort":"new"})
        if isinstance(posts, dict) and posts.get("data",{}).get("children"):
            rows = [{"title": c["data"]["title"], "score": c["data"]["score"], "sub": c["data"]["subreddit"]} for c in posts["data"]["children"]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption("Reddit (public JSON) • " + freshness(ms))
        else:
            st.info("No results.")
    except Exception as e:
        st.warning(f"Reddit fetch failed: {e}")
    st.divider(); ledger_panel()

# Astrology
def page_astrology():
    hero(); st.markdown("### 🌌 Astrology")
    st.caption("NASA SSD demo + synthetic overlay (free mode)")
    try:
        obj, ms = http_get("https://ssd-api.jpl.nasa.gov/sentry.api")
        if isinstance(obj, dict):
            st.json({"NASA feed (demo keys)": list(obj.keys())[:3]})
            st.caption("NASA SSD • " + freshness(ms))
    except Exception:
        st.info("NASA endpoint limited. Showing synthetic overlay.")
    days = st.slider("Days back", 7, 120, 45)
    xs = list(range(days))
    deg = [(math.sin(i/5.0) * 57.2958) % 360 for i in xs]
    df = pd.DataFrame({"d": xs, "degree": deg})
    (st.plotly_chart(px.line(df, x="d", y="degree", title="Planetary Degree (synthetic demo)", template="plotly_dark"), use_container_width=True)
     if HAS_PLOTLY else st.line_chart(df.set_index("d")["degree"]))
    st.divider(); ledger_panel()

# -------------------- Router --------------------
def back_button():
    if st.button("Back"):
        st.session_state.page = "home"; st.rerun()

p = st.session_state.page
if p == "home":
    arena_home()
elif p == "lottery":
    back_button(); page_lottery()
elif p == "crypto":
    back_button(); page_crypto()
elif p == "stocks":
    back_button(); page_stocks()
elif p == "options":
    back_button(); page_options()
elif p == "sports":
    back_button(); page_sports()
elif p == "real_estate":
    back_button(); page_real_estate()
elif p == "commodities":
    back_button(); page_commodities()
elif p == "human_behavior":
    back_button(); page_behavior()
elif p == "astrology":
    back_button(); page_astrology()
else:
    st.session_state.page = "home"; st.rerun()
