import os, time, json, logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
import httpx
import redis.asyncio as redis
from prometheus_fastapi_instrumentator import Instrumentator

# ------------------ CONFIG ------------------
APP_NAME = "HIS Data Gateway"
TIMEOUT = float(os.getenv("TIMEOUT", "25"))

# Provider endpoints
COINGECKO   = "https://api.coingecko.com/api/v3"
COINPAPRIKA = "https://api.coinpaprika.com/v1"
PUSHSHIFT   = "https://api.pushshift.io/reddit/search/submission/"
YF_CHART    = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"

# External keys (optional)
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
EIA_API_KEY  = os.getenv("EIA_API_KEY", "")

# LIPE engine (when ready)
LIPE_API_URL = os.getenv("LIPE_API_URL", "")   # e.g., https://lipe-core.yourdomain/api
LIPE_API_KEY = os.getenv("LIPE_API_KEY", "")

# Gateway security (lock-in)
GATEWAY_SECRET = os.getenv("GATEWAY_SECRET", "")  # required header from Streamlit
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Redis cache (real DB cache)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_DEFAULT = int(os.getenv("CACHE_TTL_DEFAULT", "60"))

# Sentry (optional)
SENTRY_DSN = os.getenv("SENTRY_DSN", "")

# -------------- LOGGING/SENTRY --------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("his-gateway")

if SENTRY_DSN:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    sentry_sdk.init(dsn=SENTRY_DSN, integrations=[FastApiIntegration()], traces_sample_rate=0.1)
    logger.info("Sentry enabled")

# -------------- APP & MIDDLEWARE --------------
app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS] if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus /metrics
Instrumentator().instrument(app).expose(app)

# -------------- REDIS ------------------------
rds: Optional[redis.Redis] = None
async def get_redis() -> redis.Redis:
    global rds
    if rds is None:
        rds = redis.from_url(REDIS_URL, decode_responses=True)
    return rds

def now_ms() -> int:
    return int(time.time() * 1000)

def ck(path: str, params: Dict[str, Any]) -> str:
    return "cache:" + path + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))

async def cache_get(path: str, params: Dict[str, Any], ttl: int, r: redis.Redis) -> Optional[Dict[str, Any]]:
    k = ck(path, params)
    v = await r.get(k)
    if not v:
        return None
    try:
        obj = json.loads(v)
        # soft TTL enforcement is done by Redis expiry; if here, it’s fresh enough
        return obj
    except Exception:
        return None

async def cache_put(path: str, params: Dict[str, Any], data: Dict[str, Any], ttl: int, r: redis.Redis):
    k = ck(path, params)
    await r.setex(k, ttl, json.dumps(data))

# -------------- SECURITY (API KEY) --------------
async def require_gateway_key(request: Request):
    """Every call must include header: X-HIS-KEY: <GATEWAY_SECRET>"""
    if not GATEWAY_SECRET:
        # no secret set -> open for development
        return
    h = request.headers.get("X-HIS-KEY", "")
    if h != GATEWAY_SECRET:
        raise HTTPException(status_code=401, detail="Missing or invalid X-HIS-KEY")

# -------------- HTTP CLIENT -------------------
async def http_get(url: str, params=None, headers=None):
    async with httpx.AsyncClient(timeout=TIMEOUT) as cli:
        res = await cli.get(url, params=params, headers=headers)
        if res.status_code >= 400:
            logger.warning("HTTP error %s for %s params=%s", res.status_code, url, params)
            raise HTTPException(res.status_code, res.text)
        try:
            return res.json()
        except Exception:
            return json.loads(res.text)

# -------------- ENDPOINTS ---------------------
@app.get("/health")
async def health():
    return {"ok": True, "ts": now_ms()}

@app.get("/metrics-plain")
async def metrics_plain():
    return PlainTextResponse("ok")

# ---- Crypto quotes (CoinGecko → CoinPaprika fallback) ----
@app.get("/v1/crypto/quotes", dependencies=[Depends(require_gateway_key)])
async def crypto_quotes(ids: str = Query(..., description="comma list e.g. bitcoin,ethereum"),
                        r: redis.Redis = Depends(get_redis)):
    params = {"ids": ids}
    cached = await cache_get("/v1/crypto/quotes", params, CACHE_TTL_DEFAULT, r)
    if cached:
        return cached
    try:
        data = await http_get(f"{COINGECKO}/coins/markets",
                              params={"vs_currency":"usd","ids":ids})
        out = {
            "provider": "coingecko",
            "fetched_ms": now_ms(),
            "symbols": [
                {"id":x["id"], "symbol":x["symbol"], "price":x["current_price"],
                 "mc":x.get("market_cap"), "pct_24h":x.get("price_change_percentage_24h")}
                for x in data
            ],
        }
        await cache_put("/v1/crypto/quotes", params, out, CACHE_TTL_DEFAULT, r)
        return out
    except HTTPException:
        data = await http_get(f"{COINPAPRIKA}/tickers")
        want = {s.strip().lower() for s in ids.split(",")}
        filt = [x for x in data if x["name"].lower() in want or x["symbol"].lower() in want]
        out = {
            "provider":"coinpaprika","fetched_ms":now_ms(),
            "symbols":[
                {"id":x["id"],"symbol":x["symbol"],
                 "price":x["quotes"]["USD"]["price"],
                 "mc":x["quotes"]["USD"].get("market_cap"),
                 "pct_24h":x["quotes"]["USD"].get("percent_change_24h")}
                for x in filt
            ]
        }
        await cache_put("/v1/crypto/quotes", params, out, CACHE_TTL_DEFAULT, r)
        return out

# ---- Stocks history (Yahoo chart JSON; no key) ----
@app.get("/v1/stocks/history", dependencies=[Depends(require_gateway_key)])
async def stocks_history(symbol: str, period: str="6mo", interval: str="1d",
                         r: redis.Redis = Depends(get_redis)):
    key = {"s":symbol,"p":period,"i":interval}
    cached = await cache_get("/v1/stocks/history", key, 120, r)
    if cached:
        return cached
    data = await http_get(YF_CHART.format(sym=symbol),
                          params={"range": period, "interval": interval, "includePrePost":"false"})
    try:
        result = data["chart"]["result"][0]
        ts = result["timestamp"]
        close = result["indicators"]["quote"][0]["close"]
        out = {"provider":"yahoo","fetched_ms":now_ms(),
               "bars":[{"date":int(t)*1000,"close":c} for t,c in zip(ts, close) if c is not None]}
        await cache_put("/v1/stocks/history", key, out, 120, r)
        return out
    except Exception as e:
        logger.exception("YF parse failed")
        raise HTTPException(502, f"YF parse failed: {e}")

# ---- Sports odds (TheOddsAPI or demo) ----
@app.get("/v1/sports/odds", dependencies=[Depends(require_gateway_key)])
async def sports_odds(sport: str="americanfootball_nfl", market: str="spreads", region: str="us",
                      r: redis.Redis = Depends(get_redis)):
    params = {"sport":sport,"market":market,"region":region}
    cached = await cache_get("/v1/sports/odds", params, 30, r)
    if cached:
        return cached
    if not ODDS_API_KEY:
        demo = {"provider":"demo","fetched_ms":now_ms(),
                "lines":[
                    {"team":"Hawks","moneyline":-120},
                    {"team":"Sharks","moneyline":140},
                    {"team":"Tigers","moneyline":-105},
                    {"team":"Giants","moneyline":155}
                ]}
        await cache_put("/v1/sports/odds", params, demo, 30, r)
        return demo
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
    data = await http_get(url, params={"regions": region, "markets": market,
                                       "oddsFormat":"american", "apiKey": ODDS_API_KEY})
    out = {"provider":"the-odds-api","fetched_ms":now_ms(),"raw":data}
    await cache_put("/v1/sports/odds", params, out, 30, r)
    return out

# ---- Human Behavior (Reddit via Pushshift) ----
@app.get("/v1/social/reddit", dependencies=[Depends(require_gateway_key)])
async def reddit_search(q: str="crypto", size: int=10,
                        r: redis.Redis = Depends(get_redis)):
    params = {"q": q, "size": size, "sort":"desc"}
    cached = await cache_get("/v1/social/reddit", params, 60, r)
    if cached:
        return cached
    data = await http_get(PUSHSHIFT, params=params)
    out = {"provider":"pushshift","fetched_ms":now_ms(),
           "posts":[{"title":i.get("title"),"sub":i.get("subreddit"),"score":i.get("score")}
                    for i in data.get("data",[])]}
    await cache_put("/v1/social/reddit", params, out, 60, r)
    return out

# ---- Lottery (NY Take 5 / IL Pick-4) ----
@app.get("/v1/lottery/ny_take5_latest", dependencies=[Depends(require_gateway_key)])
async def ny_take5_latest(r: redis.Redis = Depends(get_redis)):
    params = {}
    cached = await cache_get("/v1/lottery/ny_take5_latest", params, 300, r)
    if cached:
        return cached
    url = "https://data.cityofnewyork.us/resource/5xaw-6ayf.json?$limit=1&$order=draw_date DESC"
    data = await http_get(url)
    if not data:
        raise HTTPException(502, "NY empty")
    row = data[0]
    out = {"provider":"nyc_open_data","fetched_ms":now_ms(),
           "draw":row.get("winning_numbers"),"date":row.get("draw_date")}
    await cache_put("/v1/lottery/ny_take5_latest", params, out, 300, r)
    return out

@app.get("/v1/lottery/il_pick4_latest", dependencies=[Depends(require_gateway_key)])
async def il_pick4_latest(r: redis.Redis = Depends(get_redis)):
    params = {}
    cached = await cache_get("/v1/lottery/il_pick4_latest", params, 300, r)
    if cached:
        return cached
    url = "https://data.illinois.gov/api/views/ck5f-mz5z/rows.json?accessType=DOWNLOAD"
    data = await http_get(url)
    try:
        last = data["data"][-1]
        draw = next((c for c in last if isinstance(c,str) and "-" in c and " " not in c), None)
        date = last[8] if len(last) > 8 else ""
        out = {"provider":"illinois_open_data","fetched_ms":now_ms(),"draw":draw,"date":date}
        await cache_put("/v1/lottery/il_pick4_latest", params, out, 300, r)
        return out
    except Exception as e:
        logger.exception("IL parse failed")
        raise HTTPException(502, f"IL parse failed: {e}")

# ---- Macro / Energy ----
@app.get("/v1/macro/fred", dependencies=[Depends(require_gateway_key)])
async def fred_series(series_id: str="MORTGAGE30US",
                      r: redis.Redis = Depends(get_redis)):
    params = {"series_id":series_id}
    cached = await cache_get("/v1/macro/fred", params, 600, r)
    if cached:
        return cached
    if not FRED_API_KEY:
        out = {"provider":"fred-demo","fetched_ms":now_ms(),"series":[]}
        await cache_put("/v1/macro/fred", params, out, 120, r)
        return out
    url = "https://api.stlouisfed.org/fred/series/observations"
    data = await http_get(url, params={"series_id":series_id,"api_key":FRED_API_KEY,"file_type":"json"})
    obs = data.get("observations", [])
    out = {"provider":"fred","fetched_ms":now_ms(),"series":[{"date":o["date"],"value":o["value"]} for o in obs]}
    await cache_put("/v1/macro/fred", params, out, 600, r)
    return out

@app.get("/v1/energy/eia", dependencies=[Depends(require_gateway_key)])
async def eia_series(series_id: str="PET.RWTC.D",
                     r: redis.Redis = Depends(get_redis)):
    params = {"series_id":series_id}
    cached = await cache_get("/v1/energy/eia", params, 600, r)
    if cached:
        return cached
    if not EIA_API_KEY:
        out = {"provider":"eia-demo","fetched_ms":now_ms(),"series":[]}
        await cache_put("/v1/energy/eia", params, out, 120, r)
        return out
    url = "https://api.eia.gov/series/"
    data = await http_get(url, params={"api_key":EIA_API_KEY,"series_id":series_id})
    s = data.get("series",[{}])[0].get("data",[])
    out = {"provider":"eia","fetched_ms":now_ms(),"series":[{"date":d[0],"value":d[1]} for d in s]}
    await cache_put("/v1/energy/eia", params, out, 600, r)
    return out

# ---- LIPE Forecast Integration (forwarder) ----
@app.post("/v1/lipe/forecast", dependencies=[Depends(require_gateway_key)])
async def lipe_forecast(req: Request,
                        arena: str = Query(..., description="e.g., lottery|crypto|sports|stocks"),
                        model: str = Query("default"),
                        horizon: int = Query(1, ge=1, le=30)):
    """
    Forwards payload to LIPE engine when LIPE_API_URL is set.
    Body: JSON with any fields LIPE needs (recent draws, symbols, odds, etc).
    """
    body = await req.json()
    if not LIPE_API_URL or not LIPE_API_KEY:
        # Safe fallback: echo + stub forecast
        return {
            "provider": "lipe-stub",
            "received": {"arena": arena, "model": model, "horizon": horizon, "body": body},
            "forecast": [{"t": i, "score": 0.5, "note": "stub"} for i in range(horizon)]
        }
    headers = {"Authorization": f"Bearer {LIPE_API_KEY}"}
    url = f"{LIPE_API_URL.rstrip('/')}/forecast"
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=headers) as cli:
        res = await cli.post(url, params={"arena": arena, "model": model, "horizon": horizon}, json=body)
        if res.status_code >= 400:
            raise HTTPException(res.status_code, f"LIPE error: {res.text}")
        return res.json()

# -------------- GLOBAL ERROR HANDLER ----------
@app.exception_handler(Exception)
async def on_error(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})
