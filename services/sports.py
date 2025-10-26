# services/sports.py
from services.http import get
from services.cache import cached
import os

ODDS = os.environ.get("ODDS_API") or ""

@cached("odds:lines", ttl=30)
def moneylines(sport_key: str = "basketball_nba", regions: str = "us"):
    if not ODDS:
        return {"error":"The Odds API key missing"}
    return get(f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
               params={"apiKey": ODDS, "regions": regions, "markets": "h2h", "oddsFormat": "american"})
