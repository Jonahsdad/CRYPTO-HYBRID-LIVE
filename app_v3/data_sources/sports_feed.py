import requests
import pandas as pd
from datetime import datetime, timezone

BASE = "https://site.web.api.espn.com/apis/v2/sports"

SPORT_PATH = {
    "NBA":  ("basketball", "nba"),
    "NFL":  ("football",   "nfl"),
    "MLB":  ("baseball",   "mlb"),
    "NHL":  ("hockey",     "nhl"),
}

class SportsError(Exception):
    pass

def _get(url, params=None, timeout=15):
    try:
        r = requests.get(url, params=params or {}, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code >= 400:
            raise SportsError(f"ESPN error {r.status_code}: {r.text[:120]}")
        return r.json()
    except requests.RequestException as e:
        raise SportsError(f"Network error contacting ESPN: {e}")

def _parse_event(ev: dict) -> dict:
    comp = (ev.get("competitions") or [{}])[0]
    status = (comp.get("status") or {}).get("type", {})
    state = status.get("state", "pre")
    detail = status.get("shortDetail") or status.get("detail") or ""
    # Teams
    competitors = comp.get("competitors") or []
    home = next((c for c in competitors if c.get("homeAway") == "home"), {})
    away = next((c for c in competitors if c.get("homeAway") == "away"), {})
    # Scores
    home_team = (home.get("team") or {}).get("displayName") or ""
    away_team = (away.get("team") or {}).get("displayName") or ""
    home_score = int(float(home.get("score") or 0))
    away_score = int(float(away.get("score") or 0))
    # Records (win% rough trend)
    def win_pct(entry):
        recs = entry.get("records") or []
        if not recs: return None
        # pick first record set with summary like "12-5"
        s = recs[0].get("summary", "")
        if "-" in s:
            try:
                w, l = s.split("-")[:2]
                w, l = int(w), int(l)
                if w + l > 0:
                    return w / (w + l)
            except Exception:
                return None
        return None

    home_wp = win_pct(home)
    away_wp = win_pct(away)
    # Odds if present
    odds = ""
    try:
        od = (comp.get("odds") or [])[0]
        odds = od.get("details") or ""
    except Exception:
        odds = ""

    # Time
    start_iso = ev.get("date")
    try:
        start_dt = pd.to_datetime(start_iso, utc=True)
        start_local = start_dt.tz_convert(None)  # shows server local; fine for now
    except Exception:
        start_local = start_iso

    venue = (comp.get("venue") or {}).get("fullName") or ""

    return {
        "event_id": ev.get("id"),
        "league": ev.get("league", {}).get("name", ""),
        "start_time": start_local,
        "status": state.upper(),
        "status_detail": detail,
        "away": away_team,
        "away_score": away_score,
        "home": home_team,
        "home_score": home_score,
        "home_win_pct": home_wp,
        "away_win_pct": away_wp,
        "trend_gap": (home_wp - away_wp) if (home_wp is not None and away_wp is not None) else None,
        "venue": venue,
        "odds": odds,
    }

def get_scoreboard(league_key: str) -> pd.DataFrame:
    """Fetch today's scoreboard for a league (NBA/NFL/MLB/NHL)."""
    sport, league = SPORT_PATH[league_key]
    url = f"{BASE}/{sport}/{league}/scoreboard"
    data = _get(url)
    events = data.get("events", [])
    rows = [_parse_event(ev) for ev in events]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["status", "start_time"], ascending=[True, True]).reset_index(drop=True)
    return df
