import pandas as pd
from datetime import datetime, timedelta
from utils.http import fetch_json
from utils.errors import SportsError
from utils.settings import settings

HOSTS = [h.strip() for h in settings.espn_hosts.split(",")]
SPORT_PATH = {"NBA":("basketball","nba"), "NFL":("football","nfl"),
              "MLB":("baseball","mlb"), "NHL":("hockey","nhl")}

def _get(url, params=None):
    try:
        return fetch_json(url, params=params)
    except Exception as e:
        msg=str(e)
        if "HTTP 404" in msg: raise SportsError("404")
        raise SportsError(msg)

def _parse_event(ev: dict) -> dict:
    comp = (ev.get("competitions") or [{}])[0]
    status = (comp.get("status") or {}).get("type", {})
    state = (status.get("state") or "pre").upper()
    detail = status.get("shortDetail") or status.get("detail") or ""
    competitors = comp.get("competitors") or []
    home = next((c for c in competitors if c.get("homeAway")=="home"), {})
    away = next((c for c in competitors if c.get("homeAway")=="away"), {})
    home_team = (home.get("team") or {}).get("displayName") or ""
    away_team = (away.get("team") or {}).get("displayName") or ""
    def _score(x): 
        try: return int(float(x or 0))
        except: return 0
    home_score, away_score = _score(home.get("score")), _score(away.get("score"))

    def win_pct(entry):
        recs=entry.get("records") or []
        if not recs: return None
        s=recs[0].get("summary","")
        if "-" in s:
            try:
                w,l=s.split("-")[:2]; w,l=int(w),int(l)
                return w/(w+l) if (w+l)>0 else None
            except: return None
        return None

    home_wp, away_wp = win_pct(home), win_pct(away)
    odds = ""
    try: odds=(comp.get("odds") or [])[0].get("details") or ""
    except: pass
    start_iso = ev.get("date")
    try: start_dt = pd.to_datetime(start_iso, utc=True).tz_convert(None)
    except: start_dt = start_iso
    venue = (comp.get("venue") or {}).get("fullName") or ""

    return {
        "event_id": ev.get("id"),
        "start_time": start_dt,
        "status": state,
        "status_detail": detail,
        "away": away_team, "away_score": away_score,
        "home": home_team, "home_score": home_score,
        "home_win_pct": home_wp, "away_win_pct": away_wp,
        "trend_gap": (home_wp-away_wp) if (home_wp is not None and away_wp is not None) else None,
        "venue": venue, "odds": odds,
    }

def _fetch_once(sport: str, league: str, ymd: str):
    last=None
    for host in HOSTS:
        url=f"{host}/{sport}/{league}/scoreboard"
        try:
            data=_get(url, params={"dates": ymd})
            return data.get("events", [])
        except SportsError as e:
            last=e; continue
    raise last or SportsError("Unknown ESPN error")

def get_scoreboard(league_key: str, date: datetime|None=None):
    sport, league = SPORT_PATH[league_key]
    dt = date or datetime.utcnow()
    for d in range(0,7):  # today..6 days back
        ymd=(dt - timedelta(days=d)).strftime("%Y%m%d")
        try:
            events=_fetch_once(sport, league, ymd)
            rows=[_parse_event(ev) for ev in events]
            df=pd.DataFrame(rows)
            if not df.empty:
                df["scoreboard_date"]=ymd
                return df.sort_values(["status","start_time"], ascending=[True,True]).reset_index(drop=True)
        except SportsError as e:
            if str(e)!="404": raise
            continue
    return pd.DataFrame()
