import streamlit as st, pandas as pd
from datetime import datetime
from ui.widgets import PrimaryButton, Loader, StatCard, Badge
from data_sources.sports_feed import get_scoreboard, SportsError
from utils.vault import insert_sports_events
from utils.engine import action

LEAGUES = ["NBA","NFL","MLB","NHL"]

@st.cache_data(ttl=60, show_spinner=False)
def _cached_scoreboard(league: str, d: str|None) -> pd.DataFrame:
    dt = datetime.strptime(d, "%Y%m%d") if d else None
    return get_scoreboard(league, date=dt)

def view(theme):
    st.subheader("Sports Arena")
    Badge("Live Data: ESPN public scoreboard API", "success")

    league = st.selectbox("League", LEAGUES, index=0, key="sp_league")
    with st.expander("Date (optional)", expanded=False):
        d = st.date_input("Pick a date (or leave empty for auto-lookup)", value=None, key="sp_date")
        date_str = d.strftime("%Y%m%d") if d else None

    colA,colB,colC = st.columns(3)

    def _show(df):
        if df.empty: st.info("No events found (today/recent)."); return False
        show=df[["scoreboard_date","start_time","status_detail","away","away_score","home","home_score","odds","venue"]]
        st.dataframe(show, use_container_width=True); return True

    with colA:
        def _do_fetch():
            try:
                with action("Sports: Fetch Scoreboard", "sports.fetch", league=league, date=date_str or "auto"):
                    Loader(f"Fetching {league}…", 0.2)
                    df=_cached_scoreboard(league, date_str)
                    if _show(df): insert_sports_events(df, league)
                    StatCard("Games Found", str(len(df)))
            except SportsError as e: st.error(str(e))
        PrimaryButton("Fetch Games / Scores (Live)", key="s_fetch_live", run=_do_fetch)

    with colB:
        def _do_refresh():
            try:
                with action("Sports: Refresh Live", "sports.refresh", league=league, date=date_str or "auto"):
                    Loader("Refreshing…", 0.15)
                    df=_cached_scoreboard(league, date_str)
                    live=df[df["status"].isin(["IN","POST","POSTPONED","FINAL"])]
                    if live.empty: st.caption("No live/final games for that date.")
                    else: st.dataframe(live[["scoreboard_date","status_detail","away","away_score","home","home_score","odds"]], use_container_width=True)
                    StatCard("Live/Final Games", str(len(live)))
            except SportsError as e: st.error(str(e))
        PrimaryButton("Refresh Live Scores", key="s_refresh_live", run=_do_refresh)

    with colC:
        def _do_trend():
            try:
                with action("Sports: Trend Compare", "sports.trend", league=league, date=date_str or "auto"):
                    Loader("Computing win% gap…", 0.25)
                    df=_cached_scoreboard(league, date_str)
                    df2=df.dropna(subset=["trend_gap"]).copy()
                    if df2.empty: st.info("No record data available."); return
                    df2=df2.sort_values("trend_gap", ascending=False)
                    top=df2[["scoreboard_date","home","home_win_pct","away","away_win_pct","trend_gap","status_detail"]].head(10)
                    for c in ["home_win_pct","away_win_pct","trend_gap"]:
                        top[c]=(top[c]*100).round(1).astype(str)+"%"
                    st.dataframe(top, use_container_width=True)
                    StatCard("Strongest Home Edge", f"{top.iloc[0]['home']} vs {top.iloc[0]['away']}")
            except SportsError as e: st.error(str(e))
        PrimaryButton("Trend Compare (Win% Gap)", key="s_trend_gap", run=_do_trend)
