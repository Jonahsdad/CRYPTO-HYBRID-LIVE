import streamlit as st
import pandas as pd
from ui.widgets import PrimaryButton, Loader, StatCard, Badge
from data_sources.sports_feed import get_scoreboard, SportsError
from utils.audit import audit_event

LEAGUES = ["NBA", "NFL", "MLB", "NHL"]

@st.cache_data(ttl=60, show_spinner=False)
def _cached_scoreboard(league: str) -> pd.DataFrame:
    return get_scoreboard(league)

def view(theme):
    st.subheader("Sports Arena")
    Badge("Live Data: ESPN public scoreboard API", "success")

    league = st.selectbox("League", LEAGUES, index=0, key="sp_league")

    colA, colB, colC = st.columns(3)

    # --- Fetch Games / Scores ---
    with colA:
        def _do_fetch():
            try:
                Loader(f"Fetching {league} scoreboard…", 0.5)
                df = _cached_scoreboard(league)
                if df.empty:
                    st.info("No events returned for today (check league season calendar).")
                    return
                # Display main table
                show = df[["start_time","status_detail","away","away_score","home","home_score","odds","venue"]]
                st.dataframe(show, use_container_width=True)
                StatCard("Games Found", str(len(df)))
                audit_event("sports.fetch", {"league": league, "rows": len(df)})
            except SportsError as e:
                st.error(str(e))
                audit_event("sports.fetch.error", {"league": league, "error": str(e)})

        PrimaryButton("Fetch Games / Scores (Live)", key="s_fetch_live", run=_do_fetch)

    # --- Live Refresh (quick) ---
    with colB:
        def _do_refresh():
            try:
                Loader("Refreshing live scores…", 0.4)
                df = _cached_scoreboard(league)
                if df.empty:
                    st.info("No events to refresh.")
                    return
                live = df[df["status"].isin(["IN","POST","POSTPONED","FINAL"])]
                if live.empty:
                    st.caption("No live/final games right now.")
                st.dataframe(live[["status_detail","away","away_score","home","home_score","odds"]], use_container_width=True)
                StatCard("Live/Final Games", str(len(live)))
                audit_event("sports.refresh", {"league": league, "rows": len(live)})
            except SportsError as e:
                st.error(str(e))
                audit_event("sports.refresh.error", {"league": league, "error": str(e)})

        PrimaryButton("Refresh Live Scores", key="s_refresh_live", run=_do_refresh)

    # --- Trend Compare (simple, transparent) ---
    with colC:
        def _do_trend():
            try:
                Loader("Computing simple trend gap (home win% - away win%)…", 0.6)
                df = _cached_scoreboard(league)
                df2 = df.dropna(subset=["trend_gap"]).copy()
                if df2.empty:
                    st.info("No record data available to compute trends.")
                    return
                df2 = df2.sort_values("trend_gap", ascending=False)
                top = df2[["start_time","home","home_win_pct","away","away_win_pct","trend_gap","odds","status_detail"]].head(10)
                # Format win% nicely
                for c in ["home_win_pct","away_win_pct","trend_gap"]:
                    top[c] = (top[c]*100).round(1).astype(str) + "%"
                st.dataframe(top, use_container_width=True)
                StatCard("Strongest Home Edge (by win%)", f"{top.iloc[0]['home']} vs {top.iloc[0]['away']}")
                audit_event("sports.trend_compare", {"league": league, "analyzed": len(df2)})
            except SportsError as e:
                st.error(str(e))
                audit_event("sports.trend_compare.error", {"league": league, "error": str(e)})

        PrimaryButton("Trend Compare (Win% Gap)", key="s_trend_gap", run=_do_trend)
