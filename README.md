# Features
df["vol_to_mc"] = (df["total_volume"]/df["market_cap"]).replace([np.inf,-np.inf],np.nan).clip(0,2).fillna(0)
df["momo_1h01"]  = df["price_change_percentage_1h_in_currency"].apply(pct_sigmoid)
df["momo_24h01"] = df["price_change_percentage_24h_in_currency"].apply(pct_sigmoid)
df["momo_7d01"]  = df["price_change_percentage_7d_in_currency"].apply(pct_sigmoid)
mc = df["market_cap"].fillna(0)
df["liquidity01"] = 0 if mc.max()==0 else (mc-mc.min())/(mc.max()-mc.min()+1e-9)

# Scores
df["raw_heat"] = (0.5*(df["vol_to_mc"]/2).clip(0,1) + 0.5*df["momo_1h01"].fillna(0.5)).clip(0,1)
df["truth_full"] = (
    0.30*(df["vol_to_mc"]/2).clip(0,1) +
    0.25*df["momo_24h01"].fillna(0.5) +
    0.25*df["momo_7d01"].fillna(0.5) +
    0.20*df["liquidity01"].fillna(0.0)
).clip(0,1)

now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
st.write(f"Last update: **{now}** — Coins: **{len(df)}**")

c1,c2,c3 = st.columns(3)
with c1:
    st.subheader("🔥 Raw Wide Scan")
    cols=["name","symbol","current_price","market_cap","total_volume","raw_heat"]
    st.dataframe(df.sort_values("raw_heat", ascending=False).head(20)[cols], use_container_width=True)
with c2:
    st.subheader("🧭 Truth Filter (Lite)")
    cols=["name","symbol","current_price","market_cap","truth_full"]
    st.dataframe(df.sort_values("truth_full", ascending=False).head(20)[cols], use_container_width=True)
with c3:
    st.subheader("📉 Top Daily Gainers / Losers")
    g=df.sort_values("price_change_percentage_24h_in_currency", ascending=False).head(10)
    l=df.sort_values("price_change_percentage_24h_in_currency", ascending=True).head(10)
    st.markdown("**Top Gainers**")
    st.dataframe(g[["name","symbol","current_price","price_change_percentage_24h_in_currency"]], use_container_width=True)
    st.markdown("**Top Losers**")
    st.dataframe(l[["name","symbol","current_price","price_change_percentage_24h_in_currency"]], use_container_width=True)
