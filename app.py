add_to_watch = st.selectbox("Add/remove coin by symbol", ["(choose)"] + sorted(df["symbol"].str.upper().unique().tolist()))
    if add_to_watch != "(choose)":
        toggle_watch(add_to_watch)
        st.success(f"Toggled {add_to_watch} in your watchlist.")
    wl = get_profile().get("watchlist", [])
    view_cols_simple = ["name","symbol","current_price","fusion_truth","truth_full","mood_fusion"]
    view_cols_pro    = ["name","symbol","current_price","market_cap","liquidity01","truth_full","fusion_truth","divergence","social01","news01","tvl_score01","entropy01","bias_24h","mood_fusion"]
    if wl:
        wldf = df[df["symbol"].str.upper().isin(wl)].sort_values("fusion_truth", ascending=False)
        st.dataframe(wldf[view_cols_simple if simple_mode else view_cols_pro], use_container_width=True)
    else:
        st.info("Your watchlist is empty. Add symbols above.")

with right:
    st.subheader("🎯 Focus coin")
    focus = st.selectbox("Pick a coin to inspect", ["(none)"] + df["name"].head(50).tolist())
    if focus != "(none)" and PLOTLY_OK:
        row = df[df["name"] == focus].head(1).to_dict("records")[0]
        st.success(
            f"**{focus}** → Fusion **{row['fusion_truth']:.2f}** ({row['mood_fusion']}) • "
            f"Truth {row['truth_full']:.2f} • 24h {row['price_change_percentage_24h_in_currency']:+.2f}% • "
            f"7d {row['price_change_percentage_7d_in_currency']:+.2f}% • Liquidity {row['liquidity01']:.2f}"
        )
        st.caption("Why: " + lipe_explain_truth_row(row))
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(row["fusion_truth"]),
            number={'valueformat': '.2f'},
            gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#23d18b"}}
        ))
        fig.update_layout(height=230, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ====================== Snapshot (CSV) ======================
def make_snapshot_csv(df_truth, df_fusion, df_raw):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_UTC")
    buf = StringIO()
    buf.write(f"Snapshot,{ts}\n\nTop by Fusion Truth\n")
    df_fusion.to_csv(buf, index=False)
    buf.write("\nTop by Truth (LIPE)\n")
    df_truth.to_csv(buf, index=False)
    buf.write("\nTop by Raw Heat\n")
    df_raw.to_csv(buf, index=False)
    return f"snapshot_{ts}.csv", buf.getvalue().encode("utf-8")

cols_simple_fusion = ["name","symbol","current_price","fusion_truth","mood_fusion"]
cols_pro_fusion    = ["name","symbol","current_price","market_cap","fusion_truth","truth_full","divergence","social01","news01","tvl_score01","entropy01","bias_24h","mood_fusion"]
truth_cols         = ["name","symbol","current_price","market_cap","liquidity01","truth_full","divergence","entropy01","bias_24h","mood_fusion"]
raw_cols           = ["name","symbol","current_price","market_cap","total_volume","raw_heat"]

top_fusion = df.sort_values("fusion_truth", ascending=False).head(25)[cols_pro_fusion if not simple_mode else cols_simple_fusion]
top_truth  = df.sort_values("truth_full",  ascending=False).head(25)[truth_cols]
top_raw    = df.sort_values("raw_heat",    ascending=False).head(25)[raw_cols]
fname, payload = make_snapshot_csv(top_truth, top_fusion, top_raw)
st.download_button("⬇️ Download Snapshot (Fusion + Truth + Raw)", payload, file_name=fname, mime="text/csv")

# ====================== TABS ======================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🧭 Fusion Truth", "🔥 Raw", "🧭 LIPE Truth", "📉 Movers", "📈 Mini Charts", "🗞️ News", "💬 Social", "🏦 TVL / DeFi"
])

with tab1:
    st.subheader("🧭 Fusion Truth (News + Social + TVL + LIPE)")
    st.dataframe(df.sort_values("fusion_truth", ascending=False)[top_fusion.columns].reset_index(drop=True), use_container_width=True)

with tab2:
    st.subheader("🔥 Raw Wide Scan")
    st.dataframe(df.sort_values("raw_heat", ascending=False)[raw_cols].reset_index(drop=True), use_container_width=True)

with tab3:
    st.subheader("🧭 LIPE Truth (weights applied)")
    st.dataframe(df.sort_values("truth_full", ascending=False)[truth_cols].reset_index(drop=True), use_container_width=True)

with tab4:
    st.subheader("📉 Top Daily Gainers / Losers (24h)")
    g = df.sort_values("price_change_percentage_24h_in_currency", ascending=False).head(12).copy()
    l = df.sort_values("price_change_percentage_24h_in_currency", ascending=True).head(12).copy()
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Top Gainers**")
        st.dataframe(g[["name","symbol","current_price","price_change_percentage_24h_in_currency"]].reset_index(drop=True), use_container_width=True)
    with c2:
        st.write("**Top Losers**")
        st.dataframe(l[["name","symbol","current_price","price_change_percentage_24h_in_currency"]].reset_index(drop=True), use_container_width=True)

with tab5:
    st.subheader("📈 7-Day Mini Charts (Top 10 by Fusion Truth)")
    if not PLOTLY_OK or "sparkline_in_7d" not in df.columns:
        st.info("Mini charts need Plotly and sparkline data.")
    else:
        top10 = df.sort_values("fusion_truth", ascending=False).head(10)
        for _, r in top10.iterrows():
            prices = (r.get("sparkline_in_7d") or {}).get("price", [])
            if not prices: continue
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=prices, mode="lines", name=str(r["symbol"]).upper()))
            fig.update_layout(
                title=f"{r['name']} ({str(r['symbol']).upper()}) • Fusion {r['fusion_truth']:.2f}",
                height=220, margin=dict(l=10,r=10,t=30,b=10), showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("🗞️ Latest Headlines (CryptoPanic)")
    if CP_KEY:
        titles = fetch_cryptopanic_headlines()
        if titles:
            st.write(pd.DataFrame({"headline": titles}))
        else:
            st.info("No hot headlines returned right now.")
    else:
        st.info("Add CRYPTOPANIC_KEY in Streamlit secrets to enable headline scoring & display.")

with tab7:
    st.subheader("💬 Social Buzz (Reddit Hot)")
    titles = fetch_reddit_hot()
    if titles:
        st.write(pd.DataFrame({"reddit_title": titles}))
        st.caption("Buzz & sentiment are already fused into the Fusion Truth score.")
    else:
        st.info("Reddit titles not available right now.")

with tab8:
    st.subheader("🏦 DeFi Llama — Protocols / TVL")
    if defi.empty:
        st.info("TVL data not available right now.")
    else:
        show = defi[["name","symbol","tvl","category","chains"]].sort_values("tvl", ascending=False).head(50)
        st.dataframe(show, use_container_width=True)
        st.caption("We map protocol symbols to coin symbols best-effort to create tvl_score01 → Fusion Truth.")

# ====================== Alerts ======================
if FEATURES["ALERTS"]:
    matches = df[(df["fusion_truth"] >= alert_truth) | (df["divergence"].abs() >= alert_diverg)]
    if len(matches):
        st.warning(f"🚨 {len(matches)} coins matched your alert rules (Fusion / Divergence)")
        st.dataframe(matches.sort_values("fusion_truth", ascending=False)[
            ["name","symbol","fusion_truth","truth_full","divergence","mood_fusion"]
        ], use_container_width=True)

# ====================== FOOTER ======================
st.markdown("""<hr style="margin-top: 1rem; margin-bottom: 0.5rem;">""", unsafe_allow_html=True)
api_status = "🟢 APIs OK" if not df.empty else "🔴 API issue"
st.caption(f"{api_status} • {BRAND_FOOTER}")
