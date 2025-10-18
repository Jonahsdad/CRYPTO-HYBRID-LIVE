elif nav == "Scores":
    st.subheader("🧪 Scores — Truth / Raw / Divergence")

    # Choose which market you want to analyze
    src = st.selectbox("Source Table", ["Crypto", "Stocks", "FX", "Unified"], index=0)

    # Build the base table
    if src == "Crypto":
        base = crypto_df
    elif src == "Stocks":
        base = stocks_df
    elif src == "FX":
        base = fx_df
    else:
        base = unify_frames([crypto_df, stocks_df, fx_df])

    if base is None or base.empty:
        st.warning("No data available for the selected source.")
    else:
        # Big top tiles
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", len(base))
        c2.metric("Avg Truth", f"{pd.to_numeric(base['truth_full'], errors='coerce').mean():.2f}")
        c3.metric("Avg Raw", f"{pd.to_numeric(base['raw_heat'], errors='coerce').mean():.2f}")
        c4.metric("Avg |Δ|", f"{pd.to_numeric(base['divergence'], errors='coerce').abs().mean():.3f}")

        st.markdown("#### 🧭 Truth (ranked)")
        cols_t = [c for c in ["asset_type","symbol","name","current_price",
                              "price_change_percentage_24h_in_currency","truth_full","liquidity01","vol_to_mc"] if c in base.columns]
        st.dataframe(base.sort_values("truth_full", ascending=False)[cols_t],
                     use_container_width=True, height=300)

        st.markdown("#### 🔥 Raw (momentum/volume)")
        cols_r = [c for c in ["asset_type","symbol","current_price",
                              "price_change_percentage_24h_in_currency","raw_heat","vol_to_mc","momo_24h01"] if c in base.columns]
        st.dataframe(base.sort_values("raw_heat", ascending=False)[cols_r],
                     use_container_width=True, height=300)

        st.markdown("#### 🧠 Divergence (Raw − Truth)")
        cols_d = [c for c in ["asset_type","symbol","current_price","divergence",
                              "truth_full","raw_heat","price_change_percentage_24h_in_currency"] if c in base.columns]
        st.dataframe(base.sort_values("divergence", ascending=False)[cols_d],
                     use_container_width=True, height=300)

        # Optional scatter chart
        try:
            import plotly.express as px  # uses existing dependency
            fig = px.scatter(base, x="truth_full", y="raw_heat", text="symbol",
                             color="divergence", color_continuous_scale="Turbo")
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
