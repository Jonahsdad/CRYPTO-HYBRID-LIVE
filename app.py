w_edit["w_liq"]=st.slider("Weight: Liquidity",      0.0,1.0,float(w_edit["w_liq"]),0.01)
    w_edit=_normalize_weights(w_edit)

    st.markdown("---")
    st.subheader("Performance")
    prof["perf_mode"]=st.toggle("Performance Mode (skip Social/News/TVL)", value=prof["perf_mode"])

    st.markdown("---")
    st.subheader("Quick Filters")
    q_top_gainers=st.checkbox("Show only top 24h gainers", False)
    q_top_losers =st.checkbox("Show only top 24h losers",  False)
    q_large_cap  =st.checkbox("Large caps (Top 50 MC)",    False)
    q_small_cap  =st.checkbox("Smaller caps (Rank > 200)", False)

    st.markdown("---")
    st.subheader("Alerts")
    alert_truth=st.slider("Trigger: Fusion Truth ≥",0.0,1.0,0.85,0.01)
    alert_div  =st.slider("Trigger: |Raw - Truth| ≥",0.0,1.0,0.30,0.01)

# ====================== DATA PULL ======================
df=fetch_markets(vs_currency)
if df.empty:
    st.error("Could not load CoinGecko data. Try again in a minute."); st.stop()

# shrink to Top N and ensure cols
df=df.sort_values("market_cap",ascending=False).head(topn).copy()
for k in ["id","current_price","market_cap","total_volume",
          "price_change_percentage_1h_in_currency",
          "price_change_percentage_24h_in_currency",
          "price_change_percentage_7d_in_currency",
          "name","symbol","market_cap_rank"]:
    if k not in df.columns: df[k]=np.nan

# engineered features
df["vol_to_mc"]=(df["total_volume"]/df["market_cap"]).replace([np.inf,-np.inf],np.nan).clip(0,2).fillna(0)
df["momo_1h01"]=df["price_change_percentage_1h_in_currency"].apply(pct_sigmoid)
df["momo_24h01"]=df["price_change_percentage_24h_in_currency"].apply(pct_sigmoid)
df["momo_7d01"]=df["price_change_percentage_7d_in_currency"].apply(pct_sigmoid)
mc=df["market_cap"].fillna(0)
df["liquidity01"]=0 if mc.max()==0 else (mc-mc.min())/(mc.max()-mc.min()+1e-9)

TRUTH_W=dict(w_edit)
df["raw_heat"]=(0.5*(df["vol_to_mc"]/2).clip(0,1)+0.5*df["momo_1h01"].fillna(0.5)).clip(0,1)
df["truth_full"]=lipe_truth(df, TRUTH_W)
df["divergence"]=(df["raw_heat"]-df["truth_full"]).round(3)
df["entropy01"]=df.apply(lambda r: entropy01_from_changes(
    r.get("price_change_percentage_1h_in_currency"),
    r.get("price_change_percentage_24h_in_currency"),
    r.get("price_change_percentage_7d_in_currency")), axis=1)
df["mood"]=df["truth_full"].apply(mood_label)
df["COIN_SYM"]=df["symbol"].str.upper()

# optional external sources (kept light for speed)
perf_mode=ensure_profile()["perf_mode"]
def fetch_reddit_titles(limit=80):
    url=f"https://www.reddit.com/r/CryptoCurrency/hot.json?limit={limit}"
    r=safe_get(url, timeout=20)
    if not r: return []
    try: posts=r.json().get("data",{}).get("children",[]); return [p["data"].get("title","") for p in posts]
    except Exception: return []

@st.cache_data(ttl=120, show_spinner=False)
def social_scores(symbols):
    if perf_mode: return pd.DataFrame(index=pd.Index(symbols, name="symbol"))
    titles=fetch_reddit_titles()
    if not titles: return pd.DataFrame(index=pd.Index(symbols, name="symbol"))
    rows=[]
    for sym in symbols:
        pat=re.compile(rf"\b{re.escape(sym)}\b")
        hits=[t for t in titles if pat.search(t.upper())]
        buzz=len(hits); pol=float(np.mean([_polarity_safe(t) for t in hits])) if buzz>0 else 0.0
        rows.append({"symbol":sym,"buzz":buzz,"sentiment":pol})
    out=pd.DataFrame(rows).set_index("symbol")
    out["buzz01"]=(np.log1p(out["buzz"])/(np.log1p(out["buzz"]).max() or 1)).fillna(0.0)
    out["sent01"]=((out["sentiment"].clip(-1,1)+1)/2.0).fillna(0.5)
    out["social01"]=(0.7*out["buzz01"]+0.3*out["sent01"]).clip(0,1)
    return out[["social01"]]

@st.cache_data(ttl=600, show_spinner=False)
def fetch_defi_protocols():
    if perf_mode: return pd.DataFrame([])
    r=safe_get("https://api.llama.fi/protocols", timeout=30)
    if not r: return pd.DataFrame([])
    try: return pd.DataFrame(r.json())
    except Exception: return pd.DataFrame([])

def map_tvl_to_symbols(coins, llama):
    if llama.empty: return pd.DataFrame({"tvl_score01":[]}).set_index(pd.Index([], name="symbol"))
    llama=llama[["symbol","tvl"]].fillna({"symbol":""}); llama["symbol"]=llama["symbol"].str.upper()
    agg=llama.groupby("symbol", as_index=True)["tvl"].sum()
    raw=agg.reindex(coins).fillna(0.0)
    score=(np.log1p(raw)/np.log1p(raw.max())) if raw.max()>0 else raw
    score=score.clip(0,1); score.name="tvl_score01"; return score.to_frame()

# join external
df=df.set_index("COIN_SYM")
df=df.join(social_scores(df.index.tolist()), how="left")
tvl=map_tvl_to_symbols(df.index.tolist(), fetch_defi_protocols())
df=df.join(tvl, how="left")
for c in ["social01","tvl_score01"]:
    if c not in df.columns: df[c]=0.0
CP_KEY=st.secrets.get("CRYPTOPANIC_KEY") if hasattr(st,"secrets") else None
news01=np.zeros(len(df))
df["news01"]=news01

# Fusion
FW=_normalize_weights(FUSION_WEIGHTS)
df["fusion_truth"]=( FW["w_truth"]*df["truth_full"].fillna(0.0)
                   + FW["w_social"]*df["social01"]
                   + FW["w_news"]*df["news01"]
                   + FW["w_tvl"]*df["tvl_score01"] ).clip(0,1)
df["mood_fusion"]=df["fusion_truth"].apply(mood_label)
df=df.reset_index().rename(columns={"COIN_SYM":"symbol_up"})

# quick filters & search
if search:
    mask=df["name"].str.lower().str.contains(search) | df["symbol"].str.lower().str.contains(search)
    df=df[mask].copy()
if q_large_cap:  df=df.sort_values("market_cap_rank").head(50)
if q_small_cap:  df=df[df["market_cap_rank"]>200]
if q_top_gainers: df=df.sort_values("price_change_percentage_24h_in_currency", ascending=False).head(50)
if q_top_losers:  df=df.sort_values("price_change_percentage_24h_in_currency", ascending=True).head(50)

# KPIs + reload
now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
c1,c2,c3,c4=st.columns(4)
c1.metric("Coins",len(df))
c2.metric("Avg 24h Δ", f"{df['price_change_percentage_24h_in_currency'].mean():+.2f}%")
c3.metric("Avg LIPE Truth", f"{df['truth_full'].mean():.2f}")
c4.metric("Avg Fusion Truth", f"{df['fusion_truth'].mean():.2f}")
st.caption(f"Last update: {now}")

reload = st.button("🔄 Reload data (bypass cache)")
if reload:
    fetch_markets.clear(); df=fetch_markets(vs_currency); st.experimental_rerun()

if df.empty:
    st.warning("⚠️ No coins match your current filters/search. Clear filters or Reload.")
    st.stop()

# ====================== SHAREABLE LINK (weights + picks) ======================
def encode_state_to_query(picks, weights, preset_name):
    qp = {
        "p": ",".join(picks[:10]),              # up to 10 tickers
        "w": json.dumps(weights),               # weights dict
        "pr": preset_name,
        "n": str(topn),
        "c": vs_currency
    }
    return qp

def load_state_from_query():
    try:
        qp = st.query_params
        picks = qp.get("p", "").split(",") if "p" in qp else []
        weights = json.loads(qp.get("w","{}")) if "w" in qp else None
        pr = qp.get("pr", None)
        return picks, weights, pr
    except Exception:
        return [], None, None

loaded_picks, loaded_weights, loaded_pr = load_state_from_query()
if loaded_weights:
    try:
        w_edit = _normalize_weights({k: float(v) for k,v in loaded_weights.items()})
        st.info("Loaded weights from shared link.")
    except Exception:
        pass
if loaded_pr and loaded_pr in PRESETS:
    preset = loaded_pr

# ====================== TABS ======================
SIMPLE_COLS_FUSION=["name","symbol","current_price","fusion_truth","mood_fusion"]
PRO_COLS_FUSION   =["name","symbol","current_price","market_cap","fusion_truth","truth_full","social01","tvl_score01","divergence","mood_fusion"]
SIMPLE_COLS_RAW   =["name","symbol","current_price","raw_heat","total_volume"]
PRO_COLS_RAW      =["name","symbol","current_price","market_cap","total_volume","raw_heat"]
SIMPLE_COLS_TRUTH =["name","symbol","current_price","truth_full","mood"]
PRO_COLS_TRUTH    =["name","symbol","current_price","market_cap","liquidity01","truth_full","divergence","entropy01","mood"]

mode_simple = st.toggle("🧸 Easy Mode (simple columns)", value=True)
HEIGHT=600
cols_fusion=SIMPLE_COLS_FUSION if mode_simple else PRO_COLS_FUSION
cols_raw=SIMPLE_COLS_RAW if mode_simple else PRO_COLS_RAW
cols_truth=SIMPLE_COLS_TRUTH if mode_simple else PRO_COLS_TRUTH

tab_top, tab_fusion, tab_raw, tab_truth, tab_port, tab_mini, tab_watch, tab_help = st.tabs(
    ["🏆 Top Picks","🧭 Fusion Truth","🔥 Raw","🧭 LIPE Truth","📊 Portfolio Backtest","📈 Mini Charts","⭐ Watchlist","❓ Explainer"]
)

with tab_top:
    st.subheader("🏆 Top 3 by Fusion Truth")
    top3=df.sort_values("fusion_truth",ascending=False).head(3)
    cA,cB,cC=st.columns(3)
    for c, r in zip([cA,cB,cC], top3.to_dict("records")):
        c.metric(f"{r['name']} ({r['symbol'].upper()})", f"{r['fusion_truth']:.2f}", delta=f"{r['price_change_percentage_24h_in_currency']:+.2f}%")
    st.caption("These are not financial advice — just the current Fusion leaders.")

with tab_fusion:
    st.subheader("🧭 Fusion Truth (LIPE + Social + TVL)")
    st.dataframe(df.sort_values("fusion_truth",ascending=False)[[c for c in cols_fusion if c in df.columns]].reset_index(drop=True),
                 use_container_width=True, height=HEIGHT)
    # share current view
    picks = [x.upper() for x in loaded_picks] if loaded_picks else []
    q=encode_state_to_query(picks, w_edit, preset)
    if st.button("🔗 Share this view (copy URL from address bar)"):
        st.query_params.clear()
        st.query_params.update(q)
        st.success("URL updated — copy it and share.")

with tab_raw:
    st.subheader("🔥 Raw Wide Scan")
    raw_col="raw_heat" if "raw_heat" in df.columns else next((c for c in df.columns if c.startswith("raw_heat")), "raw_heat")
    st.dataframe(df.sort_values(raw_col,ascending=False)[[c for c in cols_raw if c in df.columns]].reset_index(drop=True),
                 use_container_width=True, height=HEIGHT)

with tab_truth:
    st.subheader("🧭 LIPE Truth")
    st.markdown("<span class='pill'>Volume/MC</span><span class='pill'>Momentum 24h</span><span class='pill'>Momentum 7d</span><span class='pill'>Liquidity</span>", unsafe_allow_html=True)
    st.dataframe(df.sort_values("truth_full",ascending=False)[[c for c in cols_truth if c in df.columns]].reset_index(drop=True),
                 use_container_width=True, height=HEIGHT)

# ====================== PORTFOLIO BACKTEST ======================
with tab_port:
    st.subheader("📊 90-Day Backtest (Equal-Weight Portfolio vs BTC / ETH)")
    # pick list (limit to top 30 by market cap for speed)
    small_df=df.sort_values("market_cap",ascending=False).head(30)
    sym_to_id={row["symbol"].upper(): row["id"] for _,row in small_df.iterrows()}
    default_picks = loaded_picks if loaded_picks else small_df["symbol"].head(5).str.upper().tolist()
    picks = st.multiselect("Choose 3–10 coins", options=list(sym_to_id.keys()), default=default_picks, max_selections=10)
    if len(picks)<3:
        st.info("Pick at least 3 coins to run the backtest.")
    else:
        # fetch price series
        series={}
        for sym in picks + ["BTC","ETH"]:
            cid = sym_to_id.get(sym, "bitcoin" if sym=="BTC" else "ethereum" if sym=="ETH" else None)
            if not cid: continue
            hist = get_market_chart(cid, vs_currency, HIST_DAYS)
            if hist.empty: continue
            hist = hist.set_index("ts")["price"].rename(sym)
            series[sym]=hist
        if len(series)<3:
            st.warning("Not enough price history returned. Try fewer picks or wait a minute.")
        else:
            prices=pd.concat(series.values(), axis=1).dropna()
            # normalize to 1.0 start
            norm=prices/prices.iloc[0]
            # equal-weight portfolio
            cols=[c for c in norm.columns if c not in ["BTC","ETH"]]
            if not cols: cols=norm.columns.tolist()
            portfolio = norm[cols].mean(axis=1).rename("Portfolio")
            comp=pd.concat([portfolio, norm.get("BTC",pd.Series(index=portfolio.index)), norm.get("ETH",pd.Series(index=portfolio.index))], axis=1)
            comp.columns=["Portfolio","BTC","ETH"]
            if PLOTLY_OK:
                fig=px.line(comp, x=comp.index, y=comp.columns, title="Normalized Growth (start = 1.0)")
                fig.update_layout(height=420, legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)
            last=comp.iloc[-1]
            st.write(f"**90-day returns:**  Portfolio: {last['Portfolio']-1:+.1%} | BTC: {last['BTC']-1:+.1%} | ETH: {last['ETH']-1:+.1%}")

            # set query params for sharing with picks
            q=encode_state_to_query(picks, w_edit, preset)
            if st.button("🔗 Share this portfolio (copy URL from address bar)"):
                st.query_params.clear(); st.query_params.update(q); st.success("URL updated — copy it and share.")

# ====================== MINI CHARTS ======================
with tab_mini:
    st.subheader("📈 Mini 90-Day Charts (Top 8 by Fusion)")
    top8 = df.sort_values("fusion_truth", ascending=False).head(8)[["id","name","symbol"]]
    cols = st.columns(2)
    for i, row in enumerate(top8.to_dict("records")):
        hist = get_market_chart(row["id"], vs_currency, HIST_DAYS)
        if hist.empty: 
            cols[i%2].info(f"{row['symbol'].upper()}: no data")
            continue
        if PLOTLY_OK:
            fig = px.line(hist, x="ts", y="price", title=f"{row['name']} ({row['symbol'].upper()})")
            fig.update_layout(height=220, margin=dict(l=10,r=10,t=30,b=10))
            cols[i%2].plotly_chart(fig, use_container_width=True)
        else:
            cols[i%2].line_chart(hist.set_index("ts")["price"], height=220)

# ====================== WATCHLIST ======================
with tab_watch:
    st.subheader("⭐ Watchlist")
    choices=["(add…)"]+df["symbol"].tolist()
    pick=st.selectbox("Add a coin to your watchlist", choices, index=0)
    if pick!="(add…)":
        wl=set(ensure_profile()["watchlist"]); wl.add(pick.upper())
        st.session_state["profile"]["watchlist"]=sorted(list(wl))
        st.success(f"Added **{pick}** to watchlist.")
    wl=ensure_profile()["watchlist"]
    if wl:
        st.dataframe(df[df["symbol"].str.upper().isin(wl)][["name","symbol","current_price","fusion_truth","truth_full","mood_fusion"]],
                     use_container_width=True, height=420)
    else:
        st.info("No coins yet. Add some above!")

# ====================== EXPLAINER ======================
with tab_help:
    st.subheader("❓ Truth & Fusion — simple explainer")
    st.markdown("""
    **Truth Score** is like judging a car race:
    - **Volume/MC** = size of the crowd vs track size (busy = real interest)  
    - **Momentum 24h & 7d** = how fast cars sped up recently  
    - **Liquidity** = how wide the track is (easy to move)  
    We mix these into **0..1**. Green = strong, Red = weak.  
    **Fusion Truth** adds **Social buzz**, **News**, and **TVL** on top of Truth.
    **Backtest** = If you had split $100 across your picks 90 days ago, how did it grow vs BTC/ETH?
    """)

# ====================== ALERTS ======================
if FEATURES["ALERTS"]:
    hits = df[(df["fusion_truth"]>=alert_truth) | (df["divergence"].abs()>=alert_div)]
    if len(hits):
        st.warning(f"🚨 {len(hits)} coins matched your rules")
        st.dataframe(hits.sort_values("fusion_truth",ascending=False)
            [["name","symbol","fusion_truth","truth_full","divergence","mood_fusion"]],
            use_container_width=True, height=420)

# ====================== SNAPSHOT ======================
def mk_snapshot():
    ts=datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_UTC")
    buf=StringIO()
    buf.write(f"Snapshot,{ts}\n\nTop by Fusion Truth\n")
    buf.write(df.sort_values('fusion_truth',ascending=False).head(25)
        [["name","symbol","current_price","market_cap","fusion_truth","truth_full","social01","tvl_score01","divergence","mood_fusion"]].to_csv(index=False))
    buf.write("\nTop by Truth\n")
    buf.write(df.sort_values('truth_full',ascending=False).head(25)
        [["name","symbol","current_price","market_cap","liquidity01","truth_full","divergence","entropy01","mood"]].to_csv(index=False))
    buf.write("\nTop by Raw\n")
    raw_col="raw_heat" if "raw_heat" in df.columns else next((c for c in df.columns if c.startswith("raw_heat")), "raw_heat")
    buf.write(df.sort_values(raw_col,ascending=False).head(25)
        [["name","symbol","current_price","market_cap","total_volume",raw_col]].to_csv(index=False))
    return f"snapshot_{ts}.csv", buf.getvalue().encode("utf-8")

if FEATURES["SNAPSHOT"]:
    fn,payload=mk_snapshot()
    st.download_button("⬇️ Download Snapshot (Fusion + Truth + Raw)", payload, file_name=fn, mime="text/csv")

# ====================== FOOTER ======================
st.markdown("""<hr style="margin-top:1rem;margin-bottom:0.5rem;">""", unsafe_allow_html=True)
st.caption("APIs OK if tables load • Sources: CoinGecko + Reddit + DeFiLlama (+CryptoPanic if keyed) • CHL Phase 5 ©")
