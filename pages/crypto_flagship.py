from lib.api import forecast

# ...
if st.button("Run Forecast"):
    try:
        res = forecast("crypto", symbol, horizon)
        pts = res["event"]["forecast"]["points"]
        xs  = [p["ts"] for p in pts]
        yhat= [p["yhat"] for p in pts]
        q10 = [p.get("q10", p["yhat"]) for p in pts]
        q90 = [p.get("q90", p["yhat"]) for p in pts]
        # rebuild fig with xs/yhat/q10/q90...
    except Exception as e:
        st.error(f"Forecast failed: {e}")
