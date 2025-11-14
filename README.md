![CI](https://github.com/YOURUSERNAME/CRYPTO-HYBRID-LIVE/actions/workflows/ci.yml/badge.svg)
# 🟢 Crypto Hybrid Live — Core

### Overview
This Streamlit dashboard analyzes the top 250 cryptocurrencies directly from the CoinGecko API.  
It compares **raw market heat** (volume vs. market cap) with a **Truth Filter** that blends:
- 1-hour, 24-hour, and 7-day momentum
- Volume-to-market-cap ratios
- Liquidity normalization

### Live Features
- 🧭 Truth Filter (weighted composite score)
- 🔥 Raw Wide Scan of top 20 coins
- 📉 Daily top gainers and losers
- ⏱️ Auto-refresh every 60 seconds

### How It Works
1. Pulls data from the CoinGecko `/coins/markets` API.  
2. Calculates momentum and liquidity metrics.  
3. Normalizes and scores the data between 0–1.  
4. Displays it in three panels for comparison.

### Future Add-Ons
- YouTube / Twitter sentiment tracking  
- Developer activity pulse  
- Social velocity scoring  
- On-chain signal fusion  

---

**Created by:** Jonahsdad  
**Framework:** [Streamlit.io](https://streamlit.io)  
**API Source:** [CoinGecko](https://www.coingecko.com/en/api)
# REPO: his-streamlit-flagship
# Push this whole tree to GitHub and deploy to Streamlit Cloud.

# -----------------------------------------
# FILE: README.md
# -----------------------------------------
# HIS — Streamlit Flagship (Crypto v1)
...
# HIS Flagship — NOW

Two services:
- **Backend (FastAPI)** on Render: `/v1/forecast`, `/healthz`, simple share links
- **Frontend (Streamlit)** on Streamlit Cloud: Crypto Flagship, Plans, Status

## Deploy

1) **Render (backend)**
- Connect repo → Render reads `render.yaml`
- After deploy, copy your URL, e.g. `https://his-lipe-core.onrender.com`
- Check `https://.../healthz` → `{"ok": true}`

2) **Streamlit (frontend)**
- New app → set file path: `streamlit/streamlit_app.py`
- In Streamlit **Secrets**, add:  
  `API_BASE_URL = "https://his-lipe-core.onrender.com"`
- Open app → Sidebar → **Connect** → go to **Crypto Flagship**

## Use
- Symbol: `BTCUSDT`
- Horizon: 1..30
- Run Forecast → see bands, KPIs
- Create Share Link → get `/v1/share/<token>` URL (public read-only)

> This repo runs without a DB. It uses CCXT for live OHLCV, with a synthetic fallback.
