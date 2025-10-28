# lipe_core/lipe_engine.py
from typing import Dict, Any, List, Tuple
import math, time, hashlib, random

def _seed_from_obj(obj: Any) -> int:
    s = str(obj).encode("utf-8")
    return int(hashlib.md5(s).hexdigest(), 16) % (2**31 - 1)

def _top_n(items: List[Tuple[Any, float]], n: int) -> List[Tuple[Any, float]]:
    return sorted(items, key=lambda x: x[1], reverse=True)[:n]

# ---- LOTTERY ----
def _predict_lottery(body: Dict[str, Any], horizon: int) -> Dict[str, Any]:
    draws = body.get("recent_draws", [])
    if not isinstance(draws, list) or not draws:
        return {"picks": [], "note": "No recent_draws provided"}

    seed = _seed_from_obj(draws)
    rng = random.Random(seed)

    freq = {str(d): 0 for d in range(10)}
    for d in draws:
        for ch in str(d):
            if ch.isdigit():
                freq[ch] += 1
    total = sum(freq.values()) or 1
    weight = {k: (v / total) for k, v in freq.items()}

    def score_combo(combo: str) -> float:
        s = sum(weight.get(ch, 0.0) for ch in combo)
        unique_bonus = len(set(combo)) * 0.01
        balance = -abs((combo.count(combo[0]) - 1)) * 0.005
        return s + unique_bonus + balance

    candidates = {}
    for _ in range(200):
        c = "".join(str(rng.randint(0, 9)) for _ in range(4))
        candidates[c] = score_combo(c)

    top = _top_n(list(candidates.items()), horizon)
    picks = [{"combo": c, "score": round(s, 4)} for c, s in top]
    return {
        "engine": "LIPE-LOTTERY-v0",
        "picks": picks,
        "digit_weight": weight,
        "meta": {"seed": seed, "horizon": horizon, "samples": len(candidates)},
    }

# ---- CRYPTO ----
def _predict_crypto(body: Dict[str, Any], horizon: int) -> Dict[str, Any]:
    market = body.get("market", [])
    if not market:
        syms = body.get("symbols", [])
        ranked = [(s, 0.5) for s in syms][:horizon]
        return {"engine": "LIPE-CRYPTO-v0", "ranks": [{"id": s, "score": sc} for s, sc in ranked]}

    seed = _seed_from_obj(market)
    rng = random.Random(seed)
    pcts = [m.get("pct_24h", 0) or 0 for m in market]
    mn, mx = (min(pcts), max(pcts)) if pcts else (0, 0)

    ranks = []
    for m in market:
        pct = m.get("pct_24h", 0) or 0
        norm = 0.5 if mx == mn else (pct - mn) / (mx - mn)
        stability = 1.0 - abs(rng.random() - 0.5) * 0.2
        score = 0.7 * norm + 0.3 * stability
        ranks.append((m.get("id") or m.get("symbol") or "unknown", score))

    top = _top_n(ranks, horizon)
    return {
        "engine": "LIPE-CRYPTO-v0",
        "ranks": [{"id": i, "score": round(s, 4)} for i, s in top],
        "meta": {"seed": seed, "horizon": horizon},
    }

# ---- SPORTS ----
def _predict_sports(body: Dict[str, Any], horizon: int) -> Dict[str, Any]:
    matchups = body.get("matchups", [])
    picks = []
    for m in matchups[:max(1, horizon)]:
        line = float(m.get("line", 0))
        conf = 0.55 + max(0, min(0.1, (3.5 - abs(line)) * 0.02))
        pick = m.get("home") if line <= 0 else m.get("away")
        picks.append({"pick": pick, "confidence": round(conf, 3), "line": line})
    return {"engine": "LIPE-SPORTS-v0", "picks": picks}

# ---- ENTRY POINT ----
def predict(arena: str, body: Dict[str, Any], model: str = "default", horizon: int = 3) -> Dict[str, Any]:
    arena = (arena or "").lower()
    if arena in ("lottery", "ldl", "pick3", "pick4"):
        return _predict_lottery(body, horizon)
    if arena in ("crypto", "coins", "defi"):
        return _predict_crypto(body, horizon)
    if arena in ("sports", "nfl", "nba", "nhl"):
        return _predict_sports(body, horizon)
    return {"engine": "LIPE-UNKNOWN", "note": f"Unknown arena '{arena}'", "echo": body}
