# lipe_core.py
from datetime import datetime
from typing import List, Dict, Any
import numpy as np

def _entropy_score(digits: List[int]) -> float:
    if not digits:
        return 0.0
    vals, counts = np.unique(digits, return_counts=True)
    p = counts / counts.sum()
    h = -np.sum(p * np.log2(p))
    return float(h / np.log2(max(2, len(vals))))  # 0..1

class LIPE:
    def __init__(self, tier:int=33, name:str="LIPE"):
        self.tier = tier
        self.name = name
        self.status = "Active"
        self.boot_time = datetime.now()
        self.logs: List[str] = []
        self.config = {
            "RollingMemory": 60,
            "SplitMiddayEvening": True,
            "BonusWeighting": "Moderate"
        }

    def ping(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "status": self.status,
            "boot_time": self.boot_time.isoformat(timespec="seconds")
        }

    def log(self, msg:str) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        self.logs.append(f"[{ts}] {msg}")

    # >>> MAIN ENTRY the dashboard calls <<<
    def run_forecast(self, game: str, draws: List[int], settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        REQUIRED RETURN:
        {
          "game": "Pick 4",
          "top_picks": [...],
          "alts": [...],
          "confidence": 0.00..1.00,
          "entropy": 0.00..1.00,
          "logic": "NBC|RP|Echo vX",
          "notes": "optional"
        }
        """
        # pull settings
        session         = settings.get("Session", "Midday")
        rolling_memory  = int(settings.get("RollingMemory", 60))
        bonus_weighting = settings.get("BonusWeighting", "Moderate")
        use_nbc         = bool(settings.get("UseNBC", True))
        use_rp          = bool(settings.get("UseRP", True))
        use_echo        = bool(settings.get("UseEcho", True))

        # slice memory window
        window = draws[-min(len(draws), rolling_memory):] if draws else []
        s = sum(window) if window else 0

        # simple entropy proxy from last-digit distribution
        ent = _entropy_score([d % 10 for d in window]) if window else 0.5

        # demo: tiny confidence shaping from toggles / weighting
        conf = 0.55 + (0.05 if use_nbc else 0.0) + (0.04 if use_rp else 0.0) + (0.03 if use_echo else 0.0)
        conf += {"None":0.00, "Light":0.01, "Moderate":0.02, "Heavy":0.03}.get(bonus_weighting, 0.02)
        conf += 0.10 * (ent - 0.5)   # small entropy influence
        conf = max(0.0, min(1.0, conf))

        # --- Replace below with your real LIPE pick logic ---
        if game == "Pick 3":
            base = (s or 123) % 1000
            picks = [(base + d) % 1000 for d in (7, 19, 37)]
            alts  = [(base + 73) % 1000, (base + 91) % 1000]
            top   = [f"{p:03d}" for p in picks]
            alts  = [f"{p:03d}" for p in alts]
        elif game == "Pick 4":
            base = (s or 4397) % 10000
            picks = [(base + d) % 10000 for d in (11, 29, 83)]
            alts  = [(base + 127) % 10000, (base + 241) % 10000]
            top   = [f"{p:04d}" for p in picks]
            alts  = [f"{p:04d}" for p in alts]
        else:  # Lucky Day Lotto (1..45)
            rng = np.random.default_rng((s or 90210) % 2_147_483_647)
            top = sorted(rng.choice(np.arange(1,46), size=5, replace=False).tolist())
            alts = sorted(rng.choice(np.arange(1,46), size=5, replace=False).tolist())

        logic_label = "NBC"*(1 if use_nbc else 0) + ("|" if (use_nbc and (use_rp or use_echo)) else "") + \
                      "RP"*(1 if use_rp else 0) + ("|" if (use_echo and (use_nbc or use_rp)) else "") + \
                      "Echo"*(1 if use_echo else 0)
        if not logic_label: logic_label = "Core"

        return {
            "game": game,
            "top_picks": top,
            "alts": alts,
            "confidence": round(conf, 2),
            "entropy": round(ent, 2),
            "logic": f"{logic_label} v1",
            "notes": f"Session={session}, Mem={rolling_memory}, BonusWeighting={bonus_weighting}"
        }
