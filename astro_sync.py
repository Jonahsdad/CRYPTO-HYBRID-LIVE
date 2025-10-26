# ==============================================================
# ASTROLOGY ARENA — Visible Layer
# ==============================================================
from datetime import datetime, timedelta
import math, random

def get_positions(planet:str="Mars", days_back:int=30):
    base = datetime.utcnow()
    data = []
    for i in range(days_back):
        date = base - timedelta(days=i)
        degree = (math.sin(i/5.0) * 180 / math.pi) % 360
        retro = random.choice([True, False])
        data.append({
            "date": date.date(),
            "degree": round(degree,2),
            "retrograde": retro
        })
    return data

def interpret_alignment(deg: float):
    if 0 <= deg < 90: return "New beginnings"
    elif 90 <= deg < 180: return "Growth & tension"
    elif 180 <= deg < 270: return "Reflection & correction"
    else: return "Completion & harvest"
