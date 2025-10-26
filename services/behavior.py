# services/behavior.py
from services.cache import cached
import random

@cached("behavior:sentiment_demo", ttl=600)
def sentiment_demo(keyword: str = "crypto"):
    # placeholder until you attach Twitter/Reddit/Trends
    import math
    xs = list(range(60))
    ys = [50 + 10*math.sin(i/8.0) + random.uniform(-2,2) for i in xs]
    return {"x": xs, "y": ys}
