# services/commodities.py
from services.http import get
from services.cache import cached

@cached("commod:gold_demo", ttl=900)
def gold_demo_series():
    # demo synthetic curve if no provider is configured
    return {"x": list(range(30)), "y": [100 + (i*0.2) for i in range(30)]}
