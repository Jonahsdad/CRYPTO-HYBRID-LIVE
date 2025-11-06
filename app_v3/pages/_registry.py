# app_v3/pages/_registry.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Arena:
    key: str
    title: str
    module: str   # import path to the Overview.show()

ARENAS = [
    Arena("Home",        "Home",                          "pages.01_Home"),  # optional if you have one
    Arena("Crypto",      "Crypto Arena",                  "pages.02_Crypto.Overview"),
    Arena("Sports",      "Sports Arena",                  "pages.03_Sports.Overview"),
    Arena("Lottery",     "Lottery Arena",                 "pages.04_Lottery.Overview"),
    Arena("Stocks",      "Stocks Arena",                  "pages.05_Stocks.Overview"),
    Arena("Options",     "Options Arena",                 "pages.06_Options.Overview"),
    Arena("RealEstate",  "Real Estate Arena",             "pages.07_RealEstate.Overview"),
    Arena("Commodities", "Commodities Arena",             "pages.08_Commodities.Overview"),
    Arena("Forex",       "Forex Arena",                   "pages.09_Forex.Overview"),
    Arena("RWA",         "RWA Arena",                     "pages.10_RWA.Overview"),
]
