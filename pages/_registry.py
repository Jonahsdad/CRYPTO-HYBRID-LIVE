from dataclasses import dataclass

@dataclass
class Arena:
    key: str       # label shown in the selector
    module: str    # python module path to import

# Keep it super simple for now: one arena wired
ARENAS = [
    Arena(key="crypto flagship", module="pages.crypto_flagship"),
]
