# pages/_registry.py
from dataclasses import dataclass

@dataclass
class Arena:
    key: str       # label
    module: str    # python module path that exposes show()

ARENAS = [
    Arena(key="Crypto",  module="pages.crypto_flagship"),
    # Add more when ready:
    # Arena(key="Sports",  module="pages.sports"),
    # Arena(key="Lottery", module="pages.lottery"),
]
