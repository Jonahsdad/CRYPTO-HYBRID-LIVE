# app_v3/utils/settings.py

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict   # <-- add SettingsConfigDict

class Settings(BaseSettings):
    # --- Networking
    http_timeout: int = Field(default=15, validation_alias="HTTP_TIMEOUT")
    http_backoff_min: float = Field(default=0.4, validation_alias="HTTP_BACKOFF_MIN")
    http_backoff_max: float = Field(default=1.2, validation_alias="HTTP_BACKOFF_MAX")

    # --- Local rate limiter
    rate_capacity: int = Field(default=8, validation_alias="RATE_CAPACITY")
    rate_refill_per_sec: float = Field(default=3.0, validation_alias="RATE_REFILL_PER_SEC")

    # --- Vault / logs
    vault_path: str = Field(default="app_v3/vault.duckdb", validation_alias="VAULT_PATH")
    logs_dir: str = Field(default="app_v3/logs", validation_alias="LOGS_DIR")

    # --- Endpoints
    espn_hosts: str = Field(default="https://site.api.espn.com/apis/v2/sports", validation_alias="ESPN_HOSTS")
    coingecko_base: str = Field(default="https://api.coingecko.org/api/v3", validation_alias="COINGECKO_BASE")

    # Pydantic v2 / pydantic-settings v2 configuration (replaces old `class Config`)
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False
    )

settings = Settings()
