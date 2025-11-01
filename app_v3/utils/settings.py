from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # Networking
    http_timeout: int = Field(15, env="HTTP_TIMEOUT")
    http_retries: int = Field(3,  env="HTTP_RETRIES")
    http_backoff_min: float = Field(0.4, env="HTTP_BACKOFF_MIN")
    http_backoff_max: float = Field(1.2, env="HTTP_BACKOFF_MAX")
    # Rate limiting (token bucket per host)
    rate_capacity: int = Field(8,  env="RATE_CAPACITY")
    rate_refill_per_sec: float = Field(3.0, env="RATE_REFILL_PER_SEC")

    # Vault paths
    vault_path: str = Field("app_v3/vault.duckdb", env="VAULT_PATH")
    logs_dir:  str = Field("app_v3/logs", env="LOGS_DIR")

    # ESPN/CoinGecko
    espn_hosts: str = Field("https://site.api.espn.com/apis/v2/sports,https://site.web.api.espn.com/apis/v2/sports",
                            env="ESPN_HOSTS")
    coingecko_base: str = Field("https://api.coingecko.com/api/v3", env="COINGECKO_BASE")

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
