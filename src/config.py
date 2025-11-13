"""Application configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)  # Fallback to defaults if file is missing


@dataclass
class Settings:
    """Central configuration object populated from environment variables."""

    tz: str = os.getenv("TZ", "Asia/Almaty")
    freq: str = os.getenv("FREQ", "H")
    forecast_horizon: int = int(os.getenv("FORECAST_HORIZON", "24"))
    random_state: int = 42
    use_postgres: bool = os.getenv("USE_POSTGRES", "true").lower() == "true"
    postgres_host: str = os.getenv("POSTGRES_HOST", "postgres")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "aq")
    postgres_user: str = os.getenv("POSTGRES_USER", "aq_user")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "aq_pass")
    data_path: Path = Path(os.getenv("DATA_PATH", "data/raw/almaty_pm25.csv"))
    db_url: str | None = os.getenv("DB_URL")

    def __post_init__(self) -> None:
        if not self.db_url:
            if self.use_postgres:
                self.db_url = (
                    f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
                    f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
                )
            else:
                sqlite_path = BASE_DIR / "data" / "app.db"
                sqlite_path.parent.mkdir(parents=True, exist_ok=True)
                self.db_url = f"sqlite:///{sqlite_path}"
        # Normalise data path to project root
        if not self.data_path.is_absolute():
            self.data_path = (BASE_DIR / self.data_path).resolve()
        self.data_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
