"""Configuration & paths. Reads from a local .env if present."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
CACHE_DIR = DATA_DIR / "cache"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    default_backend: str = "ollama"

    ollama_model: str = "qwen3.5:9b"
    ollama_base_url: str = "http://localhost:11434"

    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"

    max_loops: int = 4
    contact_email: str = "research@example.com"
    request_timeout: int = 30


settings = Settings()

# Make sure the data directories exist on import.
for _d in (PAPERS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
