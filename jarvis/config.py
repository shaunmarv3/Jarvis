"""Configuration & paths. Reads from a local .env if present."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
CACHE_DIR = DATA_DIR / "cache"
SUMMARIES_DIR = DATA_DIR / "summaries"
REPORTS_DIR = DATA_DIR / "reports"
VECTOR_DIR = DATA_DIR / "vectorstore"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    default_backend: str = "ollama"

    ollama_model: str = "qwen3.5:9b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 8192  # context window cap
    ollama_num_predict: int = 1024  # default max tokens generated per call
    ollama_reasoning: bool = False  # disable long "thinking" chains on reasoning models
    ollama_timeout: int = 180  # hard per-call timeout (s) so a call can't hang forever

    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"

    max_loops: int = 3
    contact_email: str = "research@example.com"
    request_timeout: int = 30

    # Multi-agent (orchestrator-worker, per Anthropic's research-system design).
    # The lead agent spawns subagents up to a ceiling that depends on the backend:
    # one local GPU runs them sequentially (a patience limit), the cloud runs them
    # in parallel (just API calls), so the cloud ceiling is higher.
    max_subagents_ollama: int = 5  # ceiling on local Ollama (subagents run sequentially)
    max_subagents_deepseek: int = 10  # ceiling on DeepSeek (subagents run in parallel)
    subagent_rounds: int = 2  # search rounds per subagent (broad query -> narrowed follow-up)
    parallel_subagents: bool = True  # run subagents concurrently (only effective on DeepSeek)

    # Embeddings / RAG
    embed_model: str = "nomic-embed-text"
    relevance_min: float = 0.60  # cosine cutoff (calibrated: on-topic >=0.62, off-topic <=0.58)
    keep_top_papers: int = 12  # cap papers kept per research run

    # Web research
    web_max_results: int = 6
    use_scrapling: bool = False  # opt-in; Scrapling requires Playwright (a browser)


settings = Settings()


def subagent_ceiling(backend: str | None = None) -> int:
    """Max subagents the lead may spawn, given the active backend.

    Local Ollama runs subagents one-at-a-time (one GPU), so the ceiling is a
    patience limit; DeepSeek runs them in parallel, so it can afford more.
    """
    b = (backend or settings.default_backend).lower()
    return settings.max_subagents_deepseek if b == "deepseek" else settings.max_subagents_ollama

# Make sure the data directories exist on import.
for _d in (PAPERS_DIR, CACHE_DIR, SUMMARIES_DIR, REPORTS_DIR, VECTOR_DIR):
    _d.mkdir(parents=True, exist_ok=True)
