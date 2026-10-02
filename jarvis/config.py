"""Configuration & paths. Reads from a local .env if present."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
CACHE_DIR = DATA_DIR / "cache"
SUMMARIES_DIR = DATA_DIR / "summaries"
REPORTS_DIR = DATA_DIR / "reports"
RUNS_DIR = DATA_DIR / "runs"
VECTOR_DIR = DATA_DIR / "vectorstore"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    default_backend: str = "ollama"

    ollama_model: str = "qwen3.5:9b"
    ollama_lead_model: str = ""  # optional stronger local model for the lead; "" = ollama_model
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 8192  # context window cap
    ollama_num_predict: int = 1024  # default max tokens generated per call
    ollama_reasoning: bool = False  # disable long "thinking" chains on reasoning models
    ollama_timeout: int = 180  # hard per-call timeout (s) so a call can't hang forever

    # DeepSeek: a strong "lead" model plans + writes the report, a cheap fast "worker"
    # model runs the many subagent tool-calling turns (Anthropic's Opus-lead /
    # Sonnet-subagent split). Thinking mode costs output tokens, so it's on for the
    # lead (few calls, where reasoning pays off) and off for workers by default.
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_lead_model: str = "deepseek-v4-pro"
    deepseek_worker_model: str = "deepseek-flash"
    deepseek_lead_thinking: bool = True
    deepseek_worker_thinking: bool = False
    # Reasoning effort for the lead's thinking: low | high | max. "high" can spend 5k+
    # reasoning tokens per call (minutes on v4-pro); "low" keeps runs fast and cheap.
    deepseek_lead_effort: str = "low"
    # USD per 1M tokens (peak rates — off-peak is half, so estimates err high).
    price_lead_in: float = 1.32
    price_lead_in_cached: float = 0.044
    price_lead_out: float = 3.96
    price_worker_in: float = 0.30
    price_worker_in_cached: float = 0.006
    price_worker_out: float = 1.20

    contact_email: str = "research@example.com"
    request_timeout: int = 30
    semantic_scholar_api_key: str = ""  # optional: a personal 1 req/s lane instead of the shared pool
    github_token: str = ""  # optional: raises GitHub search limits (10 -> 30 req/min)
    tavily_api_key: str = ""  # optional: LLM-grade web search (1,000 free credits/month)
    exa_api_key: str = ""  # optional: semantic web search
    tool_cache_hours: int = 24  # cache search API responses on disk (0 = off)

    # Multi-agent (orchestrator-worker, per Anthropic's research-system design).
    # One local GPU runs subagents sequentially (a patience limit); the cloud runs
    # them in parallel, so its ceiling is higher.
    max_subagents_ollama: int = 4
    max_subagents_deepseek: int = 8
    parallel_subagents: bool = True  # run subagents concurrently (only effective on DeepSeek)
    subagent_max_turns: int = 8  # LLM turns per subagent (each turn may call several tools)
    subagent_budget_ollama: int = 6  # default tool calls per subagent (lead may set 3-15)
    subagent_budget_deepseek: int = 10
    subagent_hard_cap: int = 15  # absolute tool-call ceiling per subagent
    subagent_max_reads: int = 3  # full-text reads (read_paper/read_web) allowed on top of the search budget
    lead_iteration: bool | None = None  # one gap-filling follow-up wave; None = on for deepseek only
    max_followup_subagents: int = 2
    report_max_tokens: int = 4096  # the final report must never be cut off mid-sentence

    # Embeddings / RAG (paper Q&A — needs Ollama for nomic-embed-text)
    embed_model: str = "nomic-embed-text"
    relevance_min: float = 0.60  # cosine cutoff (calibrated: on-topic >=0.62, off-topic <=0.58)
    keep_top_papers: int = 12  # papers listed for /read after a run

    # Web research
    web_max_results: int = 6
    use_scrapling: bool = False  # opt-in; Scrapling requires Playwright (a browser)


settings = Settings()


def subagent_ceiling(backend: str | None = None) -> int:
    """Max subagents the lead may spawn, given the active backend."""
    b = (backend or settings.default_backend).lower()
    return settings.max_subagents_deepseek if b == "deepseek" else settings.max_subagents_ollama


def default_budget(backend: str | None = None) -> int:
    """Default tool-call budget for a subagent when the lead doesn't set one."""
    b = (backend or settings.default_backend).lower()
    return settings.subagent_budget_deepseek if b == "deepseek" else settings.subagent_budget_ollama


def lead_iteration_enabled(backend: str | None = None) -> bool:
    if settings.lead_iteration is not None:
        return settings.lead_iteration
    return (backend or settings.default_backend).lower() == "deepseek"


# Make sure the data directories exist on import.
for _d in (PAPERS_DIR, CACHE_DIR, SUMMARIES_DIR, REPORTS_DIR, RUNS_DIR, VECTOR_DIR):
    _d.mkdir(parents=True, exist_ok=True)
