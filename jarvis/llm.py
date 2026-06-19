"""LLM backend switch: local Ollama or DeepSeek cloud, same call site."""

from __future__ import annotations

from .config import settings

# Module-level "active backend" so tools that need an LLM (e.g. the PDF reader)
# can honor a runtime /backend switch without threading it through every call.
_ACTIVE_BACKEND: str | None = None


def set_active_backend(backend: str) -> None:
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = (backend or "").lower() or None


def active_backend() -> str:
    return (_ACTIVE_BACKEND or settings.default_backend).lower()


def resolve_backend(requested: str | None = None) -> tuple[str, str | None]:
    """Return (backend, notice). Falls back to ollama if deepseek lacks a key."""
    backend = (requested or active_backend()).lower()
    if backend == "deepseek" and not settings.deepseek_api_key:
        return "ollama", "No DEEPSEEK_API_KEY set — falling back to local Ollama."
    return backend, None


def get_llm(backend: str | None = None, temperature: float = 0.3, **kwargs):
    """Construct a chat model for the chosen backend.

    Both backends expose `.bind_tools(...)` and the OpenAI-style tool-calling
    interface, so the rest of the app is backend-agnostic.
    """
    backend, _ = resolve_backend(backend)

    if backend == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            temperature=temperature,
            **kwargs,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        **kwargs,
    )
