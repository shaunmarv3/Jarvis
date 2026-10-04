"""LLM backend switch: local Ollama or DeepSeek cloud, same call site.

Two roles, mirroring Anthropic's research system (strong lead, cheaper subagents):
  * role="lead"   — plans, reviews gaps, writes the report (few calls, needs judgment)
  * role="worker" — subagent tool loops, summaries, Q&A (many calls, needs speed/price)

Every model is constructed with a shared usage callback, so each run can report
exactly how many tokens it burned and what that cost.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

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


def model_label(backend: str) -> str:
    """The model name to show for a backend (the lead model on DeepSeek)."""
    return settings.deepseek_lead_model if backend == "deepseek" else settings.ollama_model


# --------------------------------------------------------------------------- health check


def _ollama_has(names: set[str], model: str) -> bool:
    return model in names or (":" not in model and f"{model}:latest" in names)


def check_ollama(models: list[str]) -> str | None:
    """None if Ollama is up and has every model in `models`, else a message saying how to fix it."""
    import requests

    base = settings.ollama_base_url.rstrip("/")
    try:
        r = requests.get(f"{base}/api/tags", timeout=5)
        r.raise_for_status()
        names = {m.get("name", "") for m in r.json().get("models", [])}
    except Exception:
        return (f"Ollama isn't running at {base}. Start it (open the Ollama app or run `ollama serve`), "
                "or /backend deepseek.")
    missing = [m for m in models if not _ollama_has(names, m)]
    if missing:
        return ("Ollama is running but these models aren't pulled: " + ", ".join(missing)
                + ". Run: " + " && ".join(f"ollama pull {m}" for m in missing))
    return None


def check_embeddings() -> str | None:
    """/read indexing and /ask always embed locally, even on DeepSeek."""
    return check_ollama([settings.embed_model])


def check_backend(backend: str) -> str | None:
    """Fail fast before a run: return a clear error message, or None if the backend is usable.

    Without this, a stopped Ollama makes every step fall back to its default and the
    run "finishes" with an almost empty report.
    """
    import requests

    if backend == "deepseek":
        try:
            r = requests.get(f"{settings.deepseek_base_url.rstrip('/')}/models", timeout=10,
                             headers={"Authorization": f"Bearer {settings.deepseek_api_key}"})
        except requests.RequestException as exc:
            return (f"Can't reach DeepSeek at {settings.deepseek_base_url} ({type(exc).__name__}). "
                    "Check your internet connection.")
        if r.status_code in (401, 403):
            return f"DeepSeek rejected the API key (HTTP {r.status_code}). Check DEEPSEEK_API_KEY in .env."
        if r.status_code >= 500:
            return f"DeepSeek is having problems (HTTP {r.status_code}). Try again shortly, or /backend ollama."
        return None
    return check_ollama([settings.ollama_model] + ([settings.ollama_lead_model] if settings.ollama_lead_model else []))


# --------------------------------------------------------------------------- context budget

CHARS_PER_TOKEN = 3.2  # conservative for English + URLs; real tokenizers average closer to 4


def estimate_tokens(text: str) -> int:
    return int(len(text or "") / CHARS_PER_TOKEN) + 1


def input_char_budget(backend: str | None, max_tokens: int | None = None) -> int | None:
    """How many prompt characters fit, or None when the window is effectively unlimited.

    Ollama's num_ctx covers prompt AND output, and when the prompt is too long Ollama
    trims it silently, which can drop instructions or evidence. So callers shrink their
    own input to this budget instead. DeepSeek's window is far larger than anything
    Jarvis sends.
    """
    if (backend or active_backend()) == "deepseek":
        return None
    out = max_tokens or settings.ollama_num_predict
    return max(1000, int((settings.ollama_num_ctx - out - 256) * CHARS_PER_TOKEN))


# --------------------------------------------------------------------------- usage


class UsageTracker(BaseCallbackHandler):
    """Thread-safe token counter keyed by role (lead/worker)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._path = None
        self._key = ""
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.data = {
                r: {"calls": 0, "in": 0, "cached": 0, "out": 0} for r in ("lead", "worker")
            }
            self._path = None
            self._key = ""

    def attach(self, path, key: str) -> None:
        """Save the counts to `path` after every LLM call, tagged with `key` (the run's
        thread id), so a crashed run's spend can be restored by /resume."""
        with self._lock:
            self._path, self._key = path, key
            self._save()

    def restore(self, path, key: str) -> bool:
        """Load counts saved by `attach` for the same key. True if anything was restored."""
        try:
            saved = json.loads(open(path, encoding="utf-8").read())
        except Exception:
            return False
        if saved.get("key") != key:
            return False
        with self._lock:
            for role, d in (saved.get("usage") or {}).items():
                self.data[role] = {k: int(d.get(k, 0)) for k in ("calls", "in", "cached", "out")}
        return True

    def _save(self) -> None:  # caller holds the lock
        if not self._path:
            return
        try:
            tmp = f"{self._path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"key": self._key, "usage": self.data}, f)
            os.replace(tmp, self._path)
        except Exception:
            pass

    def record(self, role: str, usage: dict) -> None:
        details = usage.get("input_token_details") or {}
        with self._lock:
            d = self.data.setdefault(role, {"calls": 0, "in": 0, "cached": 0, "out": 0})
            d["calls"] += 1
            d["in"] += int(usage.get("input_tokens") or 0)
            d["cached"] += int(details.get("cache_read") or 0)
            d["out"] += int(usage.get("output_tokens") or 0)
            self._save()

    def snapshot(self) -> dict:
        with self._lock:
            return {r: dict(v) for r, v in self.data.items()}

    def cost_by_role(self, backend: str = "deepseek") -> dict[str, float]:
        """Estimated USD per role at peak DeepSeek rates (all 0 for local Ollama)."""
        s = settings
        prices = {
            "lead": (s.price_lead_in, s.price_lead_in_cached, s.price_lead_out),
            "worker": (s.price_worker_in, s.price_worker_in_cached, s.price_worker_out),
        }
        out = {}
        for role, d in self.snapshot().items():
            if backend != "deepseek":
                out[role] = 0.0
                continue
            p_in, p_cached, p_out = prices.get(role, prices["worker"])
            fresh = max(0, d["in"] - d["cached"])
            out[role] = (fresh * p_in + d["cached"] * p_cached + d["out"] * p_out) / 1e6
        return out

    def cost(self, backend: str = "deepseek") -> float:
        """Estimated USD at peak DeepSeek rates (0 for local Ollama)."""
        return sum(self.cost_by_role(backend).values())

    def summary(self, backend: str = "deepseek") -> str:
        parts = []
        for role, d in self.snapshot().items():
            if d["calls"]:
                parts.append(f"{role} {d['calls']} calls · {d['in'] / 1000:.1f}k in / {d['out'] / 1000:.1f}k out")
        if not parts:
            return "no LLM calls"
        cost = self.cost(backend)
        return " · ".join(parts) + (f" · ≈ ${cost:.3f}" if backend == "deepseek" else " · local (free)")


usage = UsageTracker()


class _RoleCallback(BaseCallbackHandler):
    def __init__(self, role: str) -> None:
        self.role = role

    def on_llm_end(self, response, **kwargs: Any) -> None:
        for gens in response.generations or []:
            for g in gens:
                um = getattr(getattr(g, "message", None), "usage_metadata", None)
                if um:
                    usage.record(self.role, dict(um))


# --------------------------------------------------------------------------- deepseek


def _deepseek_class():
    """ChatDeepSeek that echoes `reasoning_content` back to the API.

    DeepSeek's thinking mode requires the assistant's reasoning_content to be sent
    back on every later request of a tool-calling conversation (else HTTP 400), and
    langchain-deepseek drops it. We re-attach it from the AIMessage's additional_kwargs.
    """
    from langchain_core.messages import AIMessage
    from langchain_deepseek import ChatDeepSeek

    class _DeepSeek(ChatDeepSeek):
        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            try:
                msgs = self._convert_input(input_).to_messages()
            except Exception:
                return payload
            ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
            ai_dicts = [m for m in payload.get("messages", []) if m.get("role") == "assistant"]
            for m, d in zip(ai_msgs, ai_dicts):
                rc = (m.additional_kwargs or {}).get("reasoning_content")
                if rc:
                    d["reasoning_content"] = rc
            return payload

    return _DeepSeek


# --------------------------------------------------------------------------- factory


def get_llm(
    backend: str | None = None,
    temperature: float = 0.3,
    role: str = "worker",
    max_tokens: int | None = None,
    thinking: bool | None = None,
):
    """Construct a chat model for the chosen backend and role.

    `max_tokens` is backend-neutral (mapped to Ollama's num_predict). `thinking`
    overrides the role's DeepSeek thinking default (e.g. off for trivial lead calls,
    or for a retry when reasoning ate the whole output budget). Both backends
    expose `.bind_tools(...)`, so the rest of the app is backend-agnostic.
    """
    backend, _ = resolve_backend(backend)
    callbacks = [_RoleCallback(role)]

    if backend == "deepseek":
        lead = role == "lead"
        if thinking is None:
            thinking = settings.deepseek_lead_thinking if lead else settings.deepseek_worker_thinking
        extra = {"reasoning_effort": settings.deepseek_lead_effort} if (thinking and lead) else {}
        return _deepseek_class()(
            model=settings.deepseek_lead_model if lead else settings.deepseek_worker_model,
            api_key=settings.deepseek_api_key,
            api_base=settings.deepseek_base_url,
            temperature=temperature,
            # reasoning tokens count toward the output limit, so thinking calls get headroom
            max_tokens=(max_tokens or 4096) + (16384 if thinking else 0),
            extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
            **extra,
            timeout=240,
            max_retries=3,
            callbacks=callbacks,
        )

    from langchain_ollama import ChatOllama

    model = settings.ollama_model
    if role == "lead" and settings.ollama_lead_model:
        model = settings.ollama_lead_model
    params = dict(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        num_predict=max_tokens or settings.ollama_num_predict,
        num_ctx=settings.ollama_num_ctx,
        client_kwargs={"timeout": settings.ollama_timeout},
        callbacks=callbacks,
    )
    # `reasoning` exists only on newer langchain-ollama; degrade gracefully.
    try:
        return ChatOllama(reasoning=settings.ollama_reasoning, **params)
    except TypeError:
        return ChatOllama(**params)
