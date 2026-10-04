from langchain_core.messages import AIMessage, HumanMessage

from jarvis import config, llm
from jarvis.utils import chunk_text, extract_json, truncate


def test_roles_pick_lead_and_worker_models(monkeypatch):
    monkeypatch.setattr(config.settings, "deepseek_api_key", "sk-test")
    lead = llm.get_llm("deepseek", role="lead")
    worker = llm.get_llm("deepseek", role="worker")
    assert lead.model_name == config.settings.deepseek_lead_model
    assert worker.model_name == config.settings.deepseek_worker_model
    assert lead.extra_body == {"thinking": {"type": "enabled"}}
    assert worker.extra_body == {"thinking": {"type": "disabled"}}


def test_max_tokens_is_backend_neutral(monkeypatch):
    monkeypatch.setattr(config.settings, "deepseek_api_key", "sk-test")
    ds = llm.get_llm("deepseek", max_tokens=1200)
    payload = ds._get_request_payload([HumanMessage("hi")])
    assert payload["max_tokens"] == 1200 and "num_predict" not in payload  # the old /ask crash
    assert llm.get_llm("ollama", max_tokens=1200).num_predict == 1200


def test_reasoning_content_is_echoed_back_for_thinking_mode(monkeypatch):
    monkeypatch.setattr(config.settings, "deepseek_api_key", "sk-test")
    model = llm.get_llm("deepseek", role="lead")
    ai = AIMessage(content="", additional_kwargs={"reasoning_content": "I should search."},
                   tool_calls=[{"name": "search_web", "args": {"query": "x"}, "id": "c1", "type": "tool_call"}])
    from langchain_core.messages import ToolMessage

    payload = model._get_request_payload([HumanMessage("q"), ai, ToolMessage(content="r", tool_call_id="c1")])
    assistant = [m for m in payload["messages"] if m["role"] == "assistant"][0]
    assert assistant["reasoning_content"] == "I should search."


def test_deepseek_without_key_falls_back_to_ollama():
    backend, notice = llm.resolve_backend("deepseek")
    assert backend == "ollama" and "DEEPSEEK_API_KEY" in notice


def test_usage_tracker_cost():
    t = llm.UsageTracker()
    t.record("lead", {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "input_token_details": {"cache_read": 0}})
    t.record("worker", {"input_tokens": 2_000_000, "output_tokens": 0, "input_token_details": {"cache_read": 1_000_000}})
    s = config.settings
    expected = s.price_lead_in + s.price_lead_out + s.price_worker_in + s.price_worker_in_cached
    assert abs(t.cost("deepseek") - expected) < 1e-9
    assert t.cost("ollama") == 0.0
    assert "lead 1 calls" in t.summary()


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}
    assert extract_json("no json") == {}
    assert extract_json("") == {}


def test_chunk_and_truncate():
    assert chunk_text("") == []
    chunks = chunk_text("x" * 250, size=100, overlap=10)
    assert len(chunks) == 3 and all(len(c) <= 100 for c in chunks)
    assert truncate("hello world", 6) == "hello…"
    assert truncate("hi", 6) == "hi"
