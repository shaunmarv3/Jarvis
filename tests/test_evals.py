"""Eval tooling, offline: fact recall, statistics, summaries, human agreement, baseline."""

import json
import sys
from pathlib import Path

import pytest

EVALS = Path(__file__).resolve().parent.parent / "evals"
sys.path.insert(0, str(EVALS))

import human  # noqa: E402
import judge  # noqa: E402
import run_evals  # noqa: E402


def test_fact_recall_reads_the_body_only():
    report = ("QLoRA uses 4-bit NormalFloat and double quantization.\n\n## Sources\n"
              "- [1] QLoRA: fine-tune a 65B model on one 48GB GPU — https://arxiv.org/abs/2305.14314")
    facts = [["NF4", "NormalFloat"], ["double quantization"], ["65B", "65 billion"]]
    recall, missing = judge.fact_recall(report, facts)
    assert recall == pytest.approx(2 / 3, abs=1e-3) and missing == ["65B"]  # "65B" only in a source title
    assert judge.fact_recall(report, None) == (None, [])


def test_fact_recall_matches_at_word_starts():
    assert judge.fact_recall("introduced by Google in 2017", [["2017"], ["Google"]])[0] == 1.0
    assert judge.fact_recall("version 12017 at googleplex", [["2017"]])[0] == 0.0


def test_statistics():
    assert judge.sd([1, 1, 1]) == 0 and judge.sd([5]) == 0
    assert judge.sd([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(2.138, abs=1e-3)
    assert judge.pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1)
    assert judge.spearman([1, 2, 3, 4], [10, 20, 30, 1000]) == pytest.approx(1)  # rank-based
    assert judge.spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1)  # ties


def _row(qid, rep, overall, **kw):
    base = {c: overall for c in judge.CRITERIA}
    return {"id": qid, "type": "fact", "query": qid, "repeat": rep, "label": "jarvis", "overall": overall,
            "pass": overall >= 0.7, "fact_recall": kw.pop("fact_recall", None), "looks_truncated": False,
            "citations_in_text": 10, "seconds": 100, "cost_usd": 0.1, "judge_error": "", **base, **kw}


def test_summary_reports_mean_and_spread_across_repeats():
    rows = [_row("a", 0, 0.8, fact_recall=1.0), _row("b", 0, 0.6),
            _row("a", 1, 0.6, fact_recall=0.5), _row("b", 1, 0.4)]
    line = run_evals.summary_row(rows)
    # repeat means: 0.7 and 0.5 -> 0.60 ± 0.14 ; fact recall 1.0 and 0.5 -> 0.75 ± 0.35
    assert "| jarvis | 2×2 |" in line and "0.60 ± 0.14" in line and "0.75 ± 0.35" in line
    assert "25% ± 35" in line  # pass rate: repeat 0 = 50% (a passes), repeat 1 = 0%


def test_judge_failures_are_left_out_of_the_scores():
    rows = [_row("a", 0, 0.8), {**_row("b", 0, 0.0), "judge_error": "timeout"}]
    assert "| 0.80 |" in run_evals.summary_row(rows)


def test_per_query_table_lists_missing_facts():
    rows = [_row("a", 0, 0.8, fact_recall=0.5, facts_missing=["65B"]), _row("a", 1, 0.6, fact_recall=1.0)]
    table = run_evals.per_query_table(rows)
    assert "| a | fact | 0.70 ± 0.14 | 0.75 ± 0.35 | 65B |" in table
    assert "Run-to-run noise" in table


def test_human_agreement():
    rows = [_row("a", 0, 0.9), _row("b", 0, 0.5), _row("c", 0, 0.2)]
    grades = [{"id": "a", "repeat": "0", **{c: "0.8" for c in judge.CRITERIA}, "pass": "1"},
              {"id": "b", "repeat": "0", **{c: "0.6" for c in judge.CRITERIA}, "pass": "0"},
              {"id": "c", "repeat": "0", **{c: "0.1" for c in judge.CRITERIA}, "pass": "0"},
              {"id": "d", "repeat": "0", **{c: "" for c in judge.CRITERIA}, "pass": ""}]  # ungraded: skipped
    res = human.agreement(rows, grades)
    assert res["n"] == 3
    assert res["criteria"]["overall"]["spearman"] == pytest.approx(1)
    assert res["criteria"]["overall"]["mean_abs_diff"] == pytest.approx(0.1)
    assert res["pass_agreement"] == pytest.approx(1)  # 0.9 pass, 0.5 and 0.2 fail on both sides


def test_human_export_is_blind_and_spread_over_types(tmp_path):
    rows = [{**_row(f"q{i}", 0, 0.5), "type": t, "report": "r", "notes": "JUDGE SAYS 0.5"}
            for i, t in enumerate(["fact", "fact", "fact", "survey", "code"])]
    src = tmp_path / "res.json"
    src.write_text(json.dumps(rows), encoding="utf-8")
    human.export(str(src), 3, str(tmp_path / "sheet"))
    files = sorted(p.name for p in (tmp_path / "sheet").glob("q*.md"))
    types = {next(r["type"] for r in rows if f"{r['id']}-r0.md" == f) for f in files}
    assert len(files) == 3 and types == {"fact", "survey", "code"}
    assert "JUDGE SAYS" not in "".join(p.read_text(encoding="utf-8") for p in (tmp_path / "sheet").glob("*.md"))


def test_baseline_is_one_llm_call_with_registry_citations(monkeypatch):
    import baseline

    import jarvis.subagent as sa
    from conftest import FakeLLM, fake_paper

    class OneCall(FakeLLM):
        def invoke(self, msgs):
            self.calls.append(msgs)
            from langchain_core.messages import AIMessage

            return AIMessage(content="# Answer\nRAGAS [S2]; blog [S1]; invented [S9].")

    llm = OneCall()
    import jarvis.llm as jllm

    monkeypatch.setattr(jllm, "get_llm", lambda *a, **k: llm)
    monkeypatch.setattr(sa, "TOOL_FUNCS", {
        "search_web": lambda query, max_results=6: {"web": [{"title": "Blog", "url": "https://b.com", "snippet": "s"}],
                                                    "text": "Web results"},
        "search_arxiv": lambda query, max_results=5: {"papers": [fake_paper(1)], "text": "arXiv results"},
    })
    out = baseline.run_baseline("how is RAG evaluated?", "deepseek")
    assert len(llm.calls) == 1
    assert out["citation_stats"] == {"cited": 2, "invalid_tags": 1, "retrieved": 2}
    assert "- [1] Paper number 1 (2024)" in out["report"]


def test_ablations_only_touch_existing_settings():
    from jarvis.config import settings

    for name, overrides in run_evals.ABLATIONS.items():
        assert all(k in type(settings).model_fields for k in overrides), name


def test_query_set_is_well_formed():
    qs = [json.loads(line) for line in (EVALS / "queries.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(qs) >= 25 and len({q["id"] for q in qs}) == len(qs)
    for q in qs:
        assert q["type"] in {"survey", "comparison", "dataset", "code", "practitioner", "exact", "fact"}
        assert all(isinstance(g, list) and g for g in q.get("facts", []))


def test_claude_judge_adapter_returns_text_like_the_others(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    j = judge.make_judge("anthropic")
    assert j.model == judge.JUDGE_DEFAULT_MODELS["anthropic"]

    class Block:
        type, text = "text", '{"factual_accuracy": 0.9, "citation_accuracy": 0.8, "completeness": 0.7, ' \
                             '"source_quality": 0.6, "tool_efficiency": 0.5, "pass": true, "notes": "ok"}'

    sent = {}
    monkeypatch.setattr(j.client.messages, "create", lambda **k: sent.update(k) or type("R", (), {"content": [Block()]}))
    out = judge.judge(j, "q", "report", "evidence", "1 subagent")
    assert out["overall"] == pytest.approx(0.7) and out["pass"] is True
    assert sent["model"] == j.model and sent["messages"][0]["role"] == "user"


def test_missing_judge_key_fails_with_a_clear_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY missing"):
        judge.make_judge("anthropic")


# --------------------------------------------------------------------------- grading outside the harness


def test_judge_none_saves_reports_ungraded(monkeypatch):
    assert judge.make_judge("none") is None
    monkeypatch.setattr(run_evals, "_drive", lambda *a: {"report": "QLoRA uses NF4 [1].\n\n## Sources\n- [1] Q — https://x"})
    args = type("A", (), {"system": "jarvis", "backend": "ollama", "label": "jarvis", "ablation": "none",
                          "judge": "none"})()
    row = run_evals.run_job({"id": "q", "query": "qlora?", "facts": [["NF4"]]}, 0, args, None)
    assert row["graded"] is False and row["fact_recall"] == 1.0 and row["report"]
    line = run_evals.summary_row([row])
    assert "| 1.00 |" in line  # fact recall is measured in code, before any grading
    assert "| — | — | — | — | — | — |" in line  # judge scores wait for grades


def test_export_is_blind_and_import_matches_the_automated_judge(tmp_path):
    import grading

    rows = [{**_row("a", 0, 0.0), "graded": False, "report": "report A", "evidence": "ev A", "process": "p"},
            {**_row("b", 0, 0.0), "graded": False, "report": "report B", "evidence": "ev B", "process": "p"},
            {**_row("c", 0, 0.0), "report": "", "error": "crashed"}]  # no report: nothing to grade
    src = tmp_path / "jarvis-1.json"
    src.write_text(json.dumps(rows), encoding="utf-8")
    folder = grading.export([str(src)], out_root=tmp_path / "grading", seed=1)

    packets = sorted(p.name for p in (folder / "packets").glob("*.md"))
    assert packets == ["g01.md", "g02.md"]
    text = (folder / "packets" / "g01.md").read_text(encoding="utf-8")
    assert "Score each criterion" in text and "jarvis" not in text.lower()  # the rubric, but no system name
    key = json.loads((folder / "key.json").read_text(encoding="utf-8"))
    by_id = {v["id"]: k for k, v in key.items()}

    good = {c: 0.8 for c in judge.CRITERIA} | {"pass": True, "notes": "solid"}
    grades = {by_id["a"]: good, by_id["b"]: {**good, "completeness": 7}}  # b: out of range
    (folder / "grades.json").write_text(json.dumps(grades), encoding="utf-8")
    files, problems = grading.import_grades(folder, "manual:claude")

    out = files[str(src.resolve())]
    assert out[0]["graded"] and out[0]["overall"] == judge.parse_scores(good)["overall"] == 0.8
    assert out[0]["judge"] == "manual:claude" and out[0]["notes"] == "solid"
    assert out[1]["graded"] is False and any("completeness" in p for p in problems)
    assert json.loads(src.read_text(encoding="utf-8"))[0]["graded"] is False  # the original file is untouched
    # b (invalid grade) is left out; c (crashed, no report) counts as 0, as with a model judge.
    assert "| 0.40 |" in run_evals.summary_row(out)


def test_grade_validation():
    import grading

    ok = {c: 0.5 for c in judge.CRITERIA} | {"pass": False}
    assert grading.validate(ok) is None
    assert "pass" in grading.validate({**ok, "pass": "yes"})
    assert "factual_accuracy" in grading.validate({**ok, "factual_accuracy": True})
    assert grading.validate(None) == "not a JSON object"
