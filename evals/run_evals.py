"""Headless eval runner: research each benchmark query end-to-end, then judge it.

  python evals/run_evals.py --backend deepseek --limit 4
  python evals/run_evals.py --ids rag-eval,lora-qlora
  python evals/run_evals.py --jarvis-path ../jarvis-main --label baseline   # judge another checkout
  python evals/run_evals.py --rejudge evals/results/x.json --label x-rejudged # re-grade saved reports

The driver only relies on the graph's interrupt protocol (clarify_questions /
confirm_plan), so it can run older versions of Jarvis too — that's how the README's
before/after table is produced. The judge is always the same model (DeepSeek lead).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _judge_llm():
    from dotenv import load_dotenv
    from langchain_deepseek import ChatDeepSeek

    load_dotenv(REPO / ".env")
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        sys.exit("DEEPSEEK_API_KEY missing in .env — the judge needs it.")
    return ChatDeepSeek(
        model=os.environ.get("DEEPSEEK_LEAD_MODEL", "deepseek-v4-pro"), api_key=key,
        api_base=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        max_tokens=24000, extra_body={"thinking": {"type": "enabled"}}, reasoning_effort="low",
        timeout=300, max_retries=3,
    )


def _drive(graph, query: str, backend: str) -> dict:
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 80}
    payload = {"messages": [HumanMessage(content=query)], "query": query, "backend": backend, "max_loops": 3}
    for _ in range(6):
        intr = None
        for chunk in graph.stream(payload, config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                i = chunk["__interrupt__"]
                intr = i[0].value if isinstance(i, (list, tuple)) else i
        if not intr:
            break
        payload = Command(resume={"answers": ""}) if intr.get("type") == "clarify_questions" else Command(resume="yes")
    return dict(graph.get_state(config).values)


def _norm(u: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", (u or "").strip().lower()).rstrip("/")


def _evidence(state: dict, report: str) -> str:
    """Render each listed source with its abstract/snippet so the judge can check claims."""
    from judge import split_sources

    items = []
    for e in (state.get("sources") or {}).values():
        items.append(e["item"])
    for key in ("papers", "web", "datasets"):
        items.extend(state.get(key) or [])
    by_url = {}
    for it in items:
        for u in (it.get("url"), it.get("pdf_url")):
            if u:
                by_url.setdefault(_norm(u), it)
    _, entries = split_sources(report)
    lines = []
    for n, line in sorted(entries.items()):
        url = line.rsplit(" — ", 1)[-1].strip() if " — " in line else ""
        it = by_url.get(_norm(url), {})
        detail = it.get("abstract") or it.get("snippet") or it.get("description") or "(no abstract available)"
        lines.append(f"[{n}] {line}\n    {detail[:500]}")
    return "\n".join(lines) or "(the report lists no sources)"


def _process(state: dict, seconds: float) -> str:
    reps = state.get("subagent_reports") or []
    calls = sum(r.get("tool_calls") or len(r.get("tools") or []) for r in reps)
    return f"{len(reps)} subagents, {calls} tool calls, {seconds:.0f}s wall time"


def _write_summary(results: list[dict], label: str, criteria: list[str]) -> str:
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    (out_dir / f"{label}-{stamp}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    n = len(results) or 1
    avg = {c: sum(r[c] for r in results) / n for c in criteria + ["overall"]}
    costs = [r["cost_usd"] for r in results if r.get("cost_usd") is not None]
    row = (
        f"| {label} | " + " | ".join(f"{avg[c]:.2f}" for c in criteria + ["overall"])
        + f" | {sum(r['pass'] for r in results)}/{len(results)}"
        + f" | {sum(r['looks_truncated'] for r in results)}"
        + f" | {sum(r['dangling_citations'] for r in results)}"
        + f" | {sum(r['citations_in_text'] for r in results) / n:.0f}"
        + f" | {sum(r['seconds'] for r in results) / n:.0f}s"
        + (f" | ${sum(costs) / len(costs):.3f} |" if costs else " | n/a |")
    )
    header = (
        "| version | " + " | ".join(criteria + ["overall"])
        + " | pass | truncated | dangling cites | avg citations | avg time | avg cost |\n"
        + "|---" * (len(criteria) + 9) + "|"
    )
    summary = header + "\n" + row
    (out_dir / f"{label}-{stamp}.md").write_text(summary + "\n", encoding="utf-8")
    return summary


def _rejudge(path: str, label: str) -> int:
    """Re-grade saved results with the current judge (evidence rebuilt from data/runs artifacts)."""
    from judge import CRITERIA, deterministic_metrics, judge

    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    runs = [d for d in (REPO / "data" / "runs").iterdir() if (d / "report.md").exists()]
    judge_llm = _judge_llm()
    for r in rows:
        evidence = r.get("evidence")
        if not evidence:
            state = {}
            for d in runs:
                if r["report"].startswith((d / "report.md").read_text(encoding="utf-8")[:2000]):
                    state = {"sources": json.loads((d / "sources.json").read_text(encoding="utf-8"))}
                    break
            evidence = _evidence(state, r["report"])
        r.update(deterministic_metrics(r["report"]))
        r.update(judge(judge_llm, r["query"], r["report"], evidence, r.get("process", "")))
        r["evidence"] = evidence
        print(f"  {r['id']}: overall {r['overall']:.2f} pass={r['pass']} · {r['notes'][:200]}", flush=True)
    print("\n" + _write_summary(rows, label, CRITERIA))
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="deepseek")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--jarvis-path", default=str(REPO))
    ap.add_argument("--label", default="current")
    ap.add_argument("--rejudge", default="", help="re-grade a saved results .json with the current judge")
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    if args.rejudge:
        sys.path.insert(0, str(REPO))
        return _rejudge(args.rejudge, args.label)
    sys.path.insert(0, str(Path(args.jarvis_path).resolve()))
    from langgraph.checkpoint.memory import MemorySaver

    import jarvis.graph as jgraph
    import jarvis.llm as jllm
    from judge import CRITERIA, deterministic_metrics, judge

    if hasattr(jllm, "set_active_backend"):
        jllm.set_active_backend(args.backend)
    tracker = getattr(jllm, "usage", None)

    queries = [json.loads(line) for line in (HERE / "queries.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.ids:
        want = set(args.ids.split(","))
        queries = [q for q in queries if q["id"] in want]
    if args.limit:
        queries = queries[: args.limit]

    judge_llm = _judge_llm()
    results = []
    for q in queries:
        print(f"\n=== [{args.label}] {q['id']}: {q['query']}", flush=True)
        if tracker:
            tracker.reset()
        t0 = time.time()
        try:
            state = _drive(jgraph.build_graph(MemorySaver()), q["query"], args.backend)
            error = ""
        except Exception as exc:
            state, error = {}, f"{type(exc).__name__}: {exc}"
        secs = time.time() - t0
        report = state.get("report") or ""
        cost = tracker.cost(args.backend) if tracker else None
        metrics = deterministic_metrics(report)
        evidence = _evidence(state, report)
        if report:
            scores = judge(judge_llm, q["query"], report, evidence, _process(state, secs))
        else:
            scores = {**{c: 0.0 for c in CRITERIA}, "overall": 0.0, "pass": False, "notes": error or "no report"}
        row = {"id": q["id"], "query": q["query"], "seconds": round(secs, 1), "cost_usd": cost, "error": error,
               "process": _process(state, secs), **metrics, **scores, "report": report, "evidence": evidence}
        results.append(row)
        print(f"  overall {scores['overall']:.2f} pass={scores['pass']} · {row['process']} · "
              f"cost {'$%.3f' % cost if cost is not None else 'n/a'} · truncated={metrics['looks_truncated']}", flush=True)
        print(f"  notes: {scores['notes']}", flush=True)

    print("\n" + _write_summary(results, args.label, CRITERIA))
    total = sum(r["cost_usd"] or 0 for r in results)
    print(f"\nresearch cost (excl. judge): ${total:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
