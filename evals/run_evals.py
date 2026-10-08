"""Headless eval runner: research each benchmark query end-to-end, then grade the report.

  python evals/run_evals.py --limit 4                          # quick check
  python evals/run_evals.py --repeats 3 --workers 3            # full benchmark, mean ± spread
  python evals/run_evals.py --ablation single_agent            # switch one design piece off
  python evals/run_evals.py --system baseline                  # 1 web + 1 arXiv search + 1 LLM call
  python evals/run_evals.py --judge anthropic                  # cross-family judge (no self-preference)
  python evals/run_evals.py --jarvis-path ../old --label prev  # grade another checkout
  python evals/run_evals.py --rejudge results/x.json --judge anthropic --label x-claude
  python evals/run_evals.py --judge none --workers 6           # research only; grade later, by hand:
  python evals/run_evals.py --export-grading results/a.json results/b.json   # blinded packets (grading.py)
  python evals/run_evals.py --import-grades evals/grading/<stamp> --label claude
  python evals/run_evals.py --table results/a.json results/b.json   # one comparison table

Grading = one LLM-judge call on Anthropic's 5-criterion rubric + checks done in code
(fact recall on known-answer queries, truncation, dangling citations, cost, time).
The driver only relies on the graph's interrupt protocol (clarify_questions /
confirm_plan), so it can also run older versions of Jarvis.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "results"

# Each ablation switches off ONE piece of the design, so its effect can be measured.
ABLATIONS = {
    "none": {},
    "no_followup": {"lead_iteration": False},  # no gap-filling follow-up wave
    "no_reads": {"subagent_max_reads": 0},  # abstracts/snippets only, no full-text reads
    "single_agent": {"single_agent": True},  # one subagent gets the whole plan
}


# --------------------------------------------------------------------------- running


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

    items = [e["item"] for e in (state.get("sources") or {}).values()]
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


def _process(state: dict, seconds: float, system: str = "jarvis") -> str:
    reps = state.get("subagent_reports") or []
    calls = sum(r.get("tool_calls") or len(r.get("tools") or []) for r in reps)
    if system == "baseline":
        return f"single LLM call over {calls} searches, {seconds:.0f}s wall time"
    return f"{len(reps)} subagents, {calls} tool calls, {seconds:.0f}s wall time"


def _setup(args):
    """Import the Jarvis checkout under test and apply the ablation. Returns jarvis.llm."""
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    sys.path.insert(0, str(Path(args.jarvis_path).resolve()))
    import jarvis.config as jconfig
    import jarvis.llm as jllm

    for key, value in ABLATIONS[args.ablation].items():
        if key not in type(jconfig.settings).model_fields:
            sys.exit(f"This Jarvis checkout has no '{key}' setting, so it can't run the '{args.ablation}' ablation.")
        setattr(jconfig.settings, key, value)
    if hasattr(jllm, "set_active_backend"):
        jllm.set_active_backend(args.backend)
    return jllm


def run_job(q: dict, repeat: int, args, judge_llm) -> dict:
    """Research one query once and grade it."""
    import jarvis.llm as jllm
    from judge import CRITERIA, deterministic_metrics, fact_recall, judge

    tracker = getattr(jllm, "usage", None)
    if tracker:
        tracker.reset()
    t0 = time.time()
    try:
        if args.system == "baseline":
            from baseline import run_baseline

            state = run_baseline(q["query"], args.backend)
        else:
            from langgraph.checkpoint.memory import MemorySaver

            import jarvis.graph as jgraph

            state = _drive(jgraph.build_graph(MemorySaver()), q["query"], args.backend)
        error = ""
    except Exception as exc:
        state, error = {}, f"{type(exc).__name__}: {exc}"
    secs = time.time() - t0
    report = state.get("report") or ""
    cost = tracker.cost(args.backend) if tracker else None
    recall, missing = fact_recall(report, q.get("facts"))
    evidence = _evidence(state, report)
    process = _process(state, secs, args.system)
    judge_error, graded = "", True
    if report and judge_llm is None:  # --judge none: grade later from blinded packets (grading.py)
        scores = {**{c: 0.0 for c in CRITERIA}, "overall": 0.0, "pass": False, "notes": "not graded yet"}
        graded = False
    elif report:
        try:
            scores = judge(judge_llm, q["query"], report, evidence, process)
        except Exception as exc:
            judge_error = f"{type(exc).__name__}: {exc}"
            scores = {**{c: 0.0 for c in CRITERIA}, "overall": 0.0, "pass": False, "notes": "judge failed"}
    else:
        scores = {**{c: 0.0 for c in CRITERIA}, "overall": 0.0, "pass": False, "notes": error or "no report"}
    return {
        "id": q["id"], "type": q.get("type", ""), "query": q["query"], "repeat": repeat,
        "label": args.label, "system": args.system, "ablation": args.ablation,
        "judge": f"{args.judge}:{getattr(judge_llm, 'model', getattr(judge_llm, 'model_name', ''))}",
        "graded": graded,
        "seconds": round(secs, 1), "cost_usd": cost, "error": error, "judge_error": judge_error,
        "process": process, **deterministic_metrics(report),
        "fact_recall": recall, "facts_missing": missing, **scores,
        "report": report, "evidence": evidence,
    }


def _scored(row: dict) -> bool:
    """Rows whose scores count: graded, and the judge didn't fail on them."""
    return not row.get("judge_error") and row.get("graded", True)


def _print_row(row: dict) -> None:
    fr = "" if row["fact_recall"] is None else f" · facts {row['fact_recall']:.2f}"
    cost = f"${row['cost_usd']:.3f}" if row.get("cost_usd") is not None else "n/a"
    score = f"overall {row['overall']:.2f} pass={row['pass']}" if row.get("graded", True) else "ungraded"
    print(f"  [{row['id']} r{row['repeat']}] {score}{fr} · "
          f"{row['process']} · {cost} · truncated={row['looks_truncated']}", flush=True)
    if row.get("error") or row.get("judge_error"):
        print(f"    ! {row.get('error') or row.get('judge_error')}", flush=True)


# --------------------------------------------------------------------------- parallel workers


def _child_cmd(args, job: str, out: Path) -> list[str]:
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--job", job, "--out", str(out),
           "--backend", args.backend, "--system", args.system, "--ablation", args.ablation,
           "--judge", args.judge, "--jarvis-path", args.jarvis_path, "--label", args.label]
    if args.judge_model:
        cmd += ["--judge-model", args.judge_model]
    return cmd


def _run_parallel(jobs: list[tuple[dict, int]], args, stamp: str) -> list[dict]:
    """One subprocess per (query, repeat): every run gets its own token tracker and process
    state, so concurrent runs can't mix their costs."""
    logs = RESULTS / "logs" / f"{args.label}-{stamp}"
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUTF8": "1"}

    def go(job):
        q, r = job
        out = logs / f"{q['id']}-r{r}.json"
        with open(logs / f"{q['id']}-r{r}.log", "w", encoding="utf-8") as log:
            subprocess.run(_child_cmd(args, f"{q['id']}:{r}", out), stdout=log, stderr=subprocess.STDOUT,
                           env=env, cwd=str(REPO))
        try:
            row = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            row = {"id": q["id"], "type": q.get("type", ""), "query": q["query"], "repeat": r, "label": args.label,
                   "error": f"worker crashed — see {logs / (q['id'] + f'-r{r}.log')}", "judge_error": "",
                   "overall": 0.0, "pass": False, "fact_recall": None, "looks_truncated": False,
                   "citations_in_text": 0, "dangling_citations": 0, "seconds": 0, "cost_usd": None,
                   "process": "crashed", **{c: 0.0 for c in _criteria()}}
        _print_row(row)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        return list(ex.map(go, jobs))


def _criteria() -> list[str]:
    from judge import CRITERIA

    return CRITERIA


# --------------------------------------------------------------------------- summaries


def _fmt(values: list[float], pct: bool = False) -> str:
    from judge import mean, sd

    if not values:
        return "—"
    m = mean(values)
    s = f"{m * 100:.0f}%" if pct else f"{m:.2f}"
    if len(values) > 1:
        s += f" ± {sd(values) * 100:.0f}" if pct else f" ± {sd(values):.2f}"
    return s


def summary_row(rows: list[dict], label: str | None = None) -> str:
    """One table row: each score is the mean over repeats of the per-repeat average, ± the
    standard deviation across repeats (shown when there is more than one repeat)."""
    from judge import mean

    crit = _criteria()
    by_rep, all_by_rep = defaultdict(list), defaultdict(list)
    for r in rows:
        all_by_rep[r.get("repeat", 0)].append(r)
        if _scored(r):
            by_rep[r.get("repeat", 0)].append(r)

    def per_rep(fn, groups=by_rep):
        vals = [fn(rs) for rs in groups.values()]
        return [v for v in vals if v == v]  # drop NaN

    cells = [_fmt(per_rep(lambda rs, c=c: mean([r[c] for r in rs]))) for c in crit + ["overall"]]
    # Fact recall is measured in code, so it counts even before a report is graded.
    cells.append(_fmt(per_rep(lambda rs: mean([r["fact_recall"] for r in rs if r.get("fact_recall") is not None]),
                              all_by_rep)))
    cells.append(_fmt(per_rep(lambda rs: mean([1.0 if r["pass"] else 0.0 for r in rs])), pct=True))
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    label = label or (rows[0].get("label") if rows else None) or "?"
    n_q = len({r["id"] for r in rows})
    return (f"| {label} | {n_q}×{len(all_by_rep) or 1} | " + " | ".join(cells)
            + f" | {sum(bool(r.get('looks_truncated')) for r in rows)}"
            + f" | {mean([r.get('citations_in_text', 0) for r in rows]):.0f}"
            + f" | {mean([r.get('seconds', 0) for r in rows]):.0f}s"
            + (f" | ${mean(costs):.3f} |" if costs else " | n/a |"))


def table_header() -> str:
    cols = ["version", "queries×repeats", "factual", "citation", "complete", "source q.", "tool eff.",
            "**overall**", "fact recall", "pass", "truncated", "avg cites", "avg time", "avg cost"]
    return "| " + " | ".join(cols) + " |\n" + "|---" * len(cols) + "|"


def per_query_table(rows: list[dict]) -> str:
    from judge import mean, sd

    by_q = defaultdict(list)
    for r in rows:
        by_q[r["id"]].append(r)
    lines = ["| query | type | overall | fact recall | missing facts | cost |", "|---|---|---|---|---|---|"]
    noise = []
    for qid, rs in by_q.items():
        ov = [r["overall"] for r in rs if _scored(r)]
        if len(ov) > 1:
            noise.append(sd(ov))
        fr = [r["fact_recall"] for r in rs if r.get("fact_recall") is not None]
        missing = sorted({m for r in rs for m in (r.get("facts_missing") or [])})
        costs = [r["cost_usd"] for r in rs if r.get("cost_usd") is not None]
        lines.append(f"| {qid} | {rs[0].get('type', '')} | {_fmt(ov)} | {_fmt(fr)} | {', '.join(missing) or '—'} | "
                     + (f"${mean(costs):.3f}" if costs else "n/a") + " |")
    if noise:
        lines.append(f"\nRun-to-run noise: average per-query standard deviation of the overall score = {mean(noise):.3f}.")
    return "\n".join(lines)


def _write(rows: list[dict], label: str, stamp: str) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{label}-{stamp}.json"
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    md = f"{table_header()}\n{summary_row(rows)}\n\n{per_query_table(rows)}\n"
    path.with_suffix(".md").write_text(md, encoding="utf-8")
    return path


def _table(paths: list[str]) -> int:
    print(table_header())
    for p in paths:
        rows = json.loads(Path(p).read_text(encoding="utf-8"))
        name = re.sub(r"-\d{8}-\d{6}$", "", Path(p).stem)  # results/<label>-<stamp>.json
        print(summary_row(rows, None if rows and rows[0].get("label") else name))
    return 0


def _rejudge(path: str, args) -> int:
    """Re-grade saved reports with another judge (evidence is stored with each row)."""
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    sys.path.insert(0, str(REPO))
    from judge import deterministic_metrics, judge, make_judge

    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    judge_llm = make_judge(args.judge, args.judge_model)
    if judge_llm is None:
        sys.exit("--rejudge needs a model judge; to grade by hand use --export-grading instead.")
    for r in rows:
        r.update(deterministic_metrics(r["report"]))
        if r["report"]:
            r.update(judge(judge_llm, r["query"], r["report"], r.get("evidence", ""), r.get("process", "")))
        r["label"], r["judge"], r["judge_error"] = args.label, args.judge, ""
        print(f"  {r['id']} r{r.get('repeat', 0)}: overall {r['overall']:.2f} pass={r['pass']} · {r['notes'][:160]}",
              flush=True)
    out = _write(rows, args.label, f"{datetime.now():%Y%m%d-%H%M%S}")
    print("\n" + out.with_suffix(".md").read_text(encoding="utf-8"))
    return 0


def _import_grades(folder: str, args) -> int:
    """Write graded copies of the exported results files (originals stay untouched)."""
    from grading import import_grades

    grader = args.label if args.label not in {"", "jarvis"} else "external"  # label defaults to the system
    files, problems = import_grades(folder, judge_name=f"manual:{grader}")
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    for src, rows in files.items():
        source_label = (rows[0].get("label") if rows else "") or Path(src).stem
        label = f"{grader}-{source_label}"
        for r in rows:
            r["label"] = label
        out = _write(rows, label, stamp)
        print(out.with_suffix(".md").read_text(encoding="utf-8").split("\n\n")[0] + f"\n→ {out}\n")
    if problems:
        print(f"{len(problems)} packet(s) not imported (left out of the scores):\n  " + "\n  ".join(problems))
    return 0


# --------------------------------------------------------------------------- main


def _queries(args) -> list[dict]:
    qs = [json.loads(line) for line in (HERE / "queries.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.ids:
        want = set(args.ids.split(","))
        qs = [q for q in qs if q["id"] in want]
    if args.types:
        want = set(args.types.split(","))
        qs = [q for q in qs if q.get("type") in want]
    return qs[: args.limit] if args.limit else qs


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="deepseek")
    ap.add_argument("--system", choices=["jarvis", "baseline"], default="jarvis")
    ap.add_argument("--ablation", choices=sorted(ABLATIONS), default="none")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1, help="parallel runs (each in its own process)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--types", default="", help="e.g. fact,survey")
    ap.add_argument("--judge", default="deepseek",
                    help="deepseek | anthropic | openai | none (save ungraded; grade later with --export-grading)")
    ap.add_argument("--judge-model", default="")
    ap.add_argument("--jarvis-path", default=str(REPO))
    ap.add_argument("--label", default="")
    ap.add_argument("--rejudge", default="", help="re-grade a saved results .json")
    ap.add_argument("--table", nargs="+", default=None, help="print one comparison table from results .json files")
    ap.add_argument("--export-grading", nargs="+", default=None, metavar="RESULTS_JSON",
                    help="write blinded judge packets for grading outside the harness (Claude Code / a person)")
    ap.add_argument("--import-grades", default="", metavar="GRADING_DIR",
                    help="merge a filled-in grading folder back into results files (name the grader with --label)")
    ap.add_argument("--job", default="", help=argparse.SUPPRESS)  # internal: one worker run "id:repeat"
    ap.add_argument("--out", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.label = args.label or (args.system if args.ablation == "none" else args.ablation)

    sys.path.insert(0, str(HERE))
    if args.table:
        return _table(args.table)
    if args.rejudge:
        return _rejudge(args.rejudge, args)
    if args.export_grading:
        from grading import export

        folder = export(args.export_grading)
        n = len(list((folder / "packets").glob("*.md")))
        print(f"{n} blinded packets → {folder}\nInstructions for the grader: {folder / 'README.md'}")
        return 0
    if args.import_grades:
        return _import_grades(args.import_grades, args)

    jllm = _setup(args)
    from judge import make_judge

    queries = _queries(args)
    judge_llm = make_judge(args.judge, args.judge_model or None)

    if args.job:  # worker process: one run, result to --out
        qid, rep = args.job.rsplit(":", 1)
        row = run_job(next(q for q in queries if q["id"] == qid), int(rep), args, judge_llm)
        Path(args.out).write_text(json.dumps(row), encoding="utf-8")
        return 0

    if hasattr(jllm, "check_backend") and (problem := jllm.check_backend(args.backend)):
        sys.exit(f"Can't run evals: {problem}")
    jobs = [(q, r) for r in range(args.repeats) for q in queries]
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    print(f"== {args.label}: {len(queries)} queries × {args.repeats} repeat(s) · system={args.system} · "
          f"ablation={args.ablation} · judge={args.judge} · workers={args.workers}", flush=True)
    if args.workers > 1:
        rows = _run_parallel(jobs, args, stamp)
    else:
        rows = []
        for q, r in jobs:
            print(f"\n=== [{args.label}] {q['id']} (repeat {r}): {q['query']}", flush=True)
            row = run_job(q, r, args, judge_llm)
            _print_row(row)
            rows.append(row)

    out = _write(rows, args.label, stamp)
    print("\n" + out.with_suffix(".md").read_text(encoding="utf-8"))
    total = sum(r.get("cost_usd") or 0 for r in rows)
    print(f"research cost (excl. judge): ${total:.3f} · saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
