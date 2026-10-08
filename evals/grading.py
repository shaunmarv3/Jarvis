"""Grade saved reports outside the harness — by Claude Code (on a subscription, no API key)
or by a person — with exactly the automated judge's rubric, and blind to which system
wrote each report.

  python evals/run_evals.py --judge none --workers 6                  # research only, ungraded
  python evals/run_evals.py --export-grading results/jarvis-X.json results/baseline-Y.json
      -> evals/grading/<stamp>/packets/g01.md …  one blinded judge prompt per report
         evals/grading/<stamp>/grades.json        the grader fills this in
         evals/grading/<stamp>/key.json           packet -> results row (the grader must not open it)
  python evals/run_evals.py --import-grades evals/grading/<stamp> --label claude
      -> results/<label>-<source label>-<stamp>.json/.md, same format as an automated judge's

Packets are shuffled across all exported files and named g01, g02, …, so neither the file
name nor the order reveals whether a report came from Jarvis or the baseline. (The process
stats line can still hint at it — the rubric's tool-efficiency score needs it.)
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

from judge import CRITERIA, judge_prompt, parse_scores

HERE = Path(__file__).resolve().parent
GRADING = HERE / "grading"

GRADER_README = """# Grading batch {stamp}

{n} reports to grade. For each `packets/gNN.md`:

1. Read the whole packet: the question, the report, the evidence and the process stats.
2. Grade it exactly as the packet's instructions say (five criteria, 0.0-1.0, plus pass/fail).
3. Put the JSON reply into `grades.json` under the packet id, e.g.
   `"g01": {{"factual_accuracy": 0.9, "citation_accuracy": 0.8, "completeness": 0.85,
   "source_quality": 0.7, "tool_efficiency": 0.8, "pass": true, "notes": "..."}}`

Rules:
- Do NOT open `key.json` — it says which system wrote each report. Grade blind.
- Grade each packet on its own merits; don't compare packets or try to guess the system.
- Keep the same standard for every packet; re-read the rubric every ~10 packets.
- Leave a packet's entry as `null` to skip it; `--import-grades` reports what is missing.

When done: `python evals/run_evals.py --import-grades {folder} --label <grader name>`
"""


def export(result_files: list[str], out_root: Path = GRADING, seed: int | None = None) -> Path:
    """Write one blinded judge-prompt packet per gradable report; return the batch folder."""
    items = []
    for f in result_files:
        rows = json.loads(Path(f).read_text(encoding="utf-8"))
        for i, r in enumerate(rows):
            if r.get("report"):
                items.append((str(Path(f).resolve()), i, r))
    if not items:
        raise SystemExit("No reports to grade in " + ", ".join(result_files))
    random.Random(seed).shuffle(items)

    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    folder = out_root / stamp
    (folder / "packets").mkdir(parents=True)
    width = max(2, len(str(len(items))))
    key, grades = {}, {}
    for n, (src, idx, r) in enumerate(items, 1):
        gid = f"g{n:0{width}d}"
        prompt = judge_prompt(r["query"], r["report"], r.get("evidence", ""), r.get("process", ""))
        (folder / "packets" / f"{gid}.md").write_text(f"# Packet {gid}\n\n{prompt}\n", encoding="utf-8")
        key[gid] = {"file": src, "row": idx, "id": r["id"], "repeat": r.get("repeat", 0)}
        grades[gid] = None
    (folder / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    (folder / "grades.json").write_text(json.dumps(grades, indent=1), encoding="utf-8")
    (folder / "README.md").write_text(GRADER_README.format(stamp=stamp, n=len(items), folder=folder),
                                      encoding="utf-8")
    return folder


def validate(grade) -> str | None:
    """Why a filled-in grade is unusable, or None if it is fine."""
    if not isinstance(grade, dict):
        return "not a JSON object"
    for c in CRITERIA:
        v = grade.get(c)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 <= v <= 1:
            return f"{c} must be a number from 0.0 to 1.0 (got {v!r})"
    if not isinstance(grade.get("pass"), bool):
        return "pass must be true or false"
    return None


def import_grades(folder: str | Path, judge_name: str) -> tuple[dict[str, list[dict]], list[str]]:
    """Apply grades.json to copies of the source rows. Returns ({source file: rows}, problems).

    Rows whose packet was skipped or invalid keep `graded: False`, so summaries leave them out.
    """
    folder = Path(folder)
    key = json.loads((folder / "key.json").read_text(encoding="utf-8"))
    grades = json.loads((folder / "grades.json").read_text(encoding="utf-8"))
    files: dict[str, list[dict]] = {}
    problems = []
    for gid, ref in key.items():
        if ref["file"] not in files:
            files[ref["file"]] = json.loads(Path(ref["file"]).read_text(encoding="utf-8"))
        rows = files[ref["file"]]
        grade = grades.get(gid)
        why = "not graded" if grade is None else validate(grade)
        if why:
            problems.append(f"{gid} ({ref['id']} r{ref['repeat']}): {why}")
            continue
        rows[ref["row"]].update(parse_scores(grade), graded=True, judge=judge_name, judge_error="")
    return files, problems
