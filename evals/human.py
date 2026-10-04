"""Check the LLM judge against a human.

  python evals/human.py export evals/results/jarvis-X.json --n 10   # writes evals/human/<name>/
  ... grade each report in grades.csv (0.0-1.0 per criterion, pass = 1/0) ...
  python evals/human.py agree evals/results/jarvis-X.json evals/human/<name>/grades.csv

The export is blind: the sheet shows the question, the report and its evidence, never the
judge's scores. `agree` reports, per criterion, how closely the judge tracks the human
(Pearson and Spearman correlation, mean absolute difference) and how often they agree on
pass/fail. A judge that agrees with a person is what makes the README's numbers credible.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from judge import CRITERIA, JUDGE_PROMPT, mean, pearson, spearman  # noqa: E402

RUBRIC = JUDGE_PROMPT.split("Score each criterion from 0.0 to 1.0:", 1)[1].split("Reply with ONLY JSON", 1)[0].strip()


def pick(rows: list[dict], n: int, seed: int = 7) -> list[dict]:
    """A deterministic sample spread across query types (round-robin by type)."""
    rows = [r for r in rows if r.get("report")]
    by_type = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["id"], r.get("repeat", 0))):
        by_type[r.get("type", "")].append(r)
    rng = random.Random(seed)
    for group in by_type.values():
        rng.shuffle(group)
    out, groups = [], list(by_type.values())
    while len(out) < n and any(groups):
        for g in groups:
            if g and len(out) < n:
                out.append(g.pop())
    return out


def export(results: str, n: int, out_dir: str | None) -> int:
    rows = json.loads(Path(results).read_text(encoding="utf-8"))
    chosen = pick(rows, n)
    out = Path(out_dir) if out_dir else HERE / "human" / Path(results).stem
    out.mkdir(parents=True, exist_ok=True)
    (out / "INSTRUCTIONS.md").write_text(
        "# Grade each report\n\nOpen each `*.md` file, read the question, the report and the evidence, then fill "
        "in `grades.csv`: a score from 0.0 to 1.0 for every criterion, `pass` as 1 or 0, and optional notes. "
        "Grade without looking at the judge's scores.\n\n## Rubric (same as the LLM judge)\n\n" + RUBRIC + "\n",
        encoding="utf-8")
    with open(out / "grades.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "id", "repeat", *CRITERIA, "pass", "notes"])
        for r in chosen:
            name = f"{r['id']}-r{r.get('repeat', 0)}.md"
            (out / name).write_text(
                f"# {r['id']} (repeat {r.get('repeat', 0)})\n\n**Question:** {r['query']}\n\n"
                f"**Research process:** {r.get('process', '')}\n\n---\n\n{r['report']}\n\n---\n\n"
                f"## Evidence (the report's numbered sources)\n\n```\n{r.get('evidence', '')}\n```\n",
                encoding="utf-8")
            w.writerow([name, r["id"], r.get("repeat", 0), *[""] * len(CRITERIA), "", ""])
    print(f"wrote {len(chosen)} reports + grades.csv to {out}")
    return 0


def agreement(rows: list[dict], grades: list[dict]) -> dict:
    """Per-criterion agreement between judge scores (`rows`) and human grades."""
    judged = {(r["id"], int(r.get("repeat", 0))): r for r in rows}
    pairs = []
    for g in grades:
        key = (g["id"], int(g.get("repeat") or 0))
        if key in judged and all(str(g.get(c, "")).strip() for c in CRITERIA):
            pairs.append((judged[key], g))
    result = {"n": len(pairs), "criteria": {}}
    for c in CRITERIA + ["overall"]:
        if c == "overall":
            human = [mean([float(g[k]) for k in CRITERIA]) for _, g in pairs]
        else:
            human = [float(g[c]) for _, g in pairs]
        model = [float(j[c]) for j, _ in pairs]
        result["criteria"][c] = {
            "pearson": pearson(model, human), "spearman": spearman(model, human),
            "mean_abs_diff": mean([abs(a - b) for a, b in zip(model, human)]),
            "judge_mean": mean(model), "human_mean": mean(human),
        }
    passes = [(bool(j["pass"]), str(g.get("pass", "")).strip() in {"1", "true", "yes", "True"})
              for j, g in pairs if str(g.get("pass", "")).strip()]
    result["pass_agreement"] = mean([1.0 if a == b else 0.0 for a, b in passes]) if passes else float("nan")
    return result


def agree(results: str, grades_csv: str) -> int:
    rows = json.loads(Path(results).read_text(encoding="utf-8"))
    with open(grades_csv, newline="", encoding="utf-8") as f:
        grades = list(csv.DictReader(f))
    res = agreement(rows, grades)
    if not res["n"]:
        print("no fully graded rows found in the CSV")
        return 1
    print(f"Judge vs human on {res['n']} reports (pass/fail agreement {res['pass_agreement'] * 100:.0f}%)\n")
    print("| criterion | Pearson r | Spearman ρ | mean abs diff | judge mean | human mean |")
    print("|---|---|---|---|---|---|")
    for c, m in res["criteria"].items():
        print(f"| {c} | {m['pearson']:.2f} | {m['spearman']:.2f} | {m['mean_abs_diff']:.2f} | "
              f"{m['judge_mean']:.2f} | {m['human_mean']:.2f} |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("results")
    e.add_argument("--n", type=int, default=10)
    e.add_argument("--out", default=None)
    a = sub.add_parser("agree")
    a.add_argument("results")
    a.add_argument("grades")
    args = ap.parse_args()
    return export(args.results, args.n, args.out) if args.cmd == "export" else agree(args.results, args.grades)


if __name__ == "__main__":
    raise SystemExit(main())
