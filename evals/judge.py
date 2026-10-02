"""LLM-as-judge using Anthropic's published research-eval rubric.

One judge call per report (Anthropic found a single prompt with all criteria the
most consistent), scoring 0.0-1.0 per criterion plus a pass/fail. The judge sees the
report AND the retrieved evidence, so it can check claims against sources.
Deterministic metrics (citation validity, truncation, counts) are computed in code.
"""

from __future__ import annotations

import re

JUDGE_PROMPT = """You are an expert evaluator of AI research reports. Grade the report below for the
user's question, using the retrieved evidence (titles + abstracts/snippets of the sources the
report cites) to check claims.

QUESTION: {query}

REPORT:
<<<
{report}
>>>

EVIDENCE (the report's numbered sources):
<<<
{evidence}
>>>

Research process stats: {process}

Score each criterion from 0.0 to 1.0:
1. factual_accuracy — do the report's claims match the evidence? Penalize claims the evidence
   contradicts or that look invented (names, numbers, methods absent from the evidence).
2. citation_accuracy — do cited sources actually support the claims they're attached to?
   Uncited factual claims and citations to irrelevant sources lower this.
3. completeness — does it cover all aspects the question asks about?
4. source_quality — are sources relevant, primary and authoritative (papers, official docs,
   original authors) rather than off-topic, SEO content farms or weak secondary sources?
5. tool_efficiency — given the process stats, was the research effort reasonable for the
   question (not wildly too little or too much)?

Reply with ONLY JSON:
{{"factual_accuracy": x, "citation_accuracy": x, "completeness": x, "source_quality": x,
  "tool_efficiency": x, "pass": true|false, "notes": "<2-3 sentences: biggest strengths & problems>"}}"""

CRITERIA = ["factual_accuracy", "citation_accuracy", "completeness", "source_quality", "tool_efficiency"]


def split_sources(report: str) -> tuple[str, dict[int, str]]:
    """Split a report into (body, {n: source line}) from its '## Sources' section."""
    m = re.search(r"\n#{1,3}\s*(sources|references)\b(.*)\Z", report, re.IGNORECASE | re.DOTALL)
    if not m:
        return report, {}
    entries = {}
    for line in m.group(2).splitlines():
        lm = re.match(r"\s*[-*]?\s*\[(\d+)\]\s*(.+)", line)
        if lm:
            entries[int(lm.group(1))] = lm.group(2).strip()
    return report[: m.start()], entries


def deterministic_metrics(report: str) -> dict:
    body, entries = split_sources(report)
    body = re.sub(r"_\(saved to .*?\)_", "", body).strip()
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
    tail = body.rstrip()[-1:] if body.strip() else ""
    return {
        "words": len(body.split()),
        "citations_in_text": len(cited),
        "sources_listed": len(entries),
        "dangling_citations": len(cited - set(entries)),  # [n] with no matching source entry
        "looks_truncated": bool(body) and tail not in ".!?)]*_`|>\"'" and not body.rstrip().endswith("```"),
    }


def judge(llm, query: str, report: str, evidence: str, process: str) -> dict:
    from jarvis.utils import extract_json

    raw = llm.invoke(JUDGE_PROMPT.format(query=query, report=report[:60000], evidence=evidence[:150000],
                                         process=process)).content
    data = extract_json(raw)
    out = {}
    for c in CRITERIA:
        try:
            out[c] = max(0.0, min(1.0, float(data.get(c, 0))))
        except (TypeError, ValueError):
            out[c] = 0.0
    out["overall"] = round(sum(out[c] for c in CRITERIA) / len(CRITERIA), 3)
    out["pass"] = bool(data.get("pass", False))
    out["notes"] = str(data.get("notes", ""))[:600]
    return out
