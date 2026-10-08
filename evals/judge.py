"""LLM-as-judge using Anthropic's published research-eval rubric, plus checks done in code.

One judge call per report (Anthropic found a single prompt with all criteria the
most consistent), scoring 0.0-1.0 per criterion plus a pass/fail. The judge sees the
report AND the retrieved evidence, so it can check claims against sources.

Measured in code, no LLM involved: citation validity, truncation, and *fact recall*,
the share of known-correct facts a report states, for queries that have a ground-truth
answer (e.g. "QLoRA fine-tunes a 65B model on one 48GB GPU").

Judges from three providers can grade the same reports. Using a model family other than
the agent's (DeepSeek) removes self-preference bias. With `--judge none` the reports are
saved ungraded and graded later from blinded packets (evals/grading.py).
"""

from __future__ import annotations

import math
import os
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


def fact_recall(report: str, facts: list[list[str]] | None) -> tuple[float | None, list[str]]:
    """Share of fact groups the report body states (any alternative in a group counts).

    Matched case-insensitively at a word start, in the body only, so a paper title in the
    Sources list can't satisfy a fact the report never states. None when there are no facts.
    """
    if not facts:
        return None, []
    body, _ = split_sources(report or "")
    missing = []
    for group in facts:
        if not any(re.search(r"(?<![A-Za-z0-9])" + re.escape(alt), body, re.IGNORECASE) for alt in group):
            missing.append(group[0])
    return round(1 - len(missing) / len(facts), 3), missing


# --------------------------------------------------------------------------- statistics


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs: list[float]) -> float:
    """Sample standard deviation (0 for fewer than two values)."""
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):  # ties share their average rank
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(_ranks(xs), _ranks(ys))


# --------------------------------------------------------------------------- judges


class _AnthropicJudge:
    """Minimal adapter so a Claude judge has the same .invoke(prompt).content shape."""

    def __init__(self, model: str):
        import anthropic

        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def invoke(self, prompt: str):
        resp = self.client.messages.create(model=self.model, max_tokens=2000,
                                           messages=[{"role": "user", "content": prompt}])

        class _Out:
            content = "".join(b.text for b in resp.content if b.type == "text")

        return _Out()


JUDGE_DEFAULT_MODELS = {"deepseek": "deepseek-v4-pro", "anthropic": "claude-sonnet-5-5"}
JUDGE_KEYS = {"deepseek": "DEEPSEEK_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def make_judge(provider: str = "deepseek", model: str | None = None):
    """Build a judge model. Keys come from the environment / .env.

    "none" returns None: reports are saved ungraded, to be graded later outside the harness
    (see evals/grading.py: blinded packets for Claude Code or a person, then --import-grades).
    """
    provider = provider.lower()
    if provider == "none":
        return None
    if provider not in JUDGE_KEYS:
        raise SystemExit(f"Unknown judge provider '{provider}'. Use deepseek, anthropic, openai or none.")
    if not os.environ.get(JUDGE_KEYS[provider]):
        raise SystemExit(f"{JUDGE_KEYS[provider]} missing in .env — the {provider} judge needs it.")
    if provider == "anthropic":
        return _AnthropicJudge(model or JUDGE_DEFAULT_MODELS["anthropic"])
    if provider == "openai":
        if not model:
            raise SystemExit("Pass --judge-model for the OpenAI judge (e.g. the newest GPT model you can use).")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=os.environ["OPENAI_API_KEY"], timeout=300, max_retries=3)
    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(
        model=model or os.environ.get("DEEPSEEK_LEAD_MODEL", JUDGE_DEFAULT_MODELS["deepseek"]),
        api_key=os.environ["DEEPSEEK_API_KEY"],
        api_base=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        max_tokens=24000, extra_body={"thinking": {"type": "enabled"}}, reasoning_effort="low",
        timeout=300, max_retries=3,
    )


def judge_prompt(query: str, report: str, evidence: str, process: str) -> str:
    return JUDGE_PROMPT.format(query=query, report=report[:60000], evidence=evidence[:150000], process=process)


def judge(llm, query: str, report: str, evidence: str, process: str) -> dict:
    from jarvis.utils import extract_json

    return parse_scores(extract_json(llm.invoke(judge_prompt(query, report, evidence, process)).content))


def parse_scores(data: dict) -> dict:
    """A judge's JSON reply -> clamped criterion scores, overall, pass and notes."""
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
