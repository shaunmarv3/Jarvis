# Jarvis — Generalist Deep-Research Overhaul (Phase 4) — Design Spec

**Date:** 2026-07-17
**Status:** Implemented 2026-10-02 on branch `feat/anthropic-grade-agent` (see "Implementation notes" at the end)
**Owner:** Shaun (hobby project, resume-grade upgrade)

## 1. Goals & motivation

Jarvis today is an *academic paper agent*: subagents are prompt-biased toward
arXiv/Semantic Scholar/OpenAlex, web and dataset search mostly live as separate
slash commands, and web search itself is a scraper-tier stack (ddgs + trafilatura).

This phase turns Jarvis into a **generalist deep-research agent** faithful to
Anthropic's published multi-agent research system (the article
<https://www.anthropic.com/engineering/multi-agent-research-system> and its
open-sourced lead/subagent/citation prompts in the Anthropic cookbook), and adds
the things that make the project resume-strong for AI/ML + SWE fresher roles:

1. Research runs integrate **all source types** — academic, web (blogs/news/docs),
   GitHub code, HuggingFace models + datasets, community (HN/Reddit) — in one flow.
2. **Web search upgraded** from ddgs-scraping to an LLM-grade provider layer
   (Tavily → Exa → ddgs fallback), still runnable with zero API keys.
3. **Eval harness** using Anthropic's published methodology (LLM-as-judge,
   5-criterion rubric), with numbers in the README.
4. **Tests + GitHub Actions CI** with badge, and a **demo GIF** in the README.

Non-goals (backlog, explicitly out of scope): MCP server, Streamlit UI,
chat-across-all-papers, Kaggle search, citation-graph exploration, YouTube API
tool (videos are covered via the web-search providers, which index YouTube pages).

## 2. Guiding constraint — the Anthropic pattern stays intact

Decomposition remains **by facets of the question**, never by source type.
Sources are *fields inside a delegation* (Anthropic: "suggested starting points
and sources"), not the unit of decomposition. Scale-to-complexity is preserved
and strengthened. The graph shape is unchanged except one optional bounded
iteration edge:

```
clarify → (ask ⏸ → brief)? → plan → confirm ⏸ ⇄ plan
        → fanout → synthesize → (gap-check: fanout once more)? → finalize → cite
```

## 3. Lead agent upgrade (`nodes.py` / `prompts.py`)

### 3.1 Query classification (new)

`PLAN_PROMPT` first classifies the confirmed brief as one of:

- **straightforward** — single focused investigation → 1 subagent
- **depth-first** — one question, multiple perspectives → 2–4 subagents, one per perspective
- **breadth-first** — independent sub-questions → one subagent per sub-question, up to the ceiling

Guardrail (verbatim from Anthropic): *prefer fewer, more capable subagents over
many overly narrow ones.* Ceilings stay backend-dependent
(`MAX_SUBAGENTS_OLLAMA=5`, `MAX_SUBAGENTS_DEEPSEEK=10`).

### 3.2 Rich delegation spec (replaces `{objective, sub_query, tools}`)

Each subagent spec becomes:

```json
{
  "objective":     "one core objective for this subagent",
  "key_questions": ["1-3 concrete questions it must answer"],
  "sub_query":     "focused starting search query",
  "sources":       ["academic" | "web" | "code" | "models-data" | "community"],
  "tools":         "suggested tool names (comma-separated)",
  "tool_budget":   5,
  "output_format": "what the briefing back to the lead should contain",
  "boundaries":    "what is OUT of scope for this subagent"
}
```

The lead picks `sources` per facet (most facets get 2–3 types) and assigns
`tool_budget` by facet difficulty: simple <5, medium ~5, hard ~10, hard cap 15.
`plan_node` validates/defaults every field (missing sources → `["academic","web"]`,
missing budget → 5, budget clamped to [3, 15]).

### 3.3 Bounded lead iteration (new node behavior)

After `synthesize`, a **gap-check** step lets the lead review the merged findings
against the brief and optionally spawn ONE follow-up wave of at most
`MAX_FOLLOWUP_SUBAGENTS` (default 2) subagents targeting concrete gaps, then
re-synthesize. Strictly one extra wave (a `followup_done` flag in state), so runs
always terminate. Config: `LEAD_ITERATION=true` default on DeepSeek, off on
Ollama (patience). Implemented as a conditional edge `synthesize → (fanout | finalize)`.

## 4. Subagent upgrade (`subagent.py` / `prompts.py`)

### 4.1 Budget-driven OODA loop (replaces fixed 2 rounds)

- Loop continues while: tool-call budget remains AND the last round added new
  unique results (novelty check on paper ids / URLs), with a hard round cap
  (`SUBAGENT_MAX_ROUNDS=4`).
- Start-wide-then-narrow is kept: round 1 uses the lead's `sub_query` broadly;
  the reflect step proposes a sharper query each subsequent round.
- Stop condition in the prompt, per Anthropic: stop on diminishing returns.

### 4.2 Parallel tool execution (both backends)

Tool calls within a round run in a `ThreadPoolExecutor`. Tools are network-bound
HTTP calls, not GPU-bound — only LLM calls must serialize on the 4GB GPU, so this
is safe and fast on Ollama too. Per-tool timeout stays; a failed tool returns its
error text and never crashes the subagent (existing behavior kept).

### 4.3 Source-aware prompts + fallback

- `SUBAGENT_SYSTEM` rewritten: generalist researcher, covers its delegation's
  `sources` across its rounds; loses the hardwired "arXiv + Semantic Scholar +
  OpenAlex" framing.
- **Source-quality heuristics** added (Anthropic's exact lesson): prefer primary
  sources; distrust SEO content farms, aggregators, marketing language,
  speculation; flag conflicting info rather than presenting it as fact.
- `_FALLBACK_TOOLS` is no longer a constant: when the model declines to emit tool
  calls, the fallback toolset is **derived from the delegation's `sources`**:
  - academic → `search_arxiv`, `search_semantic_scholar`, `search_openalex`
  - web → `search_web`
  - code → `search_github`
  - models-data → `search_hf_models`, `search_datasets`
  - community → `search_hn`, `search_reddit`

## 5. New & upgraded tools (`tools/`)

All free; all keyless except optional Tavily/Exa upgrades. Every tool returns the
established structured shape `{"<kind>": [...], "text": "..."}` and degrades
gracefully (exception → error text, empty list).

### 5.1 Web search provider layer (`tools/web.py` rework)

- Provider chain: **Tavily → Exa → ddgs**, selected by which API keys exist in
  `.env` (`TAVILY_API_KEY`, `EXA_API_KEY`); no keys → ddgs, so Jarvis stays
  fully keyless-runnable.
- Tavily/Exa return extracted page content with results (solves the
  snippet-vs-full-text "extraction gap"); ddgs path keeps trafilatura extraction.
- `read_web` unchanged in interface; internally prefers the provider's extract
  endpoint when available, else trafilatura.
- Free tiers: Tavily 1,000 credits/mo; Exa ~1,000 one-time; ddgs unlimited-ish.

### 5.2 `tools/github.py` — `search_github(query, max_results=5)` (new)

GitHub REST `/search/repositories` (keyless: 10 req/min; optional `GITHUB_TOKEN`
raises limits). Returns repos: full name, description, stars, last-push date, URL.
Kind: `code`.

### 5.3 `tools/hf_models.py` — `search_hf_models(query, max_results=5)` (new)

`huggingface_hub.list_models` (same lib already used for datasets). Returns model
id, downloads, likes, pipeline tag, URL. Kind: `models`.

### 5.4 `tools/community.py` — `search_hn`, `search_reddit` (new)

- `search_hn(query, max_results=5)`: Algolia HN API (free, no key) — title,
  points, comment count, story/external URL, date.
- `search_reddit(query, max_results=5)`: Reddit public JSON search endpoint
  (no key; custom User-Agent required) — title, subreddit, score, comments, URL.
- Kind: `community`.

### 5.5 Registry (`tools/__init__.py`)

New `@tool` wrappers + `TOOL_FUNCS` entries for `search_github`,
`search_hf_models`, `search_hn`, `search_reddit`. `TOOL_SCHEMAS` grows to 14
tools. `PLAN_PROMPT`'s tool list updated to match.

## 6. State, aggregation, report & citations

### 6.1 State (`state.py`)

Add: `code: list` (repos), `models: list`, `community: list`,
`followup_done: bool`. Existing `web`, `datasets`, `papers` keep their meaning.
Delegation-spec change flows through `plan` (already a list of dicts).

### 6.2 Aggregation (`fanout_node`)

Aggregates and dedups all five result kinds (papers by id/title; others by URL).
Relevance filter (`rank_by_relevance`) extends beyond papers: web, code, and
community results are cosine-ranked against the query too (title + snippet),
with a shared `relevance_min`; models/datasets pass through capped.

### 6.3 Report (`finalize_node` / `FINALIZE_PROMPT`)

Integrated report sections, each rendered ONLY when it has content:

1. **TL;DR**
2. **Key findings** (synthesis across all sources)
3. **Papers** — title (year) — one line — link
4. **Repos & code** — repo — stars — one line — link
5. **Models & datasets** — id — one line — link
6. **Community pulse** — what practitioners say (HN/Reddit) — links
7. **Suggested next steps**

### 6.4 Citations (`cite_node`)

Source list passed to `CITE_PROMPT` now includes papers + web + code + models +
community entries (numbered continuously). Rules unchanged: cite only real
retrieved sources.

## 7. Eval harness (Anthropic's methodology)

New `evals/` directory:

- `evals/queries.jsonl` — 15–20 benchmark queries mirroring real usage
  (paper survey, comparison, dataset hunt, code hunt, practitioner question,
  straightforward fact).
- `evals/judge.py` — single-call LLM-as-judge (Anthropic's finding: one call,
  one prompt, most consistent). Rubric, scored 0.0–1.0 each + overall pass/fail:
  1. **Factual accuracy** — claims match sources
  2. **Citation accuracy** — cited sources support the claims
  3. **Completeness** — all requested aspects covered
  4. **Source quality** — primary/authoritative over SEO farms
  5. **Tool efficiency** — right tools, reasonable call count (read from the
     run's recorded tool log)
- `evals/run_evals.py` — headless runner: auto-confirms plans, runs each query
  end-to-end, judges the report, writes `evals/results/<timestamp>.json` and a
  markdown summary table. Judge backend defaults to DeepSeek (better judge, and
  avoids fully self-graded local runs) with Ollama fallback.
- README gets an **Eval results** section with the score table.

## 8. Tests + CI + polish

### 8.1 Tests (`tests/`, pytest — new dep, dev-only)

Fast, no-network, no-LLM unit tests (all external calls mocked/faked):

- `utils`: `extract_json` (fences, prose, garbage), `chunk_text`, `dedup_papers`, `truncate`
- `plan_node` spec validation: defaults, budget clamping, ceiling enforcement, empty-plan fallback
- Subagent loop: fallback-tool derivation from `sources`; novelty-based stop; budget cap (fake LLM + fake tools)
- Web provider chain: Tavily→Exa→ddgs selection by available keys (mocked)
- Tool output shapes: every `TOOL_FUNCS` entry returns `{"text", ...}` on failure
- Aggregation: multi-kind dedup + relevance-filter passthrough when embeddings unavailable

### 8.2 CI (`.github/workflows/ci.yml`)

GitHub Actions: Python 3.12, `pip install -r requirements.txt -e . pytest`,
run `pytest`, on push/PR to main. README gets the badge.

### 8.3 README & docs

- New architecture diagram (delegation fields, budgets, gap-check wave).
- Demo GIF of a real run (recorded after implementation; terminal recorder).
- Eval results table + CI badge.
- `info.md` gets a §16 documenting this phase; fix the stale §14 note
  (`relevance_min` is 0.60 in config, doc says 0.25).

### 8.4 Config additions (`.env.example` / `config.py`)

```ini
TAVILY_API_KEY=            # optional — upgrades web search
EXA_API_KEY=               # optional — semantic web search
GITHUB_TOKEN=              # optional — raises GitHub rate limits
LEAD_ITERATION=            # default: true on deepseek, false on ollama
MAX_FOLLOWUP_SUBAGENTS=2
SUBAGENT_MAX_ROUNDS=4
```

`SUBAGENT_ROUNDS` is superseded by budgets + `SUBAGENT_MAX_ROUNDS` (kept as a
deprecated alias for one release of the .env file, i.e., ignored with a note).

## 9. Known fixes rolled in

- `_FALLBACK_TOOLS` academic hardwiring (root cause of the arXiv bias) — §4.3.
- Sequential tool execution inside subagents — §4.2.
- `/web` and `/dataset` remain as quick commands, but their capabilities are now
  first-class inside research runs (the original complaint).
- `info.md` §14 stale `relevance_min` value — §8.3.

## 10. Success criteria

1. A query like "how do people evaluate RAG systems in production" produces a
   report citing papers AND blog posts AND at least one repo/community source.
2. A dataset-flavored query surfaces HF datasets/models inside the main run
   (no `/dataset` needed).
3. Subagents visibly stop early on diminishing returns; tool logs show budgets
   respected and parallel tool calls.
4. `pytest` green locally and in CI; badge on README.
5. Eval harness runs headless over all benchmark queries and produces the
   README score table.
6. Zero-key install still works end-to-end (ddgs fallback, keyless GitHub/HN/Reddit/HF).

## 11. Implementation notes (2026-10-02)

Built as specified, with these deviations — each forced by something found while building or testing:

- **The subagent loop was the real root cause.** Before adding sources, the subagent was rebuilt
  into a true tool loop: the old code never fed tool results back to the model (each round was a
  fresh single call), so "more tools" alone would not have helped.
- **Citations are deterministic, not an LLM pass.** Every result is registered as `S#`; agents cite
  ids; `sources.finalize_citations` renumbers and builds the Sources list from the registry. The old
  LLM citation rewrite truncated every saved report (Ollama `num_predict=1024`) and invented names.
- **Source kinds:** `academic | web | datasets | code | community` (models are folded into
  `datasets`). **Reddit is not offered to agents** — it now answers anonymous clients with HTTP 403;
  `search_reddit` stays in `TOOL_FUNCS` for when authenticated access is added.
- **Papers with Code removed** — shut down July 2025.
- **Budgets:** searches and full-text reads are budgeted separately (`subagent_max_reads=3`); in the
  first live run subagents spent their whole budget on one burst of parallel searches and never read
  a source. Novelty stop = two consecutive turns with no new sources → nudge to finish.
- **Lead/worker model split on DeepSeek** (`deepseek-v4-pro` lead with thinking at `reasoning_effort=low`,
  `deepseek-flash` workers without thinking). At effort `high` the lead spent its whole output budget
  on reasoning and returned an empty report; the report node now retries without thinking if empty.
  langchain-deepseek drops `reasoning_content`, which thinking-mode tool loops must echo back
  (HTTP 400 otherwise) — fixed in `llm._deepseek_class`.
- **Additions not in the spec:** per-host rate limiting + retries + disk cache (`tools/_http.py`),
  full-text paper reading with BM25 passage selection (`tools/reader.py`), SQLite checkpoints +
  `/resume`, `/sources`, `/cost`, per-run artifacts in `data/runs/<id>/`, token/cost tracking.
