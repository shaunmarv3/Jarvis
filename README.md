<div align="center">

```
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
```

**Your personal deep-research agent — in the terminal.**

One question in → a team of agents searches papers, the web, GitHub, HuggingFace and
Hacker News, reads the important sources in full, and writes a report where every
citation points at something it actually retrieved.

_Runs on local Ollama (free) or DeepSeek (≈ $0.07–0.12 per deep research run)._

![CI](https://github.com/shaunmarv3/Jarvis/actions/workflows/ci.yml/badge.svg)

</div>

---

## ✨ What it does

Jarvis implements the orchestrator-worker design from Anthropic's
[_How we built our multi-agent research system_](https://www.anthropic.com/engineering/multi-agent-research-system)
and its open-sourced [lead / subagent prompts](https://github.com/anthropics/claude-cookbooks/tree/main/patterns/agents/prompts):

- **A lead agent that plans like a researcher** — classifies your question
  (_straightforward / depth-first / breadth-first_), scales the number of subagents to it, and
  writes each one a full delegation: objective, key questions, source types, tool budget,
  output format and boundaries.
- **Subagents that are real agents** — each runs an observe → orient → decide → act loop in its
  own context: it sees every tool result, runs 2–3 tools **in parallel** per turn, rephrases
  when results are off-topic, opens the key papers/pages in **full text**, and stops when
  searches stop finding anything new (or its budget runs out).
- **Every source type in one run** — papers (arXiv, Semantic Scholar, OpenAlex, Crossref),
  web (Tavily → Exa → DuckDuckGo), code (GitHub), datasets & models (HuggingFace) and
  practitioner discussion (Hacker News). The lead mixes them per facet; no separate commands.
- **Gap-filling** — after the subagents report, the lead reviews the findings against the brief
  and may dispatch **one** targeted follow-up wave (bounded, so runs always terminate).
- **Citations that can't be fabricated** — every result gets a source id (`S12`); agents cite ids;
  the final `[1] [2]` numbering and Sources list are generated **in code from the registry**.
  Unknown ids are dropped and counted.
- **Smart model split** — on DeepSeek a strong model (`deepseek-v4-pro`, thinking on) leads and a
  cheap fast one (`deepseek-flash`) runs the many subagent turns. Every run prints its token
  usage and estimated cost.
- **Follow-up questions** — ask "now compare that with X" after a report: the lead resolves what
  "that" means, plans only what's new, and can cite the previous report's sources. `/new` starts
  a fresh topic.
- **Robust tools** — every API call goes through one layer with per-host rate limiting, retry with
  backoff on 429/5xx, and an on-disk cache. A hung tool is abandoned after 90 s, and each wave of
  subagents has a time limit, after which they write up what they have.
- **Fails fast, resumes cheaply** — before a run Jarvis checks that Ollama (and its models) or
  DeepSeek (and your key) actually work, and says how to fix it if not. Runs are checkpointed to
  SQLite, and each subagent's result is saved the moment it finishes, so `/resume` after a crash
  or Ctrl+C only redoes the unfinished subagents. Token spend is restored too.
- **Fits small context windows** — on Ollama, prompts are measured and evidence is trimmed in
  steps until it fits next to the answer, instead of letting the server silently cut the prompt.
- **Asks first** — vague questions get 1–3 clarifying questions; you approve the plan before any
  tool runs.
- **Chat with a paper** — `/read N` summarizes a paper and indexes it in its own vector store;
  `/ask` answers from that paper only.
- **Measured** — 92 offline tests + CI, and an eval harness with 28 benchmark queries, repeated
  runs, known-answer fact checks, ablations, a simple baseline, a cross-family judge and a
  human-agreement check.

> **Is it RAG?** Not during research. Subagents read search results and open the key sources in
> full (BM25-selected passages), and the lead writes from their findings. Embeddings and a vector
> store are used only for `/read` + `/ask`, which chat with one paper at a time.

## How it works

```
 your request
  │
  ▼
 clarify ──vague?──► ask you 1-3 questions ──► brief ──┐
  │ specific                                           │
  ▼                                                    │
 plan (LEAD) ◄─────────────────────────────────────────┘
  │  classifies the query (straightforward / depth-first / breadth-first)
  │  and writes one delegation per subagent
  ▼
 confirm (you) ──edit──► back to plan with your changes
  │ go
  ├───────────────────┬───────────────────┐
  ▼                   ▼                   ▼
 subagent 1          subagent 2     …    subagent N      worker model, own context;
  │                   │                   │                parallel on DeepSeek, one by one on Ollama
  ├───────────────────┴───────────────────┘
  ▼
 review (LEAD) ──gaps?──► follow-up wave (≤ 2 subagents, runs once) ──┐
  │ ok                                                                │
  ▼                                                                   │
 report (LEAD) ◄──────────────────────────────────────────────────────┘
  │  every claim keeps its [S#] source tag
  ▼
 cite (code) ──► [S#] → [1] [2] … + Sources list, built from the registry

 inside every subagent:

   ┌─────────────────────────────────────────────────────┐
   │  LLM turn ──► 2-3 tools in parallel ──► results     │  repeats until complete_task,
   │     ▲                                      │        │  budget used, or searches
   │     └────────── reads its own results ◄────┘        │  stop finding anything new
   └─────────────────────────────────────────────────────┘
   every result ──► shared SOURCE REGISTRY (S1, S2, …) ──► used by cite
```

Subagents run **in parallel on DeepSeek** and **sequentially on local Ollama** (one GPU);
tool calls inside a subagent run in parallel on both. Each run's plan, sources, subagent
transcripts and report are saved under `data/runs/<run-id>/`.

**Follow-ups.** After a run, the CLI keeps that report's body (citations mapped back to `S#` ids)
and its cited sources. Your next question goes to `clarify` together with them, and the lead
decides whether it's a follow-up. If it is, the run starts with those sources already in the
registry (same ids), `plan` is told what's already known, and `report` may cite it. If not,
nothing carries over. `/new` drops the context explicitly.

## Quick start

```bash
git clone https://github.com/shaunmarv3/Jarvis.git jarvis && cd jarvis
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .                 # registers the `jarvis` command
cp .env.example .env             # then edit .env
jarvis                           # or: python -m jarvis
```

- **DeepSeek (recommended for quality):** put your key in `.env` → `DEEPSEEK_API_KEY=sk-...`
  and set `DEFAULT_BACKEND=deepseek` (or switch at runtime with `/backend deepseek`).
- **Local:** install [Ollama](https://ollama.com) and `ollama pull qwen3.5:9b` (plus
  `ollama pull nomic-embed-text` for `/ask`).

On Windows you can also put the project folder on `PATH` and run `jarvis.bat` from any terminal.

### Models & cost (DeepSeek)

| Role                        | Default model                  | Why                       | Price / 1M tokens (peak; off-peak is ½) |
| --------------------------- | ------------------------------ | ------------------------- | --------------------------------------- |
| Lead — plan, review, report | `deepseek-v4-pro` (thinking)   | few calls, needs judgment | $1.32 in · $3.96 out                    |
| Subagents — tool loops      | `deepseek-flash` (no thinking) | many calls, needs speed   | $0.30 in · $1.20 out                    |

A deep run is typically **$0.07–0.12** (see the eval table) — a $5 balance is ~50 runs. Change models with
`DEEPSEEK_LEAD_MODEL` / `DEEPSEEK_WORKER_MODEL`; `/cost` shows the last run's spend.

### Optional keys (everything works without them)

| Key                              | Effect                                                                      |
| -------------------------------- | --------------------------------------------------------------------------- |
| `SEMANTIC_SCHOLAR_API_KEY`       | your own 1 req/s lane — the keyless shared pool is often rate-limited (429) |
| `TAVILY_API_KEY` / `EXA_API_KEY` | LLM-grade web search instead of DuckDuckGo                                  |
| `GITHUB_TOKEN`                   | higher GitHub search limits                                                 |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | evals only: a judge from another model family (`--judge anthropic`)   |

## Usage

```text
> how are RAG systems evaluated in research and in production?

  research brief (breadth_first)
  Plan — 3 subagent(s):
    1. Academic RAG evaluation metrics & benchmarks       sources: academic, code · budget 8
    2. How production teams evaluate & monitor RAG        sources: web, community · budget 7
    3. Open-source evaluation frameworks compared         sources: code, web · budget 6
  Proceed? [Y/edit/n] y

  · dispatching 3 subagent(s) (parallel)
  → subagent [1/3] Academic RAG evaluation metrics & benchmarks · budget 8
      [1/3] search_arxiv(query=RAG evaluation metrics)
      [1/3] search_semantic_scholar(query=RAG evaluation benchmark)
      [2/3] search_web(query=RAG evaluation in production)
      [1/3] read_paper(paper=S4, focus=faithfulness metric definition)
  ↳ subagent [1/3] done · 8 searches + 3 reads · 30 sources
  ⠋ lead reviewing findings for gaps… 4m 12s · $0.061 so far        ← live spinner
  · lead found gaps → 1 follow-up subagent(s): RAGChecker primary source & metrics
  · lead wrote the report
  · citations attached

  ╭─ report ─────────────────────────────────────────────────╮
  │ # How RAG systems are evaluated …  [1][2] … ## Sources   │
  ╰──────────────────────────────────────────────────────────╯
  ┌──────────────────────── run summary ─────────────────────────┐
  │ cost        $0.097  (estimated at peak rates; off-peak is half)
  │   lead      $0.067 · 4 calls · 13.6k in / 12.6k out (deepseek-v4-pro)
  │   subagents $0.030 · 24 calls · 146.1k in / 7.8k out (deepseek-flash)
  │ time        7m 31s
  │ research    4 subagents · 49 tool calls · depth_first
  │ sources     43 cited of 150 retrieved
  └───────────────────────────────────────────────────────────────┘
```

| Command                                       | Action                                                            |
| --------------------------------------------- | ----------------------------------------------------------------- |
| `<free text>`                                 | research anything — clarify → plan → confirm → research → report  |
| `/papers` · `/sources`                        | the last run's papers (cited first) · every source it retrieved   |
| `/read <N>` · `/ask [N] <q>`                  | summarize + index a paper · ask questions about one indexed paper |
| `/db` · `/use <N>` · `/forget <N>`            | manage the paper vector library                                   |
| `/save <N>`                                   | download paper N's PDF                                            |
| `/new`                                        | start a fresh topic (the next question won't build on the last report) |
| `/resume`                                     | continue the last run after a crash / Ctrl+C (finished subagents are reused) |
| `/cost`                                       | token usage & estimated cost of the last run                      |
| `/dataset <q>` · `/inspect <id>` · `/web <q>` | quick one-off lookups                                             |
| `/backend ollama\|deepseek` · `/model <name>` | switch brains                                                     |

## Evals

`evals/run_evals.py` runs the benchmark headlessly and grades every report two ways:

- **An LLM judge** makes one call per report on Anthropic's rubric: **factual accuracy, citation
  accuracy, completeness, source quality, tool efficiency** (0–1 each) plus pass/fail. It sees the
  report *and* the retrieved evidence. `--judge anthropic` (or `openai`) uses a model family other
  than the agent's, which removes self-preference.
- **Checks done in code:** **fact recall** on 24 queries with known answers (e.g. QLoRA: NF4,
  double quantization, paged optimizers, 65B on one 48GB GPU; matched in the report body, not the
  source titles), truncation, dangling citations, cost and time.

The benchmark is `evals/queries.jsonl`: 28 queries across surveys, comparisons, dataset and code
hunts, practitioner questions, exact-paper summaries and 12 fact lookups.

```bash
python evals/run_evals.py --repeats 3 --workers 3                 # Jarvis: mean ± spread over 3 runs
python evals/run_evals.py --system baseline --repeats 3           # 1 web + 1 arXiv search + 1 LLM call
python evals/run_evals.py --ablation single_agent                 # also: no_followup, no_reads
python evals/run_evals.py --rejudge evals/results/jarvis-X.json --judge anthropic --label jarvis-claude
python evals/run_evals.py --table evals/results/*.json            # one comparison table
python evals/human.py export evals/results/jarvis-X.json --n 10   # blind sheet for a human grader
python evals/human.py agree  evals/results/jarvis-X.json evals/human/jarvis-X/grades.csv
```

Each score is reported as the mean over repeats ± the spread between repeats. The per-query
table lists which facts each report missed and the run-to-run noise. The **baseline** answers
"is the multi-agent cost worth it?". The **ablations** each switch off one design piece (the
follow-up wave, full-text reads, splitting into subagents) to show what it contributes.
**Human agreement** (Pearson/Spearman per criterion, pass/fail agreement) shows whether the judge
can be trusted at all.

### Results so far

| version           | factual accuracy | citation accuracy | completeness | source quality | tool efficiency | **overall** | pass    | truncated reports | avg citations | avg time | avg cost |
| ----------------- | ---------------- | ----------------- | ------------ | -------------- | --------------- | ----------- | ------- | ----------------- | ------------- | -------- | -------- |
| previous (`main`) | 0.49             | 0.41              | 0.55         | 0.45           | 0.64            | **0.51**    | 2/4     | 2                 | 17            | 185 s    | —        |
| **phase 4**       | **0.81**         | **0.78**          | **0.89**     | **0.67**       | **0.86**        | **0.80**    | **4/4** | **0**             | 48            | 401 s    | $0.087   |

Run 2026-10-02 on DeepSeek with the earlier harness: 4 queries (`rag-eval, agent-bench, lora-qlora,
vector-db`), one run each, same judge (`deepseek-v4-pro`) for both versions. The previous version's
failures were exactly the ones the rebuild targets: off-topic sources, claims its sources didn't
support, and truncated reports. These numbers are a small sample, from a same-family judge, with
about ±0.05 re-judge noise, which is why the harness above was built. Full results on the
28-query benchmark (3 repeats, baseline, ablations, cross-family judge) will replace this table.
Weakest criterion so far: **source quality** (vendor/SEO blogs still get cited).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q          # 92 tests, no network, no LLM (fake models + fake tools)
python scripts/smoke.py deepseek "your question"   # one real end-to-end run
python scripts/smoke.py ollama "q" --follow-up "compare that with X"   # also exercises memory
```

Covered: the subagent loop (results fed back, parallel tools, budget, fallback, source-id
resolution, failure handling, hung-tool timeout, wave time limit, context trimming), the source
registry & deterministic citations, plan validation, the follow-up wave and the lead's verdict,
per-subagent resume after a crash, follow-up questions, the full graph end-to-end, backend health
checks, HTTP retry/cache/rate-limit (GET and POST), PDF validation, vector-registry recovery,
every tool's offline behavior, DeepSeek thinking-mode payloads, cost accounting and its restore,
and the eval tooling (fact recall, statistics, human agreement, baseline, judges).

## Project structure

```
Jarvis/                          (repo root)
├── jarvis/                      the Python package
│   ├── __main__.py              `python -m jarvis` entry point
│   ├── cli.py                   Rich REPL: live progress, plan approval, report, run summary
│   ├── repl.py                  prompt_toolkit input + live /command dropdown
│   ├── banner.py                JARVIS ASCII banner
│   ├── config.py                settings: lead/worker models, prices, budgets, optional keys
│   ├── llm.py                   get_llm(role) · backend health check · context budget · usage/cost tracker
│   ├── graph.py                 LangGraph flow + SQLite checkpointer
│   ├── state.py                 the graph's shared state
│   ├── nodes.py                 clarify / plan / confirm / fanout / review / followup / report / cite · follow-up memory
│   ├── subagent.py              the agentic tool loop for one delegated task
│   ├── sources.py               source registry + deterministic citations
│   ├── prompts.py               lead / subagent / review / report prompts (after Anthropic's cookbook)
│   ├── events.py                progress sink (subagents stream live lines to the CLI)
│   ├── headless.py              run a research request without a human (smoke tests, evals)
│   ├── store.py · qa.py · embeddings.py   per-paper vector stores & /ask Q&A
│   ├── utils.py                 JSON extraction, chunking, truncation, atomic JSON writes
│   └── tools/
│       ├── __init__.py          tool registry (schemas the LLM sees + raw functions)
│       ├── _http.py             GET/POST with rate limiting · retries · disk cache
│       ├── arxiv_tool.py · semantic_scholar.py · openalex.py · crossref.py   papers
│       ├── reader.py            full-text paper reading with focused passage selection
│       ├── web.py               Tavily → Exa → DuckDuckGo search · page reading
│       ├── github.py            GitHub repo search
│       ├── datasets.py · hf_models.py · hf_inspect.py   HuggingFace datasets & models
│       ├── community.py         Hacker News (+ Reddit, currently blocked)
│       └── pdf_reader.py        map-reduce paper summaries for /read
├── tests/                       92 offline tests (fake LLM + fake tools)
├── evals/                       queries.jsonl (28) · run_evals.py · judge.py · baseline.py · human.py · results/
├── scripts/                     smoke.py (real run, optional follow-up) · test_commands.py (CLI commands)
├── .github/workflows/ci.yml     pytest on push to main / PRs
├── requirements.txt · requirements-dev.txt · pyproject.toml
├── .env.example                 copy to .env and add keys
├── jarvis.bat                   run `jarvis` from any Windows terminal
└── data/                        (git-ignored) papers, reports, runs/, vectorstore, cache, checkpoints
```
