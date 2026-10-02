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
[*How we built our multi-agent research system*](https://www.anthropic.com/engineering/multi-agent-research-system)
and its open-sourced [lead / subagent prompts](https://github.com/anthropics/claude-cookbooks/tree/main/patterns/agents/prompts):

- **A lead agent that plans like a researcher** — classifies your question
  (*straightforward / depth-first / breadth-first*), scales the number of subagents to it, and
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
- **Robust tools** — per-host rate limiting, retry with backoff on 429/5xx, on-disk cache.
- **Resumable** — runs are checkpointed to SQLite; after a crash or Ctrl+C, `/resume` continues.
- **Asks first** — vague questions get 1–3 clarifying questions; you approve the plan before any
  tool runs.
- **Chat with a paper** — `/read N` summarizes a paper and indexes it in its own vector store;
  `/ask` answers from that paper only.
- **Measured** — 56 offline unit tests + CI, and an LLM-judge eval harness using Anthropic's rubric.

## How it works

```
 you
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

| Role | Default model | Why | Price / 1M tokens (peak; off-peak is ½) |
|---|---|---|---|
| Lead — plan, review, report | `deepseek-v4-pro` (thinking) | few calls, needs judgment | $1.32 in · $3.96 out |
| Subagents — tool loops | `deepseek-flash` (no thinking) | many calls, needs speed | $0.30 in · $1.20 out |

A deep run is typically **$0.07–0.12** (see the eval table) — a $5 balance is ~50 runs. Change models with
`DEEPSEEK_LEAD_MODEL` / `DEEPSEEK_WORKER_MODEL`; `/cost` shows the last run's spend.

### Optional keys (everything works without them)

| Key | Effect |
|---|---|
| `SEMANTIC_SCHOLAR_API_KEY` | your own 1 req/s lane — the keyless shared pool is often rate-limited (429) |
| `TAVILY_API_KEY` / `EXA_API_KEY` | LLM-grade web search instead of DuckDuckGo |
| `GITHUB_TOKEN` | higher GitHub search limits |

## Usage

```text
> how are RAG systems evaluated in research and in production?

  research brief (breadth_first)
  Plan — 3 subagent(s):
    1. Academic RAG evaluation metrics & benchmarks       sources: academic, code · budget 8
    2. How production teams evaluate & monitor RAG        sources: web, community · budget 7
    3. Open-source evaluation frameworks compared         sources: code, web · budget 6
  Proceed? [Y/edit/n] y

  · dispatching 3 subagent(s) [parallel]
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

| Command | Action |
|---|---|
| `<free text>` | research anything — clarify → plan → confirm → research → report |
| `/papers` · `/sources` | the last run's papers (cited first) · every source it retrieved |
| `/read <N>` · `/ask [N] <q>` | summarize + index a paper · ask questions about one indexed paper |
| `/db` · `/use <N>` · `/forget <N>` | manage the paper vector library |
| `/save <N>` | download paper N's PDF |
| `/resume` | continue the last run after a crash / Ctrl+C |
| `/cost` | token usage & estimated cost of the last run |
| `/dataset <q>` · `/inspect <id>` · `/web <q>` | quick one-off lookups |
| `/backend ollama\|deepseek` · `/model <name>` | switch brains |

## Evals

`evals/run_evals.py` runs benchmark queries headlessly and grades each report with a single
LLM-judge call on Anthropic's rubric — **factual accuracy, citation accuracy, completeness,
source quality, tool efficiency** (0–1 each) — plus code-computed checks (truncation, dangling
citations, cost). It can also grade an older checkout, which is how the table below compares
this version against the previous one.

```bash
python evals/run_evals.py --backend deepseek --limit 4
python evals/run_evals.py --jarvis-path ../old-checkout --label baseline
```

| version | factual accuracy | citation accuracy | completeness | source quality | tool efficiency | **overall** | pass | truncated reports | avg citations | avg time | avg cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| previous (`main`) | 0.49 | 0.41 | 0.55 | 0.45 | 0.64 | **0.51** | 2/4 | 2 | 17 | 185 s | — |
| **this version** | **0.81** | **0.78** | **0.89** | **0.67** | **0.86** | **0.80** | **4/4** | **0** | 48 | 401 s | $0.087 |

Run 2026-10-02 on DeepSeek, 4 queries (`rag-eval, agent-bench, lora-qlora, vector-db`), same judge
(`deepseek-v4-pro`) for both versions. The previous version's failures were exactly the ones the
rebuild targets: off-topic sources (SVM, protein folding, sign language papers in a vector-DB
report), claims its sources didn't support, and truncated reports. Caveats: 4 queries is a small
sample, the judge is from the same model family as the agent (possible self-preference), and
re-judging the same report moves scores by about ±0.05. Weakest criterion now: **source quality**
(vendor/SEO blogs still get cited) — the next thing to improve (e.g. Tavily/Exa search, a domain
quality prior).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q          # 56 tests, no network, no LLM (fake models + fake tools)
python scripts/smoke.py deepseek "your question"   # one real end-to-end run
```

Covered: the subagent loop (results fed back, parallel tools, budget, fallback, source-id
resolution, failure handling), the source registry & deterministic citations, plan validation,
the follow-up wave, the full graph end-to-end, HTTP retry/cache/rate-limit, every tool's offline
behavior, DeepSeek thinking-mode payloads, and cost accounting.

## Project structure

```
Jarvis/                          (repo root)
├── jarvis/                      the Python package
│   ├── __main__.py              `python -m jarvis` entry point
│   ├── cli.py                   Rich REPL: live progress, plan approval, report, run summary
│   ├── repl.py                  prompt_toolkit input + live /command dropdown
│   ├── banner.py                JARVIS ASCII banner
│   ├── config.py                settings: lead/worker models, prices, budgets, optional keys
│   ├── llm.py                   get_llm(role=lead|worker) · DeepSeek thinking fix · usage/cost tracker
│   ├── graph.py                 LangGraph flow + SQLite checkpointer
│   ├── state.py                 the graph's shared state
│   ├── nodes.py                 clarify / plan (lead) / confirm / fanout / review / followup / report / cite
│   ├── subagent.py              the agentic tool loop for one delegated task
│   ├── sources.py               source registry + deterministic citations
│   ├── prompts.py               lead / subagent / review / report prompts (after Anthropic's cookbook)
│   ├── events.py                progress sink (subagents stream live lines to the CLI)
│   ├── headless.py              run a research request without a human (smoke tests, evals)
│   ├── store.py · qa.py · embeddings.py   per-paper vector stores & /ask Q&A
│   ├── utils.py                 JSON extraction, chunking, truncation
│   └── tools/
│       ├── __init__.py          tool registry (schemas the LLM sees + raw functions)
│       ├── _http.py             rate limiting · retries · disk cache
│       ├── arxiv_tool.py · semantic_scholar.py · openalex.py · crossref.py   papers
│       ├── reader.py            full-text paper reading with focused passage selection
│       ├── web.py               Tavily → Exa → DuckDuckGo search · page reading
│       ├── github.py            GitHub repo search
│       ├── datasets.py · hf_models.py · hf_inspect.py   HuggingFace datasets & models
│       ├── community.py         Hacker News (+ Reddit, currently blocked)
│       └── pdf_reader.py        map-reduce paper summaries for /read
├── tests/                       56 offline tests (fake LLM + fake tools)
├── evals/                       queries.jsonl · judge.py · run_evals.py · results/
├── scripts/                     smoke.py (one real run) · test_commands.py (CLI commands)
├── .github/workflows/ci.yml     pytest on push to main / PRs
├── requirements.txt · requirements-dev.txt · pyproject.toml
├── .env.example                 copy to .env and add keys
├── jarvis.bat                   run `jarvis` from any Windows terminal
└── data/                        (git-ignored) papers, reports, runs/, vectorstore, cache, checkpoints
```

## Roadmap

- [x] Phase 1–3 — search/fetch/read, RAG chat, datasets, multi-agent orchestrator-worker.
- [x] **Phase 4** — real agentic subagents, lead classification + rich delegations, gap-filling
  wave, all source types in one run, grounded citations, lead/worker model split + cost
  tracking, resumable runs, tests + CI, eval harness.
- [ ] Next — chat across *all* saved papers, citation-graph exploration (references/cited-by),
  Streamlit UI, MCP server, Reddit via authenticated API.

## License

MIT — personal project.
