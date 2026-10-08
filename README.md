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
- **It remembers your preferences** — `/remember prefer papers from 2023 onwards` saves a note to
  your own `data/memory.md` (like Claude Code's memory file). The planner, every subagent, the
  report writer, `/ask` and `/read` keep it in mind. `/memory` shows and edits it.
- **Progress anyone can read** — one live line per research agent saying what it is doing in
  plain words ("Searching arXiv: …", "Reading: <paper title>"), collapsed into a short summary
  when the agents finish. The terminal tab shows `✦ <topic>` and spins while Jarvis works.
  `/verbose` lists every tool call instead.
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
- **Measured** — 111 offline tests + CI, and an eval harness with 28 benchmark queries, repeated
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

You need **Python 3.12+** and either a DeepSeek API key or [Ollama](https://ollama.com).

**Windows (PowerShell)**

```powershell
git clone https://github.com/shaunmarv3/Jarvis.git jarvis
cd jarvis
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -e .
copy .env.example .env          # then open .env and add your keys
.\jarvis.bat                    # start Jarvis
```

**macOS / Linux**

```bash
git clone https://github.com/shaunmarv3/Jarvis.git jarvis && cd jarvis
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
cp .env.example .env            # then open .env and add your keys
.venv/bin/jarvis                # start Jarvis
```

Then pick a brain:

- **DeepSeek (recommended for quality):** put your key in `.env` → `DEEPSEEK_API_KEY=sk-...`
  and set `DEFAULT_BACKEND=deepseek` (or switch at runtime with `/backend deepseek`).
- **Local:** install [Ollama](https://ollama.com) and `ollama pull qwen3.5:9b` (plus
  `ollama pull nomic-embed-text` for `/ask`).

### Run `jarvis` from any folder

`pip install -e .` creates a `jarvis` command inside `.venv`, but your terminal only finds it
while that venv is activated. To type just `jarvis` anywhere, put a launcher on your `PATH` once:

- **Windows:** `jarvis.bat` (in the project folder) starts Jarvis with the project's venv and
  UTF-8 output. Add the project folder to your user `PATH`: run this **from the project folder**
  in PowerShell, then open a new terminal:

  ```powershell
  [Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path", "User") + ";$PWD", "User")
  ```

  (Or: Start → "Edit environment variables for your account" → `Path` → New → the project folder.)
  Don't add `.venv\Scripts` itself: that would put the venv's `python` and `pip` ahead of your
  system ones.
- **macOS / Linux:** link the venv's command into a folder that is already on your `PATH`:

  ```bash
  mkdir -p ~/.local/bin && ln -s "$PWD/.venv/bin/jarvis" ~/.local/bin/jarvis
  ```

Check with `where jarvis` (Windows) or `which jarvis`. Jarvis always reads `.env` and `data/`
from the project folder, whichever folder you start it from.

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
jarvis ❯ how are RAG systems evaluated in research and in production?

✓ Understood the request
✓ Planned 3 research agents
╭─ research brief (breadth_first) ──────────────────────────────────────────╮
│ Plan — 3 research agents:                                                  │
│   1. RAG evaluation metrics & benchmarks                                   │
│      looks in: academic, code · up to 8 searches                           │
│   2. How production teams monitor RAG                                      │
│      looks in: web, community · up to 7 searches                           │
│   3. Open-source eval frameworks compared                                  │
│      looks in: code, web · up to 6 searches                                │
╰────────────────────────────────────────────────────────────────────────────╯
Proceed? Y = go · type to refine the brief & re-plan · n = cancel (y): y

⠹ Researching · 3 agents in parallel · 1 done          1m 42s · $0.031 · ctrl+c to pause
  ✓ RAG evaluation metrics & benchmarks   done · 30 sources          8 searches · 3 reads
  ⠹ How production teams monitor RAG      Searching Hacker News: "RAG evals in prod…"   5 searches
  ⠹ Open-source eval frameworks compared  Reading: Ragas: Automated Evaluation of R…   4 searches · 1 read
                                                      ↑ live: these lines update in place
✓ Researched with 3 agents · 21 searches · 6 reads · 95 sources (4m 12s)
    ✓ RAG evaluation metrics & benchmarks · 30 sources
    ✓ How production teams monitor RAG · 33 sources
    ✓ Open-source eval frameworks compared · 32 sources
● Found a gap → 1 more agent: RAGChecker metrics
✓ Followed up with 1 agent · 6 searches · 3 reads · +13 new sources (1m 05s)
✓ Wrote the report
✓ Linked 43 citations (150 sources retrieved)

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

`/verbose` switches back to one line per tool call (`[2] Searching arXiv: "RAG evaluation"`), and
every subagent's full transcript is saved under `data/runs/<run id>/`.

| Command                                       | Action                                                            |
| --------------------------------------------- | ----------------------------------------------------------------- |
| `<free text>`                                 | research anything — clarify → plan → confirm → research → report  |
| `/papers` · `/sources`                        | the last run's papers (cited first) · every source it retrieved   |
| `/read <N>` · `/ask [N] <q>`                  | summarize + index a paper · ask questions about one indexed paper |
| `/db` · `/use <N>` · `/forget <N>`            | manage the paper vector library                                   |
| `/save <N>`                                   | download paper N's PDF                                            |
| `/new`                                        | start a fresh topic (the next question won't build on the last report) |
| `/remember <note>`                            | save a preference every agent keeps in mind (see **Memory** below) |
| `/memory` · `/memory edit` · `/memory remove <N>` · `/memory clear` | show / edit your memory file |
| `/verbose [on\|off]`                          | list every tool call during research instead of one line per agent |
| `/resume`                                     | continue the last run after a crash / Ctrl+C (finished subagents are reused) |
| `/cost`                                       | token usage & estimated cost of the last run                      |
| `/dataset <q>` · `/inspect <id>` · `/web <q>` | quick one-off lookups                                             |
| `/backend ollama\|deepseek` · `/model <name>` | switch brains                                                     |

### Memory

Notes you want Jarvis to always know live in `data/memory.md`: a plain markdown file, one
`- ` line per note.

```text
jarvis ❯ /remember I'm new to ML: explain jargon briefly and keep reports short
remembered I'm new to ML: explain jargon briefly and keep reports short · note #1
jarvis ❯ /remember prefer papers from 2023 onwards; skip Medium posts
jarvis ❯ /memory
╭─ memory · every agent keeps these in mind ─╮
│   1. I'm new to ML: explain jargon briefly…  │
│   2. prefer papers from 2023 onwards; skip…  │
╰──────────────────────── 121 characters ─────╯
```

- **Who sees it:** the planner, every research subagent, the report writer, `/ask` and `/read`.
  Notes are treated as preferences: if one conflicts with your current question, the question
  wins, and notes are never cited as sources.
- **Yours only:** `data/` is git-ignored, so every person who clones Jarvis has their own memory
  file and it is never committed or pushed.
- **Edit it any way you like:** `/memory edit` opens it (in `$EDITOR`, else Notepad / nano), or
  open the file yourself. It is read fresh for every question, so changes apply right away.
- **Size:** the first 2,000 characters reach the agents on DeepSeek, 800 on Ollama (its context
  window is small). `/memory` warns you when your notes are longer.
- **Not in evals:** only the interactive CLI reads it; headless runs and the eval harness never
  do, so scores aren't shaped by one person's preferences.

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

**Grading without a judge API key.** `--judge none` saves the reports ungraded (research cost
only). `--export-grading` then writes one blinded packet per report: the judge's exact prompt,
named `g01, g02, …` and shuffled across the files, so the grader can't tell Jarvis from the
baseline by file name or order. Claude Code (on a normal subscription) or a person fills in
`grades.json` with the same rubric, and `--import-grades` merges the scores into new results
files that `--table` reads like any other:

```bash
python evals/run_evals.py --judge none --workers 6
python evals/run_evals.py --system baseline --judge none --workers 6
python evals/run_evals.py --export-grading evals/results/jarvis-X.json evals/results/baseline-Y.json
python evals/run_evals.py --import-grades evals/grading/<stamp> --label claude
```

Each score is reported as the mean over repeats ± the spread between repeats. The per-query
table lists which facts each report missed and the run-to-run noise. The **baseline** answers
"is the multi-agent cost worth it?". The **ablations** each switch off one design piece (the
follow-up wave, full-text reads, splitting into subagents) to show what it contributes.
**Human agreement** (Pearson/Spearman per criterion, pass/fail agreement) shows whether the judge
can be trusted at all.

### Results so far

Judged by **Claude Opus 5.5** (`claude-opus-5-5`) via Claude Code, grading blinded packets.

| version      | factual accuracy | citation accuracy | completeness | source quality | tool efficiency | **overall** | pass           | fact recall | truncated reports | avg citations | avg time | avg cost |
| ------------ | ---------------- | ----------------- | ------------ | -------------- | --------------- | ----------- | -------------- | ----------- | ----------------- | ------------- | -------- | -------- |
| **Jarvis**   | **0.89**         | 0.79              | **0.95**     | **0.79**       | **0.87**        | **0.86**    | **28/28**      | **0.98**    | 0                 | 29            | 199 s    | $0.062   |
| baseline     | 0.86             | **0.82**          | 0.37         | 0.44           | 0.39            | 0.58        | 6/28           | 0.66        | 0                 | 6             | 41 s     | $0.006   |

Run 2026-10-08 on DeepSeek: the full 28-query benchmark, one run per system, both graded by the
same judge with the same rubric. The baseline's reports are about as accurate as Jarvis's
(what they say matches their sources), but two searches usually aren't enough to answer the
question: it missed the QLoRA techniques, SWE-bench's instance count, Chinchilla's size and
ReAct's decision-making benchmarks. Jarvis's weakest criteria are **citation accuracy** (detailed
figures attached to a source whose snippet only shows the abstract) and **source quality**
(vendor/SEO blogs still get cited).

**How the grading was kept clean:**
- **Cross-family judge.** The reports were written by DeepSeek and graded by Claude, so the judge
  isn't scoring its own model family's output.
- **Blind.** Packets were named `g01…g56` and shuffled. The judge never opened `key.json` (the
  file mapping packets to systems) before `--import-grades` had merged the scores.
- **Packets only.** The judge read each packet's report and evidence as data to be checked, never
  as instructions. Retrieved text that was wrong or contaminated (e.g. a DOI that resolved to an
  unrelated document) lowered the citation score instead of being trusted.
- **One rubric for all.** Every packet was scored on the same five criteria on its own merits,
  without comparing packets.

Caveats: one repeat per system, so there's no ± spread yet. The blinding is partial, because each
packet's process-stats line ("single LLM call over 2 searches" vs "N subagents") reveals which
system wrote it. Ablations and human agreement haven't been run yet.

An earlier 4-query run (2026-10-02, same-family judge) scored the previous version 0.51 and the
rebuild 0.80.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q          # 111 tests, no network, no LLM (fake models + fake tools)
python scripts/smoke.py deepseek "your question"   # one real end-to-end run
python scripts/smoke.py ollama "q" --follow-up "compare that with X"   # also exercises memory
```

Covered: the subagent loop (results fed back, parallel tools, budget, fallback, source-id
resolution, failure handling, hung-tool timeout, wave time limit, context trimming), the source
registry & deterministic citations, plan validation, the follow-up wave and the lead's verdict,
per-subagent resume after a crash, follow-up questions, the full graph end-to-end, backend health
checks, HTTP retry/cache/rate-limit (GET and POST), PDF validation, vector-registry recovery,
every tool's offline behavior, DeepSeek thinking-mode payloads, cost accounting and its restore,
the eval tooling (fact recall, statistics, human agreement, baseline, judges), the user memory
file (and that it reaches every agent but no headless run), the live progress view, and the
slash-command dropdown.

## Project structure

```
Jarvis/                          (repo root)
├── jarvis/                      the Python package
│   ├── __main__.py              `python -m jarvis` entry point
│   ├── cli.py                   Rich REPL: plan approval, report, run summary, /memory commands
│   ├── progress.py              live progress view: one line per research agent, wave summaries
│   ├── memory.py                the user's memory file (data/memory.md)
│   ├── terminal.py              terminal tab title (✦ <topic>)
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
│   ├── events.py                structured progress events (nodes stay UI-free)
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
├── tests/                       111 offline tests (fake LLM + fake tools)
├── evals/                       queries.jsonl (28) · run_evals.py · judge.py · grading.py · baseline.py · human.py · results/
├── scripts/                     smoke.py (real run, optional follow-up) · test_commands.py (CLI commands)
├── .github/workflows/ci.yml     pytest on push to main / PRs
├── requirements.txt · requirements-dev.txt · pyproject.toml
├── .env.example                 copy to .env and add keys
├── jarvis.bat                   run `jarvis` from any Windows terminal
└── data/                        (git-ignored) memory.md, papers, reports, runs/, vectorstore, cache, checkpoints
```
