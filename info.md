# Jarvis — Project Knowledge Dump (`info.md`)

> This file is the single source of truth for *everything we know* about this project: the vision,
> every decision and the reasoning behind it, the environment, the architecture, every file's job,
> every tool, the prompts, the gotchas, and the research that grounds the design. If you (or a future
> session) read only one file, read this one.
>
> **Status:** Built & verified through **Phase 3** (multi-agent orchestrator-worker). See §14 (Phase 2)
> and §15 (Phase 3) for what's implemented; §1–§13 capture the original vision & plan.
> **Date started:** 2026-06-18 / 2026-06-19.
> **Owner:** Shaun (hobby project).

---

## 1. Vision & purpose

A **personal AI research agent** that lives in the terminal and does research grunt-work like a
tireless human researcher. The user gives it a topic or a paper; it confirms a short plan, then
autonomously searches, reads, reflects on gaps, and searches again until it can hand back a grounded
report with real citations.

This is a **hobby project** — the priority is that it's *genuinely good and fun to use*, not
enterprise-grade. It should feel "alive": loop dynamically, be dynamic/agentic, look cool.

### Core requirements (from the user, verbatim intent)
1. Search for **ideas** for the user.
2. Pull **related research papers** OR pull the **exact** research paper (by title/URL/ID).
3. **Read** papers and **tell/explain** them to the user.
4. Search for **datasets**.
5. Do "**rest all research-related stuff**".
6. Run on **either** the user's **DeepSeek token** **or locally via Ollama** — switchable.
7. Built on a **real framework** (LangChain/LangGraph) — "the best one"; must loop "**recursively /
   like a perfect madman**", be "**awesome dynamic**".
8. Encouraged to **search online for the best existing research agents and adapt/copy** them.
9. Must **write the README first** when building.
10. Must **search online for the best free, reusable tools** (GitHub etc.) before building.
11. CLI must **look cool** — ASCII "JARVIS" banner (hermes-CLI / polymarket-CLI vibe).
12. Must **ask for a brief / clarify with the user before it starts doing anything**.

---

## 2. Confirmed decisions (and why)

| Decision | Choice | Why |
|---|---|---|
| Interface | **CLI** (Rich REPL) | User chose CLI. Fast, scriptable, fits a terminal-first workflow. |
| LLM backend | **Switchable Ollama ⇄ DeepSeek** | "if I want deepseek then deepseek if not then ollama local." Both speak the OpenAI-compatible tool-calling protocol, so one code path. |
| Default backend | **Ollama (`qwen3.5:9b`)** | Free, private, offline. DeepSeek is opt-in via key/flag. |
| Capabilities | **Paper search+fetch, read+summarize, dataset search** | User picked "combination of 1,2,3". RAG library (#4) deferred to Phase 2. |
| Framework | **LangGraph** | User asked for the best framework with recursive looping. LangGraph is purpose-built for cyclic, stateful agent loops + human-in-the-loop. Still the consensus "best for control" in 2026. |
| Build order | **README first**, then code | Explicit user instruction. |
| CLI look | **pyfiglet "JARVIS" + Rich** | User wants the hermes/polymarket ASCII aesthetic. |
| Clarify-first | **Human-in-the-loop confirm node** | User: "it must ask me brief/clarify before it starts to do anything." |

### Design lineage (what we're adapting)
- **LangChain Open Deep Research** — `clarify → research brief → supervisor → iterative sub-research → report`. We borrow *clarify + brief*.
- **Ollama Deep Researcher / IterDRAG** — `query → search → summarize → reflect on knowledge gaps → new query → repeat N times → report`. We borrow the *reflective loop* (this is the "madman" loop).
- We **specialize the "search" step for academic sources** (arXiv/S2/OpenAlex/Crossref + PDF reading + dataset hubs) instead of generic web search.

---

## 3. Environment (the user's actual machine)

- **OS:** Windows 11 Home Single Language (10.0.26200). Shell: PowerShell 5.1 (primary); Git Bash available.
- **Project dir:** `D:\jarvis` (was empty; now has `README.md` + `info.md`). **Not** a git repo yet.
- **Python:** 3.12.5 and 3.12.10 present. `pip` 26.0.1.
- **Ollama models installed** (from `ollama list`):
  | Model | Size | Role here |
  |---|---|---|
  | `qwen3.5:9b` | 6.6 GB | **The brain** — reasoning + tool-calling. |
  | `qwen2.5-coder:7b` | 4.7 GB | Code-related tasks (optional). |
  | `nomic-embed-text` | 274 MB | **Embeddings** — ranking now / RAG in Phase 2. |
  | `codegate-reviewer` | 1.9 GB | Code review (unused here). |
- **DeepSeek:** user has a token. API is OpenAI-compatible (`deepseek-chat` model). Stored in `.env` as `DEEPSEEK_API_KEY`.

> ⚠️ Tool-calling note: `qwen3.5` / `qwen2.5` support Ollama tool calling. Verify the bound-tools
> path works against the exact installed tag before relying on it; if a model misbehaves with tools,
> fall back to a prompt-based ReAct loop.

---

## 4. Architecture

LangGraph orchestrator: a tool-calling agent wrapped in a clarify-confirm-then-reflect loop.

```
            ┌──────────────┐
   topic →  │   clarify    │  classify intent (find / pull exact / read / datasets),
            └──────┬───────┘  draft a 2-4 line research brief + first query
                   ▼
            ┌──────────────┐   ⏸ HUMAN-IN-THE-LOOP (LangGraph interrupt):
            │ confirm brief│      show brief + planned tools, wait for go/edit.
            └──────┬───────┘      NOTHING runs un-confirmed.
                   ▼
            ┌──────────────┐   LLM.bind_tools(ALL_TOOLS); picks & calls tools (ToolNode)
            │  act (agent) │←──── arxiv / s2 / openalex / crossref / read_pdf / datasets
            └──────┬───────┘
                   ▼
            ┌──────────────┐   fold new evidence into running `findings`
            │  synthesize  │
            └──────┬───────┘
                   ▼
            ┌──────────────┐   gaps remain AND loop_count < MAX_LOOPS?
            │   reflect    │ ── yes ──► new query, back to `act`
            └──────┬───────┘
                   │ no / max reached
                   ▼
            ┌──────────────┐   TL;DR + key papers (links) + datasets + suggested next steps
            │  finalize    │
            └──────────────┘
```

### Loop safety
- Bounded by **`MAX_LOOPS`** (config, default 4) **and** LangGraph's **`recursion_limit`**.
- A `MemorySaver` checkpointer gives per-session memory and enables the `interrupt()` resume.

### State (`AgentState`, a TypedDict)
- `messages` — chat history (LangChain messages).
- `intent` — find_papers | pull_exact | read | find_datasets (classified).
- `brief` — the confirmed research brief.
- `queries` — list of search queries issued.
- `papers` — found paper metadata (title, authors, year, abstract, ids, pdf_url, source).
- `read_papers` — id → parsed summary.
- `datasets` — found dataset metadata.
- `findings` — running synthesized summary.
- `gaps` — open knowledge gaps from reflect.
- `loop_count`, `max_loops`.

---

## 5. Project layout & per-file responsibilities

```
D:\jarvis\
├── jarvis/
│   ├── __init__.py
│   ├── config.py          # pydantic-settings: backend, model names, keys, max_loops, paths
│   ├── llm.py             # get_llm(backend) -> ChatOllama | ChatDeepSeek (both .bind_tools)
│   ├── state.py           # AgentState TypedDict (see §4)
│   ├── prompts.py         # clarify / reflect / synthesize / finalize prompt templates
│   ├── nodes.py           # node functions (clarify, confirm, act, synthesize, reflect, finalize)
│   ├── graph.py           # build_graph(): nodes + conditional edges + checkpointer
│   ├── banner.py          # pyfiglet "JARVIS" + rich styling (fallback: static ASCII art)
│   ├── cli.py             # Rich REPL entrypoint + slash commands
│   └── tools/
│       ├── __init__.py        # ALL_TOOLS = [...]
│       ├── arxiv_tool.py      # search_arxiv(), fetch_arxiv(id|url)
│       ├── semantic_scholar.py# search_semantic_scholar(), resolve_paper_by_title()
│       ├── openalex.py        # search_openalex()  (pyalex)
│       ├── crossref.py        # resolve_doi()      (habanero)
│       ├── pdf_reader.py      # read_paper(path|id) -> map-reduce summary (pymupdf4llm)
│       └── datasets.py        # search_datasets() -> HF Hub + Papers with Code
├── data/
│   ├── papers/            # downloaded PDFs
│   └── cache/            # parsed text + search-result cache
├── scripts/
│   └── smoke.py          # headless build+run one canned query, assert non-empty report
├── requirements.txt
├── .env.example          # DEEPSEEK_API_KEY, DEFAULT_BACKEND, OLLAMA_MODEL, MAX_LOOPS
├── README.md
└── info.md               # ← this file
```

---

## 6. Tools / data sources (all free, chosen after 2026 research)

| Tool file | Function(s) | Backing API / lib | Key? |
|---|---|---|---|
| `arxiv_tool.py` | `search_arxiv`, `fetch_arxiv` | `arxiv` PyPI pkg → arXiv API | No |
| `semantic_scholar.py` | `search_semantic_scholar`, `resolve_paper_by_title` | S2 Graph API (`api.semanticscholar.org/graph/v1`) | No (1 req/s) |
| `openalex.py` | `search_openalex` | `pyalex` → OpenAlex (250M+ works) | No (polite pool: set email) |
| `crossref.py` | `resolve_doi` | `habanero` → Crossref REST | No |
| `pdf_reader.py` | `read_paper` | `pymupdf4llm` (PDF→markdown) + active LLM (map-reduce summarize) | No |
| `datasets.py` | `search_datasets` | `huggingface_hub.list_datasets` + Papers with Code REST | No |

**Why these:** all free, no/low-key, well-maintained, JSON/clean output, no brittle scraping.
OpenAlex + Crossref were added specifically because mid-2026 surveys rank them as the best free
scholarly APIs for broad coverage + citation data, complementing arXiv (preprints) and Semantic
Scholar (CS-heavy + citations).

**Backend switching (`llm.py`):**
- `ollama` → `ChatOllama(model=settings.ollama_model)` (default `qwen3.5:9b`).
- `deepseek` → `ChatDeepSeek(model="deepseek-chat", api_key=settings.deepseek_api_key)`.
- Both expose `.bind_tools(ALL_TOOLS)`. Chosen by `DEFAULT_BACKEND` env → `--backend` flag →
  `/backend` REPL command (runtime wins). If DeepSeek selected but key missing → notice + fall back to Ollama.

---

## 7. CLI / UX

- **Banner:** `pyfiglet` renders "JARVIS" (fallback to the user-provided static art below), styled with
  `rich` (accent color + subtitle), shown on launch — hermes/polymarket CLI aesthetic.
- **Flow:** free-text query → graph runs → **pauses to show the brief and asks to confirm** → streams
  node-by-node progress with live spinners (🔎 arXiv, 🌐 OpenAlex, 📄 reading, 🤔 reflecting) → prints report.
- **Commands:** `/read N`, `/save N`, `/papers`, `/backend [ollama|deepseek]`, `/model <name>`, `/help`, `/quit`.
- **Entry:** `python -m jarvis`.

Static banner fallback:
```
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
```

---

## 8. Dependencies (`requirements.txt`)

`langgraph`, `langchain`, `langchain-ollama`, `langchain-deepseek`, `arxiv`, `pyalex`, `habanero`,
`requests`, `pymupdf4llm`, `huggingface_hub`, `pydantic-settings`, `rich`, `typer`, `pyfiglet`,
`python-dotenv`.

---

## 9. Build order (the plan)

1. **`README.md`** (done) — vision, install, usage, backend switch, example session.
2. **`info.md`** (this, done) — full knowledge dump.
3. Scaffold package + `requirements.txt` + venv + install.
4. `config.py` + `llm.py` — backend switch.
5. `tools/*` — the 6 tool modules.
6. `state.py` + `nodes.py` + `graph.py` — the LangGraph loop (incl. clarify + confirm/interrupt).
7. `banner.py` + `cli.py` — the cool REPL.
8. `.env.example` + `scripts/smoke.py`.
9. Verify (see §11).

---

## 10. Out of scope (Phase 2+ ideas)

- **Persistent vector library / "chat across all saved papers" (RAG)** using `nomic-embed-text` +
  Chroma. The embedding model is already installed; hooks left in place but not built now.
- Streamlit/web UI. Kaggle dataset search (needs API token). Citation-graph exploration. Multi-agent
  supervisor (parallel sub-topic research à la Open Deep Research).

---

## 11. Verification checklist (end-to-end, for when code exists)

1. `pip install -r requirements.txt`; `ollama list` shows `qwen3.5:9b`.
2. **Banner** renders on `python -m jarvis`.
3. **Clarify-first:** after a query, a brief is shown and the agent *waits* — no tool runs until "go";
   editing the brief changes behavior.
4. **Local backend:** "find recent papers on RAG evaluation" → real arXiv/S2/OpenAlex hits with links,
   loops ≥1 (reflect → 2nd query), final report.
5. `/read 1` → PDF downloads to `data/papers/`, coherent method summary returned.
6. "find datasets for QA over scientific papers" → HF + Papers with Code hits.
7. **Backend switch:** `/backend deepseek` (key set) uses DeepSeek; unset key → graceful Ollama fallback.
8. **Bounded loop:** a broad query stops at `MAX_LOOPS` and still finalizes.
9. `scripts/smoke.py` runs headless and asserts non-empty report + ≥1 paper found.

---

## 12. Gotchas & notes for future-me

- **Windows/PowerShell** is the user's shell. Use venv activation `\.venv\Scripts\activate`. Avoid
  bash-isms in any helper scripts meant to run on their machine.
- **Ollama context window:** `qwen3.5:9b` has a limited context — that's *why* PDF reading is
  **map-reduce** (chunk → summarize → combine), not "stuff the whole PDF in one prompt".
- **Rate limits:** Semantic Scholar ~1 req/s without a key; OpenAlex wants a contact email for the
  "polite pool". Cache search results in `data/cache/` to avoid hammering APIs and to speed reruns.
- **Tool-call reliability:** if a local model emits malformed tool calls, degrade to a JSON/ReAct
  prompt loop rather than the native bound-tools path.
- **DeepSeek = OpenAI-compatible**, so `langchain-deepseek` (or `ChatOpenAI` with a custom `base_url`)
  both work; prefer the dedicated `langchain-deepseek` package.
- **Don't over-build.** Hobby project: keep deps lean, prefer libraries over custom scraping/parsing.

---

## 13. Reference links (research grounding)

- LangChain Open Deep Research — https://github.com/langchain-ai/open_deep_research
- Open Deep Research blog — https://www.langchain.com/blog/open-deep-research
- Ollama Deep Researcher tutorial — https://langchain-opentutorial.gitbook.io/langchain-opentutorial/17-langgraph/03-use-cases/14-langgraph-ollama-deep-researcher-deepseek
- Deep research agent w/ Qwen3 + LangGraph + Ollama — https://composio.dev/content/deep-research-agent-qwen3-using-langgraph-and-ollama
- Awesome Deep Research (ACL 2026) — https://github.com/DavidZWZ/Awesome-Deep-Research
- Research Paper APIs 2026 (IntuitionLabs) — https://intuitionlabs.ai/articles/research-paper-apis-scientific-literature
- arXiv API — https://info.arxiv.org/help/api/ · Semantic Scholar API — https://www.semanticscholar.org/product/api
- OpenAlex — https://openalex.org · Crossref REST — https://www.crossref.org/documentation/retrieve-metadata/rest-api/
- pymupdf4llm — https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/ · Papers with Code — https://paperswithcode.com

---

## 14. Phase 2 — implemented & verified (2026-06-19)

Everything below is built and tested working end-to-end.

### New capabilities
- **Chat-with-paper (RAG), per-paper isolated stores** — `/read N` parses the PDF, summarizes
  (map-reduce), **saves the summary** to `data/summaries/`, and **indexes the full text into its OWN
  vector folder** `data/vectorstore/<slug>__<HHMMSSmmm>/` (one Chroma collection per paper), embedded
  by local **`nomic-embed-text`** (768-dim). A `registry.json` maps folder ↔ paper. `/db` lists the
  library, `/use N` selects a paper, `/ask <q>` answers scoped to ONE paper (picker if none selected,
  so content never mixes), `/forget N` deletes a paper's vectors (gc + retry to beat Windows file
  locks). Files: `embeddings.py`, `store.py`, `qa.py`. *Verified:* two papers → two separate
  timestamped folders, isolated search, scoped /ask correct, delete removes folder + registry entry.
- **Dataset inspection (no download)** — `tools/hf_inspect.py` via the free HF **datasets-server**:
  `/info` (columns+types), `/size` (row counts), `/first-rows` (samples), plus README from the Hub.
  Auto-resolves renamed/bare ids (e.g. `squad` → `rajpurkar/squad`). CLI: `/inspect <id>`, `/dataset <q>`.
  *Verified:* `rajpurkar/squad` → 5 columns, ~98,169 rows (train 87,599 / val 10,570), sample rows.
- **Web research** — `tools/web.py`: **`ddgs`** (DuckDuckGo search, free, no key) + **`trafilatura`**
  (clean article-text extraction). CLI: `/web <q>`; agent tools `search_web`, `read_web`.
  *Verified:* 3 hits + extracted 12.6K chars of clean text.
- **Relevance filter** — `embeddings.rank_by_relevance()` ranks gathered papers by cosine similarity
  to the brief and drops off-topic ones (`relevance_min=0.25`, keep top `keep_top_papers=12`). Fixes
  the earlier "tangential NLP papers" noise.
- **Live tool view** — the act node records each `tool(args)` call; the CLI streams them
  (`→ search_arxiv(query=…)`) like Claude Code.
- **Persistence** — summaries → `data/summaries/`, reports → `data/reports/` (timestamped).

### Key decisions & learnings (important)
- **qwen3.5 is a reasoning model.** Without caps it emits huge hidden "thinking" chains → calls take
  minutes / appear to hang. **Fix in `llm.py`:** `reasoning=False`, `num_predict=1024`, `num_ctx=8192`,
  and a hard `client_kwargs={"timeout": 180}` so a wedged call can't hang forever. `max_loops` lowered
  to 3. (The original 2-hour "hang" was the PC sleeping mid-run, but these caps are essential anyway.)
- **Scrapling was dropped from the default.** v0.4.x hard-imports **Playwright** (a full browser) even
  for static fetches — defeating the "lightweight, no-browser" goal. So the default web stack is
  **ddgs + trafilatura** (no browser). Scrapling remains *optional*: set `USE_SCRAPLING=true` and
  install `scrapling` + `curl_cffi` + `playwright` to enable the stealth fetch path in `web.py`.
- **crawl4ai** was considered and rejected for the same Playwright-weight reason; trafilatura is the
  better fit for *reading* research pages.
- **HF dataset ids get renamed** — always resolve via `is-valid` / Hub search before hitting the API.

### New dependencies (added to `requirements.txt`)
`ddgs`, `trafilatura`, `langchain-chroma`, `chromadb`. Optional (commented): `scrapling`, `curl_cffi`.
Embeddings reuse `langchain-ollama` (`OllamaEmbeddings`).

### Packaging & git hygiene (2026-06-19)
- **`jarvis` command** — `pyproject.toml` defines a console entry point `jarvis = jarvis.cli:main`;
  run `pip install -e .` to register it. **`jarvis.bat`** (calls `.venv\Scripts\python.exe -m jarvis`)
  lets you run `jarvis` from any cmd window without activating the venv — put the folder on PATH.
- **`.gitignore`** now ignores `.venv/`, `data/`, `.env`, `__pycache__/`, `*.pyc`, build/editor cruft.
  The repo had accidentally committed the entire `.venv` (~12k files); untracked via
  `git rm -r --cached .venv data` + pyc cleanup → 23 clean source files tracked. `data/` (PDFs,
  vectors, reports) is intentionally not committed.

### Phase 3 ideas (not built at the time)
Chat across *all* saved papers at once; Streamlit UI; Kaggle search; citation-graph exploration;
enable the Scrapling/Playwright booster. **Multi-agent supervisor → built in §15.**

---

## 15. Phase 3 — Multi-agent orchestrator-worker (implemented & verified 2026-06-19)

**This supersedes the single-agent reflect loop described in §4.** Jarvis now follows the
orchestrator-worker pattern from Anthropic's *[How we built our multi-agent research
system](https://www.anthropic.com/engineering/multi-agent-research-system)*.

### Architecture (new graph)
```
  clarify ─► plan(lead) ─► confirm(human) ──edit──► plan        (re-plan on edit)
                               │ go
                               ▼
                    ┌── subagent 1 (own context + tools) ──┐
                    ├── subagent 2 (own context + tools) ──┤─► synthesize ─► finalize ─► cite ─► END
                    └── subagent N (own context + tools) ──┘    (merge)       (draft)   (sources)
```
- **`plan` (lead agent)** — decomposes the confirmed brief into independent subagent specs
  `{objective, sub_query, tools}`. **Count scales to complexity** (1 for a fact, 2-3 for a
  comparison, more for a survey), bounded by a backend-dependent ceiling.
- **`confirm`** — human-in-the-loop. Shows the brief **and the subagent split**; "go" proceeds,
  any other text edits the brief and routes back to `plan` (re-plan + re-confirm). `after_confirm`
  is the conditional edge.
- **`fanout`** — runs the subagents. **Backend-aware execution:** sequential `for` loop on Ollama
  (one GPU), `ThreadPoolExecutor` on DeepSeek (true parallel API calls). Aggregates every subagent's
  papers/datasets/web, dedups, applies the cosine relevance filter against the original query.
- **`synthesize`** — lead merges the subagent briefings into one synthesis (`MERGE_PROMPT`).
- **`finalize`** — drafts the markdown report (TL;DR / key papers / datasets / next steps).
- **`cite` (CitationAgent)** — rewrites the draft so claims carry bracketed `[n]` citations tied to
  the **real** retrieved sources, appends a sources list, then persists to `data/reports/`.

### The reflective loop moved *inside* the subagent (`subagent.py`)
Each subagent has its **own context window** (its own message list) and runs `subagent_rounds`
broad→narrow rounds: round 1 = the lead's `sub_query` (broad), then a lightweight reflect picks a
sharper follow-up query, round 2 = narrowed. Then it writes a <150-word briefing. Isolated contexts
= no topic-bleed between facets (a key reason the article uses subagents).

### Backend-aware execution (the 4GB-GPU reality)
- **Ollama (RTX 3050 4GB):** subagents run **sequentially** — only one model is in VRAM at a time, so
  the subagent ceiling is a *patience* limit, not a memory limit. Forcing true-parallel here would
  thrash (the 9B model already spills to RAM).
- **DeepSeek:** subagents run **in parallel** (threads over blocking API calls) — same code path,
  same result quality, just faster wall-clock. This is the one branch that differs by backend.
- `config.subagent_ceiling(backend)` returns `MAX_SUBAGENTS_DEEPSEEK` (10) or `MAX_SUBAGENTS_OLLAMA` (5).

### Which article principles we applied vs. skipped
- **Applied:** orchestrator-worker decomposition; isolated subagent contexts; scale-effort-to-complexity
  (dynamic count); explicit delegation struct per subagent (objective + query + tools + boundaries);
  broad→narrow search; dedicated citation pass; tightened tool intent in prompts.
- **Skipped (deliberately, single-user hobby tool):** production async orchestration, "rainbow
  deployments", crash/state recovery beyond the existing `MemorySaver`; parallel *tool* calls *within*
  a subagent on Ollama (no benefit on 4GB); the LLM-judge eval harness (deferred to Phase 4).

### New / changed files
- **New:** `subagent.py` (isolated research loop), `events.py` (progress sink so deep nodes stream
  live to the CLI without importing Rich), prompts `PLAN_PROMPT` / `SUBAGENT_SYSTEM` / `SUBAGENT_USER`
  / `SUBAGENT_REFLECT` / `SUBAGENT_SYNTH` / `MERGE_PROMPT` / `CITE_PROMPT`.
- **Changed:** `nodes.py` (plan/confirm/fanout/synthesize/finalize/cite, dropped act/reflect),
  `graph.py` (rewired, linear top-level + confirm↻plan branch), `state.py` (+`plan`,
  `subagent_reports`, `replan`, `max_subagents`), `config.py` (+subagent settings + `subagent_ceiling()`),
  `cli.py` (shows the plan at confirm, streams subagent progress via the events sink),
  `scripts/demo_verbose.py` (new node names).

### New config (`.env`)
`MAX_SUBAGENTS_OLLAMA=5`, `MAX_SUBAGENTS_DEEPSEEK=10`, `SUBAGENT_ROUNDS=2`, `PARALLEL_SUBAGENTS=true`.

### Verification (2026-06-19)
Headless `scripts/smoke.py` (capped to 2 subagents × 1 round) on Ollama/`qwen3.5:9b`: all 7 nodes ran
in order (clarify→plan→confirm→fanout→synthesize→finalize→cite), **12 on-topic RAG-evaluation papers**
gathered across subagents and relevance-filtered, and the **CitationAgent produced a report with real
bracketed `[1]`–`[12]` citations** tied to actual arXiv/Semantic-Scholar links (Ragas [2], FAIR-RAG
[10], medicine benchmark [4], XRAG [12]). Exit 0. Report saved to `data/reports/`.

### Gotchas confirmed
- **Slow ≠ hang.** On a 4GB GPU each `qwen3.5:9b` call with `num_predict=1024` is slow (heavy CPU
  offload) — a full multi-agent run is several minutes. The per-call `timeout=180` guarantees no
  single call wedges the run. Use DeepSeek (parallel) when you want speed.
- **PowerShell pipe prepends a UTF-8 BOM to stdin.** `"/help" | jarvis` sent the command as
  `<BOM>/help`, so `startswith("/")` failed and it was treated as a research query. Worse, with
  `sys.stdin` on cp1252 the BOM decodes to the 3-char mojibake `ï»¿` (U+00EF/BB/BF), not a single
  U+FEFF. Fix in `cli.py`: reconfigure **stdin** (not just stdout/stderr) to UTF-8 **and** a
  `_read_line()` helper that strips a leading BOM in *both* forms (using `chr()` so no fragile literal
  bytes live in the source). Interactive typing is unaffected (no BOM → passthrough). Verified: piped
  `/help` and `/db` now parse as commands.

---

*The detailed implementation plan also lives at*
`C:\Users\SHAUN RODRIGUES\.claude\plans\shimmying-swimming-mountain.md`.
