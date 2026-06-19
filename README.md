<div align="center">

```
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
```

**Your personal AI research agent — in the terminal.**

Searches ideas, pulls exact papers, reads & explains them, and finds datasets —
looping like a tireless researcher until it has a real answer.

_Runs on local Ollama or DeepSeek. Your machine, your tokens, your call._

</div>

---

## ✨ What it does

Jarvis is a CLI research assistant that behaves like a relentless research team. Give it a topic or a
paper, confirm a short plan, and a **lead agent splits the question across parallel subagents** — each
researching one facet in its own context — then merges and cites their findings into a grounded report.

- **Multi-agent (orchestrator-worker)** — a **lead agent** decomposes your question into independent
  sub-questions and fans them out to **subagents**, each with its own context window and tools, then
  synthesizes their reports and runs a dedicated **citation pass**. (Pattern from Anthropic's
  [multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).)
- **Find ideas & papers** — searches arXiv, Semantic Scholar, OpenAlex & Crossref.
- **Pull the exact paper** — by title, URL, arXiv ID, or DOI.
- **Read & explain** — downloads the PDF, parses it, summarizes, explains methods.
- **Chat with a paper (RAG)** — after `/read`, ask follow-up questions; each paper gets its **own** vector store (`nomic-embed-text` + Chroma) so `/ask` targets one paper and never mixes content. `/db` lists your library; `/forget` deletes one.
- **Find & inspect datasets** — searches HuggingFace Hub & Papers with Code, and inspects a dataset's columns, row counts, sample rows & README **without downloading it**.
- **Search the web** — general internet research via DuckDuckGo + clean article extraction (trafilatura).
- **Relevance-filtered** — embedding-similarity filter drops off-topic papers from results.
- **Digs like a madman** — each subagent runs a bounded broad→narrow search loop in isolation.
- **Two brains, switchable** — local **Ollama** (`qwen3.5:9b`) for free/offline, or **DeepSeek** cloud for heavier reasoning.
- **Asks first** — drafts a research brief **and the subagent split**, and waits for your "go" before touching any tool.
- **Shows its work & saves it** — streams every subagent and tool call live; persists summaries & reports to disk.

## How it works

```
  topic ─► clarify ─► plan (lead) ─► confirm ──edit──► (re-plan)
                         │              ⏸ you
                         │            approve
                         ▼               │
              splits into N sub-questions│
                                         ▼
                        ┌── subagent 1 (own context + tools) ──┐
                        ├── subagent 2 (own context + tools) ──┤─► synthesize ─► finalize ─► cite ─► report
                        └── subagent N (own context + tools) ──┘    (merge)      (draft)   (sources)
```

The lead agent **scales the number of subagents to the question** (1 for a fact, a few for a
comparison, more for a survey — up to a configurable ceiling). Subagents run **sequentially on local
Ollama** (one GPU) and **in parallel on DeepSeek** (async API) — same result, the cloud is just faster.

Built on **[LangGraph](https://github.com/langchain-ai/langgraph)**. The orchestrator-worker pattern
follows Anthropic's _[How we built our multi-agent research
system](https://www.anthropic.com/engineering/multi-agent-research-system)_ (lead decomposes →
parallel subagents → synthesize → cite), specialized here for academic sources. The per-subagent
broad→narrow search loop is adapted from _Ollama Deep Researcher_ (IterDRAG).

## Quick start

### Prerequisites

- **Python 3.12+**
- **[Ollama](https://ollama.com)** running locally with a tool-calling model:
  ```bash
  ollama pull qwen3.5:9b
  ```
- _(Optional)_ A **DeepSeek** API key for the cloud backend.

### Install

```bash
git clone <your-repo-url> jarvis
cd jarvis
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .       # registers the `jarvis` command
cp .env.example .env   # then edit .env
```

### Configure (`.env`)

```ini
DEFAULT_BACKEND=ollama          # ollama | deepseek
OLLAMA_MODEL=qwen3.5:9b
DEEPSEEK_API_KEY=               # only needed for the deepseek backend

# Multi-agent (orchestrator-worker)
MAX_SUBAGENTS_OLLAMA=5          # ceiling on local GPU (subagents run sequentially)
MAX_SUBAGENTS_DEEPSEEK=10       # ceiling on cloud (subagents run in parallel)
SUBAGENT_ROUNDS=2              # search rounds per subagent (broad → narrowed follow-up)
PARALLEL_SUBAGENTS=true        # run subagents concurrently (only takes effect on DeepSeek)
```

> **On a 4GB GPU:** keep `MAX_SUBAGENTS_OLLAMA` modest — the ceiling is a *patience* limit, not a
> memory limit (only one subagent is in VRAM at a time), so more subagents just means a longer wait.
> Raise it when you want deeper research and don't mind waiting; on DeepSeek it runs in parallel.

### Run

```bash
jarvis              # if you ran `pip install -e .` (venv active)
python -m jarvis    # always works
```

**Run `jarvis` from any cmd window (Windows):** add the project folder to your `PATH`, or copy
`jarvis.bat` somewhere already on `PATH` — then just type `jarvis`. The `.bat` calls the venv's
Python directly, so you don't need to activate the venv first.

## Usage

```text
> find recent papers on retrieval-augmented generation evaluation

   Brief: Survey 2024-2026 work on evaluating RAG systems; prioritize
            benchmark/metric papers. Tools: arXiv, Semantic Scholar, OpenAlex.
     Proceed? [Y/edit/n] y

   arXiv …   OpenAlex …   reflecting (gap: faithfulness metrics) …   arXiv …
   Report ready — 6 papers, 2 follow-ups.

> read 1          # downloads + summarizes the first result
> /backend deepseek
> find datasets for question answering over scientific papers
```

### Commands

| Command                       | Action                                                       |
| ----------------------------- | ------------------------------------------------------------ |
| `<free text>`                 | Ask the agent anything; it plans, confirms, then researches  |
| `/read <N>`                   | Download, summarize & index paper _N_ into its **own** vector store (summary saved) |
| `/db`                         | List papers in the vector DB (each stored in a separate folder) |
| `/ask <question>`             | Ask a question — pick **which** indexed paper to query (no mixing) |
| `/use <N>`                    | Select DB paper _N_ as the target for `/ask`                 |
| `/forget <N>`                 | Delete DB paper _N_'s vectors                                 |
| `/save <N>`                   | Download paper _N_'s PDF to `data/papers`                    |
| `/papers`                     | List this session's search results                           |
| `/dataset <query>`            | Search HuggingFace + Papers with Code datasets               |
| `/inspect <hub_id>`           | Inspect a dataset (cols, rows, sample, README), e.g. `/inspect squad` |
| `/web <query>`                | Quick web search (DuckDuckGo)                                |
| `/backend [ollama\|deepseek]` | Switch the LLM brain at runtime                              |
| `/model <name>`               | Switch the Ollama model                                      |
| `/help`                       | Show commands                                                |
| `/quit`                       | Exit                                                         |

> 💡 Type `/` and a **live dropdown** of commands appears below the cursor (with descriptions),
> filtered as you type — like Claude Code. Powered by `prompt_toolkit`.

## Tools & data sources

| Source                                                                         | Use                               | Cost         |
| ------------------------------------------------------------------------------ | --------------------------------- | ------------ |
| [arXiv](https://info.arxiv.org/help/api/)                                      | Preprint search + exact PDF fetch | Free, no key |
| [Semantic Scholar](https://www.semanticscholar.org/product/api)                | Search, citations, title resolve  | Free, no key |
| [OpenAlex](https://openalex.org)                                               | 250M+ works, citation graph       | Free, no key |
| [Crossref](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) | DOI / metadata resolve            | Free         |
| [HuggingFace Hub](https://huggingface.co/datasets)                             | Dataset search                    | Free         |
| [HF datasets-server](https://huggingface.co/docs/datasets-server)              | Dataset schema / rows / size      | Free, no key |
| [Papers with Code](https://paperswithcode.com)                                 | Dataset / benchmark search        | Free         |
| [DuckDuckGo (ddgs)](https://github.com/deedy5/ddgs)                            | Web search                        | Free, no key |
| [trafilatura](https://trafilatura.readthedocs.io)                              | Web page text extraction          | Free         |
| [Chroma](https://www.trychroma.com) + nomic-embed                              | Local vector store for paper Q&A  | Free, local  |

## Project structure

```
jarvis/
├── jarvis/
│   ├── config.py          # settings + backend switch + subagent ceilings + paths
│   ├── llm.py             # get_llm() → Ollama | DeepSeek
│   ├── embeddings.py      # nomic-embed + relevance ranking
│   ├── store.py           # Chroma vector store (paper chunks)
│   ├── qa.py              # chat-with-paper RAG + summary persistence
│   ├── state.py           # LangGraph AgentState
│   ├── graph.py           # graph assembly
│   ├── nodes.py           # clarify / plan(lead) / confirm / fanout / synthesize / finalize / cite
│   ├── subagent.py        # isolated per-subagent broad→narrow research loop
│   ├── events.py          # live-progress sink (subagents stream to the CLI)
│   ├── prompts.py         # prompt templates
│   ├── tools/             # arxiv, semantic_scholar, openalex, crossref,
│   │                      #   pdf_reader, datasets, hf_inspect, web
│   ├── banner.py          # JARVIS ASCII banner (holographic gradient)
│   ├── repl.py            # prompt_toolkit input + live /command dropdown
│   └── cli.py             # Rich REPL
├── data/                  # git-ignored
│   ├── papers/            # downloaded PDFs
│   ├── summaries/         # saved paper summaries (markdown)
│   ├── reports/           # saved research reports (markdown)
│   └── vectorstore/       # one folder per paper: <slug>__<HHMMSSmmm>/ + registry.json
├── pyproject.toml         # `jarvis` console entry point
├── jarvis.bat             # run `jarvis` from any cmd (Windows)
├── requirements.txt
├── .gitignore             # ignores .venv/, data/, .env, __pycache__/ …
├── .env.example
├── README.md
└── info.md                # full project knowledge dump
```

## Roadmap

- [x] **Phase 1** — paper search/fetch, read/summarize, dataset search, reflect loop, dual backend.
- [x] **Phase 2** — chat-with-paper RAG (`nomic-embed-text` + Chroma), dataset inspection, web search, relevance filter, live tool view, persistence.
- [x] **Phase 3** — multi-agent **orchestrator-worker**: lead agent decomposes → parallel/sequential subagents with isolated contexts → synthesize → dedicated citation pass (backend-aware execution).
- [ ] **Phase 4** — chat across _all_ saved papers, Streamlit web UI, Kaggle search, citation-graph exploration, optional Scrapling/Playwright web booster, LLM-judge eval harness.

## License

MIT — personal hobby project.

---

<div align="center"><sub>Built for the joy of research. Powered by LangGraph + Ollama/DeepSeek.</sub></div>
