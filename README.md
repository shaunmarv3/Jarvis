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

Jarvis is a CLI research assistant that behaves like a relentless grad student. Give it a topic or a
paper, confirm a short plan, and it autonomously **searches → reads → reflects on what's missing →
searches again** until it can give you a grounded report with real citations.

- **Find ideas & papers** — searches arXiv, Semantic Scholar, OpenAlex & Crossref.
- **Pull the exact paper** — by title, URL, arXiv ID, or DOI.
- **Read & explain** — downloads the PDF, parses it, summarizes, explains methods, answers questions.
- **Find datasets** — searches HuggingFace Hub & Papers with Code.
- **Loops like a madman** — a bounded reflective loop digs deeper on knowledge gaps automatically.
- **Two brains, switchable** — local **Ollama** (`qwen3.5:9b`) for free/offline, or **DeepSeek** cloud for heavier reasoning.
- **Asks first** — drafts a research brief and waits for your "go" before touching any tool.

## How it works

```
   topic ─► clarify ─► confirm brief ─►  act (tools)  ─► synthesize ─► reflect ─► finalize ─► report
              │            ⏸ you           ▲                              │
              │          approve           └────── gaps remain? ──────────┘
              ▼          /edit                       (bounded by MAX_LOOPS)
        research brief
```

Built on **[LangGraph](https://github.com/langchain-ai/langgraph)** — a cyclic graph orchestrating a
tool-calling agent inside a reflect-and-retry loop. Pattern adapted from LangChain's _Open Deep
Research_ and _Ollama Deep Researcher_ (IterDRAG), specialized for academic sources.

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
cp .env.example .env   # then edit .env
```

### Configure (`.env`)

```ini
DEFAULT_BACKEND=ollama          # ollama | deepseek
OLLAMA_MODEL=qwen3.5:9b
DEEPSEEK_API_KEY=               # only needed for the deepseek backend
MAX_LOOPS=4                     # how deep the reflect loop digs
```

### Run

```bash
python -m jarvis
```

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

| Command                       | Action                                                      |
| ----------------------------- | ----------------------------------------------------------- |
| `<free text>`                 | Ask the agent anything; it plans, confirms, then researches |
| `/read <N>`                   | Download & summarize result _N_                             |
| `/save <N>`                   | Save paper _N_ to your local library                        |
| `/papers`                     | List results/saved papers in this session                   |
| `/backend [ollama\|deepseek]` | Switch the LLM brain at runtime                             |
| `/model <name>`               | Switch the Ollama model                                     |
| `/help`                       | Show commands                                               |
| `/quit`                       | Exit                                                        |

## Tools & data sources

| Source                                                                         | Use                               | Cost         |
| ------------------------------------------------------------------------------ | --------------------------------- | ------------ |
| [arXiv](https://info.arxiv.org/help/api/)                                      | Preprint search + exact PDF fetch | Free, no key |
| [Semantic Scholar](https://www.semanticscholar.org/product/api)                | Search, citations, title resolve  | Free, no key |
| [OpenAlex](https://openalex.org)                                               | 250M+ works, citation graph       | Free, no key |
| [Crossref](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) | DOI / metadata resolve            | Free         |
| [HuggingFace Hub](https://huggingface.co/datasets)                             | Dataset search                    | Free         |
| [Papers with Code](https://paperswithcode.com)                                 | Dataset / benchmark search        | Free         |

## Project structure

```
jarvis/
├── jarvis/
│   ├── config.py          # settings + backend switch
│   ├── llm.py             # get_llm() → Ollama | DeepSeek
│   ├── state.py           # LangGraph AgentState
│   ├── graph.py           # graph assembly
│   ├── nodes.py           # clarify / confirm / act / synthesize / reflect / finalize
│   ├── prompts.py         # prompt templates
│   ├── tools/             # arxiv, semantic_scholar, openalex, crossref, pdf_reader, datasets
│   ├── banner.py          # JARVIS ASCII banner
│   └── cli.py             # Rich REPL
├── data/papers/           # downloaded PDFs
├── requirements.txt
├── .env.example
├── README.md
└── info.md                # full project knowledge dump
```

## Roadmap

- [ ] **Phase 1** — paper search/fetch, read/summarize, dataset search, reflect loop, dual backend.
- [ ] **Phase 2** — persistent vector library + _chat across all saved papers_ (RAG via `nomic-embed-text` + Chroma).
- [ ] Streamlit web UI · Kaggle search · citation-graph exploration.

## License

MIT — personal hobby project.

---

<div align="center"><sub>Built for the joy of research. Powered by LangGraph + Ollama/DeepSeek.</sub></div>
