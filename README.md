# Patent Prior-Art Agent

An agentic prior-art screening assistant for patent attorneys. It extracts technical features from an invention disclosure, sources evidence through a **Corrective RAG (CRAG)** pipeline (internal vector store → web fallback), grades relevance with a **System-1 LLM judge**, and synthesizes a triage report — with full **Langfuse observability** on every step.

> Built with Streamlit · Groq · Qdrant · FastEmbed · DuckDuckGo · Langfuse

---

## Features

| Feature | Description |
|---|---|
| 🔍 **Feature Planning** | LLM extracts up to `MAX_FEATURES` core technical features from the disclosure |
| 🗄️ **Internal Retrieval** | Semantic search over an indexed prior-art portfolio (Qdrant + FastEmbed `bge-small-en-v1.5`) |
| ⚖️ **System-1 Judge** | Relevance grading (Relevant / Partial / Irrelevant) via the Codiv API (Laya/OpenJev) |
| 🌐 **Web Fallback** | DuckDuckGo search when internal evidence is insufficient (Corrective RAG) |
| 📝 **Triage Report** | Attorney-ready report stating prior-art found internally / externally / not at all, with citations |
| 📊 **Langfuse Tracing** | Every trace, span, and generation is logged to Langfuse for debugging and evaluation |
| 🖥️ **Streamlit UI** | Simple web interface to paste disclosures, index samples, and view results |

---

## How It Works

```
Disclosure Input
      │
      ▼
┌─────────────────┐
│  Feature Plan   │  ← LLM extracts core technical features
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
For each feature:
    │
    ▼
┌──────────────┐      insufficient?
│ Internal Search │──────────────┐
│  (Qdrant)      │               ▼
└──────┬───────┘        ┌──────────────┐
       │ relevant?      │   Web Search │
       ▼                │ (DuckDuckGo) │
┌──────────────┐       └──────┬───────┘
│ System-1     │              │
│ Relevance    │              ▼
│ Judge        │      ┌──────────────┐
│ (Codiv API)  │      │ Final        │
└──────┬───────┘      │ Evidence Set │
       │              └──────┬───────┘
       ▼                     │
┌──────────────┐             │
│  Synthesize  │─────────────┘
│ Report (LLM) │
└──────────────┘
```

Each step above is traced in Langfuse: one trace per screening run, one span per feature route, and generations for planning and report synthesis.

---

## Project Structure

```
patent-prior-art-agent/
├── app.py              # Streamlit UI
├── agent.py            # PriorArtAgent orchestrator (Plan → Act → Synthesize)
├── crag_pipeline.py    # CRAG decision loop: internal → grade → web fallback
├── retriever.py        # Qdrant vector store + FastEmbed retrieval
├── system1.py          # Codiv System-1 relevance judge
├── obs.py              # Langfuse tracer (with no-op fallback)
├── config.py           # Central configuration (.env backed)
├── req.txt             # Python dependencies
├── .env                # Runtime secrets (gitignored — do not commit)
├── .gitignore
├── README.md
└── data/
    ├── sample_disclosure.txt   # Example invention disclosure
    ├── portfolio/              # Prior-art documents to index
    └── qdrant_db/              # Local Qdrant storage (gitignored)
```

---

## Quick Start

### 1. Clone and install

```bash
git clone <your-repo-url>
cd patent-prior-art-agent

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r req.txt
```

### 2. Configure environment

Copy the example below into a `.env` file in the project root (`.env` is gitignored):

```env
# Required LLM / Judge keys
CODIV_API_KEY="sk-codiv-..."
GROQ_API_KEY="gsk_..."

# Langfuse observability (required for tracing to appear in Langfuse)
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_HOST="https://us.cloud.langfuse.com"
```

> **Important:** The Langfuse host variable must be `LANGFUSE_HOST` (not `LANGFUSE_BASE_URL`). `config.py` reads `LANGFUSE_HOST` and falls back to `https://cloud.langfuse.com` if unset.

### 3. Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

1. Paste an invention disclosure into the text area.
2. (Optional) Click **Index Sample Disclosure** to load `data/sample_disclosure.txt` into Qdrant so internal retrieval has content to search.
3. Click **Run Prior-Art Screening**.
4. Review extracted features, evidence sources, and the final report. The Langfuse toggle in the sidebar shows whether tracing is active.

---

## Langfuse Observability

Langfuse tracing is **on by default** when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. Each `streamlit run app.py` session:

- Creates one trace per screening run (`prior_art_screening`)
- Logs a `crag_feature_route` span per feature
- Logs `system1_grade` spans for every chunk judged by Codiv
- Logs `plan_features` and `synthesize_report` LLM generations
- Flushes to `LANGFUSE_HOST` (default `https://us.cloud.langfuse.com`)

If credentials are missing or the host is unreachable, the app degrades gracefully to a no-op tracer — the UI keeps working, nothing is sent to Langfuse.

Verify tracing is active in the Streamlit sidebar (**Langfuse Tracing: ON**) and check the Langfuse project dashboard for the traces after a run.

---

## Configuration (`config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `CODIV_API_KEY` | — | Codiv API key for System-1 judging |
| `GROQ_API_KEY` | — | Groq API key for feature planning & report synthesis |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse API host |
| `SYSTEM1_MODEL` | `laya-1.0` | Codiv System-1 model |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Groq model for planning/reporting |
| `THRESH_RELEVANT` | `0.65` | Relevance probability to count as internally relevant |
| `MIN_RELEVANT_CHUNKS` | `2` | Minimum internally relevant chunks to stop at internal route |
| `WEB_FALLBACK_THRESHOLD` | `0.4` | Relevance floor for web-sourced evidence |
| `MAX_FEATURES` | `4` | Max features extracted per disclosure |
| `QDRANT_PATH` | `./data/qdrant_db` | Local Qdrant storage path |

---

## Troubleshooting

**Langfuse not showing traces**
1. Confirm `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set in `.env`.
2. Confirm the variable is named `LANGFUSE_HOST` (not `LANGFUSE_BASE_URL`).
3. In the Streamlit sidebar, **Langfuse Tracing** should read `ON`.
4. Check the Langfuse project at your `LANGFUSE_HOST`; traces may take a few seconds to appear after `flush()`.

**No internal evidence found**
- Run **Index Sample Disclosure** first, or add your own prior-art documents to `data/portfolio/` and index them through `QdrantRetriever.index_documents`.

**Web results empty**
- DuckDuckGo may block automated requests from some networks. The app gracefully falls back to whatever internal evidence exists.

**API errors**
- Verify `CODIV_API_KEY` and `GROQ_API_KEY` are valid and have quota remaining.

---

## Requirements

Python 3.9+. See `req.txt` for pinned dependencies:

```
streamlit, qdrant-client, fastembed, duckduckgo-search, groq,
requests, python-dotenv, langfuse
```

---
