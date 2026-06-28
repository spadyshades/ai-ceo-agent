# AI CEO: Strategic Intelligence Agent — BMW

> An autonomous AI agent that collects live information about **BMW**, identifies opportunities, risks, and trends, and produces evidence-based strategic recommendations for executive decision-making.

---

## 🚧 Project Status

| Phase | Description | Status |
|---|---|---|
| 0 | Setup & Scaffolding | 🚧 In progress |
| 1 | Data Collection Layer | ⏳ Pending |
| 2 | Knowledge Repository + Processing | ⏳ Pending |
| 3 | Strategic Intelligence Tools | ⏳ Pending |
| 4 | AI CEO Agent (LangGraph) | ⏳ Pending |
| 5 | Evidence + Validation Layer | ⏳ Pending |
| 6 | Executive Dashboard | ⏳ Pending |
| 7 | Documentation | ⏳ Pending |
| 8 | Oral Exam Prep | ⏳ Pending |

---

## 🏗️ Architecture

*(Filled in during Phase 7)*

## 🔄 Data Flow

*(Filled in during Phase 7)*

## 🧰 Technology Stack

| Layer | Tool | Notes |
|---|---|---|
| LLM | **Phi-4 Mini** (via Ollama) | Open-source, CPU-friendly |
| Embeddings | **BAAI/bge-small-en-v1.5** | 384-dim, fast on CPU |
| Vector DB | **ChromaDB** | Persistent local store |
| Agent Framework | **LangGraph** | Explicit plan → tool → validate graph |
| Dashboard | **Streamlit** | 7-section executive UI |
| Data Sources | BMW Press · Google News RSS · Hacker News · arXiv | No API keys required |

## 🎯 Design Decisions

*(Filled in during Phase 7)*

## 🤖 AI Pipeline

The agent follows the workflow:

```
Goal → Plan → Retrieve → Analyze → Decide → Recommend → Validate
```

*(Detailed in Phase 7)*

---

## 🚀 Setup

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/download) (with `phi4-mini` model pulled)
- Git

### Install
```powershell
# Clone (if from git)
git clone <your-repo-url>
cd ai-ceo-agent

# Virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Dependencies
pip install -r requirements.txt

# Pull the LLM (one-time, ~2.5GB)
ollama pull phi4-mini

# Verify everything works
python test_setup.py
```

### Run
*(Updated when each phase completes)*

---

## 📁 Project Structure

```
ai-ceo-agent/
├── data/
│   ├── raw/                 # scraped documents
│   ├── processed/           # cleaned & deduped
│   └── chroma/              # vector index
├── src/
│   ├── ingestion/           # Phase 1 — scrapers
│   ├── processing/          # Phase 2 — clean/dedupe/embed
│   ├── tools/               # Phase 3 — agent tools
│   ├── agent/               # Phase 4-5 — LangGraph + validator
│   ├── dashboard/           # Phase 6 — Streamlit app
│   └── config.py            # central config
├── notebooks/               # prototyping
├── tests/                   # unit tests
├── test_setup.py            # Phase 0 verification
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```
