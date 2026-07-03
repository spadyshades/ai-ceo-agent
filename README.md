# AI CEO Strategic Intelligence Agent

An autonomous AI agent that acts as a strategic advisor to the CEO of BMW. It collects live intelligence from five public sources, processes it into a searchable knowledge base, identifies opportunities, risks, and trends, produces ranked evidence-based recommendations, validates them adversarially, and delivers an executive briefing — all running locally on CPU with open-source models.

---

## Table of Contents

- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Tool Registry](#tool-registry)
- [Agent Workflow](#agent-workflow)
- [Validation Layer](#validation-layer)
- [Executive Dashboard](#executive-dashboard)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Design Decisions](#design-decisions)

---

## Architecture

```
                         User Goal
                             |
                             v
                       +-----------+
                       |  Planner  |  LLM: structured JSON plan
                       +-----------+
                             |
                             v
                +------> +-----------+
                |        | Executor  |  Dispatches tools per plan step
                |        +-----------+
                |              |
                |     route_after_execute
                |        /          \
                +-------'            \
               (more steps)           v
                                +-----------+
                                |  Analyzer |  LLM: synthesise evidence
                                +-----------+
                                      |
                                      v
                                +-----------+
                                |  Decider  |  LLM: ranked recommendations
                                +-----------+
                                      |
                                      v
                                +-----------+
                                | Validator |  Rules + adversarial LLM
                                +-----------+
                                      |
                            route_after_validate
                            /                 \
                     (replan)             (pass / limit)
                        |                      |
                  back to Planner        +-----------+
                                         | Briefing  |  LLM: executive summary
                                         +-----------+
                                               |
                                               v
                                         Final State
```

The agent is a **LangGraph state machine** with six nodes, two conditional edges, and a validation-driven replan cycle. The workflow follows: **Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend -> Validate**. When validation fails, the agent autonomously replans with the failure reasons, gathers additional evidence, and tries again — up to three attempts.

A typical run makes 4-9 LLM calls and 3-13+ tool invocations, producing ranked recommendations backed by multi-source evidence, validated against contradicting corpus passages, and summarised in a CEO briefing.

---

## Data Pipeline

### Ingestion (5 sources, 4 categories)

| Source | Category | Method | Documents |
|--------|----------|--------|-----------|
| BMW Press | Company | HTML scraping from press.bmwgroup.com | ~50 |
| Google News RSS | News | RSS feeds across 6 targeted queries | ~260 |
| Hacker News | Community | Algolia search API | ~375 |
| arXiv | Research | arXiv API across 5 topic queries | ~43 |
| Yahoo Finance | Market | yfinance library | ~10 |

All scrapers are idempotent — running the scheduler again only fetches new documents.

### Processing (5 stages)

1. **Cleaning** — whitespace normalisation, HTML stripping, encoding handling
2. **Deduplication** — MinHash near-duplicate detection (Jaccard > 0.8 threshold)
3. **Entity Extraction** — spaCy NER (ORG, PERSON, GPE, PRODUCT, DATE) stored as chunk metadata
4. **Chunking** — recursive text splitting (512 tokens, 64-token overlap)
5. **Embedding and Indexing** — BAAI/bge-small-en-v1.5 (384-dim) into persistent ChromaDB with cosine distance

A sentiment pre-computation step classifies all document titles using FinBERT and stores results in SQLite for instant dashboard rendering.

### Search Modes

| Mode | Method | Best For |
|------|--------|----------|
| Semantic | BGE embedding + cosine similarity | Conceptual queries ("EV strategy") |
| BM25 | Okapi BM25 keyword matching | Exact terms, model numbers ("iX3") |
| Hybrid | Reciprocal rank fusion (0.7 semantic + 0.3 BM25) | General use, combines both strengths |

---

## Tool Registry (13 tools)

| Tool | Type | Purpose |
|------|------|---------|
| retriever | Vector search | Semantic search with metadata filtering |
| hybrid_search | Fusion search | Combined semantic + BM25 keyword search |
| web_search | External API | Live news via Google News RSS |
| financial_data | Live market data | Stock price, P/E, market cap, dividends via yfinance |
| compare_competitors | Corpus analysis | Mention frequency and FinBERT sentiment per competitor |
| detect_topic_trends | TF-IDF analysis | Rising topics across time windows |
| sentiment | ML classifier | FinBERT financial sentiment classification |
| trend_detector | Statistical | Entity mention frequency comparison across time windows |
| find_contradicting_evidence | LLM + retrieval | Claim negation + contradiction search |
| source_credibility | Rule-based | Tier-and-freshness scoring |
| detect_opportunities | Composite engine | Multi-query retrieval + LLM synthesis |
| detect_risks | Composite engine | Sentiment-weighted retrieval + LLM synthesis |
| detect_trends | Composite engine | Statistical rise detection + LLM synthesis |

The planner selects tools dynamically based on the user's goal. Parameter names are normalised with alias matching to handle LLM output variations. All LLM-facing calls use Ollama's JSON mode for constrained decoding.

---

## Agent Workflow

### Six Nodes

- **Planner** — receives the goal (and prior validation failures on replan), produces a structured JSON plan selecting from 13 tools. Includes cross-run memory: the planner sees the previous run's recommendations to build on prior analysis.
- **Executor** — dispatches one tool per plan step, tracks retrieved chunk references for downstream source attribution. Embedding results are cached in SQLite for sub-millisecond repeat lookups.
- **Analyzer** — synthesises all gathered evidence into a structured analysis with summary, key findings, and supporting chunk IDs.
- **Decider** — produces 3-5 ranked recommendations with title, rationale, priority, expected impact, risk assessment, confidence, and evidence citations. Auto-enriches evidence from retrieved chunks to ensure source diversity.
- **Validator** — four-stage validation pipeline (details below). Failures trigger autonomous re-planning.
- **Briefing** — generates a concise CEO briefing answering: What happened? Why does it matter? What should management do next?

### Autonomous Decision Points

- `route_after_execute` — loops to execute more steps or advances to analysis
- `route_after_validate` — passes to briefing, or replans (up to 3 attempts)

### Memory and Persistence

Five SQLite tables store the full audit trail: runs (with complete state JSON), plans (every version), tool calls (with parameters and summaries), recommendations (with priority and confidence), and validations (pass/fail with issues). Cross-run memory feeds the previous run's recommendations into the planner's context.

---

## Validation Layer

| Stage | Type | What It Checks |
|-------|------|----------------|
| Rule checks | Deterministic | Min 3 evidence chunks, min 2 distinct sources, min 0.2 confidence, non-empty rationale |
| Source existence | Chroma lookup | Cited chunk IDs exist in the vector store |
| Claim alignment | Embedding similarity | Recommendation text is semantically close to cited evidence (avg cosine > 0.25) |
| Adversarial review | LLM + comparator | Recommendation survives contradicting corpus passages |

Advanced checks (stages 2-4) run only on the final validation pass to optimise LLM compute during early replans.

---

## Executive Dashboard

Interactive Streamlit dashboard with 11 tabs and a sidebar agent runner:

| Tab | Content | Interactive Elements |
|-----|---------|---------------------|
| Overview | Financial snapshot (live stock data), KPI cards, source distribution, tier breakdown | Run selector |
| Market Intelligence | Latest ingested documents | Source multiselect filter |
| Competitors | Mention frequency and sentiment comparison across BMW competitors | Expandable detail cards |
| Topic Trends | TF-IDF rising topics bar chart and data table | -- |
| Opportunities | Impact-rated items with evidence and confidence | Impact level filter |
| Risks | Severity-rated items with evidence | Impact level filter |
| Sentiment | Pre-computed FinBERT distribution pie, time-series area chart, sample table | Source filter |
| Search | Side-by-side semantic vs hybrid vs BM25 search | Query input, mode toggle |
| Recommendations | Priority cards, expandable detail panels with evidence metrics | Expand/collapse |
| CEO Briefing | Executive summary | -- |
| Agent Trace | Plans, tool usage chart, tool call table, validation history, raw state | Expandable sections |

**Sidebar features:**
- Custom goal input with "Run Agent" button for on-demand analysis
- PDF report download button
- Run history selector with metadata

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Qwen 2.5 7B via Ollama (CPU, JSON mode) |
| Embedding | BAAI/bge-small-en-v1.5 (384-dim, cached) |
| Sentiment | ProsusAI/FinBERT |
| Keyword Search | BM25 via rank_bm25 |
| Topic Analysis | TF-IDF via scikit-learn |
| Vector DB | ChromaDB (persistent, cosine distance) |
| Agent Framework | LangGraph (state machine with conditional edges) |
| NER | spaCy en_core_web_sm |
| Database | SQLite (documents, chunks, embeddings cache, sentiment, agent history) |
| Dashboard | Streamlit + Plotly |
| Reports | fpdf2 |
| Financial Data | yfinance |

No paid APIs. No GPU required. Runs locally on a consumer laptop with 16 GB RAM.

---

## Project Structure

```
ai-ceo-agent/
  src/
    config.py                          Configuration constants
    utils/
      logging.py                       Structured logging
      llm.py                           Ollama client with JSON mode support
    ingestion/
      base.py                          BaseScraper abstract class
      storage.py                       SQLite document storage
      bmw_press.py                     BMW press release scraper
      google_news.py                   Google News RSS scraper
      hackernews.py                    Hacker News Algolia scraper
      arxiv_scraper.py                 arXiv API scraper
      yahoo_finance.py                 Yahoo Finance scraper
      scheduler.py                     Ingestion orchestrator
    processing/
      cleaner.py                       Text normalisation
      deduper.py                       MinHash deduplication
      extractor.py                     spaCy NER extraction
      chunker.py                       Recursive text chunking
      embedder.py                      BGE embedding
      indexer.py                       ChromaDB indexing
      pipeline.py                      Processing orchestrator
      sentiment_indexer.py             FinBERT sentiment pre-computation
    tools/
      retriever.py                     Semantic, BM25, and hybrid search
      web_search.py                    External news search
      sentiment.py                     FinBERT sentiment classification
      entity_extractor.py              spaCy NER wrapper
      trend_detector.py                Statistical entity trend detection
      topic_trends.py                  TF-IDF topic trend detection
      comparator.py                    Claim negation and contradiction retrieval
      competitor_comparison.py         Competitor mention and sentiment analysis
      financial_data.py                Live stock data via yfinance
      source_credibility.py            Tier-and-freshness scoring
      embedding_cache.py               SQLite embedding cache
      retrieval_utils.py               Query-time document deduplication
      report_generator.py              PDF executive report generation
    intelligence/
      engines.py                       Composite engines (opportunities, risks, trends)
    agent/
      schema.py                        Pydantic state and output models
      memory.py                        SQLite agent persistence
      run_memory.py                    Cross-run planner memory
      tools_registry.py                Tool catalog (13 tools) with alias normalisation
      planner.py                       LLM plan generation with cross-run context
      executor.py                      Tool dispatch with chunk tracking
      analyzer.py                      Evidence synthesis
      decider.py                       Recommendation generation with auto-enrichment
      validator.py                     Four-stage validation
      briefing.py                      CEO briefing generation
      graph.py                         LangGraph state machine
      cli.py                           Command-line entry point
    dashboard/
      app.py                           Streamlit dashboard (11 tabs)
  tests/
    test_phase3.py                     Tool and engine smoke tests
    test_phase4.py                     End-to-end agent test
    test_phase5.py                     Validation layer tests
  data/
    chroma/                            ChromaDB persistent storage
    agent.db                           SQLite database
```

---

## Setup and Installation

### Prerequisites

- Python 3.11+
- Ollama installed and running
- 16 GB RAM

### Installation

```bash
git clone https://github.com/<your-username>/ai-ceo-agent.git
cd ai-ceo-agent
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
ollama pull qwen2.5:7b
```

---

## Usage

```bash
# 1. Collect documents from all 5 sources
python -m src.ingestion.scheduler

# 2. Process, index, and pre-compute sentiment
python -m src.processing.pipeline

# 3. Run the agent with a custom goal
python -m src.agent.cli "What should BMW prioritise this quarter?"

# 4. Launch the interactive dashboard
streamlit run src/dashboard/app.py

# 5. Generate a PDF report from the latest run
python -m src.tools.report_generator
```

---

## Design Decisions

**Qwen 2.5 7B over smaller models.** Produces reliable JSON output with Ollama's constrained decoding mode, reducing the need for defensive parsing workarounds. Runs on CPU in ~5 GB RAM.

**FinBERT over generic sentiment models.** Trained on financial text rather than tweets. Correctly classifies business news with high confidence (typically >0.9), where general-purpose models frequently misclassify at the neutral-negative boundary.

**Hybrid search with reciprocal rank fusion.** Pure semantic search misses exact-match queries (model numbers, ticker symbols). Pure BM25 misses conceptual similarity. RRF combines both with configurable weights (default 0.7 semantic / 0.3 BM25) without requiring score normalisation.

**Adversarial validation.** The comparator negates each recommendation's claim and retrieves corpus passages similar to the negation. An LLM then determines whether the recommendation survives the contradicting evidence. This catches plausible-sounding recommendations that the corpus actually contradicts.

**Embedding cache.** Query embeddings are SHA256-hashed and cached in SQLite. Repeated identical queries (common during re-planning) skip the embedding model entirely, achieving sub-millisecond lookups.

**Cross-run memory.** The planner receives a summary of the previous run's recommendations and briefing. Consecutive runs build on prior analysis rather than starting from scratch. This makes the system more useful as a monitoring tool over time.

**Auto-enrichment in the decider.** Small LLMs do not reliably cite exact chunk IDs from the prompt. The decider supplements the LLM's citations with additional chunks from the retriever results to ensure source diversity, rather than failing validation every time.

**Pre-computed sentiment.** Running FinBERT on 744 documents takes ~2 minutes. Doing this during processing and storing results in SQLite means the dashboard sentiment tab renders instantly instead of blocking on model inference.

**13 tools, not 3.** The agent has genuine tool diversity: vector search, keyword search, hybrid search, live financial data, competitor analysis, topic trends, entity trends, sentiment classification, contradiction detection, source scoring, and three composite engines. The planner dynamically selects which tools to use based on the goal, rather than running a fixed pipeline.

---

## License

MIT
