# AI CEO Strategic Intelligence Agent

An autonomous AI agent that collects live BMW intelligence from five public sources, identifies strategic opportunities, risks, and trends, produces evidence-based CEO recommendations, and validates them adversarially before presenting a structured executive briefing.

Built as an NLP examination project. The core question the system answers: **"If you were the CEO today, what would you do next and why?"**

The system demonstrates planning, autonomous decision-making, tool usage beyond the LLM, retrieval-augmented generation, multi-step analysis, and validation -- using only open-source models running locally on CPU. No paid APIs. No GPU required.

---

## Table of Contents

- [Architecture](#architecture)
- [Development Phases](#development-phases)
- [Data Pipeline](#data-pipeline)
- [Tool Registry](#tool-registry)
- [Validation Layer](#validation-layer)
- [Executive Dashboard](#executive-dashboard)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Requirements Fulfillment](#requirements-fulfillment)
- [Design Decisions](#design-decisions)
- [Known Limitations](#known-limitations)

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

The agent is a **LangGraph state machine** with six nodes, two conditional edges, and a validation-driven replan cycle. Each node has a single responsibility. The graph's traversal depends on runtime state, not hardcoded sequencing. A typical run makes 4 direct LLM calls (planner, analyzer, decider, briefing) plus internal calls within the intelligence engines, totalling 5 to 9 LLM invocations per run.

The workflow follows the required pipeline: **Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend -> Validate**. When validation fails, the agent autonomously replans with the failure reasons injected into the planner prompt, gathering additional evidence to strengthen weak recommendations.

---

## Development Phases

The project was built incrementally across eight phases. Each phase delivers a distinct layer of the system and satisfies specific project requirements.

### Phase 0 -- Project Setup and Scaffolding

Established the development environment, project structure, and toolchain. Installed and verified Ollama with Phi-4 Mini (~2.5 GB, CPU-only). Created the Python virtual environment, directory layout, configuration module (`src/config.py`), SQLite database schema, `.gitignore`, and `requirements.txt`. Confirmed LLM responsiveness with a basic prompt test.

**What this delivers:** The foundation for all subsequent phases. Every configuration constant (model names, database paths, search queries, validation thresholds) lives in `src/config.py` as a single source of truth.

### Phase 1 -- Data Ingestion Layer

Built five automated scrapers that collect BMW-related documents from diverse public sources:

| Source | Type | Category | Method | Documents |
|--------|------|----------|--------|-----------|
| BMW Press | Official press releases | Company | HTML scraping from press.bmwgroup.com | ~50 |
| Google News RSS | News articles across 6 queries | News | RSS feed parsing via feedparser | ~260 |
| Hacker News | Tech community discussion | Community | Algolia search API | ~375 |
| arXiv | Academic research papers | Research | arXiv API (5 topic queries) | ~43 |
| Yahoo Finance | Financial data and news | Market | yfinance library | ~10 |

All scrapers inherit from a common `BaseScraper` class and are orchestrated by `src/ingestion/scheduler.py`. The scheduler is idempotent -- running it again only fetches new documents. Each document is stored in SQLite with its source, URL, title, content, and publication date.

**What this delivers:** Automatic collection of 744 documents from 5 independent sources across 4 categories (Company, News, Community, Research/Market). This exceeds the minimum requirement of 100 documents from 3 sources.

**Requirements fulfilled:** Task 1 (Live Data Collection) -- automatic collection, 100+ documents, 3+ sources, multiple categories.

### Phase 2 -- Processing Pipeline

Implemented the five-stage processing pipeline that transforms raw documents into searchable, indexed knowledge:

1. **Cleaning** -- normalise whitespace, strip HTML artifacts, handle encoding issues
2. **Deduplication** -- MinHash-based near-duplicate detection across sources. In practice, 14 of 744 documents were dropped as near-duplicates (e.g., the same BMW story appearing in both Google News and Yahoo Finance)
3. **Entity Extraction** -- spaCy NER (`en_core_web_sm`) extracts ORG, PERSON, GPE, PRODUCT, and DATE entities from each document. Entities are stored as semicolon-delimited metadata fields on each chunk for downstream filtering
4. **Chunking** -- recursive text splitting (512 tokens, 64-token overlap) produces 1,109 chunks from 730 deduplicated documents
5. **Embedding and Indexing** -- BAAI/bge-small-en-v1.5 (384-dimensional) generates embeddings for all chunks, which are upserted into a persistent ChromaDB collection with cosine distance

The pipeline is also idempotent. Running `python -m src.processing.pipeline --reprocess` regenerates all chunks and re-indexes without duplicating data.

**What this delivers:** A fully indexed knowledge repository of 1,109 chunks with metadata, ready for semantic retrieval. The processing pipeline is automatic and repeatable.

**Requirements fulfilled:** Task 2 (Knowledge Repository) -- ChromaDB indexed for fast retrieval. Task 3 (Information Processing) -- clean, deduplicate, extract, embed, index.

### Phase 3 -- Strategic Intelligence Tools

Built seven atomic tools and three composite intelligence engines. The tools are the agent's capabilities beyond the LLM itself -- each performs a specific, mostly deterministic operation.

**Atomic tools:**
- `retriever` -- semantic search over ChromaDB with optional metadata filters (source, date range) and entity-filtered hybrid search
- `web_search` -- live external news via Google News RSS, providing the agent a window to information outside the indexed corpus
- `sentiment` -- document-level sentiment classification using cardiffnlp/twitter-roberta-base-sentiment-latest (positive/neutral/negative with confidence)
- `entity_extractor` -- thin wrapper over spaCy NER for on-demand entity extraction
- `trend_detector` -- statistical entity-rise detection comparing mention frequencies across configurable time windows (e.g., 14-day recent vs. 28-day baseline)
- `comparator` -- generates a claim negation via the LLM, then retrieves corpus passages most similar to the negation. This is the foundation of the adversarial validation in Phase 5
- `source_credibility` -- rule-based tier-and-freshness scoring (Tier 1: BMW Press/arXiv, Tier 2: Google News/Yahoo Finance, Tier 3: Hacker News)

**Composite intelligence engines** (`src/intelligence/engines.py`):
- `detect_opportunities` -- retrieves evidence from 5 strategic queries, synthesises up to 5 structured opportunities via LLM with impact ratings and evidence chunk IDs
- `detect_risks` -- same retrieval pattern, but evidence is reordered by negative sentiment score before LLM synthesis
- `detect_trends` -- runs statistical entity-rise detection first, then retrieves supporting excerpts for the top rising entities, and synthesises strategic trends via LLM

**What this delivers:** A toolbox of 9 callable tools (7 atomic + 3 composite engines, minus `entity_extractor` which is used internally but removed from the agent registry since the planner cannot meaningfully invoke it). Each tool except the comparator operates without the LLM, satisfying the requirement for tool usage beyond the LLM.

**Requirements fulfilled:** Task 4 (Strategic Intelligence Engine) -- detects opportunities, risks, and trends. Agent Capability: Tool usage beyond the LLM. Agent Capability: Retrieval and use of evidence. Agent Capability: Analysis of risks, opportunities, and trends.

### Phase 4 -- LangGraph Agent

The centerpiece of the project. Phase 4 implements the six-node LangGraph state machine that composes the tools from Phase 3 into a planned, autonomous, multi-step workflow.

**Six nodes:**
- `planner` -- receives the user's goal (and validation issues on replan) and produces a structured JSON plan with 3 to 7 steps, each calling one registered tool. Falls back to a deterministic default plan if the LLM emits invalid JSON
- `executor` -- dispatches one tool per plan step, accumulates results in state, and tracks retrieved chunk references for downstream source attribution
- `analyzer` -- synthesises all gathered evidence (opportunities, risks, trends, retriever results) into a structured `AnalysisOutput` with summary, key findings, and supporting chunk IDs
- `decider` -- produces 3 to 5 ranked strategic recommendations, each with title, rationale, priority, expected impact, risk assessment, confidence, and evidence citations. Auto-enriches evidence from retrieved chunks to ensure source diversity
- `validator` -- applies rule-based checks (minimum evidence count, minimum distinct sources, minimum confidence, non-empty rationale). On failure, triggers re-planning up to 3 attempts
- `briefing` -- generates a concise CEO briefing answering: What happened? Why does it matter? What should management do next?

**Two conditional edges:**
- `route_after_execute` -- loops back to execute more plan steps, or moves to analysis when all steps are done
- `route_after_validate` -- routes to briefing on pass, back to planner on fail (up to `MAX_REPLAN_ATTEMPTS=3`), or to briefing on limit reached

**Memory persistence:** Five SQLite tables store the complete audit trail: `agent_runs` (with full state JSON), `agent_plans` (every plan version including replans), `agent_tool_calls` (each tool execution with parameters and summary), `agent_recommendations` (with priority and confidence), `agent_validations` (pass/fail with issue list).

**Parameter normalisation:** The tool registry includes alias matching and key canonicalisation to handle Phi-4 Mini's tendency to emit parameter names like `"search query"` instead of `"query"`. Unknown parameters are silently dropped rather than causing crashes.

**What this delivers:** A fully autonomous agent that plans before executing, composes tools dynamically, makes conditional decisions based on validation results, and persists its entire execution trace for auditability. A typical run takes 5 to 15 minutes on CPU.

**Requirements fulfilled:** Task 5 (AI CEO Agent) -- analyzes, reasons, prioritizes, recommends, justifies. Agent Capabilities: Planning before execution, Autonomous decision-making, Multi-step task execution, Memory. Workflow Pipeline: Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend -> Validate.

### Phase 5 -- Evidence and Validation Layer

Strengthened the validation system from simple rule checks to a four-stage pipeline:

1. **Rule-based checks** -- minimum 3 evidence chunks per recommendation, minimum 2 distinct sources per recommendation, minimum confidence of 0.2, non-empty rationale
2. **Source existence verification** -- cited chunk IDs are looked up in the Chroma collection. If more than 50% of cited IDs do not exist in the corpus, the recommendation is flagged
3. **Claim-evidence alignment** -- embedding cosine similarity between the recommendation text and its cited evidence chunks. Average similarity below 0.25 triggers a warning
4. **Adversarial LLM review** -- each of the top 3 recommendations is challenged. The comparator finds contradicting corpus passages, and the LLM determines whether the recommendation remains defensible given the contradictions

Advanced checks (stages 2-4) run only on the final validation pass to save LLM time during early replans, where rule checks alone are sufficient to drive evidence-gathering.

The decider was also updated to auto-enrich recommendations with evidence from retrieved chunks. When the LLM's cited chunk IDs do not map to enough distinct sources, the decider supplements with additional chunks from the retriever results, ensuring source diversity without relying entirely on the LLM's citation accuracy.

**What this delivers:** A multi-layer validation system that catches under-evidenced recommendations, hallucinated chunk IDs, misaligned claims, and contradicted assertions. The adversarial check is one of the system's most distinctive features -- it demonstrates that the agent does not just generate recommendations but actively challenges them.

**Requirements fulfilled:** Task 6 (Evidence-Based Recommendations) -- every recommendation includes recommendation, evidence, expected impact, and risk assessment. Agent Capability: Validation of recommendations before presenting them.

### Phase 6 -- Executive Dashboard

Built a Streamlit dashboard with 8 tabs covering all 7 required sections plus an Agent Trace tab:

1. **Overview** -- total documents, indexed count, data sources, source distribution bar chart, source tier pie chart
2. **Market Intelligence** -- latest ingested documents with source filter
3. **Opportunity Monitor** -- opportunities with impact rating, confidence, evidence chunks, and source attribution. Filterable by impact level
4. **Risk Monitor** -- risks with severity, confidence, evidence, sources. Filterable by impact level
5. **Sentiment Analysis** -- pie chart of sentiment distribution across a 100-document sample, area chart of sentiment over time, expandable table of individual classifications
6. **Strategic Recommendations** -- priority summary cards, expandable recommendation details with rationale, expected impact, risk assessment, evidence chunk count, and source list
7. **CEO Briefing** -- the agent's executive summary in a styled box
8. **Agent Trace** -- plan versions with reasoning, tool usage distribution chart, tool call table with errors, validation pass/fail history with issue details, and raw state summary

The sidebar shows run selection, run metadata, replan count, tool call count, and validation status. Custom CSS provides gradient metric cards, color-coded impact labels, and styled section explanations.

**What this delivers:** An interactive executive intelligence dashboard that renders all 7 required sections plus full agent execution traceability.

**Requirements fulfilled:** Dashboard Sections 1-7. Technical Requirement: Streamlit dashboard.

### Phase 7 -- Documentation

This README. Architecture diagram, data flow, tech stack, design decisions, AI pipeline walkthrough, phase-by-phase development explanation, and requirements fulfillment mapping.

**Requirements fulfilled:** Deliverable 3 (Architecture Documentation) -- architecture diagram, data flow, tech stack, design decisions, AI pipeline.

### Phase 8 -- Oral Exam Preparation

Demo dry-run, live coding scenarios, and concept Q&A practice. Not committed to the repository.

---

## Data Pipeline

### Flow

```
5 Scrapers  -->  SQLite (744 docs)  -->  Clean  -->  Dedupe (730 docs)
    -->  spaCy NER  -->  Chunk (1109 chunks)  -->  BGE Embed  -->  ChromaDB
```

### Ingestion Details

The scheduler runs all 5 scrapers sequentially. Each scraper returns a list of `Document` objects. `storage.save_documents()` checks the SQLite primary key (content hash) before inserting -- documents already in the database are skipped. This makes the scheduler safe to run on a cron schedule.

Google News uses 6 targeted queries: "BMW", "BMW Group", "BMW EV", "BMW Neue Klasse", "BMW China", "BMW autonomous driving". arXiv uses 5 research queries: "electric vehicle battery", "autonomous driving perception", "lithium-ion battery management", "automotive AI", "vehicle-to-grid".

### Processing Details

The deduper uses MinHash with 128 permutations and a Jaccard threshold of 0.8. In practice, 14 documents were flagged as near-duplicates -- typically the same story surfacing through both Google News and Hacker News.

Chunks are created with `RecursiveCharacterTextSplitter` at 512 tokens with 64-token overlap. Each chunk inherits metadata from its parent document (source, URL, title, published_at) plus the extracted entity fields.

---

## Tool Registry

9 tools registered in `src/agent/tools_registry.py`:

| Tool | Type | Input | Output |
|------|------|-------|--------|
| retriever | Vector search | query string, optional k | List of RetrievalHit with chunk_id, text, source, similarity |
| web_search | External API | query string, optional limit | List of SearchResult with title, URL, snippet |
| sentiment | ML classifier | text string | SentimentResult with label and score |
| trend_detector | Statistical | optional label, time windows | List of TrendingEntity with growth rate |
| find_contradicting_evidence | LLM + retrieval | claim string, optional k | List of RetrievalHit (passages contradicting the claim) |
| source_credibility | Rule-based | source name, optional datetime | Float score 0.0 to 1.0 |
| detect_opportunities | Composite engine | (none) | List of StrategicItem |
| detect_risks | Composite engine | (none) | List of StrategicItem |
| detect_trends | Composite engine | (none) | Tuple of TrendingEntity list + StrategicItem list |

The registry formats a catalog string for inclusion in planner prompts, with parameter names explicitly quoted and marked as REQUIRED or optional. This guides the LLM toward correct parameter formatting.

---

## Validation Layer

Four stages, executed conditionally:

| Stage | Type | What it checks | When it runs |
|-------|------|----------------|--------------|
| Rule checks | Deterministic | Evidence count, source diversity, confidence, rationale | Every validation pass |
| Source existence | Chroma lookup | Cited chunk IDs exist in the vector store | Final pass only |
| Claim alignment | Embedding similarity | Recommendation text is semantically close to cited evidence | Final pass only |
| Adversarial review | LLM + comparator | Recommendation survives contradicting evidence | Final pass only |

Validation failures on early passes trigger re-planning. The planner receives the specific failure reasons and produces a revised plan targeting the evidence gaps. After 3 replan attempts, the agent proceeds to briefing with whatever recommendations it has, and the failing validation result is preserved in the audit trail.

---

## Executive Dashboard

8 tabs in a Streamlit app (`src/dashboard/app.py`):

| Tab | Content | Interactive Elements |
|-----|---------|---------------------|
| Overview | KPI cards, source bar chart, tier pie chart | Run selector in sidebar |
| Market Intelligence | Latest documents by source | Source multiselect filter |
| Opportunities | Impact-rated items with evidence | Impact level filter |
| Risks | Severity-rated items with evidence | Impact level filter |
| Sentiment | Distribution pie, time-series area chart | Expandable sample table |
| Recommendations | Priority cards, expandable detail panels | Expand/collapse |
| CEO Briefing | Executive summary text | -- |
| Agent Trace | Plans, tool calls, validations, raw state | Expandable sections, tool usage chart |

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | Phi-4 Mini via Ollama | Planning, analysis, synthesis, briefing (CPU, ~2.5 GB) |
| Embedding | BAAI/bge-small-en-v1.5 | 384-dim document and query embeddings |
| Sentiment | cardiffnlp/twitter-roberta-base-sentiment-latest | 3-class sentiment classification |
| Vector DB | ChromaDB | Persistent semantic search with metadata filtering |
| Agent framework | LangGraph | State machine with conditional edges and cycles |
| NER | spaCy en_core_web_sm | Named entity recognition |
| Database | SQLite | Document storage, agent run history, audit trail |
| Dashboard | Streamlit + Plotly | Interactive executive dashboard |
| Language | Python 3.13 | -- |

No paid APIs. No GPU required. Entire stack runs locally on a consumer laptop with 16 GB RAM.

---

## Project Structure

```
ai-ceo-agent/
  src/
    config.py                          Single source of truth for all constants
    utils/
      logging.py                       Structured logging
      llm.py                           Shared Ollama client (cached, configurable temperature)
    ingestion/
      base.py                          BaseScraper abstract class
      storage.py                       SQLite document storage with idempotent insert
      bmw_press.py                     BMW press release scraper
      google_news.py                   Google News RSS scraper (6 queries)
      hackernews.py                    Hacker News Algolia API scraper
      arxiv_scraper.py                 arXiv API scraper (5 queries)
      yahoo_finance.py                 Yahoo Finance scraper via yfinance
      scheduler.py                     Ingestion orchestrator
    processing/
      cleaner.py                       Text normalisation
      deduper.py                       MinHash near-duplicate detection
      extractor.py                     spaCy NER extraction
      chunker.py                       Recursive text chunking (512 tokens, 64 overlap)
      embedder.py                      BGE-small embedding
      indexer.py                       ChromaDB upsert and collection management
      pipeline.py                      Processing orchestrator
    tools/
      retriever.py                     Semantic + metadata-filtered + entity-filtered search
      web_search.py                    External live news via Google News RSS
      sentiment.py                     HuggingFace sentiment classification
      entity_extractor.py              spaCy NER wrapper
      trend_detector.py                Statistical entity-rise detection
      comparator.py                    Claim negation + contradiction retrieval
      source_credibility.py            Tier-and-freshness scoring
    intelligence/
      engines.py                       Composite engines (opportunities, risks, trends)
    agent/
      schema.py                        Pydantic state, plan, recommendation, validation models
      memory.py                        SQLite persistence (5 tables for full audit trail)
      tools_registry.py                Tool catalog with alias normalisation
      planner.py                       LLM-based plan generation with fallback
      executor.py                      Tool dispatch with retrieved-chunk tracking
      analyzer.py                      Evidence synthesis
      decider.py                       Recommendation generation with auto-enrichment
      validator.py                     4-stage validation (rules, existence, alignment, adversarial)
      briefing.py                      CEO briefing generation
      graph.py                         LangGraph state machine wiring
      cli.py                           Command-line entry point
    dashboard/
      app.py                           Streamlit executive dashboard (8 tabs)
  tests/
    test_phase3.py                     Tool and engine smoke tests (9 tests)
    test_phase4.py                     End-to-end agent smoke test (3 tests)
    test_phase5.py                     Validation layer tests (4 tests)
  data/
    raw/                               Raw scraped data
    processed/                         Processed outputs
    chroma/                            ChromaDB persistent storage
    agent.db                           SQLite database
```

---

## Setup and Installation

### Prerequisites

- Python 3.11 or later
- Ollama installed and running
- 16 GB RAM recommended (Phi-4 Mini uses ~2.5 GB, embedding model uses ~130 MB, sentiment model uses ~500 MB)

### Installation

```bash
git clone https://github.com/<your-username>/ai-ceo-agent.git
cd ai-ceo-agent
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
ollama pull phi4-mini
```

---

## Usage

### 1. Collect documents

```bash
python -m src.ingestion.scheduler
```

Fetches documents from all 5 sources. Safe to run repeatedly -- only new documents are added.

### 2. Process and index

```bash
python -m src.processing.pipeline
```

Cleans, deduplicates, extracts entities, chunks, embeds, and indexes into ChromaDB. Use `--reprocess` to force a full rebuild.

### 3. Run the agent

```bash
python -m src.agent.cli "What should BMW prioritise this quarter?"
```

Or with the default goal:

```bash
python -m src.agent.cli
```

Typical runtime: 5 to 15 minutes on CPU. The CLI prints the full plan, tool calls, analysis, recommendations, validation result, and CEO briefing.

### 4. Launch the dashboard

```bash
streamlit run src/dashboard/app.py
```

Opens in your browser at `localhost:8501`. Select an agent run from the sidebar and explore the 8 tabs.

### 5. Run tests

```bash
python -m tests.test_phase3    # Tool and engine smoke tests
python -m tests.test_phase4    # End-to-end agent test
python -m tests.test_phase5    # Validation layer tests
```

---

## Requirements Fulfillment

### Functional Requirements (Tasks 1-6)

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Task 1: At least 100 documents from 3+ sources, automatic | Complete | 744 documents from 5 sources (BMW Press, Google News, Hacker News, arXiv, Yahoo Finance). Fully automatic via scheduler |
| Task 2: Knowledge repository, indexed for retrieval | Complete | ChromaDB persistent collection with 1,109 chunks. Cosine distance, metadata filtering |
| Task 3: Clean, deduplicate, extract, embed, index | Complete | 5-stage pipeline: cleaner, MinHash deduper, spaCy NER, recursive chunker, BGE embedder |
| Task 4: Detect opportunities, risks, trends | Complete | 3 composite engines in `src/intelligence/engines.py`, each combining retrieval + classification + LLM synthesis |
| Task 5: AI CEO agent that analyzes, reasons, prioritizes, recommends, justifies | Complete | 6-node LangGraph agent with planning, execution, analysis, decision, validation, and briefing |
| Task 6: Evidence-based recommendations with evidence, impact, risk | Complete | Pydantic `Recommendation` schema with all 4 required blocks. Auto-enriched evidence. Adversarial validation |

### Agent Capabilities

| Capability | Status | Implementation |
|------------|--------|----------------|
| Planning before execution | Complete | `planner.py` produces structured JSON plans visible in the Agent Trace tab |
| Autonomous decision-making | Complete | Two conditional edges in the graph. Validation-driven replan cycle. Plan varies with goal |
| Tool usage beyond the LLM | Complete | 9 registered tools. Only the comparator uses the LLM; all others are deterministic or use non-LLM models |
| Memory | Complete | 5 SQLite tables persist full audit trail: runs, plans, tool calls, recommendations, validations |
| Multi-step task execution | Complete | 3 to 10+ tool calls per run, accumulated in state across plan steps and replans |
| Retrieval and use of evidence | Complete | Semantic search with source attribution. Every recommendation carries evidence_chunk_ids and evidence_sources |
| Analysis of risks, opportunities, trends | Complete | Three intelligence engines + analyzer node |
| Validation before presenting | Complete | 4-stage validator: rules, source existence, claim alignment, adversarial LLM review |

### Workflow Pipeline

| Stage | Status | Implementation |
|-------|--------|----------------|
| Goal | Complete | `AgentState.goal` from CLI or API |
| Plan | Complete | `planner_node` produces Plan with reasoning and steps |
| Retrieve | Complete | `executor` dispatches retriever and web_search per plan |
| Analyze | Complete | `analyze_node` synthesises AnalysisOutput |
| Decide | Complete | `decide_node` produces ranked Recommendations |
| Recommend | Complete | Pydantic schema with all required fields |
| Validate | Complete | 4-stage validation with autonomous replan on failure |

### Dashboard (7 Required Sections)

| Section | Status | Tab |
|---------|--------|-----|
| Company Overview | Complete | Overview tab |
| Market Intelligence | Complete | Market Intelligence tab |
| Opportunity Monitor | Complete | Opportunities tab |
| Risk Monitor | Complete | Risks tab |
| Sentiment Analysis | Complete | Sentiment tab (pie chart + time series + sample table) |
| Strategic Recommendations | Complete | Recommendations tab |
| CEO Briefing | Complete | CEO Briefing tab |

### Deliverables

| Deliverable | Weight | Status |
|-------------|--------|--------|
| Working Prototype | 30% | Complete |
| Executive Dashboard | 10% | Complete |
| Architecture Documentation | 10% | Complete (this README) |
| Oral Examination | 50% | Prepared |

### Anti-Patterns Avoided

| Anti-Pattern | How Avoided |
|--------------|-------------|
| "User -> Prompt -> LLM + RAG -> Response" | 6-node graph with planning, tools, analysis, validation. The agent is not a single RAG call |
| Single LLM call with no planning/tools/validation | Each LLM call has a specific narrow role. Tool calls bracket every LLM synthesis. Validation enforces output quality |
| Hardcoded sequence with no autonomous decisions | Two conditional edges. Replan cycle driven by validation state. Plan varies with goal |
| Paid commercial LLMs | Phi-4 Mini via Ollama only. No OpenAI/Anthropic/Gemini APIs |

---

## Design Decisions

**Why Phi-4 Mini?** The project requires an open-source LLM with no paid APIs. Phi-4 Mini runs on CPU in ~2.5 GB RAM, produces coherent JSON output most of the time, and handles strategic synthesis adequately for a demonstration. Its main weakness is inconsistent JSON formatting, which the system handles with multi-strategy parsing (direct parse, key quoting, trailing comma removal, individual object extraction) and parameter alias normalisation.

**Why LangGraph over a simple chain?** The project explicitly penalises hardcoded sequences without autonomous decision points. LangGraph provides conditional edges and cycles, enabling the validation-driven replan loop. The graph structure also makes the execution trace inspectable via the Agent Trace dashboard tab.

**Why ChromaDB over FAISS?** ChromaDB provides persistent storage and metadata filtering out of the box. The retriever uses metadata filters (source, date range) alongside semantic search, which FAISS does not support natively without additional infrastructure.

**Why 5 sources instead of 3?** Five sources across 4 categories (Company, News, Community, Research/Market) provide genuine source diversity. The validator's "minimum 2 distinct sources per recommendation" rule is meaningful when recommendations can draw from BMW Press, Google News, Hacker News, arXiv, and Yahoo Finance.

**Why adversarial validation?** The comparator uses LLM-generated claim negation followed by similarity search. This catches recommendations that sound plausible but are contradicted by evidence in the corpus. It demonstrates that the agent actively challenges its own outputs rather than trusting them uncritically.

**Why auto-enrichment in the decider?** Phi-4 Mini does not reliably cite exact chunk IDs from the prompt. The decider supplements the LLM's citations with additional chunks from the retriever results to ensure source diversity. This is a pragmatic design choice for working with a small model on CPU.

---

## Known Limitations

- **Phi-4 Mini JSON adherence:** The model occasionally emits unquoted property names, trailing commas, or incorrect parameter keys. The engines and tool registry handle this with multi-strategy JSON parsing and alias normalisation, but some runs produce fewer structured items than expected
- **spaCy entity quality on headlines:** The small English model sometimes mislabels news headlines as organisation names (e.g., "BMW Made A Cheaper X5" tagged as ORG). This affects entity-filtered search and trend detection but does not break the pipeline
- **Sentiment model domain mismatch:** cardiffnlp was trained on tweets, not news articles. Confidence scores for neutral-vs-negative boundary cases are occasionally unreliable (e.g., an obviously negative headline classified as neutral with 0.527 confidence)
- **CPU inference speed:** A full agent run takes 5 to 15 minutes on CPU. Acceptable for a prototype; would require GPU for production use
- **Validation may still fail on final pass:** The adversarial checker can flag recommendations that have weak corpus support. This is by design -- it shows the validator is doing real work -- but means some runs complete with a FAILED validation status
- **No real-time streaming:** The corpus is a snapshot. The scheduler can be run periodically, but the system does not stream live data

---

## License

Academic project. Not intended for production use.
