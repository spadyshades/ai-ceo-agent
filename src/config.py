"""
Central configuration for the AI CEO Agent.
All other modules import from here so we have a single source of truth.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env if present (we don't strictly need it for Phase 0)
load_dotenv()

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "agent.db"  # SQLite for ingestion log + agent memory

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Target company — BMW
# ============================================================
COMPANY_NAME = "BMW"
COMPANY_FULL_NAME = "Bayerische Motoren Werke AG"
COMPANY_TICKER = "BMW.DE"
COMPANY_INDUSTRY = "Automotive"

# Search queries used by scrapers
SEARCH_QUERIES = [
    "BMW",
    "BMW Group",
    "BMW EV",
    "BMW Neue Klasse",
    "BMW China",
    "BMW autonomous driving",
]

# Known competitors — used by analyzer for competitive intel
COMPETITORS = [
    "Mercedes-Benz",
    "Audi",
    "Tesla",
    "BYD",
    "Volkswagen",
    "Porsche",
    "NIO",
    "Hyundai",
]

# ============================================================
# LLM (Ollama)
# ============================================================
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "phi4-mini")  # fallback: "phi3:mini"
LLM_TEMPERATURE = 0.3
LLM_NUM_CTX = 4096

# ============================================================
# Embeddings
# ============================================================
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# ============================================================
# Vector store (ChromaDB)
# ============================================================
CHROMA_COLLECTION_NAME = f"{COMPANY_NAME.lower()}_corpus"

# ============================================================
# Retrieval
# ============================================================
RETRIEVAL_K = 5  # top-k for vector search
SIMILARITY_THRESHOLD = 0.5

# ============================================================
# Ingestion
# ============================================================
MIN_DOC_LENGTH_CHARS = 200  # discard docs shorter than this

GOOGLE_NEWS_RSS_URL = (
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
)
BMW_PRESS_LIST_URL = "https://www.press.bmwgroup.com/global/list.html"
HN_ALGOLIA_API = "https://hn.algolia.com/api/v1/search"

USER_AGENT = "ai-ceo-agent/0.1 (educational project)"

# ============================================================
# Agent
# ============================================================
MAX_REPLAN_ATTEMPTS = 3
MIN_EVIDENCE_PER_RECOMMENDATION = 3
MIN_DISTINCT_SOURCES_PER_RECOMMENDATION = 2
