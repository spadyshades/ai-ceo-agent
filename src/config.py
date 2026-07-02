"""
Central configuration for the AI CEO Agent.
All other modules import from here so we have a single source of truth.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "agent.db"

for d in [RAW_DIR, PROCESSED_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Target company
COMPANY_NAME = "BMW"
COMPANY_FULL_NAME = "Bayerische Motoren Werke AG"
COMPANY_TICKER = "BMW.DE"
COMPANY_INDUSTRY = "Automotive"

SEARCH_QUERIES = [
    "BMW",
    "BMW Group",
    "BMW EV",
    "BMW Neue Klasse",
    "BMW China",
    "BMW autonomous driving",
]

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

# LLM (Ollama)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_TEMPERATURE = 0.3
LLM_NUM_CTX = 4096

# Embeddings
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Vector store
CHROMA_COLLECTION_NAME = f"{COMPANY_NAME.lower()}_corpus"

# Retrieval
RETRIEVAL_K = 5
SIMILARITY_THRESHOLD = 0.5

# Ingestion
MIN_DOC_LENGTH_CHARS = 200

GOOGLE_NEWS_RSS_URL = (
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
)
BMW_PRESS_LIST_URL = "https://www.press.bmwgroup.com/global"
HN_ALGOLIA_API = "https://hn.algolia.com/api/v1/search"

USER_AGENT = "ai-ceo-agent/0.1 (educational project)"

# Agent
MAX_REPLAN_ATTEMPTS = 3
MIN_EVIDENCE_PER_RECOMMENDATION = 3
MIN_DISTINCT_SOURCES_PER_RECOMMENDATION = 2
