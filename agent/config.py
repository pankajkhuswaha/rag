import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# File & Storage Paths
DOCUMENTS_DIR = "./documents"
PDF_PATH = "./documents/prospectous.pdf"
DB_DIR = "./chroma_db"

# LLM Provider: 'ollama' (Local) or 'groq' (Cloud)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq" if os.getenv("GROQ_API_KEY") else "ollama").lower()

# Embedding Provider: 'auto', 'huggingface' (Cloud/No Ollama), or 'ollama'
# 'huggingface' runs completely inside Python CPU using sentence-transformers (Zero Ollama needed!)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "auto").lower()
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Local Ollama Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Cloud Groq Settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# Local Reranker Model
RERANKER_MODEL = "BAAI/bge-reranker-base"

# Chunking & Retrieval Parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
INITIAL_FETCH_K = 9        # Broad candidates fetched from ChromaDB
FINAL_TOP_K = 3            # High-precision chunks passed to LLM after reranking
SCORE_THRESHOLD = 0.25     # Minimum Cross-Encoder score; chunks below this are discarded
STRICT_GROUNDING = True    # When True, immediately refuse if no chunks meet threshold

# Debug Flag
DEBUG_PROMPTS = False      # Set to False in production to hide prompt logs
