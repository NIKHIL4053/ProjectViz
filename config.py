"""
config.py
---------
# * Central configuration for the entire Loan Dashboard project.
# * Every other file imports constants from here — never hardcode values elsewhere.
# * Sensitive values (credentials, secrets) are loaded from .env via python-dotenv.

# ! NEVER commit the .env file to git.
# ! NEVER hardcode credentials directly in this file.
# ? Why centralise config here?
# ? If a path, model name, or setting changes — you change it in ONE place.
# ? Every other file just imports from config.py and picks it up automatically.

Usage in any file:
    from config import (
        CODER_MODEL, FAST_MODEL,
        OLLAMA_BASE_URL,
        CONTEXT_DIR,
        DATA_TEMP_DIR,
        LOG_DIR,
        PG_DSN, PG_TABLE_NAME, PG_SCHEMA,
 )
"""


import os
from pathlib import Path
from dotenv import load_dotenv

# * Load .env file from project root before reading any env variables
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


# ──────────────────────────────────────────────────────────────────────────────
# * PROJECT PATHS
# * All paths are absolute — derived from this file's location.
# * Works regardless of where you run the app from.
# ──────────────────────────────────────────────────────────────────────────────

# * Root of the project — D:\Loan_Dashboard\
ROOT_DIR      = Path(__file__).resolve().parent

# * Subfolder paths
LOG_DIR       = ROOT_DIR / "Logs"
DATA_DIR      = ROOT_DIR / "Data"
CONTEXT_DIR   = ROOT_DIR / "Context"
UTILS_DIR     = ROOT_DIR / "Utils"

# * Temporary data folder — DAX results stored here as parquet
# ! This folder is wiped on session end — do NOT store permanent files here
DATA_TEMP_DIR = DATA_DIR / "uploads"

# * ChromaDB persistent storage
CHROMADB_PATH = DATA_DIR / "chromadb"

# * Create folders if they don't exist on first run
for _dir in [LOG_DIR, DATA_DIR, DATA_TEMP_DIR, CONTEXT_DIR, CHROMADB_PATH]:
    _dir.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# * CONTEXT JSON PATHS
# ──────────────────────────────────────────────────────────────────────────────

SCHEMA_CONTEXT_PATH   = CONTEXT_DIR / "01_schema_context.json"
BUSINESS_LOGIC_PATH   = CONTEXT_DIR / "02_business_logic.json"
LANGUAGE_MAPPING_PATH = CONTEXT_DIR / "03_language_mapping.json"
VISUALIZATION_PATH    = CONTEXT_DIR / "04_visualization_context.json"
SQL_PATTERNS_PATH     = CONTEXT_DIR / "05_sql_patterns.json"


# ──────────────────────────────────────────────────────────────────────────────
# * OLLAMA — MODEL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
OLLAMA_CHAT_URL    = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_URL    = f"{OLLAMA_BASE_URL}/api/tags"

# * Primary model — intent analysis, clarifying questions, DAX generation
# ? Change to qwen2.5-coder:32b if your GPU has 24GB+ VRAM
CODER_MODEL        = os.getenv("CODER_MODEL",      "qwen2.5-coder:14b")

# * Secondary model — chart type decisions only
FAST_MODEL         = os.getenv("FAST_MODEL",       "qwen2.5:7b")

OLLAMA_TIMEOUT     = int(os.getenv("OLLAMA_TIMEOUT",     "120"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE","0.1"))
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES",  "2"))



# ──────────────────────────────────────────────────────────────────────────────
# * POSTGRESQL — DATABASE CONFIGURATION (Docker)
# ──────────────────────────────────────────────────────────────────────────────

PG_HOST         = os.getenv("PG_HOST",      "localhost")
PG_PORT         = int(os.getenv("PG_PORT",  "5432"))
PG_DATABASE     = os.getenv("PG_DATABASE",  "loan_db")
PG_USER         = os.getenv("PG_USER",      "postgres")
PG_PASSWORD     = os.getenv("PG_PASSWORD",  "")
PG_SCHEMA       = os.getenv("PG_SCHEMA",    "public")
PG_TABLE_NAME   = os.getenv("PG_TABLE_NAME","final_table")
PG_MAX_ROWS     = int(os.getenv("PG_MAX_ROWS", "100000"))

# * Full DSN string — used by psycopg2 / SQLAlchemy
PG_DSN          = (
    f"postgresql://{PG_USER}:{PG_PASSWORD}"
    f"@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
)

USE_MOCK_DATA   = os.getenv("USE_MOCK_DATA", "false").lower() == "true"

# ──────────────────────────────────────────────────────────────────────────────
# * CHROMADB
# ──────────────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
CHROMA_COLLECTION  = "loan_dashboard_context"
CHROMA_TOP_K       = int(os.getenv("CHROMA_TOP_K", "10"))



# ──────────────────────────────────────────────────────────────────────────────
# * DATA STORAGE
# ? Parquet + snappy chosen over pickle/xlsx/csv:
# ?   89K rows reads in ~0.3s vs ~3s for Excel
# ?   Snappy compression cuts file size ~70%
# ?   Type-safe across save/load — pickle is not cross-platform safe
# ──────────────────────────────────────────────────────────────────────────────

PARQUET_COMPRESSION     = os.getenv("PARQUET_COMPRESSION", "snappy")


# ──────────────────────────────────────────────────────────────────────────────
# * DATA HANDLING
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_ROWS_FOR_MODEL   = int(os.getenv("SAMPLE_ROWS_FOR_MODEL",  "5"))
MAX_CONTEXT_TOKENS      = int(os.getenv("MAX_CONTEXT_TOKENS",     "6000"))
TEMP_DATA_TTL_MINUTES   = int(os.getenv("TEMP_DATA_TTL_MINUTES",  "60"))


# ──────────────────────────────────────────────────────────────────────────────
# * CHART SETTINGS
# ──────────────────────────────────────────────────────────────────────────────

CHART_DPI               = int(os.getenv("CHART_DPI",             "150"))
MAX_CHART_CATEGORIES    = int(os.getenv("MAX_CHART_CATEGORIES",   "15"))
SCATTER_SAMPLE_ROWS     = int(os.getenv("SCATTER_SAMPLE_ROWS",    "600"))
PDF_PAGE_SIZE           = os.getenv("PDF_PAGE_SIZE",              "A4")
PDF_ORIENTATION         = os.getenv("PDF_ORIENTATION",            "landscape")


# ──────────────────────────────────────────────────────────────────────────────
# * STREAMLIT UI
# ──────────────────────────────────────────────────────────────────────────────

APP_TITLE               = "Loan Collection Analytics"
APP_ICON                = "📊"
APP_LAYOUT              = "wide"
MAX_SIDEBAR_SLICERS     = int(os.getenv("MAX_SIDEBAR_SLICERS",    "5"))
MAX_CHAT_HISTORY        = int(os.getenv("MAX_CHAT_HISTORY",       "20"))


# ──────────────────────────────────────────────────────────────────────────────
# * LOGGING
# ──────────────────────────────────────────────────────────────────────────────

LOG_LEVEL               = os.getenv("LOG_LEVEL",    "DEBUG")
LOG_ROTATION            = os.getenv("LOG_ROTATION", "10 MB")
LOG_RETENTION           = int(os.getenv("LOG_RETENTION", "5"))


# ──────────────────────────────────────────────────────────────────────────────
# * VALIDATION — called by app.py on startup
# ──────────────────────────────────────────────────────────────────────────────

def validate_config() -> list[str]:
    """
    # * Check critical config values are set correctly.
    # * Returns list of warning strings — empty list means all good.
    # ? Warns instead of crashing so app still runs in mock/demo mode.
    """
    warnings = []

    # * Check context JSONs exist
    for name, path in [
        ("Schema Context",    SCHEMA_CONTEXT_PATH),
        ("Business Logic",    BUSINESS_LOGIC_PATH),
        ("Language Mapping",  LANGUAGE_MAPPING_PATH),
        ("Visualization",     VISUALIZATION_PATH),
        ("SQL Patterns",      SQL_PATTERNS_PATH),
    ]:
        if not path.exists():
            warnings.append(f"Context JSON missing: {name} → {path}")
         # * Check PostgreSQL password is set
        if not PG_PASSWORD and not USE_MOCK_DATA:
                warnings.append("PG_PASSWORD is not set in .env — database connection will fail")


    return warnings