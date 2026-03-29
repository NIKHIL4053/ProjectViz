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
ROOT_DIR    = Path(__file__).resolve().parent

# * Subfolder paths
LOG_DIR     = ROOT_DIR / "Logs"
DATA_DIR    = ROOT_DIR / "Data"
CONTEXT_DIR = ROOT_DIR / "Context"            # stores all 5 context JSONs
UTILS_DIR   = ROOT_DIR / "Utils"

# * Temporary data folder — uploaded Excel and DAX results go here
# ! This folder is wiped on session end — do NOT store permanent files here
DATA_TEMP_DIR = DATA_DIR / "uploads"

# * ChromaDB persistent storage location
# ? ChromaDB stores embeddings locally so it doesn't re-embed on every restart
CHROMADB_PATH = DATA_DIR / "chromadb"

# * Create folders if they don't exist on first run
for _dir in [LOG_DIR, DATA_DIR, DATA_TEMP_DIR, CONTEXT_DIR, CHROMADB_PATH]:
    _dir.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# * CONTEXT JSON PATHS
# * The 5 JSON files that get loaded into ChromaDB and injected into prompts.
# ──────────────────────────────────────────────────────────────────────────────

SCHEMA_CONTEXT_PATH      = CONTEXT_DIR / "01_schema_context.json"
BUSINESS_LOGIC_PATH      = CONTEXT_DIR / "02_business_logic.json"
LANGUAGE_MAPPING_PATH    = CONTEXT_DIR / "03_language_mapping.json"
VISUALIZATION_PATH       = CONTEXT_DIR / "04_visualization_context.json"
DAX_PATTERNS_PATH        = CONTEXT_DIR / "05_dax_patterns.json"


# ──────────────────────────────────────────────────────────────────────────────
# * OLLAMA — MODEL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# * Base URL for locally running Ollama instance
# ? Default port is 11434 — change only if you've configured Ollama differently
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_URL  = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_URL  = f"{OLLAMA_BASE_URL}/api/tags"       # used to check if model is loaded

# * Primary model — Qwen Coder for intent analysis, clarifying questions, DAX generation
# ? Change to qwen2.5-coder:32b if your GPU has 24GB+ VRAM
CODER_MODEL      = os.getenv("CODER_MODEL", "qwen2.5-coder:14b")

# * Secondary model — Qwen 7B for chart type decisions (lightweight task)
FAST_MODEL       = os.getenv("FAST_MODEL", "qwen2.5:7b")

# * Ollama request settings
OLLAMA_TIMEOUT        = int(os.getenv("OLLAMA_TIMEOUT", "120"))       # seconds
OLLAMA_TEMPERATURE    = float(os.getenv("OLLAMA_TEMPERATURE", "0.1")) # low = deterministic
OLLAMA_MAX_RETRIES    = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))     # retry on timeout


# ──────────────────────────────────────────────────────────────────────────────
# * CHROMADB — VECTOR STORE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# * Embedding model used to embed the data dictionary chunks
# ? all-MiniLM-L6-v2 is small (~80MB), fast, runs fully locally, good enough for this use case
EMBEDDING_MODEL       = "all-MiniLM-L6-v2"

# * Collection name inside ChromaDB
CHROMA_COLLECTION     = "loan_dashboard_context"

# * How many relevant chunks to fetch per query
# ? 10 is a good balance — enough context without overloading the prompt
CHROMA_TOP_K          = int(os.getenv("CHROMA_TOP_K", "10"))


# ──────────────────────────────────────────────────────────────────────────────
# * POWER BI — CONFIGURATION
# * TODO: Update these values once Power BI fetcher code is found
# ! All values below are placeholders — replace with real credentials in .env
# ──────────────────────────────────────────────────────────────────────────────

# * Azure AD credentials for Service Principal authentication
POWERBI_TENANT_ID     = os.getenv("POWERBI_TENANT_ID",     "YOUR_TENANT_ID_HERE")
POWERBI_CLIENT_ID     = os.getenv("POWERBI_CLIENT_ID",     "YOUR_CLIENT_ID_HERE")
POWERBI_CLIENT_SECRET = os.getenv("POWERBI_CLIENT_SECRET", "YOUR_CLIENT_SECRET_HERE")

# * Power BI workspace and dataset
POWERBI_WORKSPACE_ID  = os.getenv("POWERBI_WORKSPACE_ID",  "YOUR_WORKSPACE_ID_HERE")
POWERBI_DATASET_ID    = os.getenv("POWERBI_DATASET_ID",    "YOUR_DATASET_ID_HERE")

# * Power BI API endpoints
# ? These are standard Microsoft endpoints — unlikely to need changing
POWERBI_AUTH_URL      = f"https://login.microsoftonline.com/{POWERBI_TENANT_ID}/oauth2/v2.0/token"
POWERBI_SCOPE         = "https://analysis.windows.net/powerbi/api/.default"
POWERBI_EXECUTE_URL   = (
    "https://api.powerbi.com/v1.0/myorg/groups/"
    f"{POWERBI_WORKSPACE_ID}/datasets/"
    f"{POWERBI_DATASET_ID}/executeQueries"
)

# * Power BI table name used in DAX queries
# ! Replace with the actual table name from your Power BI dataset
# ? This is the table name you see in Power BI Desktop → Data view
POWERBI_TABLE_NAME    = os.getenv("POWERBI_TABLE_NAME", "YOUR_TABLE_NAME_HERE")

# * Flag to use mock data instead of real Power BI API
# ? Set USE_MOCK_DATA=true in .env during development / when PBI code is not ready
USE_MOCK_DATA         = os.getenv("USE_MOCK_DATA", "true").lower() == "true"


# ──────────────────────────────────────────────────────────────────────────────
# * DATA HANDLING
# ──────────────────────────────────────────────────────────────────────────────

# * Allowed Excel file extensions for upload
ALLOWED_EXTENSIONS    = [".xlsx", ".xls"]

# * Max file size for upload in MB
# ! Files larger than this will be rejected at upload to prevent memory issues
MAX_UPLOAD_SIZE_MB    = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))

# * Max rows to sample when sending data preview to Qwen
# ? We never send full data to the model — only a sample for context
SAMPLE_ROWS_FOR_MODEL = int(os.getenv("SAMPLE_ROWS_FOR_MODEL", "5"))

# * Token budget for context injection into prompts
# ? Qwen 14B context window is 128K tokens — we cap at 6000 to leave room for response
MAX_CONTEXT_TOKENS    = int(os.getenv("MAX_CONTEXT_TOKENS", "6000"))

# * Temp data TTL — how long temp session files live (in minutes)
# ? After this, the session cleanup will delete the files from Data/uploads/
TEMP_DATA_TTL_MINUTES = int(os.getenv("TEMP_DATA_TTL_MINUTES", "60"))


# ──────────────────────────────────────────────────────────────────────────────
# * CHART SETTINGS
# ──────────────────────────────────────────────────────────────────────────────

# * Default chart output DPI — used for PDF export
CHART_DPI             = int(os.getenv("CHART_DPI", "150"))

# * Max categories to show on x-axis before truncating
# ? More than 15 categories on an axis becomes unreadable
MAX_CHART_CATEGORIES  = int(os.getenv("MAX_CHART_CATEGORIES", "15"))

# * Max rows to sample for scatter plots (scatter gets slow with 89K rows)
SCATTER_SAMPLE_ROWS   = int(os.getenv("SCATTER_SAMPLE_ROWS", "600"))

# * PDF export page size
PDF_PAGE_SIZE         = os.getenv("PDF_PAGE_SIZE", "A4")   # A4 or A3
PDF_ORIENTATION       = os.getenv("PDF_ORIENTATION", "landscape")


# ──────────────────────────────────────────────────────────────────────────────
# * STREAMLIT UI SETTINGS
# ──────────────────────────────────────────────────────────────────────────────

APP_TITLE             = "Loan Collection Analytics"
APP_ICON              = "📊"
APP_LAYOUT            = "wide"

# * Max number of sidebar slicers to show at once
MAX_SIDEBAR_SLICERS   = int(os.getenv("MAX_SIDEBAR_SLICERS", "5"))

# * Max chat history messages to keep in session state
MAX_CHAT_HISTORY      = int(os.getenv("MAX_CHAT_HISTORY", "20"))


# ──────────────────────────────────────────────────────────────────────────────
# * LOGGING SETTINGS
# ──────────────────────────────────────────────────────────────────────────────

# * Log level for console output during development
# ? Set to INFO in production to reduce noise
LOG_LEVEL             = os.getenv("LOG_LEVEL", "DEBUG")

# * Max log file size before rotation
LOG_ROTATION          = os.getenv("LOG_ROTATION", "10 MB")

# * How many rotated log files to keep
LOG_RETENTION         = int(os.getenv("LOG_RETENTION", "5"))


# ──────────────────────────────────────────────────────────────────────────────
# * VALIDATION — Run on import to catch missing config early
# ──────────────────────────────────────────────────────────────────────────────

def validate_config() -> list[str]:
    """
    # * Check that critical config values are set.
    # * Returns a list of warning messages — empty list means all good.
    # * Called by app.py on startup to surface config issues in the UI.

    # ? We warn instead of crash so the app can still run in mock/demo mode
    # ? even when Power BI credentials are not yet configured.
    """
    warnings = []

    # * Check context JSON files exist
    for name, path in [
        ("Schema Context",       SCHEMA_CONTEXT_PATH),
        ("Business Logic",       BUSINESS_LOGIC_PATH),
        ("Language Mapping",     LANGUAGE_MAPPING_PATH),
        ("Visualization",        VISUALIZATION_PATH),
        ("DAX Patterns",         DAX_PATTERNS_PATH),
    ]:
        if not path.exists():
            warnings.append(f"Context JSON missing: {name} → {path}")

    # * Check Ollama is likely reachable (just URL format check here — real check in ollama_client.py)
    if not OLLAMA_BASE_URL.startswith("http"):
        warnings.append(f"OLLAMA_BASE_URL looks wrong: {OLLAMA_BASE_URL}")

    # * Warn if Power BI credentials are still placeholders
    # TODO: Remove this check once Power BI code is integrated
    if not USE_MOCK_DATA:
        for name, val in [
            ("POWERBI_TENANT_ID",     POWERBI_TENANT_ID),
            ("POWERBI_CLIENT_ID",     POWERBI_CLIENT_ID),
            ("POWERBI_CLIENT_SECRET", POWERBI_CLIENT_SECRET),
            ("POWERBI_WORKSPACE_ID",  POWERBI_WORKSPACE_ID),
            ("POWERBI_DATASET_ID",    POWERBI_DATASET_ID),
            ("POWERBI_TABLE_NAME",    POWERBI_TABLE_NAME),
        ]:
            if "YOUR_" in val:
                warnings.append(f"{name} is still a placeholder — update in .env")

    return warnings