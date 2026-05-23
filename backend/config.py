"""
config.py – Centralised configuration for Brain Checker AI backend.
All environment variables and shared clients are initialised here.

Phase 4 changes:
  - Added GEMINI_API_KEY  (required for Gemini Embedding API)
  - Added EMBEDDING_DIMENSIONS = 768  (Gemini supports up to 3072; 768 is
    the sweet spot — much better than MiniLM's 384 while keeping Supabase
    vector storage compact)
  - Removed EMBEDDING_MODEL string (no longer a local model path)
  - CHUNK_SIZE / CHUNK_OVERLAP / TOP_K_CHUNKS unchanged — RAG logic is same
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

# ─────────────────────────────────────────
# LOAD .env  (local dev only — Render injects env vars directly)
# ─────────────────────────────────────────
_dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_dotenv_path)


# ─────────────────────────────────────────
# STARTUP VALIDATION
# ─────────────────────────────────────────
_REQUIRED_ENV_VARS = [
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "OPENROUTER_API_KEY",
    "GEMINI_API_KEY",          # ← NEW: for Gemini Embedding API
]

_missing = [k for k in _REQUIRED_ENV_VARS if not os.environ.get(k)]
if _missing:
    _msg = (
        "\n\n"
        "══════════════════════════════════════════════════\n"
        "  Brain Checker AI — MISSING ENVIRONMENT VARIABLES\n"
        "══════════════════════════════════════════════════\n"
        f"  The following required env vars are not set:\n\n"
        + "\n".join(f"    ✗  {k}" for k in _missing)
        + "\n\n"
        "  For local dev:  add them to your .env file.\n"
        "  For Render:     add them in the Environment tab\n"
        "                  of your Web Service dashboard.\n"
        "══════════════════════════════════════════════════\n"
    )
    print(_msg, file=sys.stderr)
    raise RuntimeError(_msg)


# ─────────────────────────────────────────
# SUPABASE
# ─────────────────────────────────────────
SUPABASE_URL: str              = os.environ["SUPABASE_URL"]
SUPABASE_ANON_KEY: str         = os.environ["SUPABASE_ANON_KEY"]
SUPABASE_SERVICE_ROLE_KEY: str = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

supabase: Client       = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# ─────────────────────────────────────────
# AI / OPENROUTER  (chat completions — unchanged)
# ─────────────────────────────────────────
OPENROUTER_API_KEY: str  = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
AI_MODEL: str = os.getenv("AI_MODEL", "openai/gpt-oss-120b:free")


# ─────────────────────────────────────────
# GEMINI EMBEDDING API  (replaces local sentence-transformers + torch)
# ─────────────────────────────────────────
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]

# Model: gemini-embedding-001
#   - Free tier: free of charge (official Google pricing page, May 2026)
#   - Rate limits: 1,000 requests/day, 100 requests/minute (free tier)
#   - Output dimensions: configurable 1–3072 via output_dimensionality param
#   - Task type: RETRIEVAL_DOCUMENT for storing, RETRIEVAL_QUERY for querying
#
# Why 768 dimensions?
#   - Old model (all-MiniLM-L6-v2) used 384 dims — low quality for reports
#   - Gemini max is 3072 — overkill and wastes Supabase vector storage
#   - 768 is the recommended balanced setting per Google docs
#   - Supabase pgvector handles 768-dim vectors efficiently
GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
EMBEDDING_DIMENSIONS: int   = 768   # must match vector(768) in Supabase schema


# ─────────────────────────────────────────
# APP SETTINGS
# ─────────────────────────────────────────
MONTHLY_QUESTION_LIMIT: int = int(os.getenv("MONTHLY_QUESTION_LIMIT", "40"))
SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-change-in-production")

_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5500,http://localhost:8000"
)
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]


# ─────────────────────────────────────────
# PDF PROCESSING LIMITS
# ─────────────────────────────────────────
MAX_PDF_BYTES: int = int(os.getenv("MAX_PDF_MB", "20")) * 1024 * 1024


# ─────────────────────────────────────────
# RAG PIPELINE SETTINGS
# Chunking strategy is unchanged — only the embedding step moves to API.
# ─────────────────────────────────────────
CHUNK_SIZE: int    = 1200   # chars — keeps full report sections intact
CHUNK_OVERLAP: int = 200    # chars — prevents data loss at boundaries
TOP_K_CHUNKS: int  = 10     # candidates fetched before MMR trims to best


# ─────────────────────────────────────────
# STARTUP SUMMARY
# ─────────────────────────────────────────
print(
    f"\n"
    f"  Brain Checker AI — config loaded\n"
    f"  ├─ Supabase URL        : {SUPABASE_URL}\n"
    f"  ├─ AI model            : {AI_MODEL}\n"
    f"  ├─ Embedding model     : {GEMINI_EMBEDDING_MODEL} (Gemini API, free tier)\n"
    f"  ├─ Embedding dims      : {EMBEDDING_DIMENSIONS}\n"
    f"  ├─ Monthly Q limit     : {MONTHLY_QUESTION_LIMIT}\n"
    f"  ├─ Max PDF size        : {MAX_PDF_BYTES // (1024*1024)} MB\n"
    f"  └─ CORS origins        : {ALLOWED_ORIGINS}\n"
)
