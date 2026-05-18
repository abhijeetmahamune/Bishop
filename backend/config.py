"""
config.py – Centralised configuration for Brain Checker AI backend.
All environment variables and shared clients are initialised here.

Production changes (pre-deploy):
  - Startup validation: raises RuntimeError with a clear message if any
    required env var is missing, instead of a cryptic KeyError.
  - ALLOWED_ORIGINS default changed from localhost:5500 to the Render URL
    placeholder so the server does not silently block all CORS in production.
  - PDF limits exported here so main.py and any future route can import them
    from a single source of truth.
  - Old commented-out Phase 1 code removed — file is now clean.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

# ─────────────────────────────────────────
# LOAD .env  (local dev only — Render injects env vars directly)
# ─────────────────────────────────────────
# .env lives one level above backend/ at the project root.
# On Render this file does not exist and load_dotenv() is a no-op — safe.
_dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_dotenv_path)


# ─────────────────────────────────────────
# STARTUP VALIDATION
# Fail loudly at boot with a clear message rather than a cryptic KeyError
# mid-request when the server has been running for hours.
# ─────────────────────────────────────────
_REQUIRED_ENV_VARS = [
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "OPENROUTER_API_KEY",
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

# Public client  — respects RLS, use for user-scoped read/writes
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Admin client   — bypasses RLS, use only in trusted server-side code
supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# ─────────────────────────────────────────
# AI / OPENROUTER
# ─────────────────────────────────────────
OPENROUTER_API_KEY: str  = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# Override via env var if you want to switch models without a redeploy.
# Defaults to the free GPT OSS 120B model on OpenRouter.
AI_MODEL: str = os.getenv("AI_MODEL", "openai/gpt-oss-120b:free")


# ─────────────────────────────────────────
# APP SETTINGS
# ─────────────────────────────────────────
MONTHLY_QUESTION_LIMIT: int = int(os.getenv("MONTHLY_QUESTION_LIMIT", "40"))

# Not currently used for JWT signing (Supabase handles that) but kept for
# any future HMAC / webhook verification needs.
SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-change-in-production")

# CORS allowed origins.
# Single-service Render deploy: set ALLOWED_ORIGINS to your Render URL.
# Split deploy (Render backend + Netlify frontend): set to Netlify URL.
# Multiple origins supported — separate with commas in the env var:
#   ALLOWED_ORIGINS=https://bishop.onrender.com,https://bishop.netlify.app
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5500,http://localhost:8000,https://bishop-brainchecker-di7m.onrender.com"   # safe local dev defaults
)
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]


# ─────────────────────────────────────────
# PDF PROCESSING LIMITS
# Centralised here so main.py and any future route import from one place.
# ─────────────────────────────────────────
MAX_PDF_BYTES: int = int(os.getenv("MAX_PDF_MB", "20")) * 1024 * 1024
# Default 20 MB. Override with MAX_PDF_MB env var (e.g. MAX_PDF_MB=10).


# ─────────────────────────────────────────
# RAG PIPELINE SETTINGS  (Phase 2 tuned)
# ─────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # 384-dim, runs locally, no API cost
CHUNK_SIZE: int      = 1200   # Larger chunks keep full report sections intact
CHUNK_OVERLAP: int   = 200    # Overlap prevents data loss at chunk boundaries
TOP_K_CHUNKS: int    = 10     # Candidates fetched before MMR trims to best ones


# ─────────────────────────────────────────
# STARTUP SUMMARY  (printed to Render logs for easy debugging)
# ─────────────────────────────────────────
print(
    f"\n"
    f"  Brain Checker AI — config loaded\n"
    f"  ├─ Supabase URL      : {SUPABASE_URL}\n"
    f"  ├─ AI model          : {AI_MODEL}\n"
    f"  ├─ Monthly Q limit   : {MONTHLY_QUESTION_LIMIT}\n"
    f"  ├─ Max PDF size      : {MAX_PDF_BYTES // (1024*1024)} MB\n"
    f"  ├─ CORS origins      : {ALLOWED_ORIGINS}\n"
    f"  └─ Embedding model   : {EMBEDDING_MODEL}\n"
)
