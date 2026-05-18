"""
main.py – Brain Checker AI Backend (Phase 3 — Production Ready)
===============================================================
All Phase 3 features intact, plus pre-deployment upgrades:

  New in this version:
  ├─ PDF size limit check       (MAX_PDF_BYTES from config)
  ├─ PDF magic byte validation  (%PDF header check)
  ├─ User-friendly error msgs   (every HTTPException speaks plain English)
  ├─ Custom 404 HTML page       (serves 404.html instead of JSON)
  ├─ /privacy and /terms routes (serve legal pages)
  ├─ Startup path sanity check  (clear log if index.html is missing)
  └─ Import of MAX_PDF_BYTES    (single source of truth from config)

Phase 3 features (unchanged):
  ├─ Serves index.html at GET /
  ├─ Serves assets/ for logo
  ├─ DELETE /api/session/{id}/document/{idx}
  ├─ DELETE /api/session/{id}/chunks
  └─ Duplicate upload guard
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path
import datetime

from backend.config import (
    supabase_admin,
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, AI_MODEL,
    ALLOWED_ORIGINS, MONTHLY_QUESTION_LIMIT,
    MAX_PDF_BYTES,
)
from backend.auth import router as auth_router, get_current_user
from backend.rag import (
    process_and_store_pdf,
    retrieve_three_contexts,
    get_session_pdf_status,
)
from backend.prompts import build_system_prompt
from openai import OpenAI


# ─────────────────────────────────────────
# PATHS
# Repo structure expected on Render:
#   /  (repo root)
#   ├── backend/
#   │   ├── main.py       ← this file
#   │   ├── config.py
#   │   └── ...
#   ├── index.html
#   ├── 404.html
#   ├── privacy.html
#   ├── terms.html
#   └── assets/
#       └── logo.jpg
# ─────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent.parent   # repo root


# ─────────────────────────────────────────
# STARTUP PATH CHECK
# Print clear warnings in Render logs if expected files are missing.
# This saves the "why is the page blank?" debugging session.
# ─────────────────────────────────────────
_EXPECTED_FILES = ["index.html", "404.html", "privacy.html", "terms.html"]
for _f in _EXPECTED_FILES:
    _p = BASE_DIR / _f
    if not _p.exists():
        print(f"⚠️  WARNING: {_f} not found at {_p}. Create this file before going live.")

_assets_dir = BASE_DIR / "assets"
if not _assets_dir.exists():
    print(f"⚠️  WARNING: assets/ directory not found at {_assets_dir}. Logo will not load.")


# ─────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────
app = FastAPI(
    title="Brain Checker AI API",
    version="3.1.0",
    # Disable the automatic /docs and /redoc in production to avoid
    # exposing internal API structure. Set SHOW_DOCS=true to re-enable.
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

security = HTTPBearer()


# ─────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    card_mode: str
    message: str
    history: list = []


# ─────────────────────────────────────────
# CUSTOM EXCEPTION HANDLER
# Returns the 404.html page instead of a JSON {"detail":"Not found"} response
# when any route or static file is not found.
# ─────────────────────────────────────────
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Serve the branded 404 page for any missing URL."""
    page = BASE_DIR / "404.html"
    if page.exists():
        return FileResponse(str(page), status_code=404, media_type="text/html")
    # Fallback if someone deletes 404.html — still readable JSON
    return JSONResponse(
        status_code=404,
        content={
            "error": "Page not found",
            "message": "The page you are looking for does not exist.",
            "home": "/"
        }
    )


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def check_and_increment_usage(user_id: str) -> int:
    """
    Check monthly question limit. If within limit, increment counter and
    return remaining questions. Raises 429 with a plain-English message
    if the limit is reached.
    """
    profile_res = supabase_admin.table("profiles").select(
        "questions_used, questions_reset"
    ).eq("id", user_id).single().execute()

    profile = profile_res.data or {}
    questions_used = profile.get("questions_used", 0)

    # Reset counter if a new calendar month has started
    reset_date = profile.get("questions_reset")
    if isinstance(reset_date, str):
        reset_date = datetime.datetime.fromisoformat(reset_date.replace("Z", "+00:00"))
    now = datetime.datetime.now(datetime.timezone.utc)
    if reset_date and (now.year > reset_date.year or now.month > reset_date.month):
        questions_used = 0
        supabase_admin.table("profiles").update({
            "questions_used": 0,
            "questions_reset": now.isoformat()
        }).eq("id", user_id).execute()

    if questions_used >= MONTHLY_QUESTION_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=(
                f"You have used all {MONTHLY_QUESTION_LIMIT} questions for this month. "
                "Your limit resets automatically on the 1st of next month. "
                "Contact Brain Checker support if you need an early reset."
            )
        )

    supabase_admin.table("profiles").update({
        "questions_used": questions_used + 1
    }).eq("id", user_id).execute()

    return MONTHLY_QUESTION_LIMIT - (questions_used + 1)


def validate_session_ownership(session_id: str, user_id: str) -> dict:
    """
    Verify the session belongs to this user and return it with product info.
    Raises 403 with a plain-English message if the session is not found or
    belongs to a different user.
    """
    res = supabase_admin.table("sessions").select(
        "*, products(slug, name, required_pdfs, cards)"
    ).eq("id", session_id).eq("user_id", user_id).single().execute()

    if not res.data:
        raise HTTPException(
            status_code=403,
            detail=(
                "Session not found or you do not have permission to access it. "
                "Please go back and select your product again."
            )
        )
    return res.data


def _delete_document_at_index(session_id: str, doc_index: int) -> None:
    """
    Delete any existing document record (and its chunks via CASCADE) at a
    given slot index within a session. Called before re-uploading to the
    same slot so we never accumulate duplicate chunks.
    """
    existing = supabase_admin.table("pdf_documents").select("id").eq(
        "session_id", session_id
    ).eq("doc_index", doc_index).execute()

    if existing.data:
        for row in existing.data:
            supabase_admin.table("pdf_documents").delete().eq("id", row["id"]).execute()
            print(f"🗑️  Cleared document id={row['id']} at slot {doc_index}")


# ─────────────────────────────────────────
# API ROUTES
# Must all be registered BEFORE the static file mounts at the bottom,
# otherwise FastAPI routes requests to static files before hitting the API.
# ─────────────────────────────────────────

@app.get("/api/health", tags=["system"])
def health():
    """
    Health check endpoint — used by UptimeRobot to keep the Render
    free-tier service warm and by Render itself as a liveness probe.
    """
    return {"status": "ok", "version": "3.1.0"}


@app.post("/api/upload", tags=["documents"])
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str  = Query(...),
    doc_index: int   = Query(0),
    doc_label: str   = Query("Report"),
    user=Depends(get_current_user)
):
    """
    Upload and process a PDF report for a session slot.

    Validations (in order):
      1. Filename extension must be .pdf
      2. Session must belong to this user
      3. doc_index must be within the product's allowed range
      4. File must not be empty
      5. File must not exceed MAX_PDF_BYTES (default 20 MB)
      6. File must begin with the %PDF magic bytes (real PDF check)

    The PDF is processed entirely in memory — never written to disk.
    Only the text embeddings are stored in Supabase.
    """

    # ── 1. Extension check ──────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF files are accepted. "
                "Please make sure you are uploading a file that ends in .pdf."
            )
        )

    # ── 2. Session ownership ────────────────────────────────────────────
    session      = validate_session_ownership(session_id, user.id)
    product      = session.get("products", {})
    required_pdfs = product.get("required_pdfs", 2)

    # ── 3. Slot range check ─────────────────────────────────────────────
    if doc_index > required_pdfs:
        raise HTTPException(
            status_code=400,
            detail=(
                f"This product requires {required_pdfs} PDF(s). "
                f"Slot {doc_index} is outside the allowed range. "
                "Please refresh and try again."
            )
        )

    # ── 4. Read file bytes ──────────────────────────────────────────────
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "The file you uploaded appears to be empty. "
                "Please check the file and try again."
            )
        )

    # ── 5. Size limit ───────────────────────────────────────────────────
    if len(contents) > MAX_PDF_BYTES:
        max_mb = MAX_PDF_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=(
                f"This PDF is too large ({len(contents) // (1024*1024)} MB). "
                f"Maximum allowed size is {max_mb} MB. "
                "Try compressing the PDF (use ilovepdf.com or similar) "
                "and upload again."
            )
        )

    # ── 6. Magic byte check (real PDF validation) ───────────────────────
    if not contents.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail=(
                "This file does not appear to be a valid PDF, even though its "
                "name ends in .pdf. The file may be corrupted or the wrong "
                "format. Please try opening the file on your device first to "
                "confirm it is a readable PDF."
            )
        )

    # ── 7. Clear existing document in this slot (re-upload guard) ───────
    _delete_document_at_index(session_id, doc_index)

    # ── 8. Process and store ────────────────────────────────────────────
    try:
        result = await process_and_store_pdf(
            pdf_bytes=contents,
            session_id=session_id,
            user_id=user.id,
            doc_label=doc_label,
            doc_index=doc_index,
            original_name=file.filename,
        )
    except ValueError as e:
        # ValueError from rag.py = unreadable PDF (scanned image, etc.)
        raise HTTPException(
            status_code=400,
            detail=(
                f"{str(e)} "
                "This usually means the PDF is a scanned image without selectable "
                "text. Please use the original digital PDF from Brain Checker, "
                "not a photographed or printed-then-scanned copy."
            )
        )
    except MemoryError:
        raise HTTPException(
            status_code=500,
            detail=(
                "The server ran out of memory while processing this PDF. "
                "Please try a smaller file (under 10 MB) or contact support."
            )
        )
    except Exception as e:
        print(f"❌ PDF processing error: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500,
            detail=(
                "Something went wrong while processing your PDF. "
                "Please try again. If the problem continues, contact Brain Checker support "
                "with the name of the report you were trying to upload."
            )
        )

    status = get_session_pdf_status(session_id, required_pdfs)

    return {
        "status": "success",
        "filename": file.filename,
        "doc_label": doc_label,
        "chunks_stored": result["chunks_stored"],
        "characters": result["characters"],
        "upload_progress": {
            "uploaded":    status["uploaded_count"],
            "required":    required_pdfs,
            "is_complete": status["is_complete"],
            "documents":   status["documents"],
        }
    }


@app.get("/api/session/{session_id}/pdf-status", tags=["documents"])
async def get_pdf_status(session_id: str, user=Depends(get_current_user)):
    """Return upload status for a session — how many PDFs are done vs required."""
    session = validate_session_ownership(session_id, user.id)
    product = session.get("products", {})
    required_pdfs = product.get("required_pdfs", 2)
    status = get_session_pdf_status(session_id, required_pdfs)
    return {"status": "success", **status}


@app.delete("/api/session/{session_id}/document/{doc_index}", tags=["documents"])
async def delete_document(session_id: str, doc_index: int, user=Depends(get_current_user)):
    """Clear a single PDF slot so the user can re-upload to it."""
    validate_session_ownership(session_id, user.id)
    _delete_document_at_index(session_id, doc_index)
    return {"status": "success", "message": f"Slot {doc_index} cleared successfully."}


@app.delete("/api/session/{session_id}/chunks", tags=["documents"])
async def delete_session_chunks(session_id: str, user=Depends(get_current_user)):
    """
    Delete all PDF documents and their chunks for a session.
    Called by the frontend on logout BEFORE the auth token is invalidated,
    so Supabase RLS still allows the delete.
    """
    validate_session_ownership(session_id, user.id)
    try:
        supabase_admin.table("pdf_documents").delete().eq(
            "session_id", session_id
        ).execute()
        print(f"🧹 Wiped all documents for session {session_id}")
    except Exception as e:
        print(f"❌ Chunk cleanup error for session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not clean up session data. "
                "Your account is still safe — please try logging out again."
            )
        )
    return {"status": "success", "message": "Session data cleared."}


@app.post("/api/ask", tags=["chat"])
async def ask(req: ChatRequest, user=Depends(get_current_user)):
    """
    Main chat endpoint.
    Validates card access, checks usage limit, runs 3-context RAG retrieval,
    builds the product+card system prompt, calls the AI, and stores history.
    """

    # ── Validate session and card ───────────────────────────────────────
    session       = validate_session_ownership(req.session_id, user.id)
    product       = session.get("products", {})
    product_slug  = product.get("slug", "dmit")
    allowed_cards = product.get("cards", [])

    if req.card_mode not in allowed_cards:
        raise HTTPException(
            status_code=403,
            detail=(
                f"The '{req.card_mode}' card is not available for your current product. "
                "Please select a different analysis card from the sidebar."
            )
        )

    # ── Usage limit ─────────────────────────────────────────────────────
    questions_remaining = check_and_increment_usage(user.id)

    # ── RAG: retrieve 3-context blocks ──────────────────────────────────
    try:
        contexts = retrieve_three_contexts(req.message, req.session_id)
    except Exception as e:
        print(f"❌ RAG retrieval error: {e}")
        # Non-fatal — proceed with empty context rather than failing the request
        contexts = {"semantic": "", "keyword": "", "overview": ""}

    # ── Build system prompt ─────────────────────────────────────────────
    system_prompt = build_system_prompt(
        product_slug=product_slug,
        card_mode=req.card_mode,
        contexts=contexts,
    )

    # ── Build conversation history ──────────────────────────────────────
    # Limit to last 10 turns to stay within model context window
    recent_history = req.history[-10:] if len(req.history) > 10 else req.history
    chat_history   = [{"role": m["role"], "content": m["content"]} for m in recent_history]
    chat_history.append({"role": "user", "content": req.message})

    # ── Call AI via OpenRouter ──────────────────────────────────────────
    try:
        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "system", "content": system_prompt}] + chat_history,
            temperature=0.7,
            max_tokens=2048,
        )
        reply = response.choices[0].message.content

    except Exception as e:
        err_str = str(e)
        print(f"❌ AI call error: {type(e).__name__}: {err_str}")

        if "429" in err_str or "rate_limit" in err_str.lower():
            raise HTTPException(
                status_code=429,
                detail=(
                    "The AI service is temporarily busy. "
                    "Please wait 30 seconds and try your question again."
                )
            )
        if "401" in err_str or "unauthorized" in err_str.lower():
            raise HTTPException(
                status_code=500,
                detail=(
                    "There is a configuration issue with the AI service. "
                    "Please contact Brain Checker support — this is not your fault."
                )
            )
        if "timeout" in err_str.lower() or "timed out" in err_str.lower():
            raise HTTPException(
                status_code=504,
                detail=(
                    "The AI took too long to respond. "
                    "Please try again with a shorter or simpler question."
                )
            )
        raise HTTPException(
            status_code=500,
            detail=(
                "The AI assistant is temporarily unavailable. "
                "Please try again in a moment. If this keeps happening, "
                "contact Brain Checker support."
            )
        )

    # ── Store chat history (best-effort — do not fail the request) ──────
    try:
        supabase_admin.table("chat_history").insert([
            {
                "session_id": req.session_id,
                "user_id":    user.id,
                "card_mode":  req.card_mode,
                "role":       "user",
                "content":    req.message,
            },
            {
                "session_id": req.session_id,
                "user_id":    user.id,
                "card_mode":  req.card_mode,
                "role":       "assistant",
                "content":    reply,
            },
        ]).execute()
    except Exception as e:
        print(f"⚠️  Chat history storage failed (non-fatal): {e}")

    # ── Log usage (best-effort) ─────────────────────────────────────────
    try:
        supabase_admin.table("usage_logs").insert({
            "user_id":    user.id,
            "session_id": req.session_id,
            "card_mode":  req.card_mode,
        }).execute()
    except Exception as e:
        print(f"⚠️  Usage log write failed (non-fatal): {e}")

    return {
        "reply":               reply,
        "card_mode":           req.card_mode,
        "model":               AI_MODEL,
        "questions_remaining": questions_remaining,
    }


# ─────────────────────────────────────────
# HTML PAGE ROUTES
# Registered before the catch-all static mount so FastAPI handles them
# as proper named routes, not as static file lookups.
# ─────────────────────────────────────────

@app.get("/", response_class=FileResponse, include_in_schema=False)
def serve_index():
    """Serve the Bishop single-page application."""
    path = BASE_DIR / "index.html"
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "index.html not found on the server. "
                "This is a deployment configuration issue — "
                "please contact Brain Checker support."
            )
        )
    return FileResponse(str(path), media_type="text/html")


@app.get("/privacy", response_class=FileResponse, include_in_schema=False)
def serve_privacy():
    """Serve the Privacy Policy page."""
    path = BASE_DIR / "privacy.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Privacy policy page not found.")
    return FileResponse(str(path), media_type="text/html")


@app.get("/terms", response_class=FileResponse, include_in_schema=False)
def serve_terms():
    """Serve the Terms of Service page."""
    path = BASE_DIR / "terms.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Terms of service page not found.")
    return FileResponse(str(path), media_type="text/html")


# ─────────────────────────────────────────
# STATIC FILE MOUNT  (MUST be last)
# Serves assets/logo.jpg etc. at /assets/
# Must come after all route registrations so FastAPI does not shadow any
# API or page route with a static file lookup.
# ─────────────────────────────────────────
if _assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")
else:
    print("⚠️  assets/ not mounted — /assets/* requests will 404.")


# ─────────────────────────────────────────
# ENTRY POINT  (local dev only — Render uses the startCommand in render.yaml)
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn, os
    print("\n" + "═" * 58)
    print("  Bishop by Brain Checker — Backend v3.1 (local dev)")
    print(f"  Project root : {BASE_DIR}")
    print(f"  index.html   : {BASE_DIR / 'index.html'}")
    print(f"  assets/      : {_assets_dir}")
    print("═" * 58 + "\n")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)