# """
# rag.py – RAG pipeline for Brain Checker AI.
# Handles: PDF chunking → embedding → Supabase pgvector storage → retrieval.
# Uses sentence-transformers (free, runs locally, no API cost).
# """
# import io
# import uuid
# from typing import Optional
# import numpy as np

# from backend.config import (
#     supabase_admin,
#     CHUNK_SIZE,
#     CHUNK_OVERLAP,
#     TOP_K_CHUNKS,
#     EMBEDDING_MODEL,
# )

# # Lazy-load the embedding model (loads once, reuses across requests)
# _embedder = None

# def get_embedder():
#     global _embedder
#     if _embedder is None:
#         from sentence_transformers import SentenceTransformer
#         print(f"🔄 Loading embedding model: {EMBEDDING_MODEL}")
#         _embedder = SentenceTransformer(EMBEDDING_MODEL)
#         print(f"✅ Embedding model loaded")
#     return _embedder


# # ─────────────────────────────────────────
# # TEXT CHUNKING
# # ─────────────────────────────────────────

# def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
#     if not text or not text.strip():
#         print("❌ No text provided to chunk_text")
#         return []

#     chunks = []
#     start = 0
#     text = text.strip()
#     print(f"📏 Text length: {len(text)} | chunk_size={chunk_size}, overlap={overlap}")

#     while start < len(text):
#         end = start + chunk_size
#         print(f"➡️ Attempting chunk from {start} to {end}")

#         # Try to break at a sentence boundary
#         if end < len(text):
#             for sep in [". ", ".\n", "\n\n", "\n", " "]:
#                 idx = text.rfind(sep, start, end)
#                 if idx > start:
#                     print(f"🔎 Found separator '{sep}' at {idx}")
#                     end = idx + len(sep)
#                     break

#         chunk = text[start:end].strip()
#         print(f"📦 Chunk length={len(chunk)} | Preview='{chunk[:60]}...'")

#         if chunk:
#             chunks.append(chunk)

#         # ✅ Ensure forward progress
#         new_start = end - overlap
#         if new_start <= start:
#             print(f"⚠️ Overlap too large, forcing forward move")
#             new_start = start + chunk_size
#         start = new_start
#         print(f"➡️ Next start index: {start}")

#     print(f"✅ Total chunks created: {len(chunks)}")
#     return chunks



# # ─────────────────────────────────────────
# # PDF TEXT EXTRACTION
# # ─────────────────────────────────────────

# def extract_pdf_text(pdf_bytes: bytes) -> str:
#     """Extract text from PDF bytes. Tries pdfplumber first, pypdf as fallback."""
#     try:
#         import pdfplumber
#         text = ""
#         with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
#             for page in pdf.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text + "\n"
#         if text.strip():
#             return text.strip()
#     except ImportError:
#         pass
#     except Exception:
#         pass

#     try:
#         from pypdf import PdfReader
#         reader = PdfReader(io.BytesIO(pdf_bytes))
#         text = ""
#         for page in reader.pages:
#             text += page.extract_text() or ""
#         if text.strip():
#             return text.strip()
#     except ImportError:
#         return ""
#     except Exception:
#         return ""

#     return ""


# # ─────────────────────────────────────────
# # EMBED + STORE
# # ─────────────────────────────────────────

# async def process_and_store_pdf(
#     pdf_bytes: bytes,
#     session_id: str,
#     user_id: str,
#     doc_label: str,
#     doc_index: int,
#     original_name: str
# ) -> dict:
#     print("📥 Step 1: Starting PDF processing")

#     # Step 1: Extract
#     print("➡️ Extracting text from PDF...")
#     text = extract_pdf_text(pdf_bytes)
#     print(f"✅ Extracted {len(text)} characters of text")
#     if not text.strip():
#         raise ValueError("Could not extract text from PDF. Make sure it is not a scanned image.")

#     # Step 2: Chunk
#     print("➡️ Chunking text...")
#     chunks = chunk_text(text)
#     print(f"✅ Created {len(chunks)} chunks")
#     if not chunks:
#         raise ValueError("PDF appears to be empty after text extraction.")

#     # Step 3: Embed
#     print("➡️ Generating embeddings...")
#     embedder = get_embedder()
#     embeddings = embedder.encode(chunks, batch_size=32, show_progress_bar=False)
#     print(f"✅ Generated {len(embeddings)} embeddings")

#     # Step 4: Store document record
#     print("➡️ Inserting document record into pdf_documents...")
#     doc_res = supabase_admin.table("pdf_documents").insert({
#         "session_id": session_id,
#         "user_id": user_id,
#         "doc_label": doc_label,
#         "doc_index": doc_index,
#         "original_name": original_name,
#         "chunk_count": len(chunks)
#     }).execute()
#     print(f"✅ Insert result: {doc_res.data}")

#     if not doc_res.data:
#         raise RuntimeError("Failed to create document record in database.")

#     document_id = doc_res.data[0]["id"]
#     print(f"📄 Document ID: {document_id}")

#     # Step 5: Store chunks
#     print("➡️ Preparing chunk rows for insertion...")
#     chunk_rows = []
#     for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
#         chunk_rows.append({
#             "document_id": document_id,
#             "session_id": session_id,
#             "chunk_index": i,
#             "content": chunk,
#             "embedding": embedding.tolist()
#         })
#     print(f"✅ Prepared {len(chunk_rows)} chunk rows")

#     batch_size = 100
#     for i in range(0, len(chunk_rows), batch_size):
#         batch = chunk_rows[i:i + batch_size]
#         print(f"➡️ Inserting batch {i//batch_size+1} with {len(batch)} chunks...")
#         supabase_admin.table("pdf_chunks").insert(batch).execute()
#     print(f"✅ Stored all chunks for '{doc_label}'")

#     return {
#         "document_id": document_id,
#         "doc_label": doc_label,
#         "chunks_stored": len(chunks),
#         "characters": len(text)
#     }


# # ─────────────────────────────────────────
# # RETRIEVE (query time)
# # ─────────────────────────────────────────

# def retrieve_relevant_chunks(query: str, session_id: str, top_k: int = TOP_K_CHUNKS) -> str:
#     """
#     Given a user query, find the most relevant chunks from their session's PDFs.
#     Returns formatted context string to inject into the AI prompt.
#     """
#     embedder = get_embedder()
#     query_embedding = embedder.encode([query])[0].tolist()

#     try:
#         result = supabase_admin.rpc("match_chunks", {
#             "query_embedding": query_embedding,
#             "match_session_id": session_id,
#             "match_count": top_k
#         }).execute()

#         if not result.data:
#             return "(No relevant information found in the uploaded reports.)"

#         # Format chunks with their source label
#         context_parts = []
#         for row in result.data:
#             similarity = row.get("similarity", 0)
#             if similarity > 0.2:  # Only include reasonably relevant chunks
#                 label = row.get("doc_label", "Report")
#                 content = row.get("content", "").strip()
#                 context_parts.append(f"[Source: {label}]\n{content}")

#         if not context_parts:
#             return "(No highly relevant sections found. The question may be outside the report's scope.)"

#         return "\n\n---\n\n".join(context_parts)

#     except Exception as e:
#         print(f"❌ RAG retrieval error: {e}")
#         return "(Error retrieving report context. Please try again.)"


# # ─────────────────────────────────────────
# # CHECK IF SESSION HAS ALL PDFS UPLOADED
# # ─────────────────────────────────────────

# def get_session_pdf_status(session_id: str, required_count: int) -> dict:
#     """Check how many PDFs have been uploaded for a session."""
#     res = supabase_admin.table("pdf_documents").select(
#         "id, doc_label, doc_index, original_name, chunk_count"
#     ).eq("session_id", session_id).order("doc_index").execute()

#     uploaded = res.data or []
#     return {
#         "uploaded_count": len(uploaded),
#         "required_count": required_count,
#         "is_complete": len(uploaded) >= required_count,
#         "documents": uploaded
#     }

"""
rag.py – Brain Checker AI · Phase 2 RAG Pipeline
================================================
Upgrades over Phase 1:
  1. Section-aware chunking  — never splits mid-section
  2. Bigger chunks (1200 chars) — keeps full data blocks intact
  3. 200-char overlap         — boundary data never lost
  4. Header injection         — section name prepended to every chunk
  5. MMR deduplication        — removes near-duplicate retrieved chunks
  6. 3-context retrieval      — semantic + keyword + section-sweep contexts
     sent to the AI in clearly labelled blocks so it can cross-reference
"""

import io
import re
from typing import Optional

import numpy as np

from backend.config import (
    supabase_admin,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_CHUNKS,
    EMBEDDING_MODEL,
)

# ─────────────────────────────────────────
# EMBEDDING MODEL  (lazy-load, singleton)
# ─────────────────────────────────────────

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print(f"🔄 Loading embedding model: {EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Embedding model loaded")
    return _embedder


# ─────────────────────────────────────────
# SECTION HEADER PATTERNS
# Brain Checker reports use consistent heading styles across all products.
# We detect them so we never split inside a section.
# ─────────────────────────────────────────

# Matches lines that look like report section headers, e.g.:
#   "RIASEC PROFILE", "Recommended Streams:", "2. Career Suggestions"
SECTION_HEADER_RE = re.compile(
    r'^(?:'
    r'\d+[\.\)]\s+'                          # numbered  "1. Section"
    r'|[A-Z][A-Z\s&/\-]{3,}(?:\s*:)?$'      # ALL-CAPS  "RIASEC PROFILE"
    r'|[A-Z][a-z].*(?:Report|Profile|Score'
    r'|Analysis|Summary|Recommendation'
    r'|Plan|Guide|Assessment|Result'
    r'|Traits|Strengths|Weaknesses'
    r'|Career|Intelligence|Stream'
    r'|Personality|Aptitude|Interest'
    r'|Behavior|Growth|Skill).*:?\s*$'       # Title-case headings
    r')',
    re.MULTILINE,
)

# Keywords that nearly always signal a new logical section in BC reports
SECTION_KEYWORDS = [
    "IQ Score", "RIASEC", "Recommended Stream", "Personality Trait",
    "Multiple Intelligence", "Brain Dominance", "Learning Style",
    "Career Suggestion", "Aptitude", "Strengths", "Weakness",
    "Action Plan", "Counselor Remark", "Recommendation", "Summary",
    "Overall Profile", "Emotional Quotient", "Work Style",
    "Engineering Branch", "Entrepreneur", "Leadership", "Skill Gap",
    "Job Fit", "Competency", "Candidate", "Hiring",
]


# ─────────────────────────────────────────
# PDF TEXT EXTRACTION
# ─────────────────────────────────────────

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes.
    Strategy: pdfplumber (best layout preservation) → pypdf (fallback).
    We also normalise whitespace so chunking works cleanly.
    """
    raw = ""

    # Method 1 — pdfplumber (preserves table layout better)
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    raw += page_text + "\n\n"   # double-newline = page break
        if raw.strip():
            return _normalise(raw)
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ pdfplumber error: {e}")

    # Method 2 — pypdf fallback
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            raw += (page.extract_text() or "") + "\n\n"
        if raw.strip():
            return _normalise(raw)
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ pypdf error: {e}")

    return ""


def _normalise(text: str) -> str:
    """
    Clean up extracted text so section detection works reliably.
    - Collapse 3+ blank lines to 2 (keeps section breaks)
    - Strip trailing spaces per line
    - Ensure section keywords always start on their own line
    """
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)

    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Force section keywords to start on a new line if they don't already
    for kw in SECTION_KEYWORDS:
        # Only add newline if keyword is mid-line and preceded by non-newline text
        text = re.sub(
            r'([^\n])(' + re.escape(kw) + r')',
            r'\1\n\2',
            text
        )

    return text.strip()


# ─────────────────────────────────────────
# SECTION-AWARE CHUNKING
# ─────────────────────────────────────────

def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split text into (header, body) pairs at detected section boundaries.
    Returns list of (section_header, section_text).
    If no headers detected, the whole text is one section with header "".
    """
    lines = text.splitlines()
    sections = []
    current_header = ""
    current_lines = []

    for line in lines:
        stripped = line.strip()
        is_header = bool(SECTION_HEADER_RE.match(stripped)) or any(
            kw.lower() in stripped.lower() and len(stripped) < 80
            for kw in SECTION_KEYWORDS
        )

        if is_header and current_lines:
            # Save what we have so far
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_header, body))
            current_header = stripped
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    body = "\n".join(current_lines).strip()
    if body:
        sections.append((current_header, body))

    # If nothing split, return the whole text as one section
    if not sections:
        sections = [("", text)]

    return sections


def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Section-aware chunker.

    Steps:
      1. Split text into logical sections using header detection.
      2. For each section, prepend the header to every chunk created from it.
         This ensures the embedding knows WHAT section the content belongs to.
      3. If a section body fits in one chunk → single chunk.
         If it's longer → sliding window with overlap, always breaking at
         sentence/paragraph boundaries.
    """
    if not text or not text.strip():
        print("❌ No text to chunk")
        return []

    sections = _split_into_sections(text)
    print(f"📂 Detected {len(sections)} sections")

    all_chunks = []

    for header, body in sections:
        # Prefix that gets prepended to every chunk from this section
        prefix = f"[Section: {header}]\n" if header else ""
        max_body = chunk_size - len(prefix)

        if len(body) <= max_body:
            # Whole section fits in one chunk
            chunk = (prefix + body).strip()
            if chunk:
                all_chunks.append(chunk)
        else:
            # Sliding window within the section
            sub_chunks = _sliding_window(body, max_body, overlap)
            for sc in sub_chunks:
                chunk = (prefix + sc).strip()
                if chunk:
                    all_chunks.append(chunk)

    print(f"✅ Total chunks: {len(all_chunks)}")
    return all_chunks


def _sliding_window(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Standard sliding-window chunker that respects sentence boundaries.
    Used for long section bodies after the section header is accounted for.
    """
    chunks = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = start + chunk_size

        # Try to break at a natural boundary
        if end < len(text):
            for sep in ["\n\n", ".\n", ". ", ".\t", "\n", " "]:
                idx = text.rfind(sep, start + overlap, end)
                if idx > start:
                    end = idx + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance — guarantee forward progress
        new_start = end - overlap
        if new_start <= start:
            new_start = start + max(chunk_size // 2, 1)
        start = new_start

    return chunks


# ─────────────────────────────────────────
# EMBED + STORE
# ─────────────────────────────────────────

async def process_and_store_pdf(
    pdf_bytes: bytes,
    session_id: str,
    user_id: str,
    doc_label: str,
    doc_index: int,
    original_name: str,
) -> dict:
    """
    Full pipeline: extract → chunk → embed → store in Supabase pgvector.
    Raw PDF bytes are never written to disk; they live only in memory during
    this function and are discarded automatically when it returns.
    """
    print(f"📥 Processing '{original_name}' (doc_index={doc_index})")

    # 1. Extract
    text = extract_pdf_text(pdf_bytes)
    print(f"✅ Extracted {len(text):,} characters")
    if not text.strip():
        raise ValueError(
            "Could not extract text from this PDF. "
            "Please ensure it is not a scanned image-only file."
        )

    # 2. Chunk  (section-aware, bigger, with header injection)
    chunks = chunk_text(text)
    print(f"✅ Created {len(chunks)} chunks")
    if not chunks:
        raise ValueError("PDF appears empty after text extraction.")

    # 3. Embed
    embedder = get_embedder()
    print(f"➡️ Embedding {len(chunks)} chunks…")
    embeddings = embedder.encode(
        chunks, batch_size=32, show_progress_bar=False, normalize_embeddings=True
    )
    print(f"✅ Embeddings ready")

    # 4. Store document record
    doc_res = supabase_admin.table("pdf_documents").insert({
        "session_id": session_id,
        "user_id": user_id,
        "doc_label": doc_label,
        "doc_index": doc_index,
        "original_name": original_name,
        "chunk_count": len(chunks),
    }).execute()

    if not doc_res.data:
        raise RuntimeError("Failed to create document record in database.")

    document_id = doc_res.data[0]["id"]
    print(f"📄 document_id={document_id}")

    # 5. Store chunks in batches of 50
    chunk_rows = [
        {
            "document_id": document_id,
            "session_id": session_id,
            "chunk_index": i,
            "content": chunk,
            "embedding": emb.tolist(),
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    batch_size = 50
    for i in range(0, len(chunk_rows), batch_size):
        batch = chunk_rows[i : i + batch_size]
        supabase_admin.table("pdf_chunks").insert(batch).execute()
        print(f"  ↳ Stored batch {i // batch_size + 1} ({len(batch)} chunks)")

    print(f"✅ All chunks stored for '{doc_label}'")

    # Raw PDF bytes go out of scope here — nothing is persisted to disk.

    return {
        "document_id": document_id,
        "doc_label": doc_label,
        "chunks_stored": len(chunks),
        "characters": len(text),
    }


# ─────────────────────────────────────────
# MMR  (Maximal Marginal Relevance)
# ─────────────────────────────────────────

def _mmr(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
    candidates: list[dict],
    top_k: int,
    lambda_: float = 0.6,
) -> list[dict]:
    """
    Select top_k chunks that are:
      - Relevant to the query  (high cosine sim with query_emb)
      - Diverse from each other (penalise redundancy)

    lambda_ = 0.6 means 60% relevance, 40% diversity.
    Lower lambda_ → more diversity; higher → more relevance.
    """
    if len(candidates) == 0:
        return []

    selected_indices = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = -np.inf

        for i in remaining:
            # Relevance: cosine sim between chunk and query
            relevance = float(np.dot(candidate_embs[i], query_emb))

            # Redundancy: max cosine sim with already-selected chunks
            if selected_indices:
                redundancy = max(
                    float(np.dot(candidate_embs[i], candidate_embs[j]))
                    for j in selected_indices
                )
            else:
                redundancy = 0.0

            score = lambda_ * relevance - (1 - lambda_) * redundancy

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]


# ─────────────────────────────────────────
# QUERY EXPANSION
# Brain Checker-specific synonym maps so a parent asking "which stream
# should my child choose?" also retrieves chunks about "branch" or "subject".
# ─────────────────────────────────────────

_SYNONYMS: dict[str, list[str]] = {
    "stream":        ["branch", "subject", "stream recommendation", "course"],
    "iq":            ["intelligence quotient", "iq score", "cognitive ability", "mental ability"],
    "riasec":        ["career interest", "holland code", "realistic investigative artistic social enterprising conventional"],
    "career":        ["profession", "job", "occupation", "vocation", "field"],
    "strength":      ["talent", "ability", "strong area", "aptitude", "competency"],
    "weakness":      ["area of improvement", "gap", "challenge", "limitation"],
    "personality":   ["trait", "behavior", "temperament", "character"],
    "college":       ["university", "institution", "institute", "engineering college", "medical college"],
    "engineering":   ["b.tech", "be ", "technical", "engineering branch"],
    "entrepreneur":  ["business", "startup", "venture", "tycoon"],
    "learning style":["visual", "auditory", "kinesthetic", "reading writing"],
    "parent":        ["family", "mother", "father", "guardian"],
    "plan":          ["action plan", "roadmap", "schedule", "timeline", "strategy"],
    "recommend":     ["suggest", "advise", "counselor remark", "recommendation"],
    "score":         ["percentile", "marks", "result", "rating", "rank"],
    "brain dominance": ["left brain", "right brain", "cortical", "limbic"],
}


def _expand_query(query: str) -> str:
    """
    Append synonym terms to the query so the embedding captures a wider
    semantic net. Returns original + relevant synonyms as a single string.
    """
    q_lower = query.lower()
    extras = []
    for key, synonyms in _SYNONYMS.items():
        if key in q_lower:
            extras.extend(synonyms)
    if extras:
        return query + " " + " ".join(set(extras))
    return query


# ─────────────────────────────────────────
# 3-CONTEXT RETRIEVAL
# ─────────────────────────────────────────

def retrieve_three_contexts(
    query: str,
    session_id: str,
    top_k: int = TOP_K_CHUNKS,
) -> dict[str, str]:
    """
    Return THREE distinct context blocks for the AI:

    Context A — SEMANTIC (primary)
        Standard vector similarity on the expanded query.
        Best for direct factual questions about scores, traits, careers.

    Context B — KEYWORD (supplementary)
        Pull chunks that contain the most important words from the query
        even if their embedding distance is slightly lower.
        Catches data buried in tables or lists where embedding is weaker.

    Context C — SECTION SWEEP (broad overview)
        Pull the top chunk from EACH document in the session.
        Ensures the AI always has at least a summary-level view of every
        uploaded report, so it can cross-reference between DMIT and
        Recommendation Report, for example.

    Each context goes through MMR to remove near-duplicates within itself.
    The three contexts are kept separate so the AI prompt can label them
    clearly and the model knows which block to trust for which type of info.
    """
    embedder = get_embedder()

    # ── 1. Build query embeddings ──────────────────────────────────────
    expanded_query = _expand_query(query)
    q_emb_orig = embedder.encode([query], normalize_embeddings=True)[0]
    q_emb_exp = embedder.encode([expanded_query], normalize_embeddings=True)[0]

    # ── Context A: Semantic (expanded query, more chunks, then MMR) ────
    ctx_a = _fetch_by_embedding(q_emb_exp, session_id, fetch_k=top_k * 3)
    ctx_a = _apply_mmr(q_emb_exp, ctx_a, top_k)

    # ── Context B: Keyword-boosted ─────────────────────────────────────
    # Extract meaningful words (>3 chars, not stop words) from the query
    keywords = _extract_keywords(query)
    ctx_b_raw = _fetch_by_embedding(q_emb_orig, session_id, fetch_k=top_k * 4)
    # Re-rank by keyword presence in content
    ctx_b_raw = _keyword_rerank(ctx_b_raw, keywords)
    ctx_b = _apply_mmr(q_emb_orig, ctx_b_raw, top_k)

    # ── Context C: Section sweep ───────────────────────────────────────
    ctx_c = _fetch_section_sweep(session_id, top_chunks_per_doc=3)

    # ── Format each context as a labelled string ───────────────────────
    return {
        "semantic":  _format_context(ctx_a, "Primary Context (Semantic Match)"),
        "keyword":   _format_context(ctx_b, "Supplementary Context (Keyword Match)"),
        "overview":  _format_context(ctx_c, "Report Overview (Section Sweep)"),
    }


# ── keep old single-context function for backward compat ──────────────
def retrieve_relevant_chunks(query: str, session_id: str, top_k: int = TOP_K_CHUNKS) -> str:
    """
    Backward-compatible wrapper. Returns merged context string.
    New code should use retrieve_three_contexts() directly.
    """
    contexts = retrieve_three_contexts(query, session_id, top_k)
    return "\n\n".join(contexts.values())


# ─────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────

def _fetch_by_embedding(
    query_emb: np.ndarray,
    session_id: str,
    fetch_k: int,
    min_similarity: float = 0.15,
) -> list[dict]:
    """Call Supabase match_chunks RPC and return raw rows above threshold."""
    try:
        result = supabase_admin.rpc("match_chunks", {
            "query_embedding": query_emb.tolist(),
            "match_session_id": session_id,
            "match_count": fetch_k,
        }).execute()

        rows = result.data or []
        return [r for r in rows if r.get("similarity", 0) >= min_similarity]
    except Exception as e:
        print(f"❌ _fetch_by_embedding error: {e}")
        return []


def _apply_mmr(query_emb: np.ndarray, rows: list[dict], top_k: int) -> list[dict]:
    """Run MMR on a list of DB rows. Rows must have a 'content' field."""
    if not rows:
        return []
    embedder = get_embedder()
    contents = [r.get("content", "") for r in rows]
    embs = embedder.encode(contents, normalize_embeddings=True)
    return _mmr(query_emb, embs, rows, top_k)


def _extract_keywords(query: str) -> list[str]:
    """Return meaningful words from the query (length > 3, not stop words)."""
    STOP = {
        "what", "which", "where", "when", "how", "does", "should",
        "tell", "give", "show", "about", "with", "from", "that",
        "this", "their", "have", "will", "would", "could", "please",
        "explain", "describe", "child", "student", "report", "based",
    }
    words = re.findall(r'\b[a-zA-Z]{4,}\b', query.lower())
    return [w for w in words if w not in STOP]


def _keyword_rerank(rows: list[dict], keywords: list[str]) -> list[dict]:
    """
    Boost rows that contain more of the query keywords.
    Returns rows sorted by (keyword_hits DESC, similarity DESC).
    Rows with zero keyword hits are kept — they may still be useful.
    """
    if not keywords:
        return rows

    def score(row: dict) -> tuple:
        content_lower = row.get("content", "").lower()
        hits = sum(1 for kw in keywords if kw in content_lower)
        sim = row.get("similarity", 0)
        return (hits, sim)

    return sorted(rows, key=score, reverse=True)


def _fetch_section_sweep(session_id: str, top_chunks_per_doc: int = 3) -> list[dict]:
    """
    Fetch the first N chunks of every document in the session.
    This gives the AI a broad overview of all uploaded reports,
    not just the parts most similar to the current question.
    """
    try:
        # Get all documents in this session
        docs_res = supabase_admin.table("pdf_documents").select(
            "id, doc_label"
        ).eq("session_id", session_id).execute()

        docs = docs_res.data or []
        sweep_rows = []

        for doc in docs:
            doc_id = doc["id"]
            doc_label = doc.get("doc_label", "Report")

            # Fetch the first top_chunks_per_doc chunks (lowest chunk_index)
            chunks_res = supabase_admin.table("pdf_chunks").select(
                "content, chunk_index"
            ).eq("document_id", doc_id).order("chunk_index").limit(top_chunks_per_doc).execute()

            for row in (chunks_res.data or []):
                sweep_rows.append({
                    "content": row.get("content", ""),
                    "doc_label": doc_label,
                    "similarity": 1.0,   # label as "definite include"
                })

        return sweep_rows
    except Exception as e:
        print(f"❌ _fetch_section_sweep error: {e}")
        return []


def _format_context(rows: list[dict], label: str) -> str:
    """Format a list of DB rows into a labelled context block for the AI."""
    if not rows:
        return f"[{label}]\n(No relevant content found.)"

    parts = []
    for row in rows:
        doc_label = row.get("doc_label", "Report")
        content = row.get("content", "").strip()
        if content:
            parts.append(f"[Source: {doc_label}]\n{content}")

    body = "\n\n---\n\n".join(parts)
    return f"[{label}]\n{body}"


# ─────────────────────────────────────────
# SESSION PDF STATUS
# ─────────────────────────────────────────

def get_session_pdf_status(session_id: str, required_count: int) -> dict:
    """Check how many PDFs have been uploaded for a session."""
    res = supabase_admin.table("pdf_documents").select(
        "id, doc_label, doc_index, original_name, chunk_count"
    ).eq("session_id", session_id).order("doc_index").execute()

    uploaded = res.data or []
    return {
        "uploaded_count": len(uploaded),
        "required_count": required_count,
        "is_complete": len(uploaded) >= required_count,
        "documents": uploaded,
    }