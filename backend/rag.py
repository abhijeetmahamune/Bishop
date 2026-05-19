"""
rag.py – Brain Checker AI · RAG Pipeline (HuggingFace API Edition)
==================================================================
Key change from Phase 2:
  - Removed local sentence-transformers model (was killing Render free tier RAM)
  - Embeddings now generated via HuggingFace Inference API (free, zero RAM cost)
  - Same 384-dim all-MiniLM-L6-v2 model → no Supabase schema changes needed
  - All other logic unchanged: section-aware chunking, MMR, 3-context retrieval

Setup:
  1. Get a free HuggingFace token at https://huggingface.co/settings/tokens
  2. Add HF_TOKEN to your Render environment variables
  3. Remove sentence-transformers, torch, transformers from requirements.txt
"""

import io
import re
import time
from typing import Optional

import numpy as np
import requests

from backend.config import (
    supabase_admin,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_CHUNKS,
    HF_TOKEN,
    HF_API_URL,
)


# ─────────────────────────────────────────
# HUGGING FACE EMBEDDING API
# ─────────────────────────────────────────

def get_embeddings(texts: list[str], retries: int = 3) -> list[list[float]]:
    """
    Generate embeddings via HuggingFace Inference API.
    Uses all-MiniLM-L6-v2 → 384-dim vectors, same as before.
    Retries up to 3 times with 20s wait if the model is loading.
    """
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": texts,
        "options": {"wait_for_model": True}
    }

    for attempt in range(retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                # HF returns list of embeddings directly
                if isinstance(result, list) and len(result) > 0:
                    # Handle both flat [float] and nested [[float]] formats
                    if isinstance(result[0], list):
                        return result
                    else:
                        return [result]

            elif response.status_code == 503:
                # Model is loading — wait and retry
                wait = 20 * (attempt + 1)
                print(f"⏳ HF model loading, waiting {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)

            else:
                print(f"❌ HF API error {response.status_code}: {response.text[:200]}")
                if attempt < retries - 1:
                    time.sleep(5)

        except requests.exceptions.Timeout:
            print(f"⚠️ HF API timeout (attempt {attempt+1}/{retries})")
            if attempt < retries - 1:
                time.sleep(10)
        except Exception as e:
            print(f"❌ HF API exception: {e}")
            if attempt < retries - 1:
                time.sleep(5)

    raise RuntimeError(
        "Could not generate embeddings after multiple attempts. "
        "Please check your HF_TOKEN and try again."
    )


def get_single_embedding(text: str) -> list[float]:
    """Convenience wrapper for embedding a single string."""
    return get_embeddings([text])[0]


# ─────────────────────────────────────────
# SECTION HEADER PATTERNS
# ─────────────────────────────────────────

SECTION_HEADER_RE = re.compile(
    r'^(?:'
    r'\d+[\.\)]\s+'
    r'|[A-Z][A-Z\s&/\-]{3,}(?:\s*:)?$'
    r'|[A-Z][a-z].*(?:Report|Profile|Score'
    r'|Analysis|Summary|Recommendation'
    r'|Plan|Guide|Assessment|Result'
    r'|Traits|Strengths|Weaknesses'
    r'|Career|Intelligence|Stream'
    r'|Personality|Aptitude|Interest'
    r'|Behavior|Growth|Skill).*:?\s*$'
    r')',
    re.MULTILINE,
)

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
    pdfplumber first (best layout), pypdf as fallback.
    """
    raw = ""

    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    raw += page_text + "\n\n"
        if raw.strip():
            return _normalise(raw)
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ pdfplumber error: {e}")

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
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    for kw in SECTION_KEYWORDS:
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
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_header, body))
            current_header = stripped
            current_lines = []
        else:
            current_lines.append(line)

    body = "\n".join(current_lines).strip()
    if body:
        sections.append((current_header, body))

    if not sections:
        sections = [("", text)]

    return sections


def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    if not text or not text.strip():
        print("❌ No text to chunk")
        return []

    sections = _split_into_sections(text)
    print(f"📂 Detected {len(sections)} sections")

    all_chunks = []

    for header, body in sections:
        prefix = f"[Section: {header}]\n" if header else ""
        max_body = chunk_size - len(prefix)

        if len(body) <= max_body:
            chunk = (prefix + body).strip()
            if chunk:
                all_chunks.append(chunk)
        else:
            sub_chunks = _sliding_window(body, max_body, overlap)
            for sc in sub_chunks:
                chunk = (prefix + sc).strip()
                if chunk:
                    all_chunks.append(chunk)

    print(f"✅ Total chunks: {len(all_chunks)}")
    return all_chunks


def _sliding_window(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            for sep in ["\n\n", ".\n", ". ", ".\t", "\n", " "]:
                idx = text.rfind(sep, start + overlap, end)
                if idx > start:
                    end = idx + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

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
    Full pipeline: extract → chunk → embed (via HF API) → store in Supabase.
    No model loaded into RAM — embeddings are generated remotely.
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

    # 2. Chunk
    chunks = chunk_text(text)
    print(f"✅ Created {len(chunks)} chunks")
    if not chunks:
        raise ValueError("PDF appears empty after text extraction.")

    # 3. Embed via HF API in batches of 32
    print(f"➡️ Generating embeddings via HuggingFace API...")
    all_embeddings = []
    batch_size = 32

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"  ↳ Embedding batch {i // batch_size + 1} ({len(batch)} chunks)")
        batch_embeddings = get_embeddings(batch)
        all_embeddings.extend(batch_embeddings)
        # Small delay to avoid HF rate limits
        if i + batch_size < len(chunks):
            time.sleep(0.5)

    print(f"✅ Got {len(all_embeddings)} embeddings")

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
            "embedding": emb,
        }
        for i, (chunk, emb) in enumerate(zip(chunks, all_embeddings))
    ]

    store_batch = 50
    for i in range(0, len(chunk_rows), store_batch):
        batch = chunk_rows[i:i + store_batch]
        supabase_admin.table("pdf_chunks").insert(batch).execute()
        print(f"  ↳ Stored batch {i // store_batch + 1} ({len(batch)} chunks)")

    print(f"✅ All chunks stored for '{doc_label}'")

    return {
        "document_id": document_id,
        "doc_label": doc_label,
        "chunks_stored": len(chunks),
        "characters": len(text),
    }


# ─────────────────────────────────────────
# MMR (Maximal Marginal Relevance)
# ─────────────────────────────────────────

def _mmr(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
    candidates: list[dict],
    top_k: int,
    lambda_: float = 0.6,
) -> list[dict]:
    if len(candidates) == 0:
        return []

    selected_indices = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = -np.inf

        for i in remaining:
            relevance = float(np.dot(candidate_embs[i], query_emb))
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
# ─────────────────────────────────────────

_SYNONYMS: dict[str, list[str]] = {
    "stream":          ["branch", "subject", "stream recommendation", "course"],
    "iq":              ["intelligence quotient", "iq score", "cognitive ability"],
    "riasec":          ["career interest", "holland code", "realistic investigative"],
    "career":          ["profession", "job", "occupation", "vocation", "field"],
    "strength":        ["talent", "ability", "strong area", "aptitude", "competency"],
    "weakness":        ["area of improvement", "gap", "challenge", "limitation"],
    "personality":     ["trait", "behavior", "temperament", "character"],
    "college":         ["university", "institution", "engineering college"],
    "engineering":     ["b.tech", "be ", "technical", "engineering branch"],
    "entrepreneur":    ["business", "startup", "venture", "tycoon"],
    "learning style":  ["visual", "auditory", "kinesthetic", "reading writing"],
    "parent":          ["family", "mother", "father", "guardian"],
    "plan":            ["action plan", "roadmap", "schedule", "timeline"],
    "recommend":       ["suggest", "advise", "counselor remark", "recommendation"],
    "score":           ["percentile", "marks", "result", "rating", "rank"],
    "brain dominance": ["left brain", "right brain", "cortical", "limbic"],
}


def _expand_query(query: str) -> str:
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
    Returns three context blocks for the AI prompt:
      - semantic:  vector similarity on expanded query (primary)
      - keyword:   keyword-boosted retrieval (catches tables/lists)
      - overview:  first chunks of every document (broad context)
    """
    expanded_query = _expand_query(query)

    print(f"➡️ Embedding query via HF API...")
    q_emb_orig = np.array(get_single_embedding(query))
    q_emb_exp  = np.array(get_single_embedding(expanded_query))

    # Normalise
    q_emb_orig = q_emb_orig / (np.linalg.norm(q_emb_orig) + 1e-10)
    q_emb_exp  = q_emb_exp  / (np.linalg.norm(q_emb_exp)  + 1e-10)

    # Context A — Semantic
    ctx_a = _fetch_by_embedding(q_emb_exp, session_id, fetch_k=top_k * 3)
    ctx_a = _apply_mmr(q_emb_exp, ctx_a, top_k)

    # Context B — Keyword-boosted
    keywords = _extract_keywords(query)
    ctx_b_raw = _fetch_by_embedding(q_emb_orig, session_id, fetch_k=top_k * 4)
    ctx_b_raw = _keyword_rerank(ctx_b_raw, keywords)
    ctx_b = _apply_mmr(q_emb_orig, ctx_b_raw, top_k)

    # Context C — Section sweep
    ctx_c = _fetch_section_sweep(session_id, top_chunks_per_doc=3)

    return {
        "semantic": _format_context(ctx_a, "Primary Context (Semantic Match)"),
        "keyword":  _format_context(ctx_b, "Supplementary Context (Keyword Match)"),
        "overview": _format_context(ctx_c, "Report Overview (Section Sweep)"),
    }


def retrieve_relevant_chunks(query: str, session_id: str, top_k: int = TOP_K_CHUNKS) -> str:
    """Backward-compatible wrapper."""
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
    if not rows:
        return []
    contents = [r.get("content", "") for r in rows]
    print(f"➡️ Embedding {len(contents)} candidates for MMR via HF API...")
    emb_list = get_embeddings(contents)
    embs = np.array(emb_list)
    # Normalise
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    embs = embs / norms
    return _mmr(query_emb, embs, rows, top_k)


def _extract_keywords(query: str) -> list[str]:
    STOP = {
        "what", "which", "where", "when", "how", "does", "should",
        "tell", "give", "show", "about", "with", "from", "that",
        "this", "their", "have", "will", "would", "could", "please",
        "explain", "describe", "child", "student", "report", "based",
    }
    words = re.findall(r'\b[a-zA-Z]{4,}\b', query.lower())
    return [w for w in words if w not in STOP]


def _keyword_rerank(rows: list[dict], keywords: list[str]) -> list[dict]:
    if not keywords:
        return rows

    def score(row: dict) -> tuple:
        content_lower = row.get("content", "").lower()
        hits = sum(1 for kw in keywords if kw in content_lower)
        sim = row.get("similarity", 0)
        return (hits, sim)

    return sorted(rows, key=score, reverse=True)


def _fetch_section_sweep(session_id: str, top_chunks_per_doc: int = 3) -> list[dict]:
    try:
        docs_res = supabase_admin.table("pdf_documents").select(
            "id, doc_label"
        ).eq("session_id", session_id).execute()

        docs = docs_res.data or []
        sweep_rows = []

        for doc in docs:
            doc_id    = doc["id"]
            doc_label = doc.get("doc_label", "Report")

            chunks_res = supabase_admin.table("pdf_chunks").select(
                "content, chunk_index"
            ).eq("document_id", doc_id).order("chunk_index").limit(top_chunks_per_doc).execute()

            for row in (chunks_res.data or []):
                sweep_rows.append({
                    "content":   row.get("content", ""),
                    "doc_label": doc_label,
                    "similarity": 1.0,
                })

        return sweep_rows
    except Exception as e:
        print(f"❌ _fetch_section_sweep error: {e}")
        return []


def _format_context(rows: list[dict], label: str) -> str:
    if not rows:
        return f"[{label}]\n(No relevant content found.)"

    parts = []
    for row in rows:
        doc_label = row.get("doc_label", "Report")
        content   = row.get("content", "").strip()
        if content:
            parts.append(f"[Source: {doc_label}]\n{content}")

    body = "\n\n---\n\n".join(parts)
    return f"[{label}]\n{body}"


# ─────────────────────────────────────────
# SESSION PDF STATUS
# ─────────────────────────────────────────

def get_session_pdf_status(session_id: str, required_count: int) -> dict:
    res = supabase_admin.table("pdf_documents").select(
        "id, doc_label, doc_index, original_name, chunk_count"
    ).eq("session_id", session_id).order("doc_index").execute()

    uploaded = res.data or []
    return {
        "uploaded_count": len(uploaded),
        "required_count": required_count,
        "is_complete":    len(uploaded) >= required_count,
        "documents":      uploaded,
    }
