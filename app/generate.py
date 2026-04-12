"""
Stage 6 — Answer Generation via Groq (llama-3.3-70b-versatile)
Generates grounded answers from retrieved chunks using the Groq cloud API.

Requirements:
  - GROQ_API_KEY set in .env
  - GROQ_MODEL set in .env (default: llama-3.3-70b-versatile)
  - retrieve.py must be in the same folder as this file

Usage:
  from generate import generate
  result = generate("What is classical conditioning?", top_k=5)
  # returns dict with keys: answer, context, references, rate_limited
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Ensure app/ directory is on sys.path ──────────────────────────────────────
_THIS_DIR = Path(__file__).parent.absolute()
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from retrieve import retrieve

# ── Groq client setup ─────────────────────────────────────────────────────────
try:
    from groq import Groq, RateLimitError, APIStatusError
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
    RateLimitError  = None
    APIStatusError  = None

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama-3.3-70b-versatile")

MAX_CONTEXT_CHARS = 6000   # llama-3.3-70b has a large context window
FALLBACK_ANSWER   = "Not found in the provided textbook."

RATE_LIMIT_MESSAGE = (
    "⚠️ Service Temporarily Unavailable — API Capacity Reached\n\n"
    "The AI service is currently at capacity due to high demand. "
    "This is a temporary situation and normal service will resume shortly.\n\n"
    "What you can do:\n"
    "• Wait a few minutes and try your question again\n"
    "• Try a shorter or more specific question\n"
    "• Check back later — capacity resets periodically\n\n"
    "We apologise for the inconvenience."
)

# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the "
    "provided textbook context. "
    "Do NOT use any knowledge outside the context. "
    "If the answer cannot be found in the context, respond with exactly: "
    f'"{FALLBACK_ANSWER}" '
    "Keep your answer concise and factual. Do not add any preamble."
)


def _build_user_prompt(question: str, context_text: str) -> str:
    return (
        f"Context from the textbook:\n"
        f"---\n{context_text}\n---\n\n"
        f"Question: {question}\n\n"
        "Answer (based only on the context above):"
    )


# ─── Groq call ────────────────────────────────────────────────────────────────

def _call_groq(question: str, context_text: str) -> tuple[str, bool]:
    """
    Returns (answer_text, rate_limited).
    rate_limited=True means the API quota was exhausted.
    """
    if not _GROQ_AVAILABLE:
        raise RuntimeError(
            "groq package not installed. Run: pip install groq"
        )
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    client = Groq(api_key=GROQ_API_KEY)

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(question, context_text)},
            ],
            temperature=0.2,
            max_tokens=512,
            top_p=0.9,
        )
        return completion.choices[0].message.content.strip(), False

    except Exception as e:
        # Detect rate-limit errors robustly (works even before groq is imported)
        err_str = str(e).lower()
        if (
            (RateLimitError and isinstance(e, RateLimitError))
            or "rate_limit" in err_str
            or "rate limit" in err_str
            or "429" in err_str
            or "quota" in err_str
            or "tokens per" in err_str
        ):
            return RATE_LIMIT_MESSAGE, True
        raise


# ─── Reference builder ────────────────────────────────────────────────────────

def _build_references(chunks: list) -> dict:
    """
    Converts section paths to slash-separated lowercase format.
    "1 Introduction to Psychology > 1.1 What Is Psychology?"
    → "1_introduction_to_psychology/1.1_what_is_psychology"
    """
    sections_seen = []
    pages_seen    = []

    for chunk in chunks:
        raw_path = chunk.get("section_path", "")
        parts    = [p.strip() for p in raw_path.split(">")]
        normalised = "/".join(
            p.lower().replace(" ", "_").replace("?", "").replace(":", "")
            for p in parts if p
        )
        if normalised and normalised not in sections_seen:
            sections_seen.append(normalised)

        for pg in chunk.get("page_numbers", []):
            if pg not in pages_seen:
                pages_seen.append(pg)

    return {
        "sections": sections_seen,
        "pages":    sorted(pages_seen),
    }


# ─── Public API ───────────────────────────────────────────────────────────────

def generate(question: str, top_k: int = 5) -> dict:
    """
    Full RAG pipeline for a single question.

    Returns:
        {
          "answer":       str,
          "context":      str,
          "references":   {"sections": [...], "pages": [...]},
          "rate_limited": bool   ← True if API quota was exhausted
        }
    """
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return {
            "answer":       FALLBACK_ANSWER,
            "context":      "",
            "references":   {"sections": [], "pages": []},
            "rate_limited": False,
        }

    # Assemble context — deduplicate chunk_ids, cap at MAX_CONTEXT_CHARS
    seen_ids      = set()
    context_parts = []
    total_chars   = 0

    for chunk in chunks:
        cid = chunk["chunk_id"]
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        text = chunk["full_text"].strip()
        if total_chars + len(text) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 100:
                context_parts.append(text[:remaining])
            break
        context_parts.append(text)
        total_chars += len(text)

    context_text = "\n\n".join(context_parts)

    try:
        answer, rate_limited = _call_groq(question, context_text)
    except Exception as e:
        print(f"[Groq] API error: {e}")
        answer, rate_limited = FALLBACK_ANSWER, False

    return {
        "answer":       answer,
        "context":      context_text,
        "references":   _build_references(chunks),
        "rate_limited": rate_limited,
    }


# ─── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys
    q = " ".join(_sys.argv[1:]) if len(_sys.argv) > 1 else "What is operant conditioning?"
    print(f"\nQuestion: {q}\n")
    result = generate(q)
    print("─── ANSWER ───")
    print(result["answer"])
    print("\n─── REFERENCES ───")
    print(json.dumps(result["references"], indent=2))
    print("\n─── CONTEXT PREVIEW (first 400 chars) ───")
    print(result["context"][:400])
    if result["rate_limited"]:
        print("\n[WARNING] Rate limit was hit during this request.")