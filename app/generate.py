"""
Stage 6 — Answer Generation via LM Studio (Gemma 3 4B)
Generates grounded answers from retrieved chunks using local Gemma 3 4B via LM Studio.

Requirements:
  - LM Studio running with gemma-3-4b-it-Q4_K_M.gguf loaded
  - LM Studio server reachable at http://localhost:1234
  - retrieve.py must be in the same folder as this file

Usage:
  from generate import generate
  result = generate("What is classical conditioning?", top_k=5)
  # returns dict with keys: answer, context, references
"""

import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── FIX: ensure app/ directory is on sys.path so `from retrieve import retrieve`
# works regardless of which directory the script is called from.
# Without this, calling `python app/generate.py` from the project root fails
# with ModuleNotFoundError because Python looks in the CWD, not the script dir.
_THIS_DIR = Path(__file__).parent.absolute()
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from retrieve import retrieve


# ─── Config ──────────────────────────────────────────────────────────────────

LM_STUDIO_BASE_URL  = os.getenv("LM_STUDIO_BASE_URL",  "http://localhost:1234")
LM_STUDIO_CHAT_URL  = f"{LM_STUDIO_BASE_URL}/v1/chat/completions"
LM_STUDIO_LLM_MODEL = os.getenv("LM_STUDIO_LLM_MODEL", "gemma-3-4b-it-Q4_K_M.gguf")

MAX_CONTEXT_CHARS = 4000   # Gemma 3 4B context budget — stay safe
FALLBACK_ANSWER   = "Not found in the provided textbook."


# ─── Prompt ──────────────────────────────────────────────────────────────────

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


# ─── LM Studio call ──────────────────────────────────────────────────────────

def _call_lm_studio(question: str, context_text: str) -> str:
    payload = {
        "model":       LM_STUDIO_LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(question, context_text)},
        ],
        "temperature": 0.2,
        "max_tokens":  512,
        "top_p":       0.9,
        "stream":      False,
    }
    resp = requests.post(
        LM_STUDIO_CHAT_URL,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ─── Reference builder ───────────────────────────────────────────────────────

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


# ─── Public API ──────────────────────────────────────────────────────────────

def generate(question: str, top_k: int = 5) -> dict:
    """
    Full RAG pipeline for a single question.

    Returns:
        {
          "answer":     str,
          "context":    str,
          "references": {"sections": [...], "pages": [...]}
        }
    """
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return {
            "answer":     FALLBACK_ANSWER,
            "context":    "",
            "references": {"sections": [], "pages": []},
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
        text = chunk["text"].strip()
        if total_chars + len(text) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 100:
                context_parts.append(text[:remaining])
            break
        context_parts.append(text)
        total_chars += len(text)

    context_text = "\n\n".join(context_parts)

    try:
        answer = _call_lm_studio(question, context_text)
    except Exception as e:
        print(f"[LM Studio] API error: {e}")
        answer = FALLBACK_ANSWER

    return {
        "answer":     answer,
        "context":    context_text,
        "references": _build_references(chunks),
    }


# ─── CLI test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is operant conditioning?"
    print(f"\nQuestion: {q}\n")
    result = generate(q)
    print("─── ANSWER ───")
    print(result["answer"])
    print("\n─── REFERENCES ───")
    print(json.dumps(result["references"], indent=2))
    print("\n─── CONTEXT PREVIEW (first 400 chars) ───")
    print(result["context"][:400])