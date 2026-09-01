"""
eval_suite.py — Industry-Grade RAG Evaluation Suite
Implements the RAG Triad & Key Enterprise Metrics:
  1. Faithfulness / Groundedness (Claim Extraction & Context Alignment)
  2. Answer Relevancy (Semantic + Intent Coverage)
  3. Context Precision (Signal-to-Noise Ratio & Rank Quality)
  4. Context Recall / Coverage (Information Completeness)
  5. Composite RAG Quality Index (0–100%)
"""

import re
import math
import json
import time
from typing import List, Dict, Any, Optional

FAITH_THRESHOLD = 0.72
RELEVANCY_THRESHOLD = 0.68
PRECISION_THRESHOLD = 0.55
RECALL_THRESHOLD = 0.60

def _split_into_sentences(text: str) -> List[str]:
    """Split text into distinct assertable claims/sentences."""
    if not text:
        return []
    # Split on sentence boundaries
    raw = re.split(r'(?<=[.?!])\s+', text.strip())
    cleaned = []
    for s in raw:
        s_clean = s.strip().strip('"\'')
        # Skip conversational filler, headings, or very short fragments
        if len(s_clean) > 15 and not s_clean.startswith("#"):
            cleaned.append(s_clean)
    return cleaned

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (mag_a * mag_b)))

def _extract_key_terms(text: str) -> set:
    """Extract significant domain keywords for recall estimation."""
    stopwords = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "in", "on", "at", "to", "for", "with", "by", "about", "against",
        "between", "into", "through", "during", "before", "after", "above",
        "below", "from", "up", "down", "of", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "can", "will", "just", "should", "now",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "am", "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "would", "could", "ought", "i", "you", "he", "she",
        "it", "we", "they", "them", "their", "its", "our", "your", "my"
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return {w for w in words if w not in stopwords}

def evaluate_faithfulness(answer: str, contexts: List[str], embed_fn) -> Dict[str, Any]:
    """
    Evaluates whether every claim in the generated answer is faithful
    to the retrieved textbook context (Zero-Hallucination verification).
    """
    if not answer or not contexts:
        return {"score": 0.0, "supported": 0, "total": 0, "sentences": []}

    sentences = _split_into_sentences(answer)
    if not sentences:
        return {"score": 1.0, "supported": 0, "total": 0, "sentences": []}

    # Embed all sentences and contexts in one batch
    all_texts = [f"search_document: {s}" for s in sentences] + [f"search_document: {c}" for c in contexts]
    embeddings = embed_fn(all_texts)
    
    sent_vecs = embeddings[:len(sentences)]
    ctx_vecs = embeddings[len(sentences):]

    results = []
    supported_count = 0

    for sent, sv in zip(sentences, sent_vecs):
        sims = [_cosine_similarity(sv, cv) for cv in ctx_vecs]
        max_sim = max(sims) if sims else 0.0
        best_ctx_idx = int(sims.index(max_sim)) if sims else 0
        is_supported = max_sim >= FAITH_THRESHOLD

        if is_supported:
            supported_count += 1

        results.append({
            "sentence": sent,
            "max_sim": round(max_sim, 4),
            "supported": is_supported,
            "best_context_idx": best_ctx_idx + 1
        })

    score = round(supported_count / len(sentences), 4) if sentences else 0.0
    return {
        "score": score,
        "supported": supported_count,
        "total": len(sentences),
        "sentences": results
    }

def evaluate_answer_relevancy(question: str, answer: str, embed_fn) -> Dict[str, Any]:
    """
    Evaluates semantic and intent relevancy of the answer to the user question.
    """
    if not question or not answer:
        return {"score": 0.0, "similarity": 0.0}

    vecs = embed_fn([f"search_query: {question}", f"search_document: {answer}"])
    sim = _cosine_similarity(vecs[0], vecs[1])

    # Slight penalty if answer says 'Not found in the textbook'
    if "not found in the provided textbook" in answer.lower():
        final_score = round(sim * 0.5, 4)
    else:
        final_score = round(sim, 4)

    return {
        "score": final_score,
        "raw_similarity": round(sim, 4)
    }

def evaluate_context_precision(question: str, contexts: List[str], embed_fn) -> Dict[str, Any]:
    """
    Measures signal-to-noise ratio in retrieved contexts (Context Precision).
    Higher if the most relevant chunks are placed at rank 1 & 2.
    """
    if not question or not contexts:
        return {"score": 0.0, "chunk_scores": []}

    q_vec = embed_fn([f"search_query: {question}"])[0]
    ctx_vecs = embed_fn([f"search_document: {c}" for c in contexts])

    chunk_sims = [_cosine_similarity(q_vec, cv) for cv in ctx_vecs]
    
    # Calculate Mean Average Precision / Rank-Weighted Precision
    precisions = []
    relevant_count = 0
    for rank, sim in enumerate(chunk_sims, 1):
        if sim >= 0.45:  # Relevance threshold for dense context
            relevant_count += 1
            precisions.append(relevant_count / rank)
        else:
            precisions.append(0.0)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    top_score = max(chunk_sims) if chunk_sims else 0.0

    # Composite precision score
    score = round((avg_precision * 0.6) + (top_score * 0.4), 4)

    return {
        "score": min(1.0, max(0.0, score)),
        "chunk_scores": [round(s, 4) for s in chunk_sims],
        "top_chunk_sim": round(top_score, 4),
        "relevant_chunks_ratio": round(relevant_count / len(chunk_sims), 4) if chunk_sims else 0.0
    }

def evaluate_context_recall(question: str, contexts: List[str], answer: str, embed_fn=None) -> Dict[str, Any]:
    """
    Evaluates whether the retrieved context contains all necessary factual components
    needed to satisfy the question and generate the answer.
    """
    if not contexts or not answer:
        return {"score": 0.0, "covered_ratio": 0.0, "key_terms": []}

    combined_context = " ".join(contexts).lower()
    q_terms = _extract_key_terms(question)
    ans_terms = _extract_key_terms(answer)
    target_terms = q_terms.union(ans_terms)

    if not target_terms:
        return {"score": 1.0, "covered_ratio": 1.0, "key_terms": []}

    covered = [t for t in target_terms if t in combined_context]
    coverage_ratio = len(covered) / len(target_terms)

    return {
        "score": round(coverage_ratio, 4),
        "covered_count": len(covered),
        "total_terms": len(target_terms),
        "covered_terms": covered[:8],
        "missing_terms": [t for t in target_terms if t not in combined_context][:5]
    }

def evaluate_rag_triad(
    question: str,
    answer: str,
    contexts: List[str],
    embed_fn
) -> Dict[str, Any]:
    """
    Computes full RAG Triad + Enterprise Quality Metrics.
    """
    faith = evaluate_faithfulness(answer, contexts, embed_fn)
    relevancy = evaluate_answer_relevancy(question, answer, embed_fn)
    precision = evaluate_context_precision(question, contexts, embed_fn)
    recall = evaluate_context_recall(question, contexts, answer, embed_fn)

    f_score = faith["score"]
    r_score = relevancy["score"]
    p_score = precision["score"]
    rec_score = recall["score"]

    # Composite RAG Quality Index (Weighted Average)
    composite = round(
        (0.35 * f_score) + (0.25 * r_score) + (0.20 * p_score) + (0.20 * rec_score),
        4
    )

    grade = (
        "A+ (Production Ready)" if composite >= 0.85 else
        "A (High Quality)" if composite >= 0.75 else
        "B (Acceptable)" if composite >= 0.65 else
        "C (Needs Improvement)" if composite >= 0.50 else
        "D (Low Quality)"
    )

    return {
        "composite_score": composite,
        "grade": grade,
        "faithfulness": faith,
        "answer_relevancy": relevancy,
        "context_precision": precision,
        "context_recall": recall,
    }
