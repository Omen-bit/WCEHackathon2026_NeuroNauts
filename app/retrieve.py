import os
import re
import json
import pickle
import requests
import sys
from dotenv import load_dotenv
from pymilvus import connections, Collection
from pathlib import Path

load_dotenv()

# ── Robust pathing ─────────────────────────────────────────────────────────────
_APP_DIR         = Path(__file__).parent.absolute()
_PROJECT_ROOT    = _APP_DIR.parent
_BM25_INDEX_PATH = _PROJECT_ROOT / "output" / "bm25_index.pkl"

# ── Lazy-loaded globals ────────────────────────────────────────────────────────
_milvus_collection = None
_bm25_data         = None
_lm_studio_url     = os.environ.get("LM_STUDIO_URL",   "http://localhost:1234/v1/embeddings")
_lm_studio_model   = os.environ.get("LM_STUDIO_MODEL", "nomic-ai/nomic-embed-text-v1.5-GGUF")

# ── Relevance threshold ────────────────────────────────────────────────────────
# Chunks with hybrid_score below this are dropped from results.
# Prevents weakly-matched chunks (e.g. Introduction/Figure captions) from
# appearing in the top-K when they are not genuinely relevant.
RELEVANCE_THRESHOLD = 0.25

# ── Image description stripper (mirrors embed_and_store.py) ───────────────────
IMAGE_DESC_RE = re.compile(r'\[Image Description - [^\]]+\]:.*?(?=\[Image Description -|\Z)', re.DOTALL)

def strip_image_descriptions(text: str) -> str:
    """Remove [Image Description - ...] blocks. Used before embedding the query."""
    return IMAGE_DESC_RE.sub('', text).strip()


def tokenize(text: str) -> list:
    """Tokenizer — must match build_bm25.py exactly."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()


def _get_milvus_collection():
    global _milvus_collection
    if _milvus_collection is None:
        milvus_host = os.environ.get("MILVUS_HOST", "localhost")
        milvus_port = os.environ.get("MILVUS_PORT", "19530")
        connections.connect(host=milvus_host, port=milvus_port)
        _milvus_collection = Collection("psychology2e_chunks")
        _milvus_collection.load()
    return _milvus_collection


def _get_bm25_data():
    global _bm25_data
    if _bm25_data is None:
        if not _BM25_INDEX_PATH.exists():
            raise FileNotFoundError(f"BM25 index not found at {_BM25_INDEX_PATH}")
        with open(_BM25_INDEX_PATH, 'rb') as f:
            _bm25_data = pickle.load(f)
    return _bm25_data


def embed_query(query: str) -> list:
    """Embed the query using clean text only (no image description contamination)."""
    payload = {
        "model": _lm_studio_model,
        "input": [query],
    }
    response = requests.post(_lm_studio_url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def normalize_scores(scores: list) -> list:
    """Min-max normalize a list of floats to [0, 1]."""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def retrieve(query: str, top_k: int = 5) -> list:
    """
    Stage 5 — Hybrid Retrieval.
    Combines Milvus dense (COSINE HNSW) + BM25 sparse search.
    Fusion: hybrid = 0.6 * dense_norm + 0.4 * bm25_norm

    Key fixes vs original:
      - Query is embedded as clean text (no image desc pollution)
      - Milvus returns both clean_text and full_text fields
      - full_text (with image descriptions) is passed to the LLM context
      - clean_text is what BM25 also indexes — keeps scoring consistent
      - Chunks below RELEVANCE_THRESHOLD are filtered out before returning
      - Each result clearly separates text_for_display vs text_for_llm

    Returns top_k results sorted by hybrid_score descending.
    """

    # 1. Embed query — clean text only
    query_vector = embed_query(query)

    # 2. Dense search — fetch top_k * 2 candidates
    collection    = _get_milvus_collection()
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    dense_results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k * 2,
        output_fields=[
            "chunk_id", "section_path", "page_numbers",
            "clean_text", "full_text",           # ── FIX: fetch both fields
            "has_image_context", "image_refs",
        ],
    )

    dense_candidates = {}
    for hit in dense_results[0]:
        cid = hit.entity.get("chunk_id")
        dense_candidates[cid] = {
            "chunk_id":          cid,
            "section_path":      hit.entity.get("section_path"),
            "page_numbers":      json.loads(hit.entity.get("page_numbers", "[]")),
            "clean_text":        hit.entity.get("clean_text", ""),   # for display / BM25
            "full_text":         hit.entity.get("full_text",  ""),   # for LLM context
            "has_image_context": hit.entity.get("has_image_context"),
            "image_refs":        json.loads(hit.entity.get("image_refs", "[]")),
            "dense_score_raw":   float(hit.score),
        }

    # 3. Sparse search — BM25 top_k * 2 candidates
    bm25_data   = _get_bm25_data()
    bm25_model  = bm25_data["bm25"]
    bm25_lookup = bm25_data["lookup"]

    tokens          = tokenize(query)
    bm25_scores_raw = bm25_model.get_scores(tokens)

    bm25_results = sorted(
        zip(bm25_scores_raw, bm25_lookup),
        key=lambda x: x[0],
        reverse=True,
    )[:top_k * 2]

    bm25_candidates = {}
    for score, meta in bm25_results:
        cid = meta["chunk_id"]
        bm25_candidates[cid] = {
            "chunk_id":          cid,
            "section_path":      meta["section_path"],
            "page_numbers":      meta["page_numbers"] if isinstance(meta["page_numbers"], list)
                                 else json.loads(meta["page_numbers"]),
            "clean_text":        meta.get("clean_text", meta.get("text", "")),
            "full_text":         meta.get("full_text",  meta.get("text", "")),
            "has_image_context": meta["has_image_context"],
            "image_refs":        meta["image_refs"] if isinstance(meta["image_refs"], list)
                                 else json.loads(meta.get("image_refs", "[]")),
            "bm25_score_raw":    float(score),
        }

    # 4. Fusion — deterministic pool sorted by chunk_id
    all_chunk_ids = sorted(
        set(dense_candidates.keys()) | set(bm25_candidates.keys())
    )

    dense_score_list = []
    bm25_score_list  = []
    pool             = []

    for cid in all_chunk_ids:
        meta    = dense_candidates.get(cid) or bm25_candidates.get(cid)
        d_score = dense_candidates.get(cid, {}).get("dense_score_raw", 0.0)
        b_score = bm25_candidates.get(cid, {}).get("bm25_score_raw",  0.0)

        dense_score_list.append(d_score)
        bm25_score_list.append(b_score)
        pool.append((cid, meta))

    # 5. Normalize independently
    dense_norm = normalize_scores(dense_score_list)
    bm25_norm  = normalize_scores(bm25_score_list)

    # 6. Compute hybrid scores
    final_results = []
    for i, (cid, meta) in enumerate(pool):
        dn           = dense_norm[i]
        bn           = bm25_norm[i]
        hybrid_score = 0.6 * dn + 0.4 * bn

        final_results.append({
            "chunk_id":          int(meta["chunk_id"]),
            "section_path":      meta["section_path"],
            "page_numbers":      meta["page_numbers"],

            # ── FIX: two separate text fields ─────────────────────────────────
            # clean_text → show in Sources panel / snippet previews
            # full_text  → pass to LLM as context (includes image descriptions)
            "clean_text":        meta.get("clean_text", ""),
            "full_text":         meta.get("full_text",  ""),

            "has_image_context": meta["has_image_context"],
            "image_refs":        meta["image_refs"],
            "dense_score":       round(float(dn),           4),
            "bm25_score":        round(float(bn),           4),
            "hybrid_score":      round(float(hybrid_score), 4),
        })

    final_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

    # 7. ── FIX: apply relevance threshold ─────────────────────────────────────
    # Filters out weakly-matched chunks (e.g. Introduction figure captions,
    # preamble leftovers) that pollute the top-K with irrelevant content.
    filtered = [r for r in final_results if r["hybrid_score"] >= RELEVANCE_THRESHOLD]

    # If threshold removes everything, fall back to top result only
    if not filtered:
        filtered = final_results[:1]

    return filtered[:top_k]


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_query = "What is classical conditioning?"
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])

    print(f'Hybrid Retrieval Test for: "{test_query}"\n')
    try:
        results = retrieve(test_query, top_k=5)
        for i, res in enumerate(results, 1):
            print(f"Result {i} | Hybrid: {res['hybrid_score']} | Dense: {res['dense_score']} | BM25: {res['bm25_score']}")
            print(f"  Section : {res['section_path']}")
            print(f"  Pages   : {res['page_numbers']}")
            print(f"  ImgCtx  : {res['has_image_context']}  |  ImgRefs: {res['image_refs']}")
            snippet = res["clean_text"].replace("\n", " ")[:120]
            print(f"  Preview : {snippet}...")
            if res["has_image_context"]:
                full_snippet = res["full_text"].replace("\n", " ")[len(res["clean_text"]):].strip()[:120]
                print(f"  ImgDesc : {full_snippet}...")
            print()
    except Exception as e:
        print(f"Error during retrieval: {e}")