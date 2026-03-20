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

# ── Robust pathing — works from any working directory ─────────────────────────
_APP_DIR         = Path(__file__).parent.absolute()
_PROJECT_ROOT    = _APP_DIR.parent
_BM25_INDEX_PATH = _PROJECT_ROOT / "output" / "bm25_index.pkl"

# ── Lazy-loaded globals ───────────────────────────────────────────────────────
_milvus_collection = None
_bm25_data         = None
_lm_studio_url     = os.environ.get("LM_STUDIO_URL",   "http://localhost:1234/v1/embeddings")
_lm_studio_model   = os.environ.get("LM_STUDIO_MODEL", "nomic-ai/nomic-embed-text-v1.5-GGUF")


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
    Fuses scores as: hybrid = 0.6 * dense_norm + 0.4 * bm25_norm

    Returns top_k results sorted by hybrid_score descending.
    """

    # 1. Embed query
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
            "text", "has_image_context", "image_refs"
        ],
    )

    dense_candidates = {}
    for hit in dense_results[0]:
        cid = hit.entity.get("chunk_id")
        dense_candidates[cid] = {
            "chunk_id":          cid,
            "section_path":      hit.entity.get("section_path"),
            "page_numbers":      json.loads(hit.entity.get("page_numbers", "[]")),
            "text":              hit.entity.get("text"),
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
            "text":              meta["text"],
            "has_image_context": meta["has_image_context"],
            "image_refs":        meta["image_refs"] if isinstance(meta["image_refs"], list)
                                 else json.loads(meta.get("image_refs", "[]")),
            "bm25_score_raw":    float(score),
        }

    # 4. Fusion — build unified candidate pool
    # ── FIX: sort the union set so pool order is deterministic ────────────────
    # Using an unordered set meant pool[], dense_score_list[], bm25_score_list[]
    # were built from the same iterable in one loop, which is technically safe,
    # but non-deterministic across Python runs. Sorting by chunk_id makes the
    # fusion result identical every time for the same query.
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

    # 6. Compute hybrid scores and build final list
    final_results = []
    for i, (cid, meta) in enumerate(pool):
        dn           = dense_norm[i]
        bn           = bm25_norm[i]
        hybrid_score = 0.6 * dn + 0.4 * bn

        final_results.append({
            "chunk_id":          int(meta["chunk_id"]),
            "section_path":      meta["section_path"],
            "page_numbers":      meta["page_numbers"],
            "text":              meta["text"],
            "has_image_context": meta["has_image_context"],
            "image_refs":        meta["image_refs"],
            "dense_score":       round(float(dn),           4),
            "bm25_score":        round(float(bn),           4),
            "hybrid_score":      round(float(hybrid_score), 4),
        })

    final_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return final_results[:top_k]


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
            print(f"  Path  : {res['section_path']}")
            print(f"  Pages : {res['page_numbers']}")
            print(f"  ImgCtx: {res['has_image_context']}  |  ImgRefs: {res['image_refs']}")
            snippet = res["text"].replace("\n", " ")[:120]
            print(f"  Text  : {snippet}...\n")
    except Exception as e:
        print(f"Error during retrieval: {e}")