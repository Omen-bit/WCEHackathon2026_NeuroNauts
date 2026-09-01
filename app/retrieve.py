import os
import json
import sys
import re
import pickle
from pathlib import Path
from dotenv import load_dotenv
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

load_dotenv()

ZILLIZ_URI = "https://in03-bc51ec1151acfd9.serverless.aws-eu-central-1.cloud.zilliz.com"

_collection   = None
_embed_model  = None
_bm25_bundle  = None
_bm25_checked = False

_PROJECT_ROOT = Path(__file__).parent.parent
_BM25_PKL     = _PROJECT_ROOT / "output" / "bm25_index.pkl"


def _get_token():
    token = os.getenv("ZILLIZ_TOKEN")
    if token:
        return token
    try:
        import streamlit as st
        token = st.secrets.get("ZILLIZ_TOKEN")
        if token:
            return token
    except Exception:
        pass
    raise ValueError("ZILLIZ_TOKEN not found in environment or Streamlit secrets.")


def _get_milvus_collection():
    global _collection
    if _collection is None:
        token = _get_token()
        connections.connect(alias="zilliz", uri=ZILLIZ_URI, token=token)
        _collection = Collection("psychology2e_chunks", using="zilliz")
        _collection.load()
    return _collection


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    return _embed_model


def _tokenize(text: str) -> list:
    """Tokenize text for BM25 keyword matching."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [w for w in text.split() if len(w) > 2]


def _get_bm25_bundle():
    global _bm25_bundle, _bm25_checked
    if not _bm25_checked:
        _bm25_checked = True
        if _BM25_PKL.exists():
            try:
                with open(_BM25_PKL, "rb") as f:
                    _bm25_bundle = pickle.load(f)
            except Exception as e:
                print(f"[BM25] Warning: could not load index: {e}")
                _bm25_bundle = None
    return _bm25_bundle


def _get_field(entity, field, default=None):
    """Compatible field getter for pymilvus Hit entity across versions."""
    try:
        val = getattr(entity, field, None)
        if val is not None:
            return val
    except Exception:
        pass
    try:
        val = entity.get(field)
        if val is not None:
            return val
    except Exception:
        pass
    return default


def _parse_list_field(val):
    if isinstance(val, str):
        try:
            return json.loads(val.replace("'", '"'))
        except Exception:
            return []
    return val if val else []


def retrieve_dense(question: str, top_k: int = 8) -> list:
    """Performs dense vector search against Zilliz Cloud Milvus collection."""
    model = _get_embed_model()
    col   = _get_milvus_collection()

    q_vec = model.encode(
        [f"search_query: {question}"],
        normalize_embeddings=True
    ).tolist()

    hits = col.search(
        data=q_vec,
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=top_k,
        output_fields=[
            "chunk_id", "section_path", "page_numbers",
            "clean_text", "full_text", "has_image_context", "image_refs"
        ]
    )[0]

    results = []
    for hit in hits:
        if hit.score < 0.25:
            continue
        e = hit.entity
        pn = _parse_list_field(_get_field(e, "page_numbers", "[]"))
        ir = _parse_list_field(_get_field(e, "image_refs", "[]"))

        clean_text = _get_field(e, "clean_text", "")
        full_text  = _get_field(e, "full_text", "") or clean_text

        results.append({
            "chunk_id":          _get_field(e, "chunk_id"),
            "section_path":      _get_field(e, "section_path", ""),
            "page_numbers":      pn,
            "clean_text":        clean_text,
            "full_text":         full_text,
            "has_image_context": _get_field(e, "has_image_context", False),
            "image_refs":        ir,
            "score":             float(hit.score),
            "source":            "dense"
        })
    return results


def retrieve_bm25(question: str, top_k: int = 8) -> list:
    """Performs BM25 keyword search locally."""
    bundle = _get_bm25_bundle()
    if not bundle:
        return []

    bm25   = bundle.get("bm25")
    lookup = bundle.get("lookup", [])
    if not bm25 or not lookup:
        return []

    tokens = _tokenize(question)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    scored_items = sorted(zip(scores, lookup), key=lambda x: x[0], reverse=True)

    results = []
    for score, meta in scored_items[:top_k]:
        if score <= 0.5:
            continue
        pn = meta.get("page_numbers", [])
        ir = meta.get("image_refs", [])
        text = meta.get("text", "")
        results.append({
            "chunk_id":          meta.get("chunk_id"),
            "section_path":      meta.get("section_path", ""),
            "page_numbers":      pn,
            "clean_text":        text,
            "full_text":         text,
            "has_image_context": meta.get("has_image_context", False),
            "image_refs":        ir,
            "score":             float(score),
            "source":            "bm25"
        })
    return results


def retrieve(question: str, top_k: int = 5) -> list:
    """
    Hybrid Retrieval Engine with Reciprocal Rank Fusion (RRF):
    Combines dense semantic vector search with BM25 keyword matching for optimal recall & precision.
    """
    # 1. Retrieve Dense results
    dense_results = []
    try:
        dense_results = retrieve_dense(question, top_k=top_k * 2)
    except Exception as e:
        print(f"[Retrieve] Dense search warning: {e}")

    # 2. Retrieve BM25 results
    bm25_results = []
    try:
        bm25_results = retrieve_bm25(question, top_k=top_k * 2)
    except Exception as e:
        print(f"[Retrieve] BM25 search warning: {e}")

    if not dense_results and not bm25_results:
        return []

    # If only dense or only BM25 is available
    if not bm25_results:
        for r in dense_results:
            r["hybrid_score"] = r["score"]
        return dense_results[:top_k]

    if not dense_results:
        for r in bm25_results:
            r["hybrid_score"] = r["score"]
        return bm25_results[:top_k]

    # 3. Reciprocal Rank Fusion (RRF)
    k_rrf = 60
    rrf_scores = {}
    chunk_map = {}

    for rank, item in enumerate(dense_results):
        cid = str(item.get("chunk_id") or item.get("clean_text")[:50])
        chunk_map[cid] = item
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_rrf + rank + 1)) * 1.2  # slight dense preference

    for rank, item in enumerate(bm25_results):
        cid = str(item.get("chunk_id") or item.get("clean_text")[:50])
        if cid not in chunk_map:
            chunk_map[cid] = item
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_rrf + rank + 1))

    # Sort merged results by fused score
    sorted_cids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)
    
    final_results = []
    for cid in sorted_cids[:top_k]:
        item = chunk_map[cid]
        item["hybrid_score"] = round(rrf_scores[cid], 5)
        final_results.append(item)

    return final_results


if __name__ == "__main__":
    test_query = "What is classical conditioning?"
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])

    print(f'Hybrid Retrieval Test for: "{test_query}"\n')
    try:
        results = retrieve(test_query, top_k=5)
        for i, res in enumerate(results, 1):
            print(f"Result {i} | Hybrid Score: {res.get('hybrid_score', 0):.4f} | Source: {res.get('source')}")
            print(f"  Section : {res.get('section_path')}")
            print(f"  Pages   : {res.get('page_numbers')}")
            print(f"  Preview : {str(res.get('clean_text', '')).replace(chr(10), ' ')[:120]}...")
            print()
    except Exception as e:
        print(f"Error during retrieval: {e}")