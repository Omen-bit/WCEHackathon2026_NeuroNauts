import os
import json
import sys
from dotenv import load_dotenv
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

load_dotenv()

ZILLIZ_URI = "https://in03-bc51ec1151acfd9.serverless.aws-eu-central-1.cloud.zilliz.com"

_collection  = None
_embed_model = None


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


def _get_field(entity, field, default=None):
    """Compatible field getter for pymilvus Hit entity across versions."""
    try:
        # New pymilvus versions use attribute access
        val = getattr(entity, field, None)
        if val is not None:
            return val
    except Exception:
        pass
    try:
        # Older versions support .get(field) with single argument
        val = entity.get(field)
        if val is not None:
            return val
    except Exception:
        pass
    return default


def retrieve(question: str, top_k: int = 5) -> list:
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
        if hit.score < 0.3:
            continue

        e = hit.entity

        pn_raw = _get_field(e, "page_numbers", "[]")
        if isinstance(pn_raw, str):
            try:
                pn = json.loads(pn_raw.replace("'", '"'))
            except Exception:
                pn = []
        else:
            pn = pn_raw if pn_raw else []

        ir_raw = _get_field(e, "image_refs", "[]")
        if isinstance(ir_raw, str):
            try:
                ir = json.loads(ir_raw.replace("'", '"'))
            except Exception:
                ir = []
        else:
            ir = ir_raw if ir_raw else []

        results.append({
            "chunk_id":          _get_field(e, "chunk_id"),
            "section_path":      _get_field(e, "section_path"),
            "page_numbers":      pn,
            "clean_text":        _get_field(e, "clean_text", ""),
            "full_text":         _get_field(e, "full_text", ""),
            "has_image_context": _get_field(e, "has_image_context", False),
            "image_refs":        ir,
            "hybrid_score":      hit.score,
            "score":             hit.score,
        })

    return results


if __name__ == "__main__":
    test_query = "What is classical conditioning?"
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])

    print(f'Zilliz Retrieval Test for: "{test_query}"\n')
    try:
        results = retrieve(test_query, top_k=5)
        for i, res in enumerate(results, 1):
            print(f"Result {i} | Score: {res['score']:.4f}")
            print(f"  Section : {res['section_path']}")
            print(f"  Pages   : {res['page_numbers']}")
            print(f"  Preview : {str(res['clean_text']).replace(chr(10), ' ')[:120]}...")
            print()
    except Exception as e:
        print(f"Error during retrieval: {e}")