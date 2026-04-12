import os
import json
import sys
from dotenv import load_dotenv
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

load_dotenv()

# Zilliz Configuration
ZILLIZ_URI   = "https://in03-bc51ec1151acfd9.serverless.aws-eu-central-1.cloud.zilliz.com"
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")

_collection = None
_embed_model = None

def _get_milvus_collection():
    global _collection
    if _collection is None:
        token = ZILLIZ_TOKEN
        if not token:
            try:
                import streamlit as st
                token = st.secrets["ZILLIZ_TOKEN"]
            except Exception:
                pass
                
        connections.connect(alias="zilliz", uri=ZILLIZ_URI, token=token)
        _collection = Collection("psychology2e_chunks", using="zilliz")
        _collection.load()
    return _collection

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        # User explicitly specified nomic-ai/nomic-embed-text-v1
        _embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    return _embed_model

def retrieve(question: str, top_k: int = 5) -> list:
    """
    Retrieve top chunks from Zilliz Cloud using dense vector search.
    Applies score constraints: all scores < 0.3 are discarded.
    """
    model = _get_embed_model()
    col = _get_milvus_collection()
    
    # 2. ALWAYS prefix query strings with "search_query: " before encoding
    # 3. ALWAYS use normalize_embeddings=True when encoding
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
        # Rule 5: If no relevant chunk is found (all scores < 0.3), discard
        if hit.score < 0.3:
            continue
            
        e = hit.entity
        
        # Safely parse JSON strings from Zilliz to lists
        pn_raw = e.get("page_numbers")
        if isinstance(pn_raw, str):
            try:
                pn = json.loads(pn_raw.replace("'", '"'))
            except:
                pn = []
        else:
            pn = pn_raw if pn_raw else []
            
        ir_raw = e.get("image_refs")
        if isinstance(ir_raw, str):
            try:
                ir = json.loads(ir_raw.replace("'", '"'))
            except:
                ir = []
        else:
            ir = ir_raw if ir_raw else []

        results.append({
            "chunk_id":          e.get("chunk_id"),
            "section_path":      e.get("section_path"),
            "page_numbers":      pn,
            "clean_text":        e.get("clean_text", ""),
            "full_text":         e.get("full_text", ""),
            "has_image_context": e.get("has_image_context", False),
            "image_refs":        ir,
            "hybrid_score":      hit.score, # Alias to support downstream UI
            "score":             hit.score
        })
    
    return results

if __name__ == "__main__":
    test_query = "What is classical conditioning?"
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])

    print(f'Zilliz Retrieval Test for: "{test_query}"\\n')
    try:
        results = retrieve(test_query, top_k=5)
        for i, res in enumerate(results, 1):
            print(f"Result {i} | Score: {res['score']}")
            print(f"  Section : {res['section_path']}")
            print(f"  Pages   : {res['page_numbers']}")
            print(f"  ImgCtx  : {res['has_image_context']}  |  ImgRefs: {res['image_refs']}")
            snippet = str(res["clean_text"]).replace("\\n", " ")[:120]
            print(f"  Preview : {snippet}...")
            print()
    except Exception as e:
        print(f"Error during retrieval: {e}")