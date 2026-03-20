import os
import re
import time
import json
import requests
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1/embeddings")
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# -- Path discovery -------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(BASE_DIR, "output", "psychology2e_chunks.json")
CONFIG_OUT  = os.path.join(BASE_DIR, "output", "milvus_config.json")

MODEL_NAME      = os.getenv("LM_STUDIO_MODEL", "nomic-ai/nomic-embed-text-v1.5-GGUF")
COLLECTION_NAME = "psychology2e_chunks"

# ── Image description stripper ────────────────────────────────────────────────
IMAGE_DESC_RE = re.compile(r'\[Image Description - [^\]]+\]:.*?(?=\[Image Description -|\Z)', re.DOTALL)

def strip_image_descriptions(text: str) -> str:
    """
    Removes all [Image Description - filename]: ... blocks from chunk text.
    Returns clean academic text only — used for embedding so image
    description content never pollutes the vector space.
    """
    clean = IMAGE_DESC_RE.sub('', text)
    return clean.strip()


def embed_text(text: str) -> list:
    """
    Sends a string to LM Studio embedding server and returns a 768-dim vector.
    Implements 3 retry attempts with a 2-second wait between each.
    Always embeds clean_text (no image descriptions) for uncontaminated vectors.
    """
    payload = {
        "model": MODEL_NAME,
        "input": text
    }

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = requests.post(LM_STUDIO_URL, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data['data'][0]['embedding']
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise e


def setup_collection():
    """
    Creates the Milvus collection with the predefined schema.
    If the collection already exists, it is dropped and recreated (idempotent).

    Two text fields:
      - clean_text : pure academic text, no image descriptions (used for embedding)
      - full_text  : complete chunk text including image descriptions (used for LLM context)
    """
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="chunk_id",          dtype=DataType.INT64,         is_primary=True, auto_id=False),
        FieldSchema(name="section_path",       dtype=DataType.VARCHAR,       max_length=512),
        FieldSchema(name="page_numbers",       dtype=DataType.VARCHAR,       max_length=128),
        FieldSchema(name="clean_text",         dtype=DataType.VARCHAR,       max_length=65535),  # for embedding
        FieldSchema(name="full_text",          dtype=DataType.VARCHAR,       max_length=65535),  # for LLM context
        FieldSchema(name="token_count",        dtype=DataType.INT64),
        FieldSchema(name="has_image_context",  dtype=DataType.BOOL),
        FieldSchema(name="image_refs",         dtype=DataType.VARCHAR,       max_length=512),
        FieldSchema(name="embedding",          dtype=DataType.FLOAT_VECTOR,  dim=768),
    ]

    schema     = CollectionSchema(fields, description="Psychology chunks collection")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    print(f"Collection '{COLLECTION_NAME}' created successfully.")
    print("Schema fields:")
    for field in collection.schema.fields:
        try:
            dtype_name = field.dtype.name
        except AttributeError:
            dtype_name = str(field.dtype)
        print(f"  - {field.name}: {dtype_name}")
    return collection


def process_and_store():
    # Load chunks
    if not os.path.exists(CHUNKS_PATH):
        print(f"ERROR: Chunks file not found at {CHUNKS_PATH}")
        return

    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks       = data['chunks']
    total_chunks = len(chunks)
    print(f"Found {total_chunks} chunks to embed.")

    # Set up collection
    collection = setup_collection()

    # Embedding and mapping
    embedded_data = []
    print("\nStarting embedding process...")

    for i, chunk in enumerate(chunks, 1):
        chunk_id  = chunk['chunk_id']
        full_text = chunk['text']                        # includes image descriptions
        clean_text = strip_image_descriptions(full_text) # pure text only → for vector

        print(f"[{i}/{total_chunks}] Embedding chunk_id={chunk_id} "
              f"| clean={len(clean_text)} chars | full={len(full_text)} chars")

        # ── FIX: embed clean_text only, NOT full_text ─────────────────────────
        vector = embed_text(clean_text)

        entry = {
            "chunk_id":         chunk_id,
            "section_path":     chunk['section_path'],
            "page_numbers":     json.dumps(chunk['page_numbers']),
            "clean_text":       clean_text,              # stored for reference / BM25
            "full_text":        full_text,               # stored for LLM context pack
            "token_count":      chunk['token_count'],
            "has_image_context": chunk['has_image_context'],
            "image_refs":       json.dumps(chunk['image_refs']),
            "embedding":        vector,
        }
        embedded_data.append(entry)

    # Batch insert
    print(f"\nInserting {len(embedded_data)} chunks into Milvus...")
    collection.insert(embedded_data)
    collection.flush()
    print("Insertion complete.")

    # Indexing
    print("\nBuilding HNSW index on 'embedding' field...")
    index_params = {
        "metric_type": "COSINE",
        "index_type":  "HNSW",
        "params":      {"M": 16, "efConstruction": 200},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("Index built successfully.")

    # Load collection
    collection.load()
    print("Collection loaded into memory.")

    # Verification
    total_entities = collection.num_entities
    print(f"\nVerification — Total entities in collection: {total_entities}")

    # Test search
    search_query  = "What is psychology?"
    print(f"Running test search for: '{search_query}'")
    query_vector  = embed_text(strip_image_descriptions(search_query))

    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=3,
        output_fields=["section_path", "clean_text"],
    )

    print("Top 3 Search Results:")
    test_results = []
    for hit in results[0]:
        res_info = f"  - Score: {hit.score:.4f} | Section: {hit.entity.get('section_path')}"
        print(res_info)
        test_results.append({
            "score":        hit.score,
            "section_path": hit.entity.get("section_path"),
        })

    # Save milvus_config.json
    config = {
        "collection_name": COLLECTION_NAME,
        "embedding_dim":   768,
        "metric_type":     "COSINE",
        "top_k":           5,
        "lm_studio_url":   LM_STUDIO_URL,
        "milvus_host":     MILVUS_HOST,
        "milvus_port":     MILVUS_PORT,
    }
    with open(CONFIG_OUT, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {CONFIG_OUT}")

    # Final summary
    print("\n" + "=" * 40)
    print("FINAL SUMMARY - STAGE 3")
    print("=" * 40)
    print(f"Chunks Embedded   : {total_chunks}")
    print(f"Vectors Stored    : {total_entities}")
    print(f"Collection Name   : {COLLECTION_NAME}")
    print(f"Index Type        : HNSW (Metric: COSINE)")
    print(f"Embedded field    : clean_text  (no image descriptions)")
    print(f"LLM context field : full_text   (with image descriptions)")
    print(f"Test Search       : Top match at score {results[0][0].score:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    milvus_ready = True
    lm_ready     = True

    # Verify LM Studio
    try:
        requests.get("http://localhost:1234/v1/models", timeout=2)
    except Exception:
        print("ERROR: Start LM Studio server on port 1234 first.")
        lm_ready = False

    # Verify Milvus
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=2)
    except Exception:
        print("ERROR: Start Milvus via Docker first.")
        milvus_ready = False

    if milvus_ready and lm_ready:
        process_and_store()
    else:
        print("Missing services. Aborting.")