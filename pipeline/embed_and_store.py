import os
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

# Model name from environment variable or default
MODEL_NAME = os.getenv("LM_STUDIO_MODEL", "nomic-ai/nomic-embed-text-v1.5-GGUF")
COLLECTION_NAME = "psychology2e_chunks"

def embed_text(text: str) -> list:
    """
    Sends a string to LM Studio embedding server and returns a 768-dimensional vector.
    Implements 3 retry attempts with a 2-second wait between each.
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
    """
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
    
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="section_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="page_numbers", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="token_count", dtype=DataType.INT64),
        FieldSchema(name="has_image_context", dtype=DataType.BOOL),
        FieldSchema(name="image_refs", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    
    schema = CollectionSchema(fields, description="Psychology chunks collection")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    
    print(f"Collection {COLLECTION_NAME} created successfully.")
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
    chunks_file = "../output/psychology2e_chunks.json"
    if not os.path.exists(chunks_file):
        print(f"ERROR: Chunks file {chunks_file} not found.")
        return

    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data['chunks']
    total_chunks = len(chunks)
    print(f"Found {total_chunks} chunks to embed.")

    # Set up collection
    collection = setup_collection()

    # Embedding and mapping
    embedded_data = []
    print("\nStarting embedding process...")
    for i, chunk in enumerate(chunks, 1):
        chunk_id = chunk['chunk_id']
        print(f"[{i}/{total_chunks}] Embedding chunk_id={chunk_id}")
        
        vector = embed_text(chunk['text'])
        
        # Mapping to Milvus schema
        entry = {
            "chunk_id": chunk_id,
            "section_path": chunk['section_path'],
            "page_numbers": json.dumps(chunk['page_numbers']),
            "text": chunk['text'],
            "token_count": chunk['token_count'],
            "has_image_context": chunk['has_image_context'],
            "image_refs": json.dumps(chunk['image_refs']),
            "embedding": vector
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
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("Index built successfully.")
    
    # Load collection
    collection.load()
    print("Collection loaded into memory.")

    # Verification
    total_entities = collection.num_entities
    print(f"\nVerification — Total entities in collection: {total_entities}")

    # Test Search
    search_query = "What is psychology?"
    print(f"Running test search for: '{search_query}'")
    
    query_vector = embed_text(search_query)
    
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=3,
        output_fields=["section_path"]
    )
    
    print("Top 3 Search Results:")
    test_results = []
    for hit in results[0]:
        res_info = f"  - Score: {hit.score:.4f} | Section: {hit.entity.get('section_path')}"
        print(res_info)
        test_results.append({"score": hit.score, "section_path": hit.entity.get("section_path")})

    # Save milvus_config.json
    config = {
        "collection_name": COLLECTION_NAME,
        "embedding_dim": 768,
        "metric_type": "COSINE",
        "top_k": 5,
        "lm_studio_url": LM_STUDIO_URL,
        "milvus_host": MILVUS_HOST,
        "milvus_port": MILVUS_PORT
    }
    
    config_path = "../output/milvus_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_path}")

    # Final Summary
    print("\n" + "="*40)
    print("FINAL SUMMARY - STAGE 3")
    print("="*40)
    print(f"Chunks Embedded:   {total_chunks}")
    print(f"Vectors Stored:    {total_entities}")
    print(f"Collection Name:   {COLLECTION_NAME}")
    print(f"Index Type:        HNSW (Metric: COSINE)")
    print(f"Test Search:       Top match at score {results[0][0].score:.4f}")
    print("="*40)

if __name__ == "__main__":
    # Verify connections at startup
    milvus_ready = True
    lm_ready = True
    
    # Verify LM Studio
    try:
        requests.get("http://localhost:1234/v1/models", timeout=2)
    except:
        print("ERROR: Start LM Studio server on port 1234 first.")
        lm_ready = False

    # Verify Milvus
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=2)
    except:
        print("ERROR: Start Milvus via Docker first.")
        milvus_ready = False
    
    if milvus_ready and lm_ready:
        process_and_store()
    else:
        print("Missing services. Aborting process.")
