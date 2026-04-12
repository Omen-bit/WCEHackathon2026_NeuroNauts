from pymilvus import connections, Collection, utility
import json, time, os
from dotenv import load_dotenv

load_dotenv()
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")

ZILLIZ_URI      = "https://in03-bc51ec1151acfd9.serverless.aws-eu-central-1.cloud.zilliz.com"
LOCAL_HOST      = "127.0.0.1"
LOCAL_PORT      = "19530"
COLLECTION_NAME = "psychology2e_chunks"
BATCH_SIZE      = 200


def parse_embedding(raw):
    # Embedding is already a list of floats — just enforce float type
    return [float(x) for x in raw]


# ── STEP 1: Export from local Milvus ──────────────────────────
def export_local():
    print("\n🔌  Connecting to local Milvus (Attu)...")
    connections.connect("local", host=LOCAL_HOST, port=LOCAL_PORT)

    if not utility.has_collection(COLLECTION_NAME, using="local"):
        raise Exception(f"❌ Collection '{COLLECTION_NAME}' not found in local Milvus!")

    col = Collection(COLLECTION_NAME, using="local")
    col.load()
    total = col.num_entities
    print(f"📦  Found {total} entities — exporting with vectors...")

    all_rows = []
    offset   = 0

    while offset < total:
        batch = col.query(
            expr=f"chunk_id >= {offset} && chunk_id < {offset + BATCH_SIZE}",
            output_fields=[
                "chunk_id", "section_path", "page_numbers",
                "clean_text", "full_text", "token_count",
                "has_image_context", "image_refs", "embedding"
            ],
            limit=BATCH_SIZE
        )
        all_rows.extend(batch)
        offset += BATCH_SIZE
        print(f"    Exported {min(offset, total)}/{total}", end="\r")

    connections.disconnect("local")
    print(f"\n✅  Exported {len(all_rows)} chunks\n")
    return all_rows


# ── STEP 2: Upload to Zilliz Cloud ────────────────────────────
def upload_zilliz(rows):
    print("☁️   Connecting to Zilliz Cloud...")
    connections.connect(alias="zilliz", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)

    if not utility.has_collection(COLLECTION_NAME, using="zilliz"):
        raise Exception(
            f"❌ Collection '{COLLECTION_NAME}' not found on Zilliz!\n"
            "   Make sure you created it in the Zilliz UI first."
        )

    col   = Collection(COLLECTION_NAME, using="zilliz")
    total = len(rows)
    print(f"⬆️   Uploading {total} chunks in batches of {BATCH_SIZE}...")

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]

        # Build list of row-dicts — most reliable format across pymilvus versions
        row_data = []
        for c in batch:
            row_data.append({
                "chunk_id":          int(c["chunk_id"]),
                "section_path":      str(c.get("section_path", "")),
                "page_numbers":      str(c.get("page_numbers", "[]")),
                "clean_text":        str(c.get("clean_text", ""))[:8192],
                "full_text":         str(c.get("full_text", ""))[:8192],
                "token_count":       int(c.get("token_count", 0)),
                "has_image_context": bool(c.get("has_image_context", False)),
                "image_refs":        json.dumps(c.get("image_refs", []))[:1024],
                "embedding":         parse_embedding(c["embedding"]),
            })

        col.insert(row_data)
        print(f"    Uploaded {min(i + BATCH_SIZE, total)}/{total}", end="\r")

    col.flush()
    print(f"\n✅  Upload complete — {col.num_entities} entities on Zilliz Cloud\n")
    return col


# ── STEP 3: Build vector index ─────────────────────────────────
def build_index(col):
    print("🔍  Building COSINE vector index...")
    col.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",
            "index_type":  "IVF_FLAT",
            "params":      {"nlist": 128}
        }
    )
    col.load()
    print("✅  Index built and collection loaded\n")


# ── STEP 4: Test query ─────────────────────────────────────────
def test_query():
    question = "What is operant conditioning?"
    print(f"🧪  Test query: '{question}'")
    print("─" * 60)

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        q_vec = model.encode([question]).tolist()
    except ImportError:
        print("⚠️  sentence-transformers not installed — skipping test.")
        print("    pip install sentence-transformers   (optional)")
        return

    connections.connect(alias="zilliz", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    col = Collection(COLLECTION_NAME, using="zilliz")
    col.load()

    hits = col.search(
        data=q_vec,
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=3,
        output_fields=["chunk_id", "section_path", "page_numbers", "clean_text"]
    )[0]

    for i, hit in enumerate(hits, 1):
        e = hit.entity
        print(f"\n[{i}] Score : {hit.score:.4f}")
        print(f"    Section: {e.get('section_path')}")
        print(f"    Pages  : {e.get('page_numbers')}")
        print(f"    Text   : {e.get('clean_text')[:200]}...")

    connections.disconnect("zilliz")
    print("\n🎉  All good! Your data is live on Zilliz Cloud.")


# ── MAIN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    start = time.time()

    rows = export_local()       # Step 1
    col  = upload_zilliz(rows)  # Step 2
    build_index(col)            # Step 3
    test_query()                # Step 4

    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes")
    print(f"🌐  Live at: {ZILLIZ_URI}")