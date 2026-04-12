"""
Fix double-serialized image_refs in Zilliz Cloud.

The data is stored as a string like:
  '["[", "\\"", "i", "m", "g", "_", "p", "1", "9", "_", "0", ".", "j", "p", "e", "g", "\\"", "]"]'

Fix: json.loads(raw) → list of chars → join → '["img_p19_0.jpeg"]' → json.loads → ['img_p19_0.jpeg']

Run:
    python fix_image_refs.py
"""

import os, json
from pymilvus import connections, Collection
from dotenv import load_dotenv

load_dotenv()

ZILLIZ_URI      = "https://in03-bc51ec1151acfd9.serverless.aws-eu-central-1.cloud.zilliz.com"
ZILLIZ_TOKEN    = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = "psychology2e_chunks"


def extract_filenames(raw: str) -> list:
    """
    raw is a string like: '["[", "\\"", "i", "m", "g", ...]'
    Step 1: json.loads(raw)  → ['[', '"', 'i', 'm', 'g', '_', 'p', '1', '9', ...]
    Step 2: "".join(chars)   → '["img_p19_0.jpeg"]'
    Step 3: json.loads(...)  → ['img_p19_0.jpeg']
    """
    try:
        # Step 1: parse the outer JSON string → list of single chars
        chars = json.loads(raw)
        if not isinstance(chars, list):
            return []

        # Step 2: join all chars back into the original JSON string
        rejoined = "".join(chars)

        # Step 3: parse the rejoined string → actual filename list
        filenames = json.loads(rejoined)
        if isinstance(filenames, list):
            return [f for f in filenames if f]
    except Exception as e:
        print(f"  parse error: {e} | raw[:60]={raw[:60]}")

    return []


def fix_image_refs():
    if not os.path.exists("image_url_map.json"):
        print("❌ image_url_map.json not found!")
        return

    with open("image_url_map.json") as f:
        url_map = json.load(f)
    print(f"✅ Loaded {len(url_map)} Cloudinary URLs\n")

    print("☁️  Connecting to Zilliz Cloud...")
    connections.connect(alias="zilliz", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    col = Collection(COLLECTION_NAME, using="zilliz")
    col.load()

    print("🔍  Fetching all chunks with images...")
    results = col.query(
        expr="has_image_context == true",
        output_fields=[
            "chunk_id", "section_path", "page_numbers",
            "clean_text", "full_text", "token_count",
            "has_image_context", "image_refs", "embedding"
        ],
        limit=2000
    )
    print(f"   Found {len(results)} chunks\n")

    fixed   = 0
    skipped = 0
    batch   = []

    for row in results:
        raw_refs = row.get("image_refs", "[]")

        # Extract original filenames using double json.loads + join
        filenames = extract_filenames(raw_refs)

        if not filenames:
            skipped += 1
            continue

        # Map filenames → Cloudinary URLs
        new_refs = []
        for fname in filenames:
            fname = fname.strip()
            if fname.startswith("http"):
                new_refs.append(fname)
            elif fname in url_map:
                new_refs.append(url_map[fname])
            else:
                print(f"  ⚠️  No URL for: '{fname}'")

        if not new_refs:
            skipped += 1
            continue

        print(f"  chunk {row['chunk_id']}: {filenames} → {[u.split('/')[-1] for u in new_refs]}")

        batch.append({
            "chunk_id":          int(row["chunk_id"]),
            "section_path":      str(row.get("section_path", "")),
            "page_numbers":      str(row.get("page_numbers", "[]")),
            "clean_text":        str(row.get("clean_text", ""))[:8192],
            "full_text":         str(row.get("full_text", ""))[:8192],
            "token_count":       int(row.get("token_count", 0)),
            "has_image_context": bool(row.get("has_image_context", False)),
            "image_refs":        json.dumps(new_refs)[:1024],
            "embedding":         [float(x) for x in row["embedding"]],
        })
        fixed += 1

        if len(batch) == 50:
            col.upsert(batch)
            col.flush()
            print(f"  ✅ Upserted {fixed} chunks so far...")
            batch = []

    if batch:
        col.upsert(batch)
        col.flush()

    connections.disconnect("zilliz")
    print(f"\n✅  Fixed   : {fixed} chunks")
    print(f"   Skipped : {skipped} chunks")
    print(f"\n🎉  Done! Restart Streamlit — images should now display correctly.")


if __name__ == "__main__":
    fix_image_refs()