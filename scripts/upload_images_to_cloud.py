"""
Update image_refs in Zilliz Cloud to use Cloudinary URLs.
Images already uploaded — this only updates Zilliz.

Run:
    python upload_images_to_cloud.py
"""

import os, json, cloudinary, cloudinary.uploader
from pymilvus import connections, Collection
from dotenv import load_dotenv

load_dotenv()

# ── Cloudinary config ──────────────────────────────────────────
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key    = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
)

# ── Zilliz config ──────────────────────────────────────────────
ZILLIZ_URI      = "https://in03-bc51ec1151acfd9.serverless.aws-eu-central-1.cloud.zilliz.com"
ZILLIZ_TOKEN    = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = "psychology2e_chunks"
IMAGES_DIR      = "./extracted_images"


# ── STEP 1: Upload images (skip if already done) ───────────────
def upload_all_images():
    # If mapping already exists, load it and skip re-uploading
    if os.path.exists("image_url_map.json"):
        print("💾  Found existing image_url_map.json — skipping upload step")
        with open("image_url_map.json") as f:
            return json.load(f)

    image_url_map = {}
    files = os.listdir(IMAGES_DIR)
    print(f"📸  Found {len(files)} images — uploading to Cloudinary...")

    for i, filename in enumerate(files, 1):
        filepath  = os.path.join(IMAGES_DIR, filename)
        public_id = f"psychology2e/{os.path.splitext(filename)[0]}"
        try:
            result = cloudinary.uploader.upload(
                filepath, public_id=public_id,
                overwrite=True, resource_type="image"
            )
            image_url_map[filename] = result["secure_url"]
            print(f"  [{i}/{len(files)}] ✅ {filename}")
        except Exception as e:
            print(f"  [{i}/{len(files)}] ❌ {filename}: {e}")

    with open("image_url_map.json", "w") as f:
        json.dump(image_url_map, f, indent=2)

    print(f"✅  Uploaded {len(image_url_map)} images\n")
    return image_url_map


# ── STEP 2: Update image_refs in Zilliz ───────────────────────
def update_zilliz_image_refs(image_url_map):
    print("☁️   Connecting to Zilliz Cloud...")
    connections.connect(alias="zilliz", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    col = Collection(COLLECTION_NAME, using="zilliz")
    col.load()

    print("🔍  Fetching chunks with images (including embeddings)...")
    results = col.query(
        expr="has_image_context == true",
        output_fields=[
            "chunk_id", "section_path", "page_numbers",
            "clean_text", "full_text", "token_count",
            "has_image_context", "image_refs", "embedding"
        ],
        limit=2000
    )
    print(f"   Found {len(results)} chunks with images\n")

    updated = 0
    skipped = 0
    batch   = []

    for row in results:
        raw_refs = row.get("image_refs", "[]")
        try:
            old_refs = json.loads(raw_refs)
        except:
            old_refs = []

        if not old_refs:
            skipped += 1
            continue

        # Replace local filenames with Cloudinary URLs
        new_refs = [image_url_map.get(f, f) for f in old_refs]

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
        updated += 1

        # Upsert in batches of 50
        if len(batch) == 50:
            col.upsert(batch)
            col.flush()
            print(f"  Upserted {updated} chunks so far...")
            batch = []

    # Upsert remaining
    if batch:
        col.upsert(batch)
        col.flush()

    connections.disconnect("zilliz")
    print(f"\n✅  Updated {updated} chunks in Zilliz Cloud")
    print(f"   Skipped {skipped} chunks (no image refs)")


# ── MAIN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    image_url_map = upload_all_images()       # Step 1 (skips if already done)
    update_zilliz_image_refs(image_url_map)   # Step 2
    print("\n🎉  Done! image_refs in Zilliz now point to Cloudinary URLs.")