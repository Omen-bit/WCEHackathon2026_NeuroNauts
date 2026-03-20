from dotenv import load_dotenv
load_dotenv()  # load .env into os.environ before anything else


import json
import re
import tiktoken

MERGED_PATH   = "../output/psychology2e_merged.json"
OUTPUT_PATH   = "../output/psychology2e_chunks.json"
TOKEN_LIMIT   = 400
ENCODING_NAME = "cl100k_base"

enc = tiktoken.get_encoding(ENCODING_NAME)

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

# ── Noise line patterns ───────────────────────────────────────────────────────
NOISE_LINE_RE = re.compile(
    r'^\s*$'
    r'|^\d+\s*$'
    r'|^\d+\s+\w+$'
    r'|Access for free at openstax\.org'
    r'|^\d+\s*[•·]\s*.+',
    re.IGNORECASE
)

def strip_noise_lines(text: str) -> str:
    lines = text.split('\n')
    kept  = [ln for ln in lines if not NOISE_LINE_RE.match(ln)]
    return '\n'.join(kept).strip()

# ── Image description lookup ──────────────────────────────────────────────────
def build_image_desc_lookup(merged_data: dict) -> dict:
    """Return {filename: image_description} for all images in merged JSON."""
    lookup = {}
    for img in merged_data.get("images", []):
        fname = img.get("filename", "")
        desc  = img.get("image_description", "")
        if fname:
            lookup[fname] = desc
    return lookup

NON_ACADEMIC_MARKER = "Non-academic visual element"

def inject_image_descriptions(text: str, image_refs: list, img_desc_lookup: dict) -> str:
    """Append image descriptions to chunk text for any referenced image."""
    additions = []
    for fname in image_refs:
        desc = img_desc_lookup.get(fname, "")
        if desc and NON_ACADEMIC_MARKER not in desc:
            additions.append(f"[Image Description - {fname}]: {desc}")
    if additions:
        text = text + "\n" + "\n".join(additions)
    return text

# ── FIX: has_image_context helper ────────────────────────────────────────────
def compute_has_image_context(all_image_refs: list, final_text: str) -> bool:
    """
    Returns True if:
      - The chunk has direct image_refs (caption-linked images), OR
      - The chunk text contains an injected [Image Description - ...] block.
    
    This catches chunks whose blocks had no caption link but whose text
    was enriched with an image description via inject_image_descriptions().
    """
    if len(all_image_refs) > 0:
        return True
    if "[Image Description -" in final_text:
        return True
    return False

# ── Core chunk builder ────────────────────────────────────────────────────────
def make_chunk(chunk_id, section_path, block_list, img_desc_lookup,
               split_from_block=False, sub_index=None):
    raw_text   = "\n".join(b["text"] for b in block_list)
    clean_text = strip_noise_lines(raw_text)

    page_numbers = sorted(set(b["page_number"] for b in block_list))

    # Collect deduplicated, sorted image refs
    seen           = set()
    all_image_refs = []
    for b in block_list:
        for ref in b.get("image_refs", []):
            if ref not in seen:
                seen.add(ref)
                all_image_refs.append(ref)
    all_image_refs.sort()

    # Inject image descriptions before token counting
    final_text  = inject_image_descriptions(clean_text, all_image_refs, img_desc_lookup)
    token_count = count_tokens(final_text)

    # block_id naming
    raw_block_ids = [b["block_id"] for b in block_list]
    if split_from_block and sub_index is not None:
        block_ids = [f"{bid}_s{sub_index}" for bid in raw_block_ids]
    else:
        block_ids = raw_block_ids

    # ── FIX: use compute_has_image_context instead of bare len() check ────────
    has_img_ctx = compute_has_image_context(all_image_refs, final_text)

    return {
        "chunk_id":          chunk_id,
        "section_path":      section_path,
        "text":              final_text,
        "page_numbers":      page_numbers,
        "token_count":       token_count,
        "image_refs":        all_image_refs,
        "has_image_context": has_img_ctx,   # FIXED
        "block_ids":         block_ids,
        "split_from_block":  split_from_block,
    }


def main():
    with open(MERGED_PATH, encoding="utf-8") as f:
        merged = json.load(f)

    img_desc_lookup = build_image_desc_lookup(merged)
    blocks          = merged["blocks"]

    # ── Primary chunking pass ─────────────────────────────────────────────────
    chunks          = []
    chunk_id        = 0
    current_section = None
    current_blocks  = []
    current_tokens  = 0

    def flush():
        nonlocal chunk_id, current_blocks, current_tokens
        if current_blocks:
            chunks.append(make_chunk(
                chunk_id, current_section, current_blocks, img_desc_lookup,
                split_from_block=False
            ))
            chunk_id += 1
        current_blocks = []
        current_tokens = 0

    for block in blocks:
        text = block.get("text", "")
        if not text or not text.strip():
            continue

        section      = block.get("section_path", "")
        block_tokens = count_tokens(text)

        if section != current_section:
            flush()
            current_section = section

        if block_tokens > TOKEN_LIMIT:
            flush()
            chunks.append(make_chunk(
                chunk_id, section, [block], img_desc_lookup,
                split_from_block=False
            ))
            chunk_id += 1
            continue

        if current_blocks:
            combined_tokens = count_tokens(
                "\n".join(b["text"] for b in current_blocks) + "\n" + text
            )
            if combined_tokens > TOKEN_LIMIT:
                flush()

        current_blocks.append(block)
        current_tokens = count_tokens("\n".join(b["text"] for b in current_blocks))

    flush()

    # ── Post-processing: sentence-split oversized single-block chunks ─────────
    processed = []
    for chunk in chunks:
        if chunk["token_count"] > TOKEN_LIMIT and len(chunk["block_ids"]) == 1:
            raw_text = chunk["text"]
            orig_bid = chunk["block_ids"][0]

            parts     = raw_text.split(". ")
            sentences = [p + ("." if i < len(parts) - 1 else "") for i, p in enumerate(parts)]

            sub_texts       = []
            current_sub     = []
            current_sub_tok = 0
            for sent in sentences:
                sent_tokens = count_tokens(sent)
                if current_sub and current_sub_tok + sent_tokens + 1 > TOKEN_LIMIT:
                    sub_texts.append(" ".join(current_sub))
                    current_sub     = [sent]
                    current_sub_tok = sent_tokens
                else:
                    current_sub.append(sent)
                    current_sub_tok += sent_tokens + (1 if current_sub else 0)
            if current_sub:
                sub_texts.append(" ".join(current_sub))

            for s_idx, sub_text in enumerate(sub_texts):
                sub_text = sub_text.strip()
                sub_text = strip_noise_lines(sub_text)
                # Inject image descriptions only on first sub-chunk
                if s_idx == 0:
                    sub_text = inject_image_descriptions(
                        sub_text, chunk["image_refs"], img_desc_lookup
                    )
                token_count = count_tokens(sub_text)

                # ── FIX: apply compute_has_image_context on sub-chunks too ───
                has_img_ctx = compute_has_image_context(chunk["image_refs"], sub_text)

                processed.append({
                    "chunk_id":          -1,
                    "section_path":      chunk["section_path"],
                    "text":              sub_text,
                    "page_numbers":      chunk["page_numbers"],
                    "token_count":       token_count,
                    "image_refs":        chunk["image_refs"],
                    "has_image_context": has_img_ctx,   # FIXED
                    "block_ids":         [f"{orig_bid}_s{s_idx}"],
                    "split_from_block":  True,
                })
        else:
            processed.append(chunk)

    # Filter noise chunks (< 20 tokens)
    processed = [c for c in processed if c["token_count"] >= 20]

    # Re-assign sequential chunk_ids
    for i, c in enumerate(processed):
        c["chunk_id"] = i
    chunks = processed

    # ── Build metadata ────────────────────────────────────────────────────────
    token_counts         = [c["token_count"] for c in chunks]
    section_chunk_counts = {}
    for c in chunks:
        sp = c["section_path"]
        section_chunk_counts[sp] = section_chunk_counts.get(sp, 0) + 1

    total_chunks        = len(chunks)
    total_tokens        = sum(token_counts)
    average_tokens      = round(total_tokens / total_chunks, 2) if total_chunks else 0
    min_tokens          = min(token_counts) if token_counts else 0
    max_tokens          = max(token_counts) if token_counts else 0
    chunks_with_img_ctx = sum(1 for c in chunks if c["has_image_context"])

    output = {
        "metadata": {
            "total_chunks":              total_chunks,
            "total_tokens":              total_tokens,
            "average_token_count":       average_tokens,
            "min_token_count":           min_tokens,
            "max_token_count":           max_tokens,
            "chunks_with_image_context": chunks_with_img_ctx,
            "section_chunk_counts":      section_chunk_counts,
        },
        "chunks": chunks,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Total chunks produced     : {total_chunks}")
    print(f"Total tokens across all   : {total_tokens}")
    print(f"Average tokens per chunk  : {average_tokens}")
    print(f"Min tokens in a chunk     : {min_tokens}")
    print(f"Max tokens in a chunk     : {max_tokens}")
    print(f"Chunks with image context : {chunks_with_img_ctx}")
    print("\nChunks per section:")
    for sp, count in section_chunk_counts.items():
        print(f"  {sp:<60}: {count}")

    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)

    if len(chunks) > 32:
        c32 = chunks[32]
        print(f"\n--- section_path of chunk 32 ---")
        print(f"  {c32['section_path']}")
    else:
        print(f"\n[Chunk 32 does not exist — only {len(chunks)} chunks total]")

    if len(chunks) > 30:
        c30 = chunks[30]
        print(f"\n--- Full text of chunk 30 ---")
        print(f"(section_path: {c30['section_path']})")
        print(c30['text'])
    else:
        print(f"\n[Chunk 30 does not exist — only {len(chunks)} chunks total]")


if __name__ == "__main__":
    main()