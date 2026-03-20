from dotenv import load_dotenv
load_dotenv()  # load .env into os.environ before anything else

import json
import os
import re

# ── patterns for Fix 1: footer/header noise ───────────────────────────────────
NOISE_PATTERNS = [
    re.compile(r'Access for free at openstax\.org', re.IGNORECASE),
]
INLINE_BULLET = re.compile(r'^\d+\s*[•·]\s*.+')   # "8 • Introduction to Psychology"
STANDALONE_DIGIT = re.compile(r'^\s*\d+\s*$')      # lone page numbers / chapter nums


def clean_block_text(text: str) -> str:
    """Fix 1 + Fix 2: strip footer noise lines and normalise whitespace."""
    lines = text.split('\n')
    kept = []
    for line in lines:
        # Drop "Access for free at openstax.org" footer
        if any(p.search(line) for p in NOISE_PATTERNS):
            continue
        # Drop inline bullet lines like "8 • Introduction to Psychology"
        if INLINE_BULLET.match(line):
            continue
        # Drop standalone digit lines (page numbers floating alone)
        if STANDALONE_DIGIT.match(line):
            continue
        kept.append(line)

    cleaned = '\n'.join(kept).strip()
    # Fix 2: collapse runs of 2+ spaces into one (justified-text artifact)
    cleaned = re.sub(r'  +', ' ', cleaned)
    return cleaned


def generate_markdown(blocks, images_dict):
    lines = []

    for block in blocks:
        btype = block.get('block_type', 'paragraph')
        text = block.get('text', '')

        if btype == 'heading':
            level = text.count('>') if text else 1
            prefix = '#' * min(level + 1, 6)
            lines.append(f"\n{prefix} {text}")
        elif btype == 'list_item':
            lines.append(f"- {text}")
        elif btype == 'caption':
            for img_fname in block.get('image_refs', []):
                lines.append(f"\n[Figure: {img_fname}]")
            lines.append(f"*{text}*")
            for img_fname in block.get('image_refs', []):
                if img_fname in images_dict:
                    desc = images_dict[img_fname].get('image_description', '')
                    if desc:
                        desc_formatted = desc.replace('\n', '\n> ')
                        lines.append(f"\n> **[Image Description]** {desc_formatted}\n")
        elif btype == 'figure':
            if text.strip():
                lines.append(f"\n[Figure Placeholder: {text}]\n")
            for img_fname in block.get('image_refs', []):
                lines.append(f"\n[Figure: {img_fname}]\n")
                if img_fname in images_dict:
                    desc = images_dict[img_fname].get('image_description', '')
                    if desc:
                        desc_formatted = desc.replace('\n', '\n> ')
                        lines.append(f"\n> **[Image Description]** {desc_formatted}\n")
        else:
            lines.append(text)

        lines.append("")  # paragraph break

    return "\n".join(lines)


def main():
    # Discover the PDF filename and page range dynamically so the script works
    # regardless of which PDF the pipeline was invoked with.
    pdf_path   = os.environ.get('PIPELINE_PDF_PATH', '../data/Psychology2e_WEB-1-100.pdf')
    pdf_name   = os.path.basename(pdf_path)

    # Determine the page range from the files that ingest.py actually wrote.
    page_files = sorted(
        f for f in os.listdir('../page_outputs')
        if f.startswith('page_') and f.endswith('.json')
    )
    if page_files:
        # filenames are page_001.json … page_NNN.json
        first_page = int(page_files[0].split('_')[1].split('.')[0])
        last_page  = int(page_files[-1].split('_')[1].split('.')[0])
    else:
        first_page, last_page = 1, 16   # safe fallback

    merged = {
        "title": "Psychology 2e - OpenStax",
        "source_file": pdf_name,
        "processed_pages": last_page - first_page + 1,
        "page_range": [first_page, last_page],
        "section_index": {},
        "blocks": [],
        "images": []
    }

    pymupdf_layer = []
    images_dict = {}

    # 1. Load pages and assemble master layout
    for p in range(first_page, last_page + 1):
        filepath = f"../page_outputs/page_{p:03d}.json"
        if not os.path.exists(filepath):
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Collect docling blocks
        for block in data.get("blocks", []):
            # Rename legacy field
            if "image_filenames" in block:
                block["image_refs"] = block.pop("image_filenames")

            # Fix clean spaced small-caps headings (verbatim match, no cleaning)
            t = block.get("text", "")
            if t == "O PEN S TAX":
                block["text"] = "OpenStax"
            elif t == "R ICE U NIVERSITY":
                block["text"] = "Rice University"
            elif t == "P HILANTHROPIC S UPPORT":
                block["text"] = "Philanthropic Support"
            else:
                # Fix 1 + Fix 2: strip noise lines and normalise spacing
                block["text"] = clean_block_text(block.get("text", ""))

            merged["blocks"].append(block)

            # Update section index
            spath = block["section_path"]
            if spath not in merged["section_index"]:
                merged["section_index"][spath] = []
            if p not in merged["section_index"][spath]:
                merged["section_index"][spath].append(p)

        # Collect docling images
        for img in data.get("images", []):
            merged["images"].append(img)
            images_dict[img["filename"]] = img

        # Collect pure PyMuPDF geometry spatial representation
        pymupdf_layer.append({
            "page_number": p,
            "text_blocks": data.get("pymupdf_blocks", []),
            "images": [
                {
                    "filename": img["filename"],
                    "bbox": img.get("bbox")
                } for img in data.get("images", [])
            ]
        })

    # Normalize sorting
    for spath in merged["section_index"]:
        merged["section_index"][spath].sort()

    merged["blocks"].sort(key=lambda x: (x["page_number"], int(x["block_id"].split("_b")[1])))

    # Fix 3: Populate caption field on image objects from linked caption blocks
    caption_lookup = {}  # filename -> caption text
    for block in merged["blocks"]:
        if block.get("block_type") == "caption":
            for fname in block.get("image_refs", []):
                caption_lookup[fname] = block.get("text", "")
    for img in merged["images"]:
        fname = img.get("filename", "")
        if fname in caption_lookup:
            img["caption"] = caption_lookup[fname]

    # Fix 4: Merge cross-page broken sentences.
    # Pairs where block[i] text ends without a full stop and block[i+1] starts
    # with a lowercase letter or closing bracket are merged into one block.
    # Also explicitly handles p15_b9 / p16_b0 (Nash narrative split).
    kept_blocks = []
    skip_next = False
    for i, block in enumerate(merged["blocks"]):
        if skip_next:
            skip_next = False
            continue

        if i + 1 < len(merged["blocks"]):
            next_block = merged["blocks"][i + 1]
            curr_text = block.get("text", "").rstrip()
            next_text = next_block.get("text", "").lstrip()

            is_explicit_pair = (
                block["block_id"] == "p15_b9" and next_block["block_id"] == "p16_b0"
            )
            ends_without_stop = curr_text and curr_text[-1] not in '.!?'
            starts_lowercase_or_bracket = next_text and (
                next_text[0].islower() or next_text[0] in ')],;'
            )

            if is_explicit_pair or (ends_without_stop and starts_lowercase_or_bracket):
                merged_text = curr_text + " " + next_text
                merged_text = re.sub(r'  +', ' ', merged_text)
                block = dict(block)  # shallow copy
                block["text"] = merged_text
                # Merge image_refs from both blocks
                combined_refs = list(block.get("image_refs", []))
                for ref in next_block.get("image_refs", []):
                    if ref not in combined_refs:
                        combined_refs.append(ref)
                block["image_refs"] = combined_refs
                skip_next = True

        kept_blocks.append(block)

    merged["blocks"] = kept_blocks

    os.makedirs("../output", exist_ok=True)

    # Output 1: Master JSON Layout
    with open("../output/psychology2e_merged.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    # Output 2: PyMuPDF Spatial Format
    with open("../output/pymupdf_output.json", "w", encoding="utf-8") as f:
        json.dump(pymupdf_layer, f, indent=2)

    # Output 3: Docling Markdown
    final_images_dict = {img["filename"]: img for img in merged["images"]}
    markdown_out = generate_markdown(merged["blocks"], final_images_dict)
    with open("../output/docling_output.md", "w", encoding="utf-8") as f:
        f.write(markdown_out)

    print("Files successfully written to output/")


if __name__ == '__main__':
    main()
