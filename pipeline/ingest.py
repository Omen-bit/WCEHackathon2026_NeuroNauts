from dotenv import load_dotenv
load_dotenv()

import os
import gc
import json
import re
import fitz
import base64
import requests
import time
import ctypes
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import DocItemLabel

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

# ── Page geometry constants ───────────────────────────────────────────────────
HEADER_CUTOFF        = 50
FOOTER_CUTOFF        = 742
BIG_IMAGE_RATIO      = 0.70
LARGE_FONT_THRESHOLD = 16.0

# ── Heading classification regexes ───────────────────────────────────────────
CHAPTER_HEADING_RE = re.compile(
    r'^chapter\s+\d+'
    r'|^\d+\s+[A-Z]'
    r'|^[IVXLCDM]+\s+[A-Z]',
    re.IGNORECASE
)
SECTION_NUM_RE = re.compile(r'^\d+\.\d+')
TOC_HEADING_RE = re.compile(r'^(contents|table of contents)$', re.IGNORECASE)
TOC_ENTRY_RE   = re.compile(r'.+\s+\d+\s*$')


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def call_groq_vision(api_key, base64_image, mime_type, prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                {"type": "text", "text": prompt}
            ]
        }]
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content'].strip()


def build_groq_prompt(page_num, section_path, caption_text):
    """
    Build a rich pipeline-agnostic image description prompt.
    Uses section_path and caption — both always available from the pipeline.
    No hardcoded content — works for any PDF.
    """
    context_hint = (
        f"The figure caption reads: '{caption_text}'. "
        if caption_text and caption_text.strip()
        else ""
    )
    return (
        f"This image appears on page {page_num} of an academic textbook "
        f"in the section '{section_path}'. {context_hint}"
        f"Provide a detailed description of this image covering all of the following: "
        f"1) What type of visual it is — photograph, diagram, chart, graph, illustration, collage, or screenshot. "
        f"2) What people, objects, scenes, or concepts are depicted and how they are arranged. "
        f"3) The mood, composition, or visual emphasis of the image. "
        f"4) How this image relates to or illustrates the academic concepts in this section. "
        f"5) Any visible text, labels, axis titles, figure numbers, legends, or annotations. "
        f"Write 5 to 8 complete sentences. Return only the description, nothing else."
    )


# ── Dynamic section path tracker ─────────────────────────────────────────────
class SectionTracker:
    """
    Tracks section path from structural signals only — no hardcoded content.

    States:
        pre_chapter  - before any chapter heading       → Preamble
        in_toc       - inside Table of Contents         → Preamble > Table of Contents
        in_chapter   - inside a chapter                 → Chapter > Section

    _toc_seen: set True the first time a TOC heading is detected.
    Font-size promotion is ONLY applied after _toc_seen is True.
    This prevents promotional preamble text (e.g. OpenStax slogans with
    large fonts) from being mistakenly promoted to chapter headings.
    """

    def __init__(self):
        self.state         = "pre_chapter"
        self.preamble_sub  = "OpenStax Credits and Promotional"
        self.chapter_parts = []
        self.section_parts = []
        self._toc_seen     = False   # KEY: guards font-size promotion

    def is_chapter_heading(self, text):
        return bool(CHAPTER_HEADING_RE.match(text))

    def is_section_heading(self, text):
        return bool(SECTION_NUM_RE.match(text))

    def is_toc_heading(self, text):
        return bool(TOC_HEADING_RE.match(text.strip()))

    def is_toc_entry(self, text):
        return bool(TOC_ENTRY_RE.match(text.strip()))

    def toc_has_been_seen(self):
        return self._toc_seen

    def update(self, heading_text, font_promoted=False):
        """
        Update tracker state from a heading.

        font_promoted=True means PyMuPDF detected a large-font heading.
        This is only honoured AFTER the TOC has been seen (_toc_seen=True)
        to prevent false promotion of preamble decorative text.
        """
        _SMALLCAPS_MAP = {
            "O PEN S TAX":             "OpenStax",
            "R ICE U NIVERSITY":       "Rice University",
            "P HILANTHROPIC S UPPORT": "Philanthropic Support",
        }
        raw = heading_text.strip()
        t   = _SMALLCAPS_MAP.get(raw, raw)

        # TOC detection — set _toc_seen flag permanently
        if self.is_toc_heading(t):
            self.state     = "in_toc"
            self._toc_seen = True   # CRITICAL: enables font promotion from now on
            return

        if self.state == "in_toc":
            if self.is_toc_entry(t):
                return
            self.state = "pre_chapter"

        # Chapter heading — regex match OR font-size promotion (only after TOC)
        if self.is_chapter_heading(t) or (font_promoted and self._toc_seen):
            self.state         = "in_chapter"
            self.chapter_parts = [t]
            self.section_parts = ["Introduction"]
            return

        # Section number
        if self.is_section_heading(t):
            if self.state == "in_chapter":
                self.section_parts = [t]
            else:
                self.state         = "in_chapter"
                self.chapter_parts = [t]
                self.section_parts = ["Introduction"]
            return

        # Any other heading while in pre_chapter = preamble sub-label
        if self.state == "pre_chapter":
            self.preamble_sub = t

    def current_path(self):
        if self.state == "pre_chapter":
            return ["Preamble", self.preamble_sub]
        if self.state == "in_toc":
            return ["Preamble", "Table of Contents"]
        return self.chapter_parts + self.section_parts

    def is_preamble(self):
        return self.state in ("pre_chapter", "in_toc")


def extract_large_font_headings(page):
    """
    Collect large-font text from a page, grouping spans on the same line.
    Returns [(text, size)] sorted by font size descending.
    Works on any PDF regardless of content.
    """
    found_lines = {} # y_coord -> [(text, size)]
    # We maintain the natural PDF block/line ordering for grouping
    raw = page.get_text("dict")["blocks"]

    for b in raw:
        if "lines" not in b:
            continue
        x0_b, y0_b, x1_b, y1_b = b["bbox"]
        if y1_b < HEADER_CUTOFF or y0_b > FOOTER_CUTOFF:
            continue

        for line in b["lines"]:
            # Use the middle of the line's bbox as y-coordinate for grouping
            ly0, ly1 = line["bbox"][1], line["bbox"][3]
            mid_y = (ly0 + ly1) / 2
            
            for span in line["spans"]:
                size = span.get("size", 0)
                text = span["text"].strip()
                if size >= LARGE_FONT_THRESHOLD and text:
                    # Find if there's an existing group within 10 points
                    # (Observed 8.87pt difference on page 15 between different spans on same logical line)
                    target_y = None
                    for y in found_lines.keys():
                        if abs(y - mid_y) < 10:
                            target_y = y
                            break
                    if target_y is None:
                        target_y = mid_y
                        found_lines[target_y] = []
                    
                    found_lines[target_y].append((text, size))
    
    # Process grouped lines into combined headings
    final_headings = []
    # Ordering is preserved within the grouped list to match PDF reading order
    for y in found_lines:
        texts = [s[0] for s in found_lines[y]]
        combined_text = " ".join(texts).strip()
        max_size = max(s[1] for s in found_lines[y])
        # Filter by length to avoid stray punctuation or artifacts
        if len(combined_text) > 3:
            final_headings.append((combined_text, max_size))
            
    final_headings.sort(key=lambda x: x[1], reverse=True)
    return final_headings


def make_converter():
    """Fresh Docling converter per page — prevents RAM accumulation."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr             = False
    pipeline_options.do_table_structure = True
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )


def process_page_with_docling(doc, page_num, tmp_dir):
    """
    Write one page to temp PDF, run Docling, collect items,
    delete temp file, force GC. Returns [(item, level, page_num)].
    """
    tmp_path = os.path.join(tmp_dir, f"tmp_page_{page_num:03d}.pdf")

    single = fitz.open()
    single.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)
    single.save(tmp_path)
    single.close()

    items = []
    try:
        converter   = make_converter()
        result      = converter.convert(tmp_path)
        docling_doc = result.document

        for item, level in docling_doc.iterate_items():
            if hasattr(item, 'prov') and item.prov:
                item.prov[0].page_no = page_num
            items.append((item, level, page_num))

        del result
        del docling_doc
        del converter
        gc.collect()
        try:
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
        except Exception:
            pass

    except Exception as e:
        print(f"  Docling failed on page {page_num}: {e}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return items


def main():
    os.makedirs('../extracted_images', exist_ok=True)
    os.makedirs('../page_outputs',     exist_ok=True)
    os.makedirs('../output',           exist_ok=True)
    tmp_dir = 'tmp_pages'
    os.makedirs(tmp_dir, exist_ok=True)

    pdf_path     = os.environ.get('PIPELINE_PDF_PATH', '../data/Psychology2e_WEB.pdf')
    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key:
        print("Warning: GROQ_API_KEY not set — Groq Vision calls will be skipped.")

    # ── Pass 1: PyMuPDF — text blocks, images, large-font heading scan ───────
    print("Running PyMuPDF analysis and image extraction...")
    doc         = fitz.open(pdf_path)
    total_pages = len(doc)

    ocr_flags          = {}
    pymupdf_blocks     = {p: [] for p in range(1, total_pages + 1)}
    pymupdf_images     = []
    large_font_by_page = {}

    for i, page in enumerate(doc):
        page_num = i + 1
        if page_num > total_pages:
            break

        full_text = page.get_text()
        ocr_flags[page_num] = len(full_text.strip()) < 30

        raw_blocks = page.get_text("dict")["blocks"]
        for b in raw_blocks:
            if "lines" not in b:
                continue
            x0, y0, x1, y1 = b["bbox"]
            if y1 < HEADER_CUTOFF or y0 > FOOTER_CUTOFF:
                continue
            text_content = " ".join(
                span["text"]
                for line in b["lines"]
                for span in line["spans"]
            ).strip()
            if text_content:
                pymupdf_blocks[page_num].append({
                    "bbox":       list(b["bbox"]),
                    "text":       text_content,
                    "block_type": "text"
                })

        large_font_by_page[page_num] = extract_large_font_headings(page)

        for img_index, img in enumerate(page.get_images(full=True)):
            xref  = img[0]
            rects = page.get_image_rects(xref)
            bbox  = [rects[0].x0, rects[0].y0, rects[0].x1, rects[0].y1] if rects else None

            base_img  = doc.extract_image(xref)
            img_bytes = base_img["image"]
            ext       = "jpeg" if base_img["ext"] == "jpg" else base_img["ext"]
            fname     = f"img_p{page_num}_{img_index}.{ext}"
            img_path  = os.path.join("../extracted_images", fname)

            with open(img_path, "wb") as f:
                f.write(img_bytes)

            pymupdf_images.append({
                "page_number": page_num,
                "filename":    fname,
                "index":       img_index,
                "bbox":        bbox,
                "img_path":    img_path,
                "ext":         ext,
                "image_description": "",
            })

    # ── Pass 2: Docling — one page at a time ─────────────────────────────────
    print("Running Docling extraction (one page at a time)...")

    tracker            = SectionTracker()
    blocks_by_page     = {p: [] for p in range(1, total_pages + 1)}
    images_meta_by_pg  = {p: [] for p in range(1, total_pages + 1)}
    block_counters     = {p: 0  for p in range(1, total_pages + 1)}
    page_section_cache = {}
    last_path          = ["Preamble", "OpenStax Credits and Promotional"]

    for img_info in pymupdf_images:
        pnum = img_info["page_number"]
        if pnum <= total_pages:
            images_meta_by_pg[pnum].append({
                "image_id":          f"img_p{pnum}_{img_info['index']}",
                "filename":          img_info["filename"],
                "page_number":       pnum,
                "section_path":      "",
                "bbox":              img_info["bbox"],
                "caption":           None,
                "image_description": "",
                "_unmatched":        True,
                "_img_path":         img_info["img_path"],
                "_ext":              img_info["ext"],
            })

    for page_num in range(1, total_pages + 1):
        print(f"  [Page {page_num:2d}/{total_pages}] Running Docling...")

        if page_num not in page_section_cache:
            page_section_cache[page_num] = list(last_path)

        # ── Pre-seed tracker from large-font headings ─────────────────────────
        # Only applied AFTER TOC has been seen (_toc_seen=True).
        # This prevents promotional preamble text from firing chapter promotion.
        # After the TOC, large-font text = chapter title structural signal.
        if tracker.toc_has_been_seen():
            for heading_text, font_size in large_font_by_page.get(page_num, []):
                if tracker.is_toc_entry(heading_text):
                    continue
                if len(heading_text.strip()) <= 3:
                    continue
                if tracker.is_toc_heading(heading_text):
                    tracker.update(heading_text)
                    page_section_cache[page_num] = tracker.current_path()
                    last_path = tracker.current_path()
                    break
                if tracker.is_preamble():
                    tracker.update(heading_text, font_promoted=True)
                    page_section_cache[page_num] = tracker.current_path()
                    last_path = tracker.current_path()
                    break
                if tracker.is_chapter_heading(heading_text):
                    tracker.update(heading_text, font_promoted=True)
                    page_section_cache[page_num] = tracker.current_path()
                    last_path = tracker.current_path()
                    break

        page_items = process_page_with_docling(doc, page_num, tmp_dir)
        time.sleep(0.5)

        for item, level, pnum in page_items:

            text_content = ""
            block_type   = "paragraph"

            if item.label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
                block_type   = "heading"
                text_content = getattr(item, 'text', '')
                tracker.update(text_content)
                page_section_cache[page_num] = tracker.current_path()
                last_path = tracker.current_path()

            elif item.label in (DocItemLabel.PARAGRAPH, DocItemLabel.TEXT):
                block_type   = "paragraph"
                text_content = getattr(item, 'text', '')

            elif item.label == DocItemLabel.LIST_ITEM:
                block_type   = "list_item"
                text_content = getattr(item, 'text', '')

            elif item.label == DocItemLabel.CAPTION:
                block_type   = "caption"
                text_content = getattr(item, 'text', '')

            elif item.label == DocItemLabel.TABLE:
                block_type   = "table"
                text_content = (
                    getattr(item, 'export_to_markdown', lambda: '')()
                    or getattr(item, 'text', '')
                )

            elif item.label == DocItemLabel.PICTURE:
                block_type   = "figure"
                text_content = getattr(item, 'text', '')

            else:
                block_type   = "paragraph"
                text_content = getattr(item, 'text', '')

            if block_type == "paragraph" and text_content.strip().upper().startswith("FIGURE"):
                block_type = "caption"

            if not text_content and block_type not in ("table", "figure"):
                continue

            section_path_str = " > ".join(tracker.current_path())

            linked_images = []
            if block_type == "caption":
                for img_meta in images_meta_by_pg.get(page_num, []):
                    if img_meta.get("_unmatched", True):
                        img_meta["caption"]    = text_content
                        img_meta["_unmatched"] = False
                        linked_images.append(img_meta["filename"])
                        break

            block_id = f"p{page_num}_b{block_counters[page_num]}"
            block_counters[page_num] += 1

            blocks_by_page[page_num].append({
                "block_id":     block_id,
                "page_number":  page_num,
                "section_path": section_path_str,
                "text":         text_content,
                "block_type":   block_type,
                "ocr":          ocr_flags.get(page_num, False),
                "image_refs":   linked_images,
            })

        page_section_cache[page_num] = tracker.current_path()
        last_path = tracker.current_path()

        blocks_found = len(blocks_by_page[page_num])
        section_now  = " > ".join(tracker.current_path())
        print(f"  [Page {page_num:2d}/{total_pages}] {blocks_found} blocks — {section_now}")

        if page_num % 10 == 0:
            gc.collect()
            try:
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
            except Exception:
                pass
            print(f"  [Memory recovery at page {page_num}]")

    for p in range(1, total_pages + 1):
        if not blocks_by_page[p]:
            print(f"  Page {p} has no Docling blocks — using PyMuPDF fallback.")
            page_text  = doc[p - 1].get_text().strip()
            path_parts = page_section_cache.get(p, last_path)
            path_str   = " > ".join(path_parts)
            blocks_by_page[p].append({
                "block_id":     f"p{p}_b0",
                "page_number":  p,
                "section_path": path_str,
                "text":         page_text,
                "block_type":   "paragraph",
                "ocr":          ocr_flags.get(p, False),
                "image_refs":   [],
            })
            page_section_cache[p] = path_parts

    # ── Rule 5: Cross-page broken sentence merge ──────────────────────────────
    all_blocks_flat = []
    for p in range(1, total_pages + 1):
        all_blocks_flat.extend(blocks_by_page[p])

    merged_flat = []
    skip_next   = False
    for i, blk in enumerate(all_blocks_flat):
        if skip_next:
            skip_next = False
            continue
        if i + 1 < len(all_blocks_flat):
            nxt       = all_blocks_flat[i + 1]
            curr_text = blk.get("text", "").rstrip()
            next_text = nxt.get("text", "").lstrip()
            same_sect = blk["section_path"] == nxt["section_path"]
            ends_open = curr_text and curr_text[-1] not in '.!?)'
            starts_lc = next_text and (next_text[0].islower() or next_text[0] in ')],;')
            if same_sect and ends_open and starts_lc:
                blk         = dict(blk)
                blk["text"] = re.sub(r'  +', ' ', curr_text + " " + next_text)
                combined    = list(blk.get("image_refs", []))
                for ref in nxt.get("image_refs", []):
                    if ref not in combined:
                        combined.append(ref)
                blk["image_refs"] = combined
                skip_next = True
        merged_flat.append(blk)

    for p in range(1, total_pages + 1):
        blocks_by_page[p] = [b for b in merged_flat if b["page_number"] == p]

    # ── Patch image section_path + Groq calls ────────────────────────────────
    print("Running Groq Vision API calls for academic images...")
    groq_call_count = 0
    for p in range(1, total_pages + 1):
        path_parts       = page_section_cache.get(p, last_path)
        path_str         = " > ".join(path_parts)
        is_preamble_page = path_parts[0] == "Preamble" if path_parts else True

        page_w          = doc[p - 1].rect.width
        page_h          = doc[p - 1].rect.height
        total_page_area = page_w * page_h

        for img_meta in images_meta_by_pg[p]:
            img_meta["section_path"] = path_str

            bbox      = img_meta.get("bbox")
            big_image = False
            if bbox:
                iw        = abs(bbox[2] - bbox[0])
                ih        = abs(bbox[3] - bbox[1])
                big_image = (iw * ih) / total_page_area > BIG_IMAGE_RATIO

            img_path = img_meta.pop("_img_path", None)
            ext      = img_meta.pop("_ext", "jpeg")

            if is_preamble_page:
                img_meta["image_description"] = "Non-academic visual element from the textbook preamble."
            elif big_image:
                img_meta["image_description"] = "Full-page or near-full-page decorative image; Groq skipped due to size threshold."
            elif groq_api_key and img_path and os.path.exists(img_path):
                try:
                    b64          = encode_image(img_path)
                    mime         = "image/jpeg" if ext == "jpeg" else f"image/{ext}"
                    caption_text = img_meta.get("caption", "") or ""
                    prompt       = build_groq_prompt(p, path_str, caption_text)
                    print(f"  Calling Groq Vision for {img_meta['filename']}...")
                    img_meta["image_description"] = call_groq_vision(groq_api_key, b64, mime, prompt)
                    groq_call_count += 1
                    time.sleep(2)
                    if groq_call_count % 25 == 0:
                        print(f"  Groq rate limit pause: {groq_call_count} calls made, waiting 60 seconds...")
                        time.sleep(60)
                except Exception as e:
                    print(f"  Groq error for {img_meta['filename']}: {e}")
                    img_meta["image_description"] = f"Groq API error: {e}"
            elif not groq_api_key:
                img_meta["image_description"] = "Error: GROQ_API_KEY not set."

            img_meta.pop("_unmatched", None)

    # ── Write per-page JSON files ─────────────────────────────────────────────
    print("Writing per-page JSON files...")
    for p in range(1, total_pages + 1):
        out_data = {
            "page_number":    p,
            "ocr_flag":       ocr_flags.get(p, False),
            "blocks":         blocks_by_page[p],
            "images":         images_meta_by_pg[p],
            "pymupdf_blocks": pymupdf_blocks[p],
        }
        with open(f"../page_outputs/page_{p:03d}.json", "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)

    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print(f"Done. Processed {total_pages} pages.")


if __name__ == '__main__':
    main()