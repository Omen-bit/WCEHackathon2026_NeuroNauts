"""
run_pipeline.py
===============
End-to-end pipeline runner.

Usage:
    python run_pipeline.py --pdf Psychology2e_WEB-1-100.pdf
    python run_pipeline.py --pdf Psychology2e_WEB_compressed.pdf
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── ANSI helpers ──────────────────────────────────────────────────────────────
BOLD  = "\033[1m"
GREEN = "\033[32m"
CYAN  = "\033[36m"
RESET = "\033[0m"

BAR = "=" * 44

APP_DIR = Path(__file__).parent.absolute()


def banner(title: str) -> None:
    print(f"\n{BAR}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{BAR}")


def run_stage(label: str, script: str, env: dict) -> float:
    """
    Print stage header, run *script* as a subprocess that inherits the
    current terminal (so its stdout/stderr stream live to the console),
    measure elapsed time, print timing, and return elapsed seconds.
    Raises SystemExit on non-zero return code.
    """
    banner(label)

    t0 = time.monotonic()
    result = subprocess.run(
        [sys.executable, script],
        env=env,
        cwd=str(APP_DIR)
    )
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        print(f"\n{BOLD}ERROR:{RESET} {script} exited with code {result.returncode}",
              file=sys.stderr)
        sys.exit(result.returncode)

    print(f"\n{GREEN}✓{RESET} {label} completed in {elapsed:.1f}s")
    return elapsed


def delete_debug_artifacts() -> None:
    """Remove debug/temporary files that should not accumulate between runs."""
    targets = [
        APP_DIR.parent / "output" / "chunk_summary.txt",
        APP_DIR.parent / "output" / "chunk_verify.txt",
        APP_DIR.parent / "output" / "final_answers.txt",
        APP_DIR.parent / "output" / "ingest_debug.txt",
        APP_DIR / "tmp_query.txt",
        APP_DIR / "tmp_si.txt",
    ]
    for path in targets:
        try:
            os.remove(path)
            print(f"  Deleted: {path}")
        except FileNotFoundError:
            pass  # already gone, that's fine


def collect_stats(pdf_path: str) -> dict:
    """
    Read the generated output files and gather numbers for the final summary.
    Returns a dict with keys: pages, blocks, images, chunks.
    """
    stats = {"pages": 0, "blocks": 0, "images": 0, "chunks": 0}

    merged_path = APP_DIR.parent / "output" / "psychology2e_merged.json"
    if merged_path.exists():
        try:
            with merged_path.open(encoding="utf-8") as f:
                merged = json.load(f)
            stats["pages"]  = merged.get("processed_pages", 0)
            stats["blocks"] = len(merged.get("blocks", []))
            stats["images"] = len(merged.get("images", []))
        except Exception:
            pass

    chunks_path = APP_DIR.parent / "output" / "psychology2e_chunks.json"
    if chunks_path.exists():
        try:
            with chunks_path.open(encoding="utf-8") as f:
                chunks_data = json.load(f)
            stats["chunks"] = chunks_data.get("metadata", {}).get("total_chunks", 0)
        except Exception:
            pass

    return stats


def print_final_summary(pdf_path: str, stats: dict, total_elapsed: float) -> None:
    print(f"\n{BAR}")
    print(f"{BOLD}PIPELINE COMPLETE{RESET}")
    print(f"{BAR}")
    print(f"Input PDF        : {pdf_path}")
    print(f"Pages processed  : {stats['pages']}")
    print(f"Blocks extracted : {stats['blocks']}")
    print(f"Images extracted : {stats['images']}")
    print(f"Chunks produced  : {stats['chunks']}")
    print(f"Total time       : {total_elapsed:.1f} seconds")
    print()
    print("Output files:")
    print("  ../output/psychology2e_merged.json")
    print("  ../output/docling_output.md")
    print("  ../output/pymupdf_output.json")
    print("  ../output/psychology2e_chunks.json")
    print("  ../output/milvus_config.json")
    print("  ../output/bm25_index.pkl")
    print(f"{BAR}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full PDF ingestion → merge → chunking pipeline."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        metavar="PDF_PATH",
        help="Path to the input PDF file.",
    )
    args = parser.parse_args()

    pdf_path = str(Path(args.pdf).absolute())

    # Validate PDF exists before kicking off potentially long steps
    if not Path(pdf_path).exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # ── Pre-run cleanup ───────────────────────────────────────────────────────
    print(f"\n{CYAN}Cleaning up debug artifacts …{RESET}")
    delete_debug_artifacts()

    # ── Build subprocess environment ──────────────────────────────────────────
    # Pass the PDF path to child scripts via an environment variable so that
    # ingest.py and merge.py can pick the correct file without hardcoding.
    env = os.environ.copy()
    env["PIPELINE_PDF_PATH"] = pdf_path

    pipeline_start = time.monotonic()

    # ── Stage 1A ─────────────────────────────────────────────────────────────
    run_stage("STAGE 1A — PDF Ingestion (Docling + PyMuPDF)", "ingest.py", env)

    # ── Stage 1B ─────────────────────────────────────────────────────────────
    run_stage("STAGE 1B — Merge per-page JSONs", "merge.py", env)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    run_stage("STAGE 2  — Chunking", "chunk.py", env)

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    run_stage("STAGE 3 — Embed and Store (Milvus)", "embed_and_store.py", env)

    # ── Stage 4 ──────────────────────────────────────────────────────────────
    run_stage("STAGE 4 — BM25 Indexing", "build_bm25.py", env)

    total_elapsed = time.monotonic() - pipeline_start

    # ── Final summary ─────────────────────────────────────────────────────────
    stats = collect_stats(pdf_path)
    print_final_summary(pdf_path, stats, total_elapsed)


if __name__ == "__main__":
    main()
