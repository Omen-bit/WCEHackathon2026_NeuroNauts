"""
Stage 7 — Submission CSV Generation
Processes all 50 queries from queries.json and writes submission.csv.

Requirements:
  - retrieve.py and generate.py must be in the same folder as this file (app/)
  - queries.json must be at the project root (../queries.json)
  - Milvus running at localhost:19530
  - LM Studio running with nomic-embed + gemma-3-4b on port 1234

Usage:
  python submit.py
  python submit.py --queries ../queries.json --output ../submission.csv --top_k 5
"""

import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── FIX: ensure app/ directory is on sys.path so generate.py can import
# retrieve.py correctly regardless of which directory submit.py is called from.
_THIS_DIR = Path(__file__).parent.absolute()
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from generate import generate


# ─── Config ──────────────────────────────────────────────────────────────────

DEFAULT_QUERIES_PATH  = str(_THIS_DIR.parent / "queries.json")
DEFAULT_OUTPUT_PATH   = str(_THIS_DIR.parent / "submission.csv")
DEFAULT_TOP_K         = 5

# ── FIX: removed 1.0 second sleep — we are using local LM Studio (Gemma),
# not NVIDIA's free-tier API, so rate limiting is unnecessary and only
# wastes ~50 seconds over the full 50-query run.
SLEEP_BETWEEN_QUERIES = 0.0


# ─── CSV columns ─────────────────────────────────────────────────────────────

CSV_COLUMNS = ["ID", "context", "answer", "references"]


# ─── Main ────────────────────────────────────────────────────────────────────

def run(queries_path: str, output_path: str, top_k: int):
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"Loaded {len(queries)} queries from {queries_path}")
    print(f"Output → {output_path}  |  top_k={top_k}\n")

    rows = []
    for i, q in enumerate(queries, 1):
        query_id = q["query_id"]
        question = q["question"]

        print(f"[{i:02d}/{len(queries)}] Q{query_id}: {question}")

        try:
            result     = generate(question, top_k=top_k)
            answer     = result["answer"]
            context    = result["context"]
            references = json.dumps(result["references"])
        except Exception as e:
            print(f"  !! Error: {e}")
            answer     = "Not found in the provided textbook."
            context    = ""
            references = json.dumps({"sections": [], "pages": []})

        rows.append({
            "ID":         query_id,
            "context":    context,
            "answer":     answer,
            "references": references,
        })

        print(f"  → {answer[:80].strip()}{'...' if len(answer) > 80 else ''}")

        if SLEEP_BETWEEN_QUERIES > 0 and i < len(queries):
            time.sleep(SLEEP_BETWEEN_QUERIES)

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ submission.csv written → {output_path}  ({len(rows)} rows)")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate submission.csv from queries.json using the RAG pipeline."
    )
    parser.add_argument(
        "--queries", default=DEFAULT_QUERIES_PATH,
        help=f"Path to queries.json (default: {DEFAULT_QUERIES_PATH})"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_PATH,
        help=f"Path for output CSV (default: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--top_k", type=int, default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve per query (default: {DEFAULT_TOP_K})"
    )
    args = parser.parse_args()
    run(args.queries, args.output, args.top_k)