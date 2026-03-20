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
    banner(label)
    t0 = time.monotonic()
    
    # Use current interpreter (the venv) to run the scripts
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

def main() -> None:
    print(f"\n{CYAN}Starting Post-Ingestion Pipeline (Resume from Chunking)...{RESET}")

    env = os.environ.copy()
    pipeline_start = time.monotonic()

    # -- Stage 2: Chunking ----------------------------------------------------
    run_stage("STAGE 2  — Chunking", "chunk.py", env)

    # -- Stage 3: Embed and Store (Milvus) -------------------------------------
    run_stage("STAGE 3 — Embed and Store (Milvus)", "embed_and_store.py", env)

    # -- Stage 4: BM25 Indexing -----------------------------------------------
    run_stage("STAGE 4 — BM25 Indexing", "build_bm25.py", env)

    total_elapsed = time.monotonic() - pipeline_start
    
    print(f"\n{BAR}")
    print(f"{BOLD}POST-INGESTION PIPELINE COMPLETE{RESET}")
    print(f"{BAR}")
    print(f"Total time: {total_elapsed:.1f} seconds")
    print(f"{BAR}\n")

if __name__ == "__main__":
    main()
