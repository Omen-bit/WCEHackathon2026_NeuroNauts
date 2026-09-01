import sys
import json
import csv
from pathlib import Path

_APP_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(_APP_DIR))

# Dummy streamlit objects to trick imports if needed
class DummyST:
    def __getattr__(self, name):
        def _dummy(*args, **kwargs): return DummyST()
        return _dummy
sys.modules['streamlit'] = DummyST()

from retrieve import retrieve, _get_embed_model
from generate import generate
from eval_suite import evaluate_rag_triad, FAITH_THRESHOLD, RELEVANCY_THRESHOLD

PROJECT_ROOT = _APP_DIR.parent
queries_path = PROJECT_ROOT / "queries.json"
OUTPUT_JSON  = PROJECT_ROOT / "output" / "evaluation_results.json"
OUTPUT_CSV   = PROJECT_ROOT / "output" / "evaluation_results.csv"

if not queries_path.exists():
    print(f"Error: {queries_path} not found.")
    sys.exit(1)

with open(queries_path, encoding="utf-8") as f:
    queries = json.load(f)

print(f"Starting headless industry-grade evaluation of {len(queries)} queries...")

model = _get_embed_model()
def embed_fn(texts):
    return model.encode(texts, normalize_embeddings=True).tolist()

results = []
if OUTPUT_JSON.exists():
    try:
        with open(OUTPUT_JSON, encoding="utf-8") as f:
            existing = json.load(f)
        results = existing.get("results", [])
    except Exception:
        pass

def _save_data():
    valid = [r for r in results if r.get("faithfulness_score") is not None]
    n = len(valid) if valid else 1
    avg_faith = round(sum(r.get("faithfulness_score", 0.0) for r in valid) / n, 4)
    avg_relevancy = round(sum(r.get("relevancy_score", 0.0) for r in valid) / n, 4)
    avg_precision = round(sum(r.get("context_precision_score", 0.0) for r in valid) / n, 4)
    avg_recall = round(sum(r.get("context_recall_score", 0.0) for r in valid) / n, 4)
    avg_composite = round(sum(r.get("composite_score", 0.0) for r in valid) / n, 4)

    output = {
        "summary": {
            "total_evaluated": len(valid),
            "avg_faithfulness": avg_faith,
            "avg_answer_relevancy": avg_relevancy,
            "avg_context_precision": avg_precision,
            "avg_context_recall": avg_recall,
            "avg_composite_score": avg_composite,
            "faithfulness_threshold": FAITH_THRESHOLD,
            "relevancy_threshold": RELEVANCY_THRESHOLD
        },
        "results": results
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "query_id", "question", "answer", "composite_score", "grade",
            "faithfulness_score", "relevancy_score", "context_precision_score", "context_recall_score"
        ])
        w.writeheader()
        for r in results:
            w.writerow({
                "query_id": r["query_id"],
                "question": r["question"],
                "answer": r.get("answer", ""),
                "composite_score": r.get("composite_score", ""),
                "grade": r.get("grade", ""),
                "faithfulness_score": r.get("faithfulness_score", ""),
                "relevancy_score": r.get("relevancy_score", ""),
                "context_precision_score": r.get("context_precision_score", ""),
                "context_recall_score": r.get("context_recall_score", "")
            })

done_ids = {r["query_id"] for r in results}

for i, q in enumerate(queries):
    qid, question = q["query_id"], q["question"]
    if qid in done_ids:
        print(f"[{i+1}/{len(queries)}] (Skipping, already evaluated) Q{qid}: {question}")
        continue

    print(f"[{i+1}/{len(queries)}] Evaluating Q{qid}: {question}")
    try:
        gen_res = generate(question, top_k=5)
        answer = gen_res.get("answer", "")
        chunks = retrieve(question, top_k=5)
        contexts = [c.get("clean_text", "") for c in chunks]

        metrics = evaluate_rag_triad(question, answer, contexts, embed_fn)
        results.append({
            "query_id": qid,
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "composite_score": metrics["composite_score"],
            "grade": metrics["grade"],
            "faithfulness_score": metrics["faithfulness"]["score"],
            "faithfulness_detail": metrics["faithfulness"],
            "relevancy_score": metrics["answer_relevancy"]["score"],
            "relevancy_detail": metrics["answer_relevancy"],
            "context_precision_score": metrics["context_precision"]["score"],
            "context_precision_detail": metrics["context_precision"],
            "context_recall_score": metrics["context_recall"]["score"],
            "context_recall_detail": metrics["context_recall"]
        })
    except Exception as e:
        print(f"  !! Error on Q{qid}: {e}")
        results.append({
            "query_id": qid, "question": question, "answer": "",
            "composite_score": None, "grade": "Error",
            "faithfulness_score": None, "relevancy_score": None,
            "context_precision_score": None, "context_recall_score": None,
            "error": str(e)
        })

    _save_data()

print("\n✓ Full evaluation completed seamlessly! Output written to output/evaluation_results.json & .csv")
