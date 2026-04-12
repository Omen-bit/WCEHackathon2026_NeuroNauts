import sys
import json
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()))
from app import retrieve, build_context, call_llm_stateless, score_faithfulness, score_answer_relevancy, FAITH_THRESHOLD

# Dummy streamlit objects to trick imports if needed
class DummyST:
    def __getattr__(self, name):
        def _dummy(*args, **kwargs): return DummyST()
        return _dummy
import sys
sys.modules['streamlit'] = DummyST()

PROJECT_ROOT = Path(__file__).parent.parent
queries_path = PROJECT_ROOT / "queries.json"
OUTPUT_JSON  = PROJECT_ROOT / "output" / "evaluation_results.json"
OUTPUT_CSV   = PROJECT_ROOT / "output" / "evaluation_results.csv"

with open(queries_path, encoding="utf-8") as f:
    queries = json.load(f)

print(f"Starting headless evaluation of {len(queries)} queries...")
results = []
if OUTPUT_JSON.exists():
    try:
        with open(OUTPUT_JSON, encoding="utf-8") as f:
            existing = json.load(f)
        results = existing.get("results", [])
    except Exception:
        pass

# function to write to disk
def _save_data():
    valid = [r for r in results if r.get("faithfulness_score") is not None]
    avg_faith = round(sum(r["faithfulness_score"] for r in valid)/len(valid), 4) if valid else 0.0
    avg_relevancy = round(sum(r["relevancy_score"] for r in valid)/len(valid), 4) if valid else 0.0

    output = {
        "summary": {
            "total_evaluated": len(valid),
            "avg_faithfulness": avg_faith,
            "avg_answer_relevancy": avg_relevancy,
            "faithfulness_threshold": FAITH_THRESHOLD
        },
        "results": results
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query_id","question","answer","faithfulness_score","relevancy_score"])
        w.writeheader()
        for r in results:
            w.writerow({
                "query_id": r["query_id"], "question": r["question"], "answer": r.get("answer", ""),
                "faithfulness_score": r.get("faithfulness_score", ""), "relevancy_score": r.get("relevancy_score", "")
            })

done_ids = {r["query_id"] for r in results}

for i, q in enumerate(queries):
    qid, question = q["query_id"], q["question"]
    if qid in done_ids:
        print(f"[{i+1}/50] (Skipping, already evaluated) Q{qid}: {question}")
        continue
        
    print(f"[{i+1}/50] Evaluating Q{qid}: {question}")
    try:
        chunks = retrieve(question)
        contexts = [c["clean_text"] for c in chunks]
        answer = call_llm_stateless(question, build_context(chunks))
        faith = score_faithfulness(answer, contexts)
        relevancy = score_answer_relevancy(question, answer)
        results.append({
            "query_id": qid, "question": question, "answer": answer, "contexts": contexts,
            "faithfulness_score": faith["score"], "faithfulness_detail": faith,
            "relevancy_score": relevancy["score"], "relevancy_detail": relevancy,
        })
    except Exception as e:
        print(f"  !! Error on Q{qid}: {e}")
        results.append({"query_id": qid, "question": question, "answer": "",
                        "faithfulness_score": None, "relevancy_score": None, "error": str(e)})
    
    # Live save after every query 
    _save_data()

print("\n✓ Full evaluation completed seamlessly! Output written to output/evaluation_results.json & .csv")

