import json
import os
import re
import pickle
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> list[str]:
    """
    Tokenize the text by lowercasing, stripping punctuation, and splitting on whitespace.
    """
    text = text.lower()
    # Replace anything not a word character or whitespace with space
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

def build_bm25_index():
    chunks_file = "../output/psychology2e_chunks.json"
    if not os.path.exists(chunks_file):
        print(f"ERROR: Chunks file {chunks_file} not found.")
        return

    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data['chunks']
    total_chunks = len(chunks)
    
    # Build the corpus and metadata lookup
    corpus = [chunk['text'] for chunk in chunks]
    tokenized_corpus = [tokenize(text) for text in corpus]
    
    # Updated lookup with text and has_image_context as requested
    lookup = [
        {
            "chunk_id": chunk['chunk_id'],
            "section_path": chunk['section_path'],
            "page_numbers": chunk['page_numbers'],
            "text": chunk['text'],
            "has_image_context": chunk['has_image_context'],
            "image_refs": chunk['image_refs']
        }
        for chunk in chunks
    ]

    # Build BM25Okapi index
    print(f"Building BM25 index over {total_chunks} chunk texts...")
    bm25 = BM25Okapi(tokenized_corpus)

    # Save both objects in a single pickle file
    output_pkl = "../output/bm25_index.pkl"
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    
    bundle = {
        "bm25": bm25,
        "lookup": lookup
    }
    
    with open(output_pkl, 'wb') as f:
        pickle.dump(bundle, f)
    
    print(f"BM25 index built over {total_chunks} documents")
    print(f"Saved to {output_pkl}")

    # Test Query
    test_query = "What is psychology?"
    print(f"\nTest query: {test_query}")
    
    tokenized_query = tokenize(test_query)
    scores = bm25.get_scores(tokenized_query)
    
    # Pair scores with lookup and sort descending
    results = sorted(zip(scores, lookup), key=lambda x: x[0], reverse=True)
    
    for i, (score, meta) in enumerate(results[:3], 1):
        print(f"  Rank {i} | score={score:.2f} | {meta['section_path']}")

    print("\nStage 4 complete with improved tokenization.")

if __name__ == "__main__":
    build_bm25_index()
