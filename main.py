"""
main.py -- interactive command-line interface for the patent search engine.

Run it from the project root:
    python3 main.py

Loads every patents_ipa*.json file found (however deeply nested) inside
the `data/` folder next to this script, builds the TF-IDF index once,
then lets you type queries at a prompt and see ranked results.
"""
import sys
import time
from pathlib import Path

from src.data_loader import load_patents
from src.search_engine import build_index, search, build_title_index, search_by_title, hybrid_search
from part3.two_phase_search import two_phase_search

# Resolve relative to this file's location, not the working directory --
# same reasoning as the tests: this must work the same way no matter
# where the script is launched from.
DATA_DIR = Path(__file__).resolve().parent / "data"


def print_results(results):
    if not results:
        print("  (no results)")
        return
    for rank, r in enumerate(results, start=1):
        print(f"  {rank}. [{r['score']:.3f}] {r['title']}  (doc #{r['doc_number']})")
        print(f"     {r['snippet'][:200]}")
        print()


def main():
    print(f"Loading patents from {DATA_DIR} ...")
    start = time.time()
    patents = load_patents(DATA_DIR)
    print(f"Loaded {len(patents)} patents in {time.time() - start:.2f}s")

    if not patents:
        print(f"No patents found in {DATA_DIR}. Check that your data files are there.")
        sys.exit(1)

    print("Building search index ...")
    start = time.time()
    vectorizer, tfidf_matrix = build_index(patents)
    title_index = build_title_index(patents)
    print(f"Index built in {time.time() - start:.2f}s ({tfidf_matrix.shape[1]} vocabulary terms, "
          f"{len(title_index)} unique titles)")

    print()
    print("Type a natural language query to search patents.")
    print("Type 'title:<exact title>' to look up a patent by its exact title.")
    print("Type 'rerank:<query>' to use two-phase search (TF-IDF + feature-based rerank).")
    print("Type 'hybrid:<query>|class=<prefix>|title=<keyword>' to filter by classification")
    print("  code prefix and/or title keyword (either filter is optional, e.g.:")
    print("  'hybrid:wheel bearing|class=B60B').")
    print("Type 'quit' or 'exit' to stop.")
    print()

    while True:
        query = input("Search> ").strip()

        if query.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if not query:
            continue

        start = time.time()
        if query.lower().startswith("title:"):
            exact_title = query[len("title:"):].strip()
            results = search_by_title(exact_title, patents, title_index)
        elif query.lower().startswith("rerank:"):
            real_query = query[len("rerank:"):].strip()
            results = two_phase_search(real_query, patents, vectorizer, tfidf_matrix, phase1_k=100, final_k=5)
        elif query.lower().startswith("hybrid:"):
            rest = query[len("hybrid:"):]
            parts = [p.strip() for p in rest.split("|")]
            real_query = parts[0]
            classification_prefix = None
            title_keyword = None
            for part in parts[1:]:
                if part.lower().startswith("class="):
                    classification_prefix = part[len("class="):].strip()
                elif part.lower().startswith("title="):
                    title_keyword = part[len("title="):].strip()
            results = hybrid_search(real_query, patents, vectorizer, tfidf_matrix,
                                     classification_prefix=classification_prefix,
                                     title_keyword=title_keyword, top_k=5)
        else:
            results = search(query, patents, vectorizer, tfidf_matrix, top_k=5)
        elapsed = time.time() - start

        print(f"\n{len(results)} results in {elapsed*1000:.1f}ms:\n")
        print_results(results)


if __name__ == "__main__":
    main()
