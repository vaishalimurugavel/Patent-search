"""
benchmark.py -- times hybrid_search() (filter-first) against
naive_hybrid_search() (score-everything-first) to measure the actual
cost of the naive approach as filter selectivity changes.

Run from the project root:
    python3 benchmark.py
"""
import time
from pathlib import Path

from src.data_loader import load_patents
from src.search_engine import build_index, hybrid_search, naive_hybrid_search

DATA_DIR = Path(__file__).resolve().parent / "data"

# Each case pairs a query with a filter of different selectivity, so we
# can see how the naive approach's cost scales as filters get narrower
# (more selective = more wasted work scoring patents that get thrown away).
BENCHMARK_CASES = [
    {"label": "no filter (baseline)", "query": "vehicle wheel assembly", "filters": {}},
    {"label": "broad classification filter", "query": "vehicle wheel assembly", "filters": {"classification_prefix": "B60"}},
    {"label": "narrow classification filter", "query": "vehicle wheel assembly", "filters": {"classification_prefix": "B60B"}},
    {"label": "title keyword filter", "query": "vehicle wheel assembly", "filters": {"title_keyword": "wheel"}},
    {"label": "combined filters", "query": "vehicle wheel assembly", "filters": {"classification_prefix": "B60B", "title_keyword": "wheel"}},
]

REPEATS = 200  # run each case this many times and average, since a single
               # run on a small dataset is too fast to time reliably


def time_it(fn, *args, **kwargs):
    start = time.perf_counter()
    for _ in range(REPEATS):
        result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / REPEATS) * 1000
    return avg_ms, result


def main():
    print(f"Loading patents from {DATA_DIR} ...")
    patents = load_patents(DATA_DIR)
    print(f"Loaded {len(patents)} patents")

    if not patents:
        print(f"No patents found in {DATA_DIR}, cannot benchmark.")
        return

    vectorizer, tfidf_matrix = build_index(patents)
    print(f"Index built: {tfidf_matrix.shape[0]} patents x {tfidf_matrix.shape[1]} vocabulary terms")
    print(f"(averaging over {REPEATS} runs per case)\n")

    header = f"{'Case':<32} {'Candidates':>10} {'Hybrid (ms)':>12} {'Naive (ms)':>12} {'Speedup':>9}"
    print(header)
    print("-" * len(header))

    for case in BENCHMARK_CASES:
        query = case["query"]
        filters = case["filters"]
        label = case["label"]

        hybrid_ms, hybrid_results = time_it(
            hybrid_search, query, patents, vectorizer, tfidf_matrix, top_k=5, **filters
        )
        naive_ms, naive_results = time_it(
            naive_hybrid_search, query, patents, vectorizer, tfidf_matrix, top_k=5, **filters
        )

        # sanity check every time: if these ever disagree, something is
        # wrong with the benchmark itself, not just a performance issue
        hybrid_ids = [r["doc_number"] for r in hybrid_results]
        naive_ids = [r["doc_number"] for r in naive_results]
        assert hybrid_ids == naive_ids, f"Result mismatch for case {label!r}: {hybrid_ids} != {naive_ids}"

        n_candidates = len(hybrid_results) if not filters else "n/a"
        speedup = naive_ms / hybrid_ms if hybrid_ms > 0 else float("inf")

        print(f"{label:<32} {str(n_candidates):>10} {hybrid_ms:>12.4f} {naive_ms:>12.4f} {speedup:>8.2f}x")

    print()
    print("Commentary on hybrid vs. naive efficiency (based on an actual 620-patent run):")
    print("Results were MIXED, not a clean win for hybrid_search -- it won on broad")
    print("classification and title filters, but lost on no-filter, narrow")
    print("classification, and combined filters. The reason: filter_candidates()")
    print("itself is an O(n) Python loop -- it scans every patent regardless of how")
    print("selective the filter ends up being. A 'narrow' filter only makes the")
    print("SUBSEQUENT similarity scoring cheaper (fewer surviving rows), not the")
    print("filtering step itself. Since scipy's cosine_similarity is a single")
    print("vectorized, C-level operation, it's already fast even over the whole")
    print("corpus at this scale -- fast enough that the Python-loop filtering")
    print("overhead can outweigh the scoring savings.")
    print()
    print("What would actually fix this: replace the linear classification-prefix")
    print("scan with a sorted array + binary search (bisect) -- O(log n) instead of")
    print("O(n) -- so filtering cost stops growing linearly with corpus size, and")
    print("hybrid_search's advantage would show up consistently rather than")
    print("depending on how selective a given query's filter happens to be.")


if __name__ == "__main__":
    main()
