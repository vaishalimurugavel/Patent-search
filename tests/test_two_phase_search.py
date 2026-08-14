"""
Tests for part3/two_phase_search.py.
Run with: python3 tests/test_two_phase_search.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_patents
from src.search_engine import build_index, search
from part3.two_phase_search import (
    compute_rerank_features,
    build_docnum_index,
    two_phase_search,
    DEFAULT_WEIGHTS,
)

DATA_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_test_patents():
    return load_patents(DATA_DIR)


# ---- compute_rerank_features tests ----

def test_features_exact_phrase_in_title():
    patents = _load_test_patents()
    spoke = next(p for p in patents if p["title"] == "SPOKE")
    features = compute_rerank_features("spoke", spoke)
    assert features["exact_phrase_in_title"] == 1.0


def test_features_no_match_gives_zero():
    patents = _load_test_patents()
    spoke = next(p for p in patents if p["title"] == "SPOKE")
    features = compute_rerank_features("zzz nonexistent qqqxyz", spoke)
    assert features["exact_phrase_in_title"] == 0.0
    assert features["exact_phrase_in_abstract"] == 0.0
    assert features["term_overlap_ratio"] == 0.0


def test_features_empty_query_gives_zero_not_false_positive():
    patents = _load_test_patents()
    spoke = next(p for p in patents if p["title"] == "SPOKE")
    features = compute_rerank_features("", spoke)
    # an empty string is technically `in` every string in Python --
    # this test guards against that false-positive case specifically
    assert features["exact_phrase_in_title"] == 0.0
    assert features["exact_phrase_in_abstract"] == 0.0


def test_features_partial_term_overlap_ratio():
    patents = _load_test_patents()
    spoke = next(p for p in patents if p["title"] == "SPOKE")
    # "spoke" matches, "rocketship" and "banana" don't -- 1 of 3 terms
    features = compute_rerank_features("spoke rocketship banana", spoke)
    assert abs(features["term_overlap_ratio"] - (1 / 3)) < 1e-9


# ---- build_docnum_index tests ----

def test_docnum_index_maps_every_patent():
    patents = _load_test_patents()
    index = build_docnum_index(patents)
    assert len(index) == len(patents)
    for patent in patents:
        assert index[patent["doc_number"]] is patent


# ---- two_phase_search tests ----

def test_two_phase_search_returns_final_k_or_fewer():
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)
    results = two_phase_search("wheel", patents, vectorizer, tfidf_matrix, phase1_k=20, final_k=3)
    assert len(results) <= 3


def test_two_phase_search_results_sorted_descending():
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)
    results = two_phase_search("wheel vehicle", patents, vectorizer, tfidf_matrix, phase1_k=20, final_k=10)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_two_phase_search_empty_phase1_returns_empty_list():
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)
    results = two_phase_search("zzzznonexistentword qqqqxyzabc", patents, vectorizer, tfidf_matrix, final_k=5)
    # a nonsense query still technically returns phase1 candidates (all
    # zero-score, per Part 1's documented behavior for stopword/nonsense
    # queries) -- so this should NOT crash, and should return a list
    assert isinstance(results, list)


def test_two_phase_search_only_reranks_actual_phase1_candidates():
    # every result must have come from phase 1 -- reranking should never
    # introduce a patent that TF-IDF retrieval didn't already surface
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)

    phase1_docnums = {r["doc_number"] for r in search("wheel", patents, vectorizer, tfidf_matrix, top_k=20)}
    two_phase_docnums = {r["doc_number"] for r in
                          two_phase_search("wheel", patents, vectorizer, tfidf_matrix, phase1_k=20, final_k=20)}

    assert two_phase_docnums.issubset(phase1_docnums)


def test_two_phase_search_zero_feature_weights_equals_pure_tfidf_order():
    # with all feature weights at 0, two_phase_search's ranking should
    # be IDENTICAL to plain search() -- confirms the reranking math
    # doesn't introduce a bug that changes results even when it's
    # supposed to be a no-op
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)

    zero_feature_weights = {"tfidf": 1.0, "title_phrase": 0.0, "abstract_phrase": 0.0, "term_overlap": 0.0}
    two_phase_results = two_phase_search(
        "wheel vehicle", patents, vectorizer, tfidf_matrix,
        phase1_k=20, final_k=10, weights=zero_feature_weights,
    )
    plain_results = search("wheel vehicle", patents, vectorizer, tfidf_matrix, top_k=10)

    two_phase_order = [r["doc_number"] for r in two_phase_results]
    plain_order = [r["doc_number"] for r in plain_results]
    assert two_phase_order == plain_order

    # and the scores themselves should match the original tfidf scores exactly
    for r in two_phase_results:
        assert abs(r["score"] - r["tfidf_score"]) < 1e-9


def test_two_phase_search_phase1_k_limits_candidate_pool():
    # a very small phase1_k should limit how many candidates even make
    # it into consideration for reranking, regardless of final_k
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)
    results = two_phase_search("wheel", patents, vectorizer, tfidf_matrix, phase1_k=2, final_k=10)
    assert len(results) <= 2


def test_two_phase_search_default_weights_are_used_when_none_given():
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)
    # just confirm it runs without needing weights explicitly, and that
    # DEFAULT_WEIGHTS is actually a dict with the expected keys
    assert set(DEFAULT_WEIGHTS.keys()) == {"tfidf", "title_phrase", "abstract_phrase", "term_overlap"}
    results = two_phase_search("wheel", patents, vectorizer, tfidf_matrix, phase1_k=10, final_k=5)
    assert isinstance(results, list)


def run_all():
    import traceback

    tests = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    run_all()
