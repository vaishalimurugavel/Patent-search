"""
Pytest-style tests for search_engine.py.

Run with: pytest tests/test_search_engine.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_patents, normalize_patent
from src.search_engine import (
    get_search_text,
    build_index,
    search,
    filter_candidates,
    hybrid_search,
    naive_hybrid_search,
    build_title_index,
    search_by_title,
)

# Resolve relative to this file's location, not the working directory --
# see test_data_loader.py for why this matters.
# Points at tests/fixtures/, NOT the project-root data/ folder -- see
# test_data_loader.py for why tests use a small, fixed, decoupled dataset.
DATA_DIR = Path(__file__).resolve().parent / "fixtures"


# ---- shared fixtures (plain functions, since we're not using pytest fixtures) ----

def _load_test_patents():
    return load_patents(DATA_DIR)


# ---- get_search_text tests ----

def test_search_text_combines_title_abstract_claims():
    patent = normalize_patent({
        "title": "WIDGET",
        "abstract": "A widget abstract.",
        "claims": ["Claim one.", "Claim two."],
    })
    text = get_search_text(patent)
    assert "WIDGET" in text
    assert "A widget abstract." in text
    assert "Claim one." in text
    assert "Claim two." in text


def test_search_text_does_not_crash_on_empty_patent():
    patent = normalize_patent({"title": "Bare"})
    text = get_search_text(patent)
    assert isinstance(text, str)
    assert "Bare" in text


# ---- build_index tests ----

def test_build_index_matrix_row_count_matches_patent_count():
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)
    assert tfidf_matrix.shape[0] == len(patents)


def test_build_index_vocabulary_is_nonempty():
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)
    assert tfidf_matrix.shape[1] > 0


# ---- search tests ----

def test_search_returns_top_k_or_fewer_results():
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)
    results = search("wheel", patents, vectorizer, tfidf_matrix, top_k=3)
    assert len(results) <= 3


def test_search_results_are_sorted_descending_by_score():
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)
    results = search("wheel spoke", patents, vectorizer, tfidf_matrix, top_k=10)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_result_has_expected_keys():
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)
    results = search("wheel", patents, vectorizer, tfidf_matrix, top_k=1)
    assert len(results) == 1
    result = results[0]
    assert set(result.keys()) == {"doc_number", "title", "score", "snippet"}


def test_search_negative_top_k_means_no_limit():
    # regression test: top_k=-1 must return ALL results, not
    # corpus_size - 1 (Python's negative-slice-index behavior applied
    # to [::-1][:top_k] without this fix)
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)
    results = search("wheel", patents, vectorizer, tfidf_matrix, top_k=-1)
    assert len(results) == len(patents)


def test_hybrid_search_negative_top_k_means_no_limit():
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)
    results = hybrid_search("wheel", patents, vectorizer, tfidf_matrix,
                             classification_prefix="B60B", top_k=-1)
    expected_count = len(filter_candidates(patents, classification_prefix="B60B"))
    assert len(results) == expected_count


def test_search_relevant_query_ranks_matching_patent_first():
    # SPOKE patent should rank at or near the top for a spoke-specific query
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)
    results = search("spoke axle friction enhancing structure", patents, vectorizer, tfidf_matrix, top_k=1)
    assert results[0]["title"] == "SPOKE"


def test_search_nonsense_query_still_returns_without_crashing():
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)
    results = search("zzzznonexistentword qqqqxyzabc", patents, vectorizer, tfidf_matrix, top_k=5)
    # vectorizer just produces an all-zero vector for unknown words;
    # cosine similarity against an all-zero vector is 0, not an error
    assert isinstance(results, list)
    for r in results:
        assert r["score"] == 0.0


# ---- filter_candidates tests ----

def test_filter_by_classification_prefix():
    patents = _load_test_patents()
    idxs = filter_candidates(patents, classification_prefix="B60B")
    for i in idxs:
        assert patents[i]["classification"].startswith("B60B")


def test_filter_by_title_keyword_case_insensitive():
    patents = _load_test_patents()
    idxs = filter_candidates(patents, title_keyword="WHEEL")
    assert len(idxs) > 0
    for i in idxs:
        assert "wheel" in patents[i]["title"].lower()


def test_filter_by_classification_prefix_case_insensitive():
    # regression test: lowercase prefix should match uppercase codes
    patents = _load_test_patents()
    upper = filter_candidates(patents, classification_prefix="B60B")
    lower = filter_candidates(patents, classification_prefix="b60b")
    assert upper == lower
    assert len(upper) > 0


def test_filter_with_no_filters_returns_everything():
    patents = _load_test_patents()
    idxs = filter_candidates(patents)
    assert len(idxs) == len(patents)


def test_filter_with_impossible_combo_returns_empty():
    patents = _load_test_patents()
    idxs = filter_candidates(patents, classification_prefix="ZZZ_NOPE")
    assert idxs == []


# ---- hybrid_search tests ----

def test_hybrid_search_empty_filter_returns_empty_list_not_crash():
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)
    results = hybrid_search("wheel", patents, vectorizer, tfidf_matrix, title_keyword="zzz_nope")
    assert results == []


def test_hybrid_search_only_returns_patents_matching_filter():
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)
    results = hybrid_search("vehicle", patents, vectorizer, tfidf_matrix, classification_prefix="B60B", top_k=20)
    result_docnums = {r["doc_number"] for r in results}
    filtered_docnums = {patents[i]["doc_number"] for i in filter_candidates(patents, classification_prefix="B60B")}
    assert result_docnums.issubset(filtered_docnums)


def test_hybrid_search_result_indices_map_back_correctly():
    # regression test for the "candidate_indices[rank]" translation step --
    # this is the easiest bug to introduce in hybrid_search, so it gets
    # its own explicit test rather than relying on other tests to catch it
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)

    # filter down to a known small subset, then confirm every returned
    # doc_number actually belongs to a patent within that subset
    prefix = patents[0]["classification"][:4]
    results = hybrid_search("the", patents, vectorizer, tfidf_matrix, classification_prefix=prefix, top_k=20)
    expected_docnums = {p["doc_number"] for p in patents if p["classification"].startswith(prefix)}
    for r in results:
        assert r["doc_number"] in expected_docnums


# ---- build_title_index / search_by_title tests ----

def test_title_index_exact_match():
    patents = _load_test_patents()
    title_index = build_title_index(patents)
    results = search_by_title("SPOKE", patents, title_index)
    assert len(results) == 1
    assert results[0]["title"] == "SPOKE"
    assert results[0]["score"] == 1.0


def test_title_index_case_insensitive():
    patents = _load_test_patents()
    title_index = build_title_index(patents)
    upper = search_by_title("SPOKE", patents, title_index)
    lower = search_by_title("spoke", patents, title_index)
    assert len(upper) == len(lower) == 1


def test_title_index_strips_whitespace():
    patents = _load_test_patents()
    title_index = build_title_index(patents)
    results = search_by_title("  SPOKE  ", patents, title_index)
    assert len(results) == 1


def test_title_index_no_match_returns_empty_list():
    patents = _load_test_patents()
    title_index = build_title_index(patents)
    results = search_by_title("NONEXISTENT TITLE XYZ", patents, title_index)
    assert results == []


def test_title_index_skips_blank_titles():
    patents = _load_test_patents() + [normalize_patent({"doc_number": "999", "abstract": "no title"})]
    title_index = build_title_index(patents)
    assert "" not in title_index


def test_title_index_handles_real_duplicate_titles():
    # regression test using ACTUAL duplicate titles found in the real
    # sample data ("PNEUMATIC TIRE" x3, "TIRE" x2) -- confirms the
    # list-based index design doesn't silently drop patents when two
    # different patents share an identical title
    patents = _load_test_patents()
    title_index = build_title_index(patents)

    pneumatic_tire_results = search_by_title("PNEUMATIC TIRE", patents, title_index)
    tire_results = search_by_title("TIRE", patents, title_index)

    assert len(pneumatic_tire_results) == 3
    assert len(tire_results) == 2

    # every result must be a distinct patent (no accidental duplicates
    # or overwritten entries)
    doc_numbers = [r["doc_number"] for r in pneumatic_tire_results]
    assert len(doc_numbers) == len(set(doc_numbers))


# ---- naive_hybrid_search tests ----

def test_naive_hybrid_search_matches_hybrid_search_results():
    # CRITICAL: naive_hybrid_search only exists to be a slower baseline
    # for benchmarking. If its results ever diverge from hybrid_search,
    # the benchmark would be comparing two different algorithms, not
    # two implementations of the same one -- making any timing
    # comparison meaningless.
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)

    test_cases = [
        dict(query="wheel bearing", classification_prefix="B60B", top_k=5),
        dict(query="vehicle wheel", title_keyword="wheel", top_k=10),
        dict(query="spoke", classification_prefix="B60B", title_keyword="spoke", top_k=3),
        dict(query="tire", abstract_keyword="reinforced", top_k=5),
        dict(query="the", top_k=-1),  # no filters, no limit
    ]

    for case in test_cases:
        efficient = hybrid_search(patents=patents, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix, **case)
        naive = naive_hybrid_search(patents=patents, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix, **case)
        efficient_ids = [r["doc_number"] for r in efficient]
        naive_ids = [r["doc_number"] for r in naive]
        assert efficient_ids == naive_ids, f"Mismatch for {case}: {efficient_ids} != {naive_ids}"


def test_naive_hybrid_search_empty_filter_returns_empty_list():
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)
    results = naive_hybrid_search("wheel", patents, vectorizer, tfidf_matrix, title_keyword="zzz_nope")
    assert results == []


def test_naive_hybrid_search_negative_top_k_means_no_limit():
    patents = _load_test_patents()
    vectorizer, tfidf_matrix = build_index(patents)
    results = naive_hybrid_search("wheel", patents, vectorizer, tfidf_matrix,
                                   classification_prefix="B60B", top_k=-1)
    expected_count = len(filter_candidates(patents, classification_prefix="B60B"))
    assert len(results) == expected_count


def run_all():
    """Run every test_* function in this module and report pass/fail."""
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
