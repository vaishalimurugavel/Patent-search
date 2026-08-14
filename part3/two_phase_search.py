"""
two_phase_search.py -- Part 3 enhancement: two-phase search.

Phase 1 (broad retrieval): reuse Part 1's TF-IDF search() to cheaply
narrow the whole corpus down to a smaller candidate set (e.g. top 100).

Phase 2 (rerank): re-score just those candidates using additional
lexical signals that TF-IDF alone doesn't directly capture -- exact
phrase matches and raw term overlap -- and re-sort by a combined score.

This stands in for the cross-encoder/LLM-based rerankers the brief
mentions as examples; a feature-based reranker was chosen instead
because it needs no extra dependencies or network access to build and
test (see conversation for why sentence-transformers/torch weren't
available in this environment).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.search_engine import search


def build_docnum_index(patents):
    """
    Map doc_number -> full patent dict, for O(1) lookup. Needed because
    Phase 1's search() results only carry doc_number/title/score/snippet
    (the display-ready shape) -- Phase 2 needs the FULL patent record
    (title/abstract/claims separately) to compute rerank features.
    """
    return {p["doc_number"]: p for p in patents}


# Default weights combining the TF-IDF score with the three lexical
# features into one final ranking score. These are hand-picked, not
# learned -- see the Evaluation & Training enhancement for a more
# principled way to set these from data instead of intuition.
DEFAULT_WEIGHTS = {
    "tfidf": 1.0,
    "title_phrase": 0.5,     # exact phrase in title is a strong signal -- titles are short and specific
    "abstract_phrase": 0.2,  # exact phrase in abstract is a weaker signal -- abstracts are longer, more likely to contain it by chance
    "term_overlap": 0.3,
}


def two_phase_search(query, patents, vectorizer, tfidf_matrix, phase1_k=100, final_k=10, weights=None):
    """
    Phase 1: TF-IDF retrieval down to phase1_k candidates (cheap, broad).
    Phase 2: rerank those candidates using TF-IDF score + lexical
    features (expensive-if-run-on-everything, cheap on a small
    candidate set -- which is the whole point of doing it in two phases
    instead of computing these features for the entire corpus).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    phase1_results = search(query, patents, vectorizer, tfidf_matrix, top_k=phase1_k)
    if not phase1_results:
        return []

    docnum_index = build_docnum_index(patents)

    reranked = []
    for r in phase1_results:
        patent = docnum_index[r["doc_number"]]
        features = compute_rerank_features(query, patent)

        final_score = (
            weights["tfidf"] * r["score"]
            + weights["title_phrase"] * features["exact_phrase_in_title"]
            + weights["abstract_phrase"] * features["exact_phrase_in_abstract"]
            + weights["term_overlap"] * features["term_overlap_ratio"]
        )

        reranked.append({
            **r,
            "tfidf_score": r["score"],  # keep the original score around for inspection/debugging
            "score": final_score,
            "features": features,
        })

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:final_k]


def compute_rerank_features(query, patent):
    """
    Compute lexical features for one patent relative to a query, used
    as reranking signals on top of the TF-IDF score:

    - exact_phrase_in_title: does the WHOLE query appear verbatim in
      the title? (strong signal -- TF-IDF treats "wheel bearing" as two
      independent weighted terms, and can't tell "wheel bearing" apart
      from a document that happens to contain "wheel" and "bearing" in
      unrelated places)
    - exact_phrase_in_abstract: same idea, against the abstract
    - term_overlap_ratio: what fraction of the query's individual words
      appear literally anywhere in the patent's text? This is a raw,
      un-weighted signal -- unlike TF-IDF, it doesn't care whether a
      word is rare or common corpus-wide, just whether it's present.
    """
    query_lower = query.lower().strip()
    query_terms = query_lower.split()

    title_lower = patent["title"].lower()
    abstract_lower = patent["abstract"].lower()
    claims_text_lower = " ".join(patent["claims"]).lower()
    combined_text_lower = f"{title_lower} {abstract_lower} {claims_text_lower}"

    exact_phrase_in_title = 1.0 if query_lower and query_lower in title_lower else 0.0
    exact_phrase_in_abstract = 1.0 if query_lower and query_lower in abstract_lower else 0.0

    if query_terms:
        matched_terms = sum(1 for term in query_terms if term in combined_text_lower)
        term_overlap_ratio = matched_terms / len(query_terms)
    else:
        term_overlap_ratio = 0.0

    return {
        "exact_phrase_in_title": exact_phrase_in_title,
        "exact_phrase_in_abstract": exact_phrase_in_abstract,
        "term_overlap_ratio": term_overlap_ratio,
    }
