from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import get_snippet


def get_search_text(patent):
    """
    Combine the fields we match a query against into one string:
    title + abstract + claims. This is the text that gets vectorized
    for TF-IDF matching (separate from get_snippet, which is what gets
    displayed to the user).
    """
    claims_text = " ".join(patent["claims"])
    return f"{patent['title']} {patent['abstract']} {claims_text}"


def build_index(patents):
    """
    Fit a TF-IDF vectorizer over every patent's search text.

    Returns:
        vectorizer: the fitted TfidfVectorizer (needed later to transform
                    a new query the same way the corpus was vectorized)
        tfidf_matrix: sparse matrix, one row per patent, one column per
                      vocabulary term, values are TF-IDF weights
    """
    search_texts = [get_search_text(p) for p in patents]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(search_texts)

    return vectorizer, tfidf_matrix


def filter_candidates(patents, classification_prefix=None, title_keyword=None, abstract_keyword=None):
    """
    Return the list indices of patents matching all given filters.
    Any filter left as None is skipped (not applied).
    """
    candidate_indices = []

    for i, patent in enumerate(patents):
        if classification_prefix and not patent["classification"].upper().startswith(classification_prefix.upper()):
            continue
        if title_keyword and title_keyword.lower() not in patent["title"].lower():
            continue
        if abstract_keyword and abstract_keyword.lower() not in patent["abstract"].lower():
            continue
        candidate_indices.append(i)

    return candidate_indices


def build_title_index(patents):
    """
    Build a dict mapping normalized (lowercased, stripped) title ->
    list of patent indices with that exact title.

    A list (not a single index) because two different patents could,
    in principle, share an identical title -- we don't want to
    silently drop one of them.
    """
    index = {}
    for i, patent in enumerate(patents):
        key = patent["title"].strip().lower()
        if not key:
            continue  # can't index a patent with no title
        index.setdefault(key, []).append(i)
    return index


def search_by_title(title, patents, title_index):
    """
    Exact title lookup -- O(1) dict access instead of scanning every
    patent's title string. Case-insensitive, trims whitespace.
    """
    key = title.strip().lower()
    matching_indices = title_index.get(key, [])

    results = []
    for i in matching_indices:
        patent = patents[i]
        results.append({
            "doc_number": patent["doc_number"],
            "title": patent["title"],
            "score": 1.0,  # exact match, not a similarity score
            "snippet": get_snippet(patent),
        })

    return results


def naive_hybrid_search(query, patents, vectorizer, tfidf_matrix, classification_prefix=None,
                         title_keyword=None, abstract_keyword=None, top_k=10):
    """
    The "naive" way to do hybrid search: score EVERY patent first,
    rank them all, THEN walk the ranked list and keep only the ones
    that pass the filters. This does full corpus-wide similarity work
    regardless of how selective the filters are -- exists purely as a
    slower baseline to benchmark hybrid_search() against.
    """
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    effective_k = len(scores) if top_k < 0 else top_k

    # Rank ALL patents first, before any filtering happens.
    all_ranked_indices = scores.argsort()[::-1]

    results = []
    for idx in all_ranked_indices:
        patent = patents[idx]

        if classification_prefix and not patent["classification"].upper().startswith(classification_prefix.upper()):
            continue
        if title_keyword and title_keyword.lower() not in patent["title"].lower():
            continue
        if abstract_keyword and abstract_keyword.lower() not in patent["abstract"].lower():
            continue

        results.append({
            "doc_number": patent["doc_number"],
            "title": patent["title"],
            "score": float(scores[idx]),
            "snippet": get_snippet(patent),
        })

        if len(results) >= effective_k:
            break

    return results


def hybrid_search(query, patents, vectorizer, tfidf_matrix, classification_prefix=None,
                   title_keyword=None, abstract_keyword=None, top_k=10):
    """
    Filter patents by metadata first, then rank only the surviving
    subset by similarity to query. This avoids computing similarity
    scores for patents that were going to be filtered out anyway.
    """
    candidate_indices = filter_candidates(
        patents,
        classification_prefix=classification_prefix,
        title_keyword=title_keyword,
        abstract_keyword=abstract_keyword,
    )

    if not candidate_indices:
        return []

    # Slice the TF-IDF matrix down to only the filtered rows BEFORE
    # scoring. tfidf_matrix is (n_patents x vocab_size); this selects
    # just the candidate rows, so cosine_similarity only does work
    # proportional to len(candidate_indices), not len(patents).
    candidate_matrix = tfidf_matrix[candidate_indices]

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, candidate_matrix).flatten()

    # These ranks are positions WITHIN candidate_indices, not positions
    # in the original patents list -- that's why we map back through
    # candidate_indices[rank] below.
    #
    # Same negative-top_k fix as in search(): treat negative as "no limit"
    # rather than relying on Python's negative-slice-index behavior.
    effective_k = len(scores) if top_k < 0 else top_k
    ranked = scores.argsort()[::-1][:effective_k]

    results = []
    for rank in ranked:
        original_idx = candidate_indices[rank]
        patent = patents[original_idx]
        results.append({
            "doc_number": patent["doc_number"],
            "title": patent["title"],
            "score": float(scores[rank]),
            "snippet": get_snippet(patent),
        })

    return results


def search(query, patents, vectorizer, tfidf_matrix, top_k=10):
    """
    Rank all patents by similarity to query and return the top_k as
    result dicts.
    """
    # Transform the query using the SAME vectorizer the corpus was built
    # with, so it lands in the same vector space. Note the [query] -- it
    # has to be a list, since the vectorizer expects a list of documents,
    # even when there's only one.
    query_vector = vectorizer.transform([query])

    # cosine_similarity returns a 2D array (1 query row x N patent rows).
    # .flatten() collapses it to a plain 1D array of N scores.
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # argsort() gives indices that would sort scores ascending; [::-1]
    # reverses to descending (highest similarity first).
    #
    # NOTE: we do NOT just slice with [:top_k] directly. Python slicing
    # treats negative indices as "count from the end" -- [:-1] means
    # "everything except the last element," NOT "no limit." That means
    # a caller passing top_k=-1 expecting "give me everything" would
    # silently get back (corpus_size - 1) results instead. We handle
    # negative top_k explicitly as "no limit" here.
    effective_k = len(scores) if top_k < 0 else top_k
    ranked_indices = scores.argsort()[::-1][:effective_k]

    results = []
    for idx in ranked_indices:
        patent = patents[idx]
        results.append({
            "doc_number": patent["doc_number"],
            "title": patent["title"],
            "score": float(scores[idx]),
            "snippet": get_snippet(patent),
        })

    return results
