"""
train_reranker.py -- Part 3 enhancement: Evaluation & Training.

Instead of hand-picking the feature weights in two_phase_search.py's
DEFAULT_WEIGHTS, this learns them from data using logistic regression
over the same features (TF-IDF score, exact phrase match, term overlap).

Training data is synthesized from the patent corpus itself (no real user
query logs available):
  - POSITIVE pair: (first sentence of a patent's abstract, that same
    patent) -- a proxy for "a user searched for roughly this and meant
    to find this patent"
  - HARD NEGATIVE pair: (that same query, a DIFFERENT patent sharing the
    same classification prefix) -- same category, wrong invention; this
    is what the brief means by "seems like it should be related, but
    isn't really"
  - RANDOM NEGATIVE pair: (that same query, any other randomly chosen
    patent) -- an easier negative, included for comparison against the
    harder classification-based negatives
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from part3.two_phase_search import compute_rerank_features, DEFAULT_WEIGHTS


def build_feature_matrix(pairs, patents, vectorizer, tfidf_matrix):
    """
    Convert labeled (query, doc_number, label) pairs into a feature
    matrix X and label vector y, ready for sklearn.

    Features per pair (same 4 signals two_phase_search.py combines with
    hand-picked weights -- this is what learns the weights instead):
      0: tfidf_score           (cosine similarity between query and that patent)
      1: exact_phrase_in_title
      2: exact_phrase_in_abstract
      3: term_overlap_ratio
    """
    # doc_number -> row index in tfidf_matrix / position in patents list
    docnum_to_index = {p["doc_number"]: i for i, p in enumerate(patents)}

    X = []
    y = []
    pair_types = []

    for pair in pairs:
        idx = docnum_to_index.get(pair["doc_number"])
        if idx is None:
            continue  # doc_number not found -- skip rather than crash

        patent = patents[idx]
        query_vector = vectorizer.transform([pair["query"]])
        tfidf_score = cosine_similarity(query_vector, tfidf_matrix[idx]).flatten()[0]

        features = compute_rerank_features(pair["query"], patent)

        X.append([
            tfidf_score,
            features["exact_phrase_in_title"],
            features["exact_phrase_in_abstract"],
            features["term_overlap_ratio"],
        ])
        y.append(pair["label"])
        pair_types.append(pair["pair_type"])

    return np.array(X), np.array(y), pair_types


def score_with_weights(X, weights):
    """
    Combine feature columns into one score per row using the given
    weights -- same linear-combination idea as two_phase_search.py's
    final_score, applied here to a whole matrix at once via numpy.

    weights is a length-4 array/list matching the column order:
    [tfidf, title_phrase, abstract_phrase, term_overlap]
    """
    weights = np.array(weights)
    return X @ weights  # matrix-vector product: one weighted sum per row


def evaluate_auc(scores, y):
    """
    AUC-ROC: the probability that a randomly chosen positive pair scores
    higher than a randomly chosen negative pair. 0.5 = no better than
    random guessing, 1.0 = perfect separation. Chosen (per our earlier
    decision) because it evaluates RANKING quality directly, independent
    of any specific score threshold -- exactly what a reranker needs to
    get right.
    """
    return roc_auc_score(y, scores)


def run_baselines(X, y):
    """
    Two baselines to compare a trained model against:
      1. Raw TF-IDF score alone (column 0) -- "what if we didn't rerank at all"
      2. Part 3's hand-picked DEFAULT_WEIGHTS combination -- "what if we
         reranked with intuition-based weights instead of learned ones"
    """
    tfidf_only_scores = X[:, 0]
    tfidf_only_auc = evaluate_auc(tfidf_only_scores, y)

    handpicked_weights = [
        DEFAULT_WEIGHTS["tfidf"],
        DEFAULT_WEIGHTS["title_phrase"],
        DEFAULT_WEIGHTS["abstract_phrase"],
        DEFAULT_WEIGHTS["term_overlap"],
    ]
    handpicked_scores = score_with_weights(X, handpicked_weights)
    handpicked_auc = evaluate_auc(handpicked_scores, y)

    return {
        "tfidf_only_auc": tfidf_only_auc,
        "handpicked_weights_auc": handpicked_auc,
    }


def train_and_evaluate_reranker(X, y, n_splits=5, seed=42):
    """
    Train a logistic regression model to LEARN the feature weights
    (instead of two_phase_search.py's hand-picked DEFAULT_WEIGHTS),
    and evaluate it honestly using cross-validation.

    With only 60 examples, a single train/test split would be noisy --
    whichever 20% happened to land in the test set could swing the AUC
    a lot by chance. Cross-validation trains on multiple different folds
    and evaluates each example only when it was held OUT of training,
    giving a more honest, lower-variance estimate of how well this would
    generalize to a genuinely new query.

    Returns the out-of-fold AUC (the real evaluation metric) plus the
    final model's learned weights, fit on ALL the data (for inspection/
    use as the new weights, once the OOF AUC has confirmed it's sound).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # cross_val_predict trains n_splits separate models internally, one
    # per fold, and stitches together predictions for each example from
    # whichever fold left it OUT of training -- so oof_probs are always
    # predictions on data the model didn't see during that fold's fit.
    oof_probs = cross_val_predict(
        LogisticRegression(), X, y, cv=skf, method="predict_proba"
    )[:, 1]  # column 1 = probability of class "1" (relevant)

    oof_auc = roc_auc_score(y, oof_probs)

    # Fit one final model on ALL the data (not cross-validated) to get
    # the actual weights we'd use going forward -- the cross-validation
    # above was purely for honest evaluation, not for producing the
    # weights we ship.
    final_model = LogisticRegression()
    final_model.fit(X, y)

    return {
        "oof_auc": oof_auc,
        "learned_weights": {
            "tfidf": final_model.coef_[0][0],
            "title_phrase": final_model.coef_[0][1],
            "abstract_phrase": final_model.coef_[0][2],
            "term_overlap": final_model.coef_[0][3],
        },
        "intercept": final_model.intercept_[0],
    }


def get_first_sentence(text):
    """
    Return the first sentence of text (split on '. '), used as a
    synthetic "query" derived from a patent's own abstract. Falls back
    to the whole text if there's no sentence break to split on.
    """
    text = text.strip()
    if not text:
        return ""
    first = text.split(". ")[0].strip()
    if first and not first.endswith("."):
        first += "."
    return first


def build_training_pairs(patents, seed=42):
    """
    Build labeled (query, doc_number, label, pair_type) records for
    training/evaluating the reranker.

    Returns a list of dicts, 1 positive + up to 1 hard negative + up to
    1 random negative per patent with a non-empty abstract. Patents with
    no abstract are skipped entirely, since there's no text to build a
    synthetic query from.
    """
    rng = random.Random(seed)
    pairs = []
    n = len(patents)

    for i, patent in enumerate(patents):
        query = get_first_sentence(patent["abstract"])
        if not query:
            continue

        pairs.append({
            "query": query,
            "doc_number": patent["doc_number"],
            "label": 1,
            "pair_type": "positive",
        })

        prefix = patent["classification"][:4]
        same_category_others = [
            p for j, p in enumerate(patents)
            if j != i and prefix and p["classification"][:4] == prefix
        ]
        if same_category_others:
            hard_neg = rng.choice(same_category_others)
            pairs.append({
                "query": query,
                "doc_number": hard_neg["doc_number"],
                "label": 0,
                "pair_type": "hard_negative",
            })

        other_indices = [j for j in range(n) if j != i]
        if other_indices:
            random_neg = patents[rng.choice(other_indices)]
            pairs.append({
                "query": query,
                "doc_number": random_neg["doc_number"],
                "label": 0,
                "pair_type": "random_negative",
            })

    return pairs
