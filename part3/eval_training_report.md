# Evaluation & Training: Learning Reranker Weights

This documents the "Evaluation and Training" enhancement (Part 3): instead
of hand-picking the feature weights used in `two_phase_search.py`'s
reranker (`DEFAULT_WEIGHTS`), this learns them from labeled training data,
and reports the evaluation methodology used to justify that the learned
weights are actually sound -- not just "run once and hope."

## Problem Statement

Part 3's two-phase search combines four signals into one final ranking
score: the TF-IDF similarity, whether the query appears verbatim in the
title, whether it appears verbatim in the abstract, and what fraction of
the query's words appear anywhere in the patent's text. The weights
combining these four signals (`DEFAULT_WEIGHTS` in `two_phase_search.py`)
were chosen by intuition (e.g. "title matches feel like a stronger signal
than abstract matches, so give it a bigger weight"). This enhancement
replaces that guesswork with weights learned from actual labeled examples.

## Training Data

No real user query logs were available, so training data is synthesized
directly from the patent corpus itself (see `train_reranker.py`,
`build_training_pairs`):

- **Positive pairs**: the first sentence of a patent's own abstract, paired
  with that same patent. A proxy for "a user searched for roughly this
  text and meant to find this specific patent."
- **Hard negative pairs**: that same query, paired with a *different*
  patent sharing the same classification-code prefix -- same category,
  wrong invention. This is deliberately the harder, more informative kind
  of negative example (per the assignment's own framing: "pairs of text
  that seem like they should be related, but aren't really").
- **Random negative pairs**: that same query, paired with any other
  randomly chosen patent -- an easier negative, included for comparison.

On the 20-patent sample dataset, this produced 60 total pairs (20 of each
type).

## Features

Each `(query, patent)` pair is converted into a 4-dimensional feature
vector (`build_feature_matrix`), the same four signals `two_phase_search.py`
combines:

| Feature | Description |
|---|---|
| `tfidf_score` | Cosine similarity between the query and that specific patent's TF-IDF vector |
| `exact_phrase_in_title` | 1.0 if the whole query appears verbatim in the title, else 0.0 |
| `exact_phrase_in_abstract` | 1.0 if the whole query appears verbatim in the abstract, else 0.0 |
| `term_overlap_ratio` | Fraction of the query's individual words that appear anywhere in the patent's text |

## Evaluation Metric

**AUC-ROC**: the probability that a randomly chosen positive pair scores
higher than a randomly chosen negative pair. Chosen over plain accuracy
because it evaluates *ranking quality* directly, independent of picking a
specific score threshold -- which is exactly what a reranker needs to get
right (it doesn't need to output a calibrated "is this relevant" score, it
needs relevant results to end up ABOVE irrelevant ones).

Evaluated via 5-fold stratified cross-validation rather than a single
train/test split, since with only 60 examples, a single split would be
noisy -- whichever handful of examples happened to land in a test set
could swing the result substantially by chance.

## Results

| Method | AUC-ROC |
|---|---|
| Baseline: TF-IDF score alone (no reranking) | 0.9950 |
| Baseline: Part 3's hand-picked `DEFAULT_WEIGHTS` | 1.0000 |
| **Trained: logistic regression, learned weights (5-fold CV)** | **1.0000** |

Learned weights (logistic regression coefficients):

| Feature | Learned weight | Hand-picked weight (for comparison) |
|---|---|---|
| `tfidf` | +1.35 | 1.0 |
| `title_phrase` | **+0.00** | 0.5 |
| `abstract_phrase` | +3.04 | 0.2 |
| `term_overlap` | +1.24 | 0.3 |

## Honest Interpretation (Reading the Results Correctly)

Per the assignment's own framing, this task doesn't expect a dramatic
improvement in one hour of work -- and it didn't produce one, for a
specific, explainable reason worth stating directly rather than glossing
over: **the training data is synthetic and self-referential.** Positive
queries are literally extracted from the same patent they're paired with,
which makes them an unrealistically easy match for plain TF-IDF (0.995
AUC on its own, with essentially no room left to improve). A real
deployment would train on genuine user queries, which are messier, more
varied in phrasing, and don't share exact vocabulary with the "correct"
patent's own text -- that's where a learned combination of signals would
actually show its value over a single similarity score.

That said, one specific result IS a genuinely meaningful finding, not just
a ceiling effect: **the model learned a weight of exactly 0.0 for
`title_phrase`**, while the hand-picked weights guessed 0.5 for it. This
isn't an error -- every positive example in this training set came from
an *abstract* sentence, never a title, so `title_phrase` was always 0 for
positive examples. The model correctly learned "I have no evidence this
feature matters" rather than being assigned an arbitrary nonzero weight
on intuition alone. This is precisely the kind of thing a
training/evaluation pipeline is supposed to surface: not "the model got
better," but "here's a concrete, checkable reason why one weight was
unjustified guesswork and the data doesn't support it."

## What Would Improve This With More Time/Data

- **Real user query logs** instead of self-referential synthetic queries
  -- the single biggest limitation, and the reason the task is currently
  too easy to show real separation.
- **Title-derived queries** in the training set (e.g. also generate
  positive pairs using a patent's own title as the query) so the
  `title_phrase` feature actually has evidence to learn from, instead of
  correctly-but-uninformatively landing on 0.
- **A larger, more diverse negative set** -- right now hard negatives
  only come from the same classification prefix; at real scale, mining
  genuinely confusable negatives (e.g. via a first-pass embedding search)
  would produce a harder, more realistic evaluation.

## How to Reproduce

```bash
cd thinkstruct
python3 -c "
from src.data_loader import load_patents
from src.search_engine import build_index
from part3.train_reranker import build_training_pairs, build_feature_matrix, run_baselines, train_and_evaluate_reranker

patents = load_patents('data')
vectorizer, tfidf_matrix = build_index(patents)
pairs = build_training_pairs(patents)
X, y, pair_types = build_feature_matrix(pairs, patents, vectorizer, tfidf_matrix)

print(run_baselines(X, y))
print(train_and_evaluate_reranker(X, y))
"
```
