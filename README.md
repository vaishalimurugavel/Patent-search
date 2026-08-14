# Thinkstruct Patent Search Take-Home

A search engine over vehicle-related patent applications (2024-present),
built in three parts: a local hybrid search engine (Part 1), a design for
running it at 10M-patent scale (Part 2), and two enhancements -- two-phase
search and a learned reranker (Part 3).

## Problem Statement

**Input**: a natural language query (e.g. "carbon fiber bicycle wheel
spoke"), with optional structured filters (classification code prefix,
title keyword/exact match).

**Output**: a ranked list of similar patents, each with its title, doc
number, similarity score, and a brief descriptive snippet (the patent's
abstract, or a fallback to its first claim/description paragraph if the
abstract is missing).

Matching is done against each patent's **title + abstract + claims**
combined -- the broadest, most thorough text representation available,
chosen so the same representation could later extend to other input types
(e.g. matching a whole patent or a specific claim against the corpus) with
minimal rework.

## Demo Video

<video src="video_tutorial/VaishaliMurugavel_ThinkstructDemo.mp4" controls width="700">
  Your browser does not support embedded video. Watch it directly here:
  <a href="video_tutorial/VaishaliMurugavel_ThinkstructDemo.mp4">VaishaliMurugavel_ThinkstructDemo.mp4</a>
</video>

*(GitHub's video embed depends on file size/format -- if it doesn't render
inline above, click directly:
[video_tutorial/VaishaliMurugavel_ThinkstructDemo.mp4](video_tutorial/VaishaliMurugavel_ThinkstructDemo.mp4))*

## Project Structure

```
thinkstruct/
├── data/                          Production data -- drop your full
│                                  patents_ipa*.json dataset here (however
│                                  deeply nested; loader searches recursively)
├── src/
│   ├── data_loader.py             Loading, missing-field normalization, snippet fallback
│   └── search_engine.py           TF-IDF search, hybrid filtering, title lookup, benchmarksupport
├── main.py                        Interactive CLI -- the actual runnable entry point
├── benchmark.py                   Hybrid vs. naive search timing comparison
├── tests/                         33 tests across data_loader.py and search_engine.py
│   └── fixtures/                  Small, fixed dataset used ONLY by tests (decoupled from data/)
├── part2/
│   ├── system_design.md           Scaling design: components, pipelines, cost, error handling
│   ├── schema.sql                 Postgres DDL (patents + ingestion_jobs tables)
│   ├── ingest.py                  Real Postgres ingestion script (psycopg2)
│   └── validate_ingest_logic.py   SQLite-based logic validation (not part of the design itself)
└── part3/
    ├── two_phase_search.py        Enhancement 1: TF-IDF retrieval + feature-based rerank
    ├── train_reranker.py           Enhancement 2: learn rerank weights via logistic regression
    └── eval_training_report.md    Methodology, results, and honest limitations for enhancement 2
```

## Setup

```bash
pip install scikit-learn numpy scipy
```

(Everything in Parts 1 and 3 runs on just these -- no GPU or extra
model downloads required.)

Place your full patent dataset (all `patents_ipa*.json` files, however
deeply nested) inside `data/`.

## Part 1: Running the Search Engine

```bash
python3 main.py
```

At the `Search>` prompt:
- Plain text, e.g. `wheel bearing` -> TF-IDF ranked results
- `title:<exact title>` -> exact title lookup (O(1), handles duplicate titles)
- `rerank:<query>` -> two-phase search (TF-IDF + feature-based rerank, Part 3 enhancement)
- `hybrid:<query>|class=<prefix>|title=<keyword>` -> hybrid search with metadata
  filters; either filter is optional (e.g. `hybrid:wheel bearing|class=B60B`)
- `quit` / `exit` -> stop

**Benchmark** (hybrid vs. naive hybrid search timing):
```bash
python3 benchmark.py
```

**Tests**:
```bash
python3 tests/test_data_loader.py
python3 tests/test_search_engine.py
python3 tests/test_two_phase_search.py
```
(Plain `assert`-based functions -- runnable directly with no dependencies,
or with `pytest tests/ -v` for nicer output if you have pytest installed.)

## Key Design Decisions

- **Missing fields**: every patent is normalized to have all expected
  fields (empty string/list defaults), so no patent is ever dropped from
  the searchable set just because one field is missing. Missing abstract
  falls back to the first claim, then first description paragraph, then a
  placeholder string, for the display snippet.
- **Hybrid search efficiency**: `hybrid_search()` filters by metadata
  *first*, then only computes TF-IDF similarity on the surviving subset --
  as opposed to `naive_hybrid_search()`, which scores everything first and
  filters after. **Measured against a real 620-patent dataset, the result
  was mixed** -- hybrid won on broad classification and title filters
  (1.2-1.5x faster) but lost on no-filter, narrow classification, and
  combined filters (0.7-0.8x, i.e. slower). Root cause: `filter_candidates()`
  is itself an O(n) Python loop that scans every patent regardless of how
  selective the filter ends up being -- a narrow filter only makes the
  *subsequent* similarity scoring cheaper, not the filtering step itself,
  and scipy's vectorized cosine similarity is already fast enough at this
  scale that the Python-loop overhead can outweigh the scoring savings.
  The actual fix -- a sorted array + binary search (bisect) for O(log n)
  classification lookups instead of O(n) -- was deliberately left
  undone and documented here rather than implemented, since demonstrating
  an accurate understanding of *why* the current approach doesn't
  consistently win was judged more valuable than an untested optimization
  built under time pressure. See `benchmark.py`'s output for the full
  numbers and commentary.
- **Case-insensitive classification filtering**, **negative `top_k`
  treated as "no limit"** -- both were real bugs found via deliberate edge
  case testing and are covered by regression tests.
- **Title index** returns a list of matches per title, not a single
  match -- the sample dataset actually contains duplicate titles (e.g.
  "PNEUMATIC TIRE" appears 3 times), confirmed via testing, not assumed.

## Part 2: Scaling to 10M Patents

See `part2/system_design.md` for the full writeup (components, pipelines,
cost breakdown, error handling, challenges at scale).

Proof-of-concept: a real Postgres schema + ingestion script.

```bash
pip install psycopg2-binary
createdb thinkstruct_poc
psql -U <user> -d thinkstruct_poc -f part2/schema.sql
python3 part2/ingest.py --dsn "dbname=thinkstruct_poc user=<youruser>" --data-dir data
```

Note: this requires a real running Postgres instance to actually execute.
The data-handling logic (parsing, upsert construction, error handling) was
validated separately against SQLite (`part2/validate_ingest_logic.py`),
since Postgres wasn't available in the environment this was developed in
-- but the live Postgres connection itself should be verified by whoever
runs this.

## Part 3: Enhancements

**1. Two-phase search** (`part3/two_phase_search.py`): Phase 1 reuses
Part 1's TF-IDF search to cheaply retrieve a broad candidate set. Phase 2
reranks just those candidates using additional lexical features (exact
phrase match in title/abstract, raw term overlap) that TF-IDF alone
doesn't directly capture. Chosen over a neural cross-encoder because it
needs no extra dependencies (no `sentence-transformers`/`torch`, no
network access to download a model) while still demonstrating the same
"broad retrieval then refined rerank" architecture the brief describes.

Reachable interactively via `rerank:<query>` in `main.py`, or directly:
```python
from src.data_loader import load_patents
from src.search_engine import build_index
from part3.two_phase_search import two_phase_search

patents = load_patents("data")
vectorizer, tfidf_matrix = build_index(patents)
results = two_phase_search("protection device wheel vehicle", patents, vectorizer, tfidf_matrix,
                            phase1_k=100, final_k=5)
```

**2. Evaluation & Training** (`part3/train_reranker.py`,
`part3/eval_training_report.md`): rather than hand-picking the weights
used to combine the four reranking features, this generates labeled
positive/hard-negative/random-negative training pairs from the corpus
itself, evaluates via 5-fold cross-validated AUC-ROC, and trains a
logistic regression model to learn the weights instead. Full methodology,
results, and -- importantly -- an honest discussion of the training data's
limitations are in `eval_training_report.md`.

Reproduce the evaluation/training results directly:
```bash
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

Both enhancements were picked together (rather than one) because they
build on each other: the second literally learns the weights the first
enhancement had to guess at.

## Known Limitations / Honest Gaps

- The Evaluation & Training pipeline's synthetic training data is
  self-referential (queries are extracted from the same patent they're
  labeled positive for), which makes the task easier than a real user
  query would be -- discussed honestly in `eval_training_report.md` rather
  than glossed over.
- Patent-ID-based scraping (mentioned as an optional Part 1 extension in
  the brief) was not built -- out of scope given time spent on the core
  requirements and both Part 3 enhancements.