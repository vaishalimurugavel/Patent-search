# Scaling the Patent Search Engine to 10M Patents

This document describes how the Part 1 search engine (TF-IDF over a local
JSON dataset of ~20 patents) would need to change to handle the full patent
corpus: ~10 million patents, with new filings arriving continuously. It's
meant to be simple enough for an intern to read and start implementing, not
a final, polished architecture.

## 1. System Components

| Component | Technology | Purpose |
|---|---|---|
| Raw data landing zone | Object storage (S3) | Durable, cheap storage of raw patent JSON/XML as it arrives, before any processing. Source of truth for "what did we originally receive." |
| Ingestion queue | Kafka | Decouples "a new patent arrived" from "a patent has been processed." Lets ingestion workers scale independently and lets us replay/reprocess if something downstream breaks. |
| Metadata store | Postgres | System of record for structured patent metadata (doc_number, title, classification, filing date, bibtex, ingestion status). Relational, transactional, easy to query/join/audit. |
| Search index | OpenSearch | Text search (BM25, same idea as Part 1's TF-IDF but built for scale) AND vector search (k-NN plugin) for semantic similarity, combined in one system. Stores a denormalized copy of the fields needed for search/filtering (title, abstract, claims text, classification, embedding vector) so queries never need to join across Postgres and OpenSearch at request time. |
| Embedding service | Batch worker pool (GPU-backed) | Generates embeddings for title+abstract+claims text, consumed by ingestion workers before indexing into OpenSearch. |
| Cache | Redis | Caches results for repeated/hot queries (e.g. common classification-code browsing) in front of the API layer. |
| API layer | Flask/FastAPI | Exposes the actual search endpoints (natural language query, classification filter, title lookup, hybrid combinations) -- this is the productionized version of Part 1's `search()`/`hybrid_search()`/`search_by_title()`. |
| Monitoring | Prometheus + Grafana, structured logs | Tracks ingestion throughput, queue lag, error rates, search latency. |

## 2. Major Pipelines

### 2.1 Ingestion pipeline (new/updated patents)

```
New patent file arrives
   -> written to S3 (raw landing zone)
   -> a small "watcher" job publishes one Kafka message per patent
      to topic `patents.raw`
   -> a pool of ingestion workers (horizontally scalable, one Kafka
      consumer group) each:
        1. reads a message
        2. parses + normalizes the patent record (same logic as
           Part 1's normalize_patent -- missing fields get safe
           defaults, nothing crashes)
        3. upserts structured metadata into Postgres (by doc_number)
        4. calls the embedding service to get a vector for the
           search text (title + abstract + claims, same
           representation as Part 1)
        5. bulk-indexes the document (text fields + embedding vector
           + classification) into OpenSearch
        6. writes a row to an `ingestion_jobs` table in Postgres
           recording success/failure
   -> on failure at any step, the message is retried a few times,
      then routed to a dead-letter topic (`patents.raw.dlq`) instead
      of being dropped, so nothing silently disappears
```

This mirrors Part 1's data flow (load -> normalize -> index) but each step
is now a separate, independently scalable, fault-isolated stage instead of
one Python script run start to finish.

### 2.2 Query pipeline (a user searching)

```
User query (natural language, optionally with classification/title
filters, same shape as Part 1's search()/hybrid_search())
   -> API layer
   -> check Redis cache (skip straight to response on a hit)
   -> on a miss:
        - filters (classification prefix, title) become an OpenSearch
          filter clause -- OpenSearch applies these BEFORE scoring,
          same "filter first, then only score survivors" idea as
          Part 1's hybrid_search(), just implemented by OpenSearch's
          query engine instead of a Python list comprehension + numpy
        - text query becomes a BM25 clause; if semantic search is
          enabled, the query is also embedded and combined with a
          k-NN clause (this is where Part 3's "two-phase search"
          enhancement would plug in, as a re-ranking step after this
          initial retrieval)
   -> results returned, written to Redis cache for next time
```

### 2.3 Backfill pipeline (the initial 10M-patent bulk load)

Loading 10 million existing patents can't go through the same
one-message-per-Kafka-event path efficiently -- publishing 10 million
individual Kafka messages and having workers process them one at a time
would take a very long time. Instead:

- Read source data in large batches (e.g. 10,000 patents at a time)
- Use Postgres bulk `COPY` and OpenSearch's bulk indexing API instead of
  one-row/one-document-at-a-time inserts
- Generate embeddings in batches on GPU workers (batching is what makes
  embedding generation fast -- doing it one patent at a time wastes most
  of the GPU's capacity)
- Run this as a one-time batch job, separate from (but reusing the same
  parsing/normalization code as) the steady-state Kafka pipeline

## 3. Cost Breakdown (rough, illustrative -- not a vendor quote)

Assumptions: 10M patents, average patent text (title+abstract+claims)
~2-4 KB, embedding dimension 384 (a small/efficient sentence-transformer
model, not a huge one, to keep cost down at this scale).

| Item | Rough estimate | Notes |
|---|---|---|
| Postgres storage | ~10-20 GB | Metadata only (no full text), scales linearly, cheap |
| OpenSearch storage | ~150-300 GB | Full text index + embeddings (10M x 384 floats x 4 bytes = ~15 GB for vectors alone; BM25 inverted index + OpenSearch's own overhead is the bigger chunk) |
| OpenSearch cluster (managed, e.g. AWS OpenSearch Service) | ~$1,500-3,000/month | For a cluster sized to hold the above with room for query load; the single biggest recurring cost in this design |
| Embedding generation (one-time backfill) | ~$500-2,000 one-time | GPU batch job processing 10M patents; highly dependent on model size and GPU choice, and is a one-time cost, not recurring |
| Kafka (managed, e.g. AWS MSK) | ~$300-600/month | Sized for steady-state weekly filing volume, not backfill volume |
| Postgres (managed, e.g. RDS) | ~$100-300/month | Small compared to search infra since it's metadata-only |
| Redis cache | ~$50-150/month | Small instance, mostly for hot-query latency, not a hard requirement |

**Bottom line: OpenSearch is the dominant recurring cost**, because storing
and querying both full text and vectors at 10M-document scale is
inherently storage- and compute-heavy. This is the direct tradeoff of the
"combine text + vector search in one system" decision made in Section 1:
it's operationally simpler (one less system to run) but that system ends
up doing more work and costing more than either piece would alone.

## 4. Error Handling

- **Ingestion failures** (malformed JSON, missing required fields beyond
  what normalization can safely default, embedding service timeout):
  retried a fixed number of times, then routed to a Kafka dead-letter
  topic. Nothing is silently dropped -- a failed patent is always
  visible somewhere (the DLQ + the `ingestion_jobs` table), for someone
  to inspect and reprocess later.
- **Partial failures within one patent** (e.g. metadata write to Postgres
  succeeds but OpenSearch indexing fails): the `ingestion_jobs` table
  tracks per-step status, so a patent can be retried from the specific
  step that failed rather than redone from scratch.
- **Downstream consistency drift**: since Postgres and OpenSearch are
  separate systems holding overlapping data, they can drift out of sync
  (e.g. a patent updated in Postgres but the OpenSearch reindex failed).
  A periodic reconciliation job compares row counts / checksums between
  the two and re-syncs anything that's out of alignment.

## 5. Tracking Contents and Status

- The `ingestion_jobs` table (Postgres) is the single place to answer
  "what's the status of patent X" or "how many patents failed today" --
  columns: `doc_number`, `source_file`, `status` (pending/processing/
  success/failed), `attempts`, `last_error`, `updated_at`.
- A simple status dashboard (this is what the Part 2 proof-of-concept
  demonstrates a minimal version of) queries this table to show
  ingestion throughput and failure rates over time.
- Grafana dashboards on top of Prometheus metrics track Kafka consumer
  lag (are workers keeping up with incoming volume?), OpenSearch query
  latency (p50/p95), and error rates -- the operational, "is the system
  healthy right now" view, as opposed to the `ingestion_jobs` table's
  "what happened to this specific patent" view.

## 6. Major Challenges at Scale

- **Initial backfill volume.** Loading 10M patents through a per-message
  Kafka pipeline designed for steady-state weekly filings would be far
  too slow -- this is why Section 2.3 describes a separate batch backfill
  path reusing the same parsing code but bulk-loading instead.
- **Embedding generation throughput.** Generating embeddings for 10M
  documents is a real, non-trivial compute cost and takes real wall-clock
  time even with GPU batching -- this needs to be planned as its own
  project phase, not treated as a quick step in the pipeline.
- **Two-system consistency.** Postgres and OpenSearch holding overlapping
  data is the direct cost of the architecture choice in Section 1
  (denormalizing search-relevant fields into OpenSearch instead of
  joining at query time) -- worth acknowledging honestly rather than
  pretending it's free.
- **OpenSearch cost and shard management.** At 10M+ documents with
  vectors, index size and shard count become real operational concerns
  (too few shards limits parallelism, too many adds overhead) -- this
  needs load testing and tuning that's out of scope for this writeup but
  should be flagged as a known follow-up.
- **Query latency for hybrid queries.** Combining a classification-code
  filter, a BM25 text clause, and a k-NN vector clause in one query is
  more expensive than any one of those alone -- Part 1's benchmark
  already showed that "filter first, then only score survivors" only
  wins once the corpus is large enough; the same principle applies here,
  just at a much bigger scale where the crossover point matters a lot
  more.

## 7. What This Deliberately Doesn't Solve

Per the assignment's own framing, this is meant to be a simple system with
acknowledged gaps, not a finished one. Left out on purpose:

- Multi-region / disaster recovery design
- Detailed OpenSearch shard/replica sizing (needs real load testing)
- Authentication/authorization for the API layer
- Handling of patent *amendments* or *withdrawals* (this design assumes
  patents are append-only, which may not hold in reality)
