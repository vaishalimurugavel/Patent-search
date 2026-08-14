-- schema.sql
-- Postgres schema for the Part 2 proof-of-concept: metadata storage +
-- ingestion status tracking, matching the design in system_design.md
-- (Section 1: "Metadata store" and Section 5: "Tracking Contents and Status").
--
-- Run with: psql -U <user> -d <database> -f schema.sql

CREATE TABLE IF NOT EXISTS patents (
    doc_number      TEXT PRIMARY KEY,
    title           TEXT NOT NULL DEFAULT '',
    abstract        TEXT NOT NULL DEFAULT '',
    classification  TEXT NOT NULL DEFAULT '',
    bibtex          TEXT NOT NULL DEFAULT '',
    -- claims and detailed_description are lists in the source JSON.
    -- Stored as JSONB here rather than normalized into separate tables --
    -- this is a metadata store, not the search index (that's OpenSearch's
    -- job per the design doc), so we don't need to query into individual
    -- claims relationally. JSONB lets us keep the original list structure
    -- without a join.
    claims                JSONB NOT NULL DEFAULT '[]',
    detailed_description  JSONB NOT NULL DEFAULT '[]',
    source_file     TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Supports the classification-prefix filtering pattern from Part 1
-- (e.g. WHERE classification LIKE 'B60B%') at Postgres-metadata scale.
CREATE INDEX IF NOT EXISTS idx_patents_classification ON patents (classification);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id              SERIAL PRIMARY KEY,
    doc_number      TEXT NOT NULL,
    source_file     TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'success', 'failed')),
    attempts        INT NOT NULL DEFAULT 0,
    last_error      TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One job row per doc_number: re-ingesting the same patent updates its
-- existing job row (via upsert in ingest.py) rather than growing a new
-- row every time, so this table always reflects CURRENT status per
-- patent, not a full history log.
CREATE UNIQUE INDEX IF NOT EXISTS idx_ingestion_jobs_doc_number ON ingestion_jobs (doc_number);

-- Powers "how many patents failed" / status-dashboard style queries.
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status ON ingestion_jobs (status);
