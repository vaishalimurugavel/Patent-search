"""
ingest.py -- loads patent JSON files into Postgres, tracking per-patent
ingestion status in the ingestion_jobs table.

This is the proof-of-concept for the ingestion pipeline described in
system_design.md (Section 2.1). It's intentionally simple: single-threaded,
one connection, no Kafka -- it stands in for "one ingestion worker
processing messages," just without the queue in front of it.

Usage:
    pip install psycopg2-binary
    createdb thinkstruct_poc          # or any database name
    psql -U <user> -d thinkstruct_poc -f schema.sql
    python3 ingest.py --dsn "dbname=thinkstruct_poc user=<user>" --data-dir ../data

NOTE: this requires a real running Postgres instance. It has NOT been
run against a live Postgres server in the environment this was written
in (no Postgres available there) -- the data-handling logic (parsing,
normalization, upsert construction, status tracking) was validated
separately against SQLite, which supports the same INSERT ... ON
CONFLICT upsert syntax used here. Please run this against your own
Postgres instance before considering it verified end-to-end.
"""
import argparse
import json
import sys
from pathlib import Path

# Reuse Part 1's loading/normalization logic -- same rules about missing
# fields apply here as they did in the local search engine.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_patents


def get_connection(dsn):
    import psycopg2
    return psycopg2.connect(dsn)


UPSERT_PATENT_SQL = """
    INSERT INTO patents (doc_number, title, abstract, classification, bibtex,
                          claims, detailed_description, source_file, updated_at)
    VALUES (%(doc_number)s, %(title)s, %(abstract)s, %(classification)s, %(bibtex)s,
            %(claims)s, %(detailed_description)s, %(source_file)s, now())
    ON CONFLICT (doc_number) DO UPDATE SET
        title = EXCLUDED.title,
        abstract = EXCLUDED.abstract,
        classification = EXCLUDED.classification,
        bibtex = EXCLUDED.bibtex,
        claims = EXCLUDED.claims,
        detailed_description = EXCLUDED.detailed_description,
        source_file = EXCLUDED.source_file,
        updated_at = now();
"""

UPSERT_JOB_SQL = """
    INSERT INTO ingestion_jobs (doc_number, source_file, status, attempts, last_error, updated_at)
    VALUES (%(doc_number)s, %(source_file)s, %(status)s, 1, %(last_error)s, now())
    ON CONFLICT (doc_number) DO UPDATE SET
        source_file = EXCLUDED.source_file,
        status = EXCLUDED.status,
        attempts = ingestion_jobs.attempts + 1,
        last_error = EXCLUDED.last_error,
        updated_at = now();
"""


def ingest_patent(cursor, patent):
    """
    Insert/update one patent's metadata and record the ingestion job
    outcome. Returns True on success, False on failure -- never raises,
    since one bad patent shouldn't stop the whole batch (mirrors the
    "one message failure doesn't take down the pipeline" principle from
    the design doc).
    """
    # doc_number is our primary key and the ingestion_jobs join key --
    # if it's missing or None, we can't use it as-is (NOT NULL constraint
    # on both tables), but we still need to log a job record so the
    # patent isn't silently dropped. Fall back to a value built from the
    # source filename so at least it's traceable back to *something*.
    doc_number = patent.get("doc_number") or f"UNKNOWN_{patent.get('filename', 'no_filename')}"
    source_file = patent.get("filename") or "unknown"

    try:
        params = {
            "doc_number": doc_number,
            "title": patent.get("title", ""),
            "abstract": patent.get("abstract", ""),
            "classification": patent.get("classification", ""),
            "bibtex": patent.get("bibtex", ""),
            "claims": json.dumps(patent.get("claims", [])),
            "detailed_description": json.dumps(patent.get("detailed_description", [])),
            "source_file": source_file,
        }
        cursor.execute(UPSERT_PATENT_SQL, params)
        cursor.execute(UPSERT_JOB_SQL, {
            "doc_number": doc_number,
            "source_file": source_file,
            "status": "success",
            "last_error": None,
        })
        return True
    except Exception as e:
        # This failure-logging path itself must not raise -- if it does,
        # one malformed patent would crash the whole batch, which is
        # worse than the original failure it was trying to record.
        try:
            cursor.execute(UPSERT_JOB_SQL, {
                "doc_number": doc_number,
                "source_file": source_file,
                "status": "failed",
                "last_error": str(e),
            })
        except Exception as log_error:
            print(f"  WARNING: failed to log failure for {doc_number}: {log_error}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True, help='e.g. "dbname=thinkstruct_poc user=youruser"')
    parser.add_argument("--data-dir", required=True, help="folder containing patents_ipa*.json files")
    args = parser.parse_args()

    print(f"Loading patents from {args.data_dir} ...")
    patents = load_patents(args.data_dir)
    print(f"Loaded {len(patents)} patents")

    conn = get_connection(args.dsn)
    conn.autocommit = False
    cursor = conn.cursor()

    succeeded, failed = 0, 0
    for patent in patents:
        if ingest_patent(cursor, patent):
            succeeded += 1
        else:
            failed += 1

        conn.commit()  # commit per-patent so one failure doesn't roll back prior successes

    cursor.close()
    conn.close()

    print(f"Done. {succeeded} succeeded, {failed} failed.")
    print("Check the ingestion_jobs table for per-patent status, e.g.:")
    print("    SELECT status, COUNT(*) FROM ingestion_jobs GROUP BY status;")


if __name__ == "__main__":
    main()
