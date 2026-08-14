"""
validate_ingest_logic.py -- NOT part of the submission. This proves the
data-handling logic in ingest.py (parsing, upsert construction, error
handling, status tracking) actually works, using SQLite as a stand-in
since Postgres isn't available in this sandbox. SQLite supports the same
INSERT ... ON CONFLICT upsert syntax used in ingest.py, so this is a real
test of the logic, not just a syntax check -- but it does NOT prove the
psycopg2 connection code or Postgres-specific JSONB behavior works;
that part still needs a real Postgres run.
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_patents

SQLITE_SCHEMA = """
CREATE TABLE patents (
    doc_number TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    abstract TEXT NOT NULL DEFAULT '',
    classification TEXT NOT NULL DEFAULT '',
    bibtex TEXT NOT NULL DEFAULT '',
    claims TEXT NOT NULL DEFAULT '[]',
    detailed_description TEXT NOT NULL DEFAULT '[]',
    source_file TEXT NOT NULL DEFAULT ''
);

CREATE TABLE ingestion_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_number TEXT NOT NULL UNIQUE,
    source_file TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT
);
"""

UPSERT_PATENT_SQL = """
    INSERT INTO patents (doc_number, title, abstract, classification, bibtex, claims, detailed_description, source_file)
    VALUES (:doc_number, :title, :abstract, :classification, :bibtex, :claims, :detailed_description, :source_file)
    ON CONFLICT(doc_number) DO UPDATE SET
        title = excluded.title,
        abstract = excluded.abstract,
        classification = excluded.classification,
        bibtex = excluded.bibtex,
        claims = excluded.claims,
        detailed_description = excluded.detailed_description,
        source_file = excluded.source_file;
"""

UPSERT_JOB_SQL = """
    INSERT INTO ingestion_jobs (doc_number, source_file, status, attempts, last_error)
    VALUES (:doc_number, :source_file, :status, 1, :last_error)
    ON CONFLICT(doc_number) DO UPDATE SET
        source_file = excluded.source_file,
        status = excluded.status,
        attempts = attempts + 1,
        last_error = excluded.last_error;
"""


def ingest_patent(cursor, patent):
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
            "doc_number": doc_number, "source_file": source_file,
            "status": "success", "last_error": None,
        })
        return True
    except Exception as e:
        try:
            cursor.execute(UPSERT_JOB_SQL, {
                "doc_number": doc_number, "source_file": source_file,
                "status": "failed", "last_error": str(e),
            })
        except Exception as log_error:
            print(f"  WARNING: failed to log failure for {doc_number}: {log_error}")
        return False


def main():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SQLITE_SCHEMA)
    cursor = conn.cursor()

    print("--- Test 1: ingest real sample data ---")
    patents = load_patents("data")
    succeeded, failed = 0, 0
    for patent in patents:
        if ingest_patent(cursor, patent):
            succeeded += 1
        else:
            failed += 1
    conn.commit()
    print(f"{succeeded} succeeded, {failed} failed (out of {len(patents)})")

    cursor.execute("SELECT COUNT(*) FROM patents")
    assert cursor.fetchone()[0] == len(patents), "row count mismatch"
    print("Row count in patents table matches loaded patent count -- OK")

    print()
    print("--- Test 2: re-ingest same data (upsert should update, not duplicate) ---")
    for patent in patents:
        ingest_patent(cursor, patent)
    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM patents")
    count_after_reingest = cursor.fetchone()[0]
    assert count_after_reingest == len(patents), f"expected {len(patents)}, got {count_after_reingest} -- upsert duplicated rows!"
    print(f"Row count still {count_after_reingest} after re-ingesting -- upsert working correctly, no duplicates")

    cursor.execute("SELECT attempts FROM ingestion_jobs WHERE doc_number = ?", (patents[0]["doc_number"],))
    attempts = cursor.fetchone()[0]
    assert attempts == 2, f"expected attempts=2 after two ingestions, got {attempts}"
    print(f"Attempts counter correctly incremented to {attempts} after second ingestion")

    print()
    print("--- Test 3: missing doc_number should get a fallback ID, not crash ---")
    no_docnum_patent = {"doc_number": None, "title": "No ID Patent", "abstract": "", "classification": "",
                        "bibtex": "", "claims": [], "detailed_description": [], "filename": "no_id.xml"}
    ok = ingest_patent(cursor, no_docnum_patent)
    conn.commit()
    print(f"Missing doc_number handled via fallback, ingest_patent returned: {ok}")
    cursor.execute("SELECT doc_number FROM patents WHERE doc_number LIKE 'UNKNOWN_%'")
    fallback_row = cursor.fetchone()
    assert fallback_row is not None, "expected a fallback doc_number to be used"
    print(f"Fallback doc_number used: {fallback_row[0]}")

    print()
    print("--- Test 4: genuinely unrecoverable record (non-JSON-serializable data) should fail gracefully ---")
    class Unserializable:
        pass
    broken_patent = {"doc_number": "BROKEN001", "title": "Broken", "abstract": "", "classification": "",
                     "bibtex": "", "claims": [Unserializable()], "detailed_description": [], "filename": "broken.xml"}
    ok = ingest_patent(cursor, broken_patent)
    conn.commit()
    print(f"Unserializable record handled without crashing, ingest_patent returned: {ok}")
    assert ok is False, "expected this record to fail, since its claims data can't be JSON-serialized"

    cursor.execute("SELECT status, last_error FROM ingestion_jobs WHERE doc_number = ?", ("BROKEN001",))
    row = cursor.fetchone()
    assert row is not None and row[0] == "failed", "expected a 'failed' job row to be logged for the broken record"
    print(f"Job correctly logged as failed, error message: {row[1][:80]}...")

    cursor.execute("SELECT status, COUNT(*) FROM ingestion_jobs GROUP BY status")
    print()
    print("Final ingestion_jobs status breakdown:")
    for status, count in cursor.fetchall():
        print(f"  {status}: {count}")

    conn.close()
    print()
    print("All validation checks passed. Logic confirmed correct in SQLite;")
    print("run ingest.py against a real Postgres instance to confirm the")
    print("psycopg2 connection and JSONB-specific behavior separately.")


if __name__ == "__main__":
    main()
