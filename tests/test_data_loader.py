"""
Pytest-style tests for data_loader.py.

Run with: pytest tests/test_data_loader.py -v
(requires `pip install pytest` -- these are plain assert-based functions,
so they also work if you just import and call them directly)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import normalize_patent, get_snippet, load_patents

# Resolve the data directory relative to THIS FILE's location, not the
# process's working directory. Without this, running the test from a
# different working directory (e.g. an IDE launching it from tests/
# instead of the project root) silently finds zero files and every
# test using real data breaks -- as opposed to failing loudly with a
# clear "wrong path" message.
# Resolve relative to THIS FILE's location, not the working directory
# (see earlier fix). Also note this points at tests/fixtures/, NOT the
# project-root data/ folder -- tests should run against a small, fixed,
# known dataset that doesn't change when the real production dataset
# (all 50+ files) gets dropped into data/. Otherwise tests that assume
# specific rankings (e.g. "SPOKE ranks #1 for this query") could break
# for reasons that have nothing to do with an actual bug.
DATA_DIR = Path(__file__).resolve().parent / "fixtures"


# ---- normalize_patent tests ----

def test_normalize_fills_missing_string_fields():
    patent = {"title": "Test Widget"}
    result = normalize_patent(patent)
    assert result["title"] == "Test Widget"
    assert result["abstract"] == ""
    assert result["doc_number"] == ""
    assert result["classification"] == ""


def test_normalize_fills_missing_list_fields():
    patent = {"title": "Test Widget"}
    result = normalize_patent(patent)
    assert result["claims"] == []
    assert result["detailed_description"] == []


def test_normalize_handles_explicit_none_values():
    # real-world messy data sometimes has "abstract": null rather than
    # omitting the key entirely -- normalize_patent should catch this too
    patent = {"title": "Test", "abstract": None, "claims": None}
    result = normalize_patent(patent)
    assert result["abstract"] == ""
    assert result["claims"] == []


def test_normalize_preserves_original_dict():
    # normalize_patent should not mutate the input
    original = {"title": "Test"}
    normalize_patent(original)
    assert "abstract" not in original


def test_normalize_keeps_full_data_when_present():
    patent = {
        "title": "Full Patent",
        "abstract": "An abstract.",
        "claims": ["claim 1", "claim 2"],
        "classification": "B60B104FI",
    }
    result = normalize_patent(patent)
    assert result["title"] == "Full Patent"
    assert result["abstract"] == "An abstract."
    assert result["claims"] == ["claim 1", "claim 2"]
    assert result["classification"] == "B60B104FI"


# ---- get_snippet tests ----

def test_snippet_prefers_abstract():
    patent = normalize_patent({
        "abstract": "The abstract text.",
        "claims": ["A claim."],
        "detailed_description": ["A paragraph."],
    })
    assert get_snippet(patent) == "The abstract text."


def test_snippet_falls_back_to_first_claim():
    patent = normalize_patent({
        "claims": ["First claim.", "Second claim."],
        "detailed_description": ["A paragraph."],
    })
    assert get_snippet(patent) == "First claim."


def test_snippet_falls_back_to_first_description_paragraph():
    patent = normalize_patent({
        "detailed_description": ["First paragraph.", "Second paragraph."],
    })
    assert get_snippet(patent) == "First paragraph."


def test_snippet_final_fallback_when_nothing_available():
    patent = normalize_patent({"title": "Bare Patent"})
    assert get_snippet(patent) == "(no description available for this patent)"


def test_snippet_treats_empty_string_abstract_as_missing():
    # abstract present but empty -- should still fall through, not
    # return an empty string as the "snippet"
    patent = normalize_patent({"abstract": "", "claims": ["A claim."]})
    assert get_snippet(patent) == "A claim."


# ---- load_patents tests ----

def test_load_patents_returns_nonempty_list_from_real_data():
    patents = load_patents(DATA_DIR)
    assert isinstance(patents, list)
    assert len(patents) > 0


def test_load_patents_every_record_is_normalized():
    patents = load_patents(DATA_DIR)
    required_string_fields = ["title", "doc_number", "filename", "abstract", "bibtex", "classification"]
    required_list_fields = ["detailed_description", "claims"]
    for patent in patents:
        for field in required_string_fields:
            assert field in patent
        for field in required_list_fields:
            assert field in patent
            assert isinstance(patent[field], list)


def test_load_patents_finds_files_in_nested_directories():
    # regression test for the nested-zip-folder issue: rglob must find
    # files several directories deep, not just in the top-level folder
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as tmp:
        nested = Path(tmp) / "outer" / "middle" / "inner"
        nested.mkdir(parents=True)
        for f in DATA_DIR.glob("patents_ipa*.json"):
            shutil.copy(f, nested / f.name)

        patents = load_patents(tmp)
        assert len(patents) == len(load_patents(DATA_DIR))


def test_load_patents_returns_empty_list_for_directory_with_no_matches():
    # Decision: fail quiet. A directory with no patents_ipa*.json files
    # should return [] rather than raise -- caller is responsible for
    # checking len(result) if they want to detect a misconfigured path.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        result = load_patents(tmp)
        assert result == []


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
