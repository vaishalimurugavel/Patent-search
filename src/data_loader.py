from pathlib import Path
import json

# Every patent dict is guaranteed to have these keys after normalize_patent()
# runs, even if the source JSON left them out. String fields default to ""
# and list fields default to [] so downstream code never has to check
# "does this key exist" — it can just use the value.
STRING_FIELDS = ["title", "doc_number", "filename", "abstract", "bibtex", "classification"]
LIST_FIELDS = ["detailed_description", "claims"]


def normalize_patent(patent):
    """
    Return a copy of patent guaranteed to have every expected field,
    filling in "" or [] for anything missing.
    """
    normalized = dict(patent)  # copy, so we don't mutate the original

    for field in STRING_FIELDS:
        if not normalized.get(field):
            normalized[field] = ""

    for field in LIST_FIELDS:
        if not normalized.get(field):
            normalized[field] = []

    return normalized


def get_snippet(patent):
    """
    Return a brief paragraph describing this patent for display in search
    results. Prefers the abstract; if it's missing, falls back to the
    first claim, then the first paragraph of the detailed description,
    then a final fallback string.
    """
    if patent["abstract"]:
        return patent["abstract"]

    if patent["claims"]:
        return patent["claims"][0]

    if patent["detailed_description"]:
        return patent["detailed_description"][0]

    return "(no description available for this patent)"


def load_patents(data_dir):
    """
    Read every patents_ipa*.json file in data_dir and return one flat
    list of normalized patent dicts.
    """
    data_dir = Path(data_dir)
    patents = []

    for filepath in sorted(data_dir.rglob("patents_ipa*.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            file_patents = json.load(f)
        patents.extend(normalize_patent(p) for p in file_patents)

    return patents


if __name__ == "__main__":
    patents = load_patents("data")
    print(f"Loaded {len(patents)} patents")
    print("First title:", patents[0]["title"])
