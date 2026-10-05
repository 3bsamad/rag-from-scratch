import csv
import json
from pathlib import Path


def load_data(path):
    """Load corpus, queries, or qrels into dictionaries keyed by IDs."""
    path = Path(path)
    data = {}
    with path.open(encoding="utf-8", newline="") as file:
        if path.suffix == ".jsonl":
            for line in file:
                if not line.strip():
                    continue
                item = json.loads(line)
                data[item["_id"]] = (
                    {"title": item.get("title", ""), "text": item["text"]}
                    if path.name == "corpus.jsonl" else item["text"]
                )
        elif path.suffix == ".tsv":
            for row in csv.DictReader(file, delimiter="\t"):
                data.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
        else:
            raise ValueError(f"Unsupported data file type: {path.suffix}")
    return data
