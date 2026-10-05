import json
from pathlib import Path
import csv

DATA_DIR = Path("data/scifact")

corpus = {}

with open(DATA_DIR / "corpus.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        corpus[item["_id"]] = {
            "title": item.get("title", ""),
            "text": item["text"],
        }

print(f"Loaded {len(corpus)} documents")

queries = {}

with open(DATA_DIR / "queries.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        queries[item["_id"]] = item["text"]

print(f"Loaded {len(queries)} queries")



for query_id, query in list(queries.items())[:5]:
    print(query_id, "=>", query)

for doc_id, doc in list(corpus.items())[:1]:
    print(doc_id, "=>", doc)

qrels = {}

with open(DATA_DIR / "qrels" / "test.tsv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        query_id = row["query-id"]
        corpus_id = row["corpus-id"]
        score = int(row["score"])

        qrels.setdefault(query_id, {})[corpus_id] = score

print(f"Loaded qrels for {len(qrels)} queries")


query_id = next(iter(qrels))

print("QUERY:")
print(query_id, "===>", queries[query_id])

print("\nRELEVANT DOCUMENTS:")

for corpus_id, score in qrels[query_id].items():
    print("\nID:", corpus_id)
    print("Score:", score)
    print("Title:", corpus[corpus_id]["title"])
    print("Text:", corpus[corpus_id]["text"][:500])



def load_data(path):
    """Load a SciFact corpus, queries, or qrels file into its lookup mapping."""
    path = Path(path)

    if path.suffix == ".jsonl":
        data = {}
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                item = json.loads(line)
                if path.name == "corpus.jsonl":
                    data[item["_id"]] = {
                        "title": item.get("title", ""),
                        "text": item["text"],
                    }
                else:
                    data[item["_id"]] = item["text"]
        return data

    if path.suffix == ".tsv":
        data = {}
        with path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file, delimiter="\t"):
                query_id = row["query-id"]
                corpus_id = row["corpus-id"]
                score = int(row["score"])
                data.setdefault(query_id, {})[corpus_id] = score
        return data

    raise ValueError(f"Unsupported data file type: {path.suffix}")
