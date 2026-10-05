"""Print dataset counts and one query's relevant documents."""
import argparse
from pathlib import Path

from rag.data import load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parent / "data/scifact")
    args = parser.parse_args()
    corpus = load_data(args.data_dir / "corpus.jsonl")
    queries = load_data(args.data_dir / "queries.jsonl")
    qrels = load_data(args.data_dir / "qrels/test.tsv")
    print(f"Loaded {len(corpus)} documents, {len(queries)} queries, and {len(qrels)} labeled queries")
    query_id = next(iter(qrels))
    print(f"\nQUERY {query_id}: {queries[query_id]}")
    for doc_id, score in qrels[query_id].items():
        print(f"\nID: {doc_id} | Relevance: {score}")
        print(corpus[doc_id]["title"])
        print(corpus[doc_id]["text"][:500])


if __name__ == "__main__":
    main()
