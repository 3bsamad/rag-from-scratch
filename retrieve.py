"""Run a dense retrieval experiment and save a readable report."""
import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from rag.data import load_data
from rag.evaluation import evaluate

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="baseline-v0", help="Experiment label")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--representation", choices=["title-abstract", "abstract"],
                        default="title-abstract")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/scifact")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use an already downloaded model")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N labeled queries")
    args = parser.parse_args()
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.name):
        parser.error("--name must contain only letters, digits, hyphens, and underscores")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def main():
    args = parse_args()
    corpus = load_data(args.data_dir / "corpus.jsonl")
    queries = load_data(args.data_dir / "queries.jsonl")
    qrels = load_data(args.data_dir / "qrels" / f"{args.split}.tsv")
    if args.limit is not None:
        qrels = dict(list(qrels.items())[:args.limit])
    if not corpus:
        raise ValueError("The corpus is empty.")

    from rag.retrieval import DenseRetriever
    retriever = DenseRetriever(corpus, args.model, args.representation,
                               args.cache_dir, args.rebuild_cache, args.offline)
    metrics = evaluate(queries, qrels, retriever.retrieve)
    representation = "title + full abstract" if args.representation == "title-abstract" else "full abstract"
    timestamp = datetime.now(ZoneInfo("Europe/Berlin"))
    settings = {
        "Experiment": args.name,
        "Run time": timestamp.isoformat(timespec="seconds"),
        "Model": args.model,
        "Representation": representation,
        "Dataset": str(args.data_dir.resolve()),
        "Split": args.split,
        "Documents": len(corpus),
        "Evaluated queries": metrics["queries"],
        "Embedding dim": retriever.embeddings.shape[1],
        "Similarity": "cosine (normalized embeddings)",
        "Search": "brute-force matrix similarity",
        "Reranker": "none",
        "Chunking": "none",
        "MRR cutoff": 10,
    }
    config = "\n".join(f"{key + ':':<22}{value}" for key, value in settings.items())
    columns = ["Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR"]
    # Escape model names so slash-containing model IDs still render correctly.
    model_label = args.model.replace("|", "\\|").replace("\n", " ")
    table = (
        "| Experiment | Model | Representation | " + " | ".join(columns) + " |\n"
        "|---|---|---|" + "---:|" * len(columns) + "\n"
        f"| {args.name} | {model_label} | {representation} | "
        + " | ".join(f"{metrics[key]:.4f}" for key in columns) + " |\n"
    )
    report = "# Retrieval Experiments\n\n" + config + "\n\n" + table
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"{args.name}-{timestamp:%Y%m%d-%H%M%S-%f}.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
