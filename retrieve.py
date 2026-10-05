from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from inspect_data import load_data
from eval_utils import evaluate

DATA_DIR = Path("data/scifact")
model = SentenceTransformer("all-MiniLM-L6-v2")


corpus = load_data(DATA_DIR / 'corpus.jsonl')
queries = load_data(DATA_DIR / 'queries.jsonl')

qrels = load_data(DATA_DIR / "qrels/test.tsv")


doc_ids = list(corpus.keys())

documents = [
    corpus[doc_id]["title"] + "\n" + corpus[doc_id]["text"]
    for doc_id in doc_ids
]

print("Encoding corpus...")

doc_embeddings = model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True,
)

print(doc_embeddings.shape)

def retrieve(query: str, k: int = 5):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    # embeddings are normalized, cosine similarity becomes simple dot product
    scores = doc_embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:k]

    results = []

    for idx in top_indices:
        doc_id = doc_ids[idx]

        results.append({
            "doc_id": doc_id,
            "score": float(scores[idx]),
            "title": corpus[doc_id]["title"],
            "text": corpus[doc_id]["text"],
        })

    return results

query = "0-dimensional biomaterials show inductive properties."

results = retrieve(query)

for rank, result in enumerate(results, start=1):
    print(f"\n#{rank}")
    print("ID:", result["doc_id"])
    print("Score:", result["score"])
    print("Title:", result["title"])
    print(result["text"][:300])



evaluate(queries, qrels, retrieve)
