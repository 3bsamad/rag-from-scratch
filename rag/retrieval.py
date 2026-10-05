import hashlib
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, corpus, model_name, representation="title-abstract",
                 cache_dir="cache", rebuild_cache=False, offline=False):
        self.corpus = corpus
        self.doc_ids = list(corpus)
        self.model = SentenceTransformer(model_name, local_files_only=offline)
        documents = [
            (corpus[doc_id]["title"] + "\n" if representation == "title-abstract" else "")
            + corpus[doc_id]["text"]
            for doc_id in self.doc_ids
        ]
        # Include IDs, text, and encoding settings so different experiments
        # cannot accidentally reuse incompatible embeddings.
        fingerprint = hashlib.sha256()
        for value in [model_name, representation, "normalized-v1", *self.doc_ids, *documents]:
            fingerprint.update(value.encode("utf-8") + b"\0")
        cache_path = Path(cache_dir) / (fingerprint.hexdigest() + ".npy")
        if cache_path.exists() and not rebuild_cache:
            print(f"Loading cached corpus embeddings: {cache_path}")
            self.embeddings = np.load(cache_path, allow_pickle=False)
        else:
            print("Encoding corpus...")
            self.embeddings = self.model.encode(
                documents, normalize_embeddings=True, show_progress_bar=True,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, self.embeddings, allow_pickle=False)

    def retrieve(self, query, k=5):
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        scores = self.embeddings @ query_embedding
        return [
            {"doc_id": self.doc_ids[index], "score": float(scores[index]),
             **self.corpus[self.doc_ids[index]]}
            for index in np.argsort(scores)[::-1][:k]
        ]
