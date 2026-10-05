import numpy as np

def recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved_at_k = set(retrieved_ids[:k])
    relevant_ids = set(relevant_ids)

    if not relevant_ids:
        return 0.0

    found = retrieved_at_k.intersection(relevant_ids)

    return len(found) / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant_ids = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def evaluate(queries, qrels, retrieve):
    """Evaluate each labeled query using its own top ten retrieval results."""
    recall_1 = []
    recall_3 = []
    recall_5 = []
    recall_10 = []
    reciprocal_ranks = []

    evaluated = 0

    for query_id, relevant_docs in qrels.items():

        if query_id not in queries:
            continue

        query = queries[query_id]

        results = retrieve(query, k=10)

        retrieved_ids = [
            result["doc_id"]
            for result in results
        ]

        relevant_ids = list(relevant_docs.keys())

        recall_1.append(
            recall_at_k(retrieved_ids, relevant_ids, 1)
        )

        recall_3.append(
            recall_at_k(retrieved_ids, relevant_ids, 3)
        )

        recall_5.append(
            recall_at_k(retrieved_ids, relevant_ids, 5)
        )

        recall_10.append(
            recall_at_k(retrieved_ids, relevant_ids, 10)
        )

        reciprocal_ranks.append(
            reciprocal_rank(retrieved_ids, relevant_ids)
        )

        evaluated += 1

    if not evaluated:
        raise ValueError("No labeled queries available for evaluation.")

    return {
        "queries": evaluated,
        "Recall@1": float(np.mean(recall_1)),
        "Recall@3": float(np.mean(recall_3)),
        "Recall@5": float(np.mean(recall_5)),
        "Recall@10": float(np.mean(recall_10)),
        "MRR": float(np.mean(reciprocal_ranks)),
    }
