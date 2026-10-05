# RAG from Scratch

A small project for learning retrieval-augmented generation, starting with a dense retrieval baseline on SciFact. The current implementation embeds scientific paper titles and abstracts, ranks them with cosine similarity, and evaluates against the dataset's relevance labels.

## Setup

Use Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dataset lives in `data/scifact/`: `corpus.jsonl`, `queries.jsonl`, and `qrels/{train,test}.tsv`. The first model run downloads weights from Hugging Face; later runs reuse those downloaded files.

## Run an experiment

```bash
python retrieve.py --name baseline-v0
python retrieve.py --name abstract-only --representation abstract
python retrieve.py --name mpnet-v0 --model sentence-transformers/all-mpnet-base-v2
```

Each run writes `experiments/<name>-<timestamp>.txt` with its configuration, evaluated query count, and a Markdown-style metrics table. Repeated names create separate files. Reports stay visible to Git so useful runs can be committed.

| Experiment    | Model              | Representation        | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
| ------------- | ------------------ | --------------------- | -------: | -------: | -------: | --------: | -----: |
| `baseline-v0` | `all-MiniLM-L6-v2` | Title + full abstract |   0.4823 |   0.6603 |   0.7379 |    0.7833 | 0.6047 |

These are the previously measured baseline scores on all 300 test queries. MRR@10 uses the top ten results. Each query is retrieved independently before its metrics are calculated; reported scores are averaged across queries.

Useful flags:

| Flag | Purpose |
|---|---|
| `--name` | Label the experiment; default: `baseline-v0` |
| `--model` | Sentence Transformers model name or local path |
| `--representation` | `title-abstract` (default) or `abstract` |
| `--split` | Evaluate `test` (default) or `train` relevance labels |
| `--limit 5` | Quick check on the first five labeled queries; not a full baseline |
| `--offline` | Load already downloaded model weights without network requests |
| `--rebuild-cache` | Re-encode the corpus instead of using saved embeddings |
| `--data-dir`, `--cache-dir`, `--output-dir` | Override dataset, embedding cache, or report directories |

Corpus embeddings are saved under `cache/`. The cache key includes model name, representation, document IDs, and text. If a model changes under the same name or local path, use `--rebuild-cache`. Model weights still load into memory for every new Python process, and queries are encoded during evaluation.

Inspect the dataset with `python inspect_data.py`. Run `python retrieve.py --help` for all options.

## Structure

```text
rag/
  data.py             # Corpus, query, and qrels loaders
  retrieval.py        # Dense retrieval and corpus embedding cache
  evaluation.py       # Recall@k and MRR@10
retrieve.py           # Experiment CLI and text reports
inspect_data.py       # Dataset inspection CLI
data/scifact/         # Dataset files
experiments/          # Saved experiment reports
cache/                # Generated embeddings (ignored by Git)
requirements.txt
```

## Roadmap

### Phase 1 — Dense Retrieval Baseline

- [x] Load SciFact corpus, queries, and relevance labels
- [x] Encode documents and queries with Sentence Transformers
- [x] Implement brute-force cosine similarity search
- [x] Evaluate retrieval using Recall@k and MRR@10
- [x] Establish a baseline with `all-MiniLM-L6-v2`
- [x] Save experiment configurations and results

### Phase 2 — Retrieval Improvements

- [ ] Benchmark multiple embedding models
- [ ] Compare full-document vs. chunk-based retrieval
- [ ] Experiment with chunk size and overlap
- [ ] Add lexical retrieval with BM25
- [ ] Implement hybrid dense + sparse retrieval
- [ ] Add metadata-aware filtering

### Phase 3 — Reranking

- [ ] Add cross-encoder reranking
- [ ] Compare retrieval-only vs. reranked results
- [ ] Measure latency vs. retrieval-quality trade-offs

### Phase 4 — RAG Generation

- [ ] Integrate a local LLM with Ollama
- [ ] Build prompts from retrieved context
- [ ] Add grounded responses with source citations
- [ ] Handle insufficient-context / no-answer cases
- [ ] Add conversational query rewriting

### Phase 5 — RAG Evaluation

- [ ] Build end-to-end QA evaluation
- [ ] Measure retrieval and generation separately
- [ ] Evaluate faithfulness and answer relevance
- [ ] Add regression tests for retrieval quality
- [ ] Compare different retrieval + generation configurations

### Phase 6 — Application Layer

- [ ] Build a FastAPI backend
- [ ] Add document ingestion endpoints
- [ ] Add streaming chat responses
- [ ] Create a React / TypeScript frontend
- [ ] Add document and source inspection in the UI

### Phase 7 — Production & MLOps

- [ ] Containerize the application with Docker
- [ ] Add Qdrant as a persistent vector database
- [ ] Add logging and retrieval/LLM latency metrics
- [ ] Add automated tests and GitHub Actions CI
- [ ] Add configurable embedding and LLM providers
- [ ] Add Docker Compose for local deployment
