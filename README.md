# Tiny RAG Pipeline for Scientific Abstracts

![Project overview](assets/readme_project_overview.png)

Figure: a small retrieval benchmark with 30 abstracts, 30 labeled queries, TF-IDF ranking, and hit@k evaluation.

## Motivation

Retrieval augmented generation depends on retrieval quality. If the retriever gives the wrong document, the generated answer may be unsupported even if the language model sounds confident. For that reason, a RAG project should report retrieval metrics, not only example outputs.

## Project Goal

We built a small RAG-style retrieval pipeline and evaluated whether the correct document appears in the top retrieved results.

## Dataset

The corpus contains 30 short AI research abstracts covering federated learning, edge AI, quantization, pruning, computer vision, RAG, tool agents, graph learning, and evaluation methods.

The query set contains 30 labeled questions. Each query has one gold document ID.

## Tools

Python, pandas, scikit-learn, and matplotlib.

## Method

We used TF-IDF with unigram and bigram features. For each query, we ranked all documents by cosine similarity and evaluated whether the gold document appeared in the top results.

## Metrics

- Hit@1: gold document is ranked first
- Hit@3: gold document appears in the top 3
- Hit@5: gold document appears in the top 5
- Mean reciprocal rank: average reciprocal rank of the gold document

## Results

| Metric | Value |
|---|---:|
| Documents | 30 |
| Queries | 30 |
| Hit@1 | 0.9333 |
| Hit@3 | 0.9667 |
| Hit@5 | 0.9667 |
| Mean reciprocal rank | 0.9521 |
| Mean gold rank | 1.5333 |

![Retrieval metrics](results/retrieval_metrics.png)

![Gold rank distribution](results/gold_rank_distribution.png)

Result files:

- `results/abstract_corpus.csv`
- `results/query_set.csv`
- `results/rag_retrieval_results.csv`
- `results/retrieval_metrics_by_query.csv`
- `results/retrieval_summary.csv`

## Interpretation

The retriever works well on this controlled scientific mini-corpus. Hit@1 is high, meaning most queries retrieve the correct document first. Hit@3 improves slightly, showing that top-k retrieval helps when the first document is not perfect.

The result is not a claim that TF-IDF is enough for real RAG. The dataset is still small and carefully written. A larger corpus with overlapping topics would be harder and would likely require dense retrieval or reranking.

## Conclusion

The project now evaluates retrieval quality directly. The next step is to add more documents with similar wording and compare TF-IDF with embedding retrieval and reranking.

## How To Run

```bash
pip install -r requirements.txt
python 1_tiny_rag_pipeline.py
```
