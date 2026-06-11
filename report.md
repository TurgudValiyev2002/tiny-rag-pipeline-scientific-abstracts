# One-Page Report: Tiny RAG Retrieval Evaluation

## Motivation

RAG systems need good retrieval. If the retriever misses the correct source, the answer can become unsupported. We therefore evaluated retrieval quality with hit@k metrics.

## Dataset

The corpus has 30 short AI research abstracts and 30 labeled queries. Each query has one gold document ID.

## Method

We used TF-IDF with unigrams and bigrams, then ranked documents by cosine similarity. For each query, we measured the rank of the gold document.

## Results

The system achieved Hit@1 of 0.9333, Hit@3 of 0.9667, Hit@5 of 0.9667, mean reciprocal rank of 0.9521, and mean gold rank of 1.5333.

## Interpretation

The retriever is strong on this controlled corpus, but the task is still easier than real scientific retrieval. Some topics have clear keywords, so sparse lexical retrieval works well. A harder benchmark should include more documents with overlapping vocabulary.

## Conclusion

The project now has real retrieval evaluation instead of only example outputs. A useful next step is comparing TF-IDF against dense embeddings and reranking.
