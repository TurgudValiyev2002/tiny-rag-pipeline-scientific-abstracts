# One-Page Report: 100-Document RAG Retrieval Evaluation

## Motivation

RAG systems need good retrieval. If the retriever misses the correct source, the answer can become unsupported. We therefore evaluated retrieval quality with hit@k metrics.

## Dataset

The corpus has 100 short AI research abstracts and 100 labeled queries across 20 topic groups. Each query has one gold document ID.

## Method

We used TF-IDF with unigrams and bigrams, then ranked documents by cosine similarity. For each query, we measured the rank of the gold document.

## Results

The system achieved Hit@1 of 0.8700, Hit@3 of 0.9700, Hit@5 of 0.9800, Hit@10 of 0.9900, mean reciprocal rank of 0.9189, and mean gold rank of 1.9700.

## Interpretation

The retriever is strong but not perfect. Most queries retrieve the correct abstract first, while top-k retrieval recovers almost all remaining cases. The misses are useful because they show where related AI topics share vocabulary and confuse a lexical retriever.

## Conclusion

The project shows how to evaluate retrieval before adding generation. TF-IDF is a good baseline, but the non-perfect Hit@1 result makes a clear case for comparing lexical retrieval with dense embeddings and reranking.
