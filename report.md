# Report: Tiny RAG Pipeline for Scientific Abstracts

## Motivation

We built a small RAG pipeline to understand retrieval before adding generation.

## Dataset

The corpus contains five short AI abstract snippets and three test questions.

## Method

We used TF-IDF vectors and cosine similarity to retrieve the most relevant abstract for each query.

## Hyperparameters

The vectorizer used English stop-word removal. Retrieval used top-1 cosine similarity.

## Results

The system retrieved the expected topics for all three queries. Similarity scores were 0.5000, 0.6030, and 0.4143.

## Interpretation

Retrieval worked correctly in this small setting, but the modest scores show that larger corpora and better embeddings would be needed for a serious RAG system.

## Conclusion

The project demonstrates the core RAG idea clearly. The next step is top-k retrieval and faithfulness checking.
