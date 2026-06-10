# Tiny RAG Pipeline for Scientific Abstracts

## Motivation

Retrieval augmented generation works only if retrieval finds useful evidence. This project builds a small RAG-style pipeline to make the retrieval step visible and measurable.

## Project Goal

We built a tiny scientific-abstract search system. Given a question, the system retrieves the most relevant abstract and returns an evidence-based answer.

## Dataset

The corpus contains five short AI-related abstract snippets about federated learning, vision transformers, RAG, Edge AI, and graph neural networks.

## Tools

Python, pandas, scikit-learn, and matplotlib.

## Method

We converted abstracts and queries into TF-IDF vectors, computed cosine similarity, selected the top matching abstract, and used that abstract as the simple answer.

## Hyperparameters

- Vectorizer: `TfidfVectorizer(stop_words="english")`
- Retrieval: top-1 cosine similarity
- Number of queries: 3

## Results

| Query | Retrieved Topic | Score |
|---|---|---:|
| How does federated learning protect data? | federated learning | 0.5000 |
| What does RAG do before generation? | retrieval augmented generation | 0.6030 |
| Why run AI on edge devices? | edge ai | 0.4143 |

Result files include the corpus, retrieval table, and retrieval-score figure.

## Interpretation

The retriever selected the expected topic for all three queries. The scores are not very high because the corpus is tiny and the wording is short. This is normal for a minimal TF-IDF setup.

## Conclusion

This project shows the basic RAG workflow: build a corpus, retrieve evidence, and answer from the retrieved text. A stronger version should add top-k retrieval, larger documents, and faithfulness evaluation.

## How To Run

```bash
pip install -r requirements.txt
python 1_tiny_rag_pipeline.py
```
