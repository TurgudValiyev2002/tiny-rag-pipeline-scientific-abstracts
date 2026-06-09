# Tiny RAG Pipeline for Scientific Abstracts

## 1. Motivation

This lab builds a very small retrieval augmented generation pipeline. The motivation is to understand RAG as a system: store documents, retrieve evidence, then answer using retrieved context.

## 2. Project Goal

Build a small, reproducible AI research lab with clear outputs and honest limitations.

## 3. Dataset, Paper, Or Problem Description

Dataset: five local AI abstract snippets written for offline use.

## 4. Tools

Tools: Python, pandas, scikit-learn TF-IDF, matplotlib.

## 5. Models Or Methods

Method: TF-IDF vectors and cosine similarity retrieval; the answer is an evidence-grounded extractive sentence.

## 6. Hyperparameters When Relevant

Hyperparameters: English stop words, top-1 retrieval.

## 7. Results

Results include corpus, retrieval table, and score figure.

## 8. Interpretation Of Results

Interpretation: RAG quality depends first on retrieval quality.

## 9. Conclusion

Conclusion: before building agents around RAG, verify whether the retriever finds the right evidence.

## 10. How To Run

```bash
pip install -r requirements.txt
python 1_*.py
```
