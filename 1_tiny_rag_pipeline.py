from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RESULTS = Path("results")
DOCS = [
    ("federated learning", "Federated learning trains models across clients while keeping raw data local and sharing model updates."),
    ("vision transformer", "Vision transformers split images into patches and use self-attention instead of convolution as the main mixing operation."),
    ("retrieval augmented generation", "Retrieval augmented generation first retrieves relevant documents and then conditions generation on that evidence."),
    ("edge ai", "Edge AI runs inference near sensors to reduce latency, bandwidth use, and privacy exposure."),
    ("graph neural network", "Graph neural networks learn from nodes and edges by passing messages between neighboring nodes."),
]
QUERIES = [
    "How does federated learning protect data?",
    "What does RAG do before generation?",
    "Why run AI on edge devices?",
]

def main():
    RESULTS.mkdir(exist_ok=True)
    corpus = pd.DataFrame(DOCS, columns=["topic", "abstract"])
    corpus.to_csv(RESULTS / "abstract_corpus.csv", index=False)
    vec = TfidfVectorizer(stop_words="english")
    doc_matrix = vec.fit_transform(corpus["abstract"])
    rows = []
    for q in QUERIES:
        sims = cosine_similarity(vec.transform([q]), doc_matrix).ravel()
        best = sims.argmax()
        rows.append({"query": q, "top_topic": corpus.loc[best, "topic"], "score": round(float(sims[best]), 4), "retrieved_abstract": corpus.loc[best, "abstract"], "simple_answer": corpus.loc[best, "abstract"]})
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / "rag_retrieval_results.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.bar(results["top_topic"], results["score"], color="#3d6fb6")
    plt.ylabel("Cosine similarity")
    plt.title("Top Retrieval Score per Query")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "retrieval_scores.png", dpi=160)
    print(results[["query", "top_topic", "score"]].to_string(index=False))

if __name__ == "__main__":
    main()
