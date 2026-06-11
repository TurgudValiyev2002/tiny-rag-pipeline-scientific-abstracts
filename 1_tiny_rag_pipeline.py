from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


RESULTS = Path("results")

DOCS = [
    ("fedavg", "Federated Averaging trains local models on client data and averages their weight updates on a central server."),
    ("fedprox", "FedProx adds a proximal term to stabilize federated learning when clients have non-IID data and different computation budgets."),
    ("secure_aggregation", "Secure aggregation protects individual client updates so the server can see only the aggregated model update."),
    ("differential_privacy_fl", "Differential privacy in federated learning clips and noises client updates to reduce leakage from model gradients."),
    ("edge_ai_latency", "Edge AI runs inference near sensors to reduce network latency, bandwidth usage, and dependence on cloud connectivity."),
    ("tinyml", "TinyML deploys small neural networks on microcontrollers with strict memory, energy, and compute limits."),
    ("model_quantization", "Model quantization stores weights and activations with fewer bits to reduce memory use and speed up inference."),
    ("weight_pruning", "Weight pruning removes low-importance parameters from neural networks to reduce model size and computation."),
    ("vision_transformer", "Vision Transformers split images into patches and use self-attention to mix visual information across the image."),
    ("swin_transformer", "Swin Transformer computes attention inside shifted local windows to reduce the cost of vision transformers."),
    ("cnn_edges", "Convolutional neural networks learn local filters that detect edges, textures, object parts, and higher-level patterns."),
    ("object_detection", "Object detection predicts both class labels and bounding boxes for objects present in an image."),
    ("yolo", "YOLO performs object detection in a single forward pass and is often used when real-time speed matters."),
    ("faster_rcnn", "Faster R-CNN uses region proposals followed by classification and bounding-box refinement for object detection."),
    ("rag", "Retrieval augmented generation retrieves external documents and conditions the answer on the retrieved evidence."),
    ("dense_retrieval", "Dense retrieval maps queries and documents into embedding vectors and ranks documents by vector similarity."),
    ("bm25", "BM25 is a sparse lexical retrieval method based on term frequency, inverse document frequency, and document length."),
    ("reranking", "Reranking applies a stronger model to reorder initially retrieved documents for better relevance."),
    ("hallucination", "Language model hallucination occurs when a model gives fluent but unsupported or false information."),
    ("prompt_engineering", "Prompt engineering studies how instructions, examples, and constraints change language model behavior."),
    ("agent_tool_use", "Tool-using agents select external tools such as calculators or search functions to solve tasks more reliably."),
    ("graph_centrality", "Graph centrality measures identify important nodes based on degree, shortest paths, or connections to other important nodes."),
    ("community_detection", "Community detection finds groups of nodes that are more densely connected internally than externally."),
    ("gcn", "Graph convolutional networks learn node representations by aggregating information from neighboring graph nodes."),
    ("node_classification", "Node classification predicts labels for graph nodes using node features, graph structure, or both."),
    ("calibration", "Model calibration measures whether predicted probabilities match the true frequency of correctness."),
    ("cross_validation", "Cross-validation estimates model performance by training and testing across multiple data splits."),
    ("early_stopping", "Early stopping halts training when validation performance stops improving to reduce overfitting."),
    ("confusion_matrix", "A confusion matrix shows which classes a classifier predicts correctly and which classes it confuses."),
    ("macro_f1", "Macro F1 averages F1 scores across classes equally, making it useful when class balance matters."),
]

QUERIES = [
    ("How does federated averaging combine client models?", "fedavg"),
    ("Which federated method helps with non-IID clients?", "fedprox"),
    ("How can a server aggregate updates without seeing each client update?", "secure_aggregation"),
    ("What privacy method adds noise to client gradients?", "differential_privacy_fl"),
    ("Why deploy inference near sensors?", "edge_ai_latency"),
    ("What is machine learning on microcontrollers called?", "tinyml"),
    ("How do we store model weights with fewer bits?", "model_quantization"),
    ("What compression method removes small neural network weights?", "weight_pruning"),
    ("Which vision model uses image patches and self-attention?", "vision_transformer"),
    ("Which transformer uses shifted windows for images?", "swin_transformer"),
    ("What model family learns local visual filters?", "cnn_edges"),
    ("Which task predicts bounding boxes and labels?", "object_detection"),
    ("Which detector is known for single-pass real-time detection?", "yolo"),
    ("Which detector uses region proposals?", "faster_rcnn"),
    ("What does RAG retrieve before answering?", "rag"),
    ("What retrieval method compares embedding vectors?", "dense_retrieval"),
    ("Which sparse retriever uses term frequency and IDF?", "bm25"),
    ("What step reorders initially retrieved documents?", "reranking"),
    ("What is a fluent but unsupported LLM answer called?", "hallucination"),
    ("What studies the effect of instructions and examples on LLM behavior?", "prompt_engineering"),
    ("What kind of agent calls calculators or search tools?", "agent_tool_use"),
    ("Which graph measure finds important nodes?", "graph_centrality"),
    ("What graph task finds densely connected groups?", "community_detection"),
    ("Which graph neural network aggregates neighbors?", "gcn"),
    ("What task predicts labels for graph nodes?", "node_classification"),
    ("What checks whether probabilities match correctness frequency?", "calibration"),
    ("What evaluation repeats training across several splits?", "cross_validation"),
    ("What stops training when validation stops improving?", "early_stopping"),
    ("What table shows classifier confusions between classes?", "confusion_matrix"),
    ("Which F1 treats each class equally?", "macro_f1"),
]


def reciprocal_rank(ranked_topics: list[str], gold_topic: str) -> float:
    for idx, topic in enumerate(ranked_topics, start=1):
        if topic == gold_topic:
            return 1.0 / idx
    return 0.0


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    corpus = pd.DataFrame(DOCS, columns=["doc_id", "abstract"])
    queries = pd.DataFrame(QUERIES, columns=["query", "gold_doc_id"])
    corpus.to_csv(RESULTS / "abstract_corpus.csv", index=False)
    queries.to_csv(RESULTS / "query_set.csv", index=False)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    doc_matrix = vectorizer.fit_transform(corpus["abstract"])

    retrieval_rows = []
    metric_rows = []
    for query, gold_doc_id in QUERIES:
        sims = cosine_similarity(vectorizer.transform([query]), doc_matrix).ravel()
        order = sims.argsort()[::-1]
        ranked_doc_ids = corpus.iloc[order]["doc_id"].tolist()
        top5_ids = ranked_doc_ids[:5]
        metric_rows.append(
            {
                "query": query,
                "gold_doc_id": gold_doc_id,
                "rank": ranked_doc_ids.index(gold_doc_id) + 1,
                "hit_at_1": int(gold_doc_id in ranked_doc_ids[:1]),
                "hit_at_3": int(gold_doc_id in ranked_doc_ids[:3]),
                "hit_at_5": int(gold_doc_id in ranked_doc_ids[:5]),
                "reciprocal_rank": reciprocal_rank(ranked_doc_ids, gold_doc_id),
            }
        )
        for rank, doc_idx in enumerate(order[:5], start=1):
            retrieval_rows.append(
                {
                    "query": query,
                    "gold_doc_id": gold_doc_id,
                    "rank": rank,
                    "retrieved_doc_id": corpus.loc[doc_idx, "doc_id"],
                    "score": round(float(sims[doc_idx]), 4),
                    "is_gold": corpus.loc[doc_idx, "doc_id"] == gold_doc_id,
                    "retrieved_abstract": corpus.loc[doc_idx, "abstract"],
                }
            )

    retrieval = pd.DataFrame(retrieval_rows)
    per_query = pd.DataFrame(metric_rows)
    summary = pd.DataFrame(
        [
            {"metric": "queries", "value": len(queries)},
            {"metric": "documents", "value": len(corpus)},
            {"metric": "hit_at_1", "value": per_query["hit_at_1"].mean()},
            {"metric": "hit_at_3", "value": per_query["hit_at_3"].mean()},
            {"metric": "hit_at_5", "value": per_query["hit_at_5"].mean()},
            {"metric": "mean_reciprocal_rank", "value": per_query["reciprocal_rank"].mean()},
            {"metric": "mean_gold_rank", "value": per_query["rank"].mean()},
        ]
    )
    retrieval.to_csv(RESULTS / "rag_retrieval_results.csv", index=False)
    per_query.to_csv(RESULTS / "retrieval_metrics_by_query.csv", index=False)
    summary.to_csv(RESULTS / "retrieval_summary.csv", index=False)

    plot_summary(summary)
    plot_rank_distribution(per_query)
    plot_readme_overview()
    print(summary.to_string(index=False))


def plot_summary(summary: pd.DataFrame) -> None:
    scores = summary[summary["metric"].isin(["hit_at_1", "hit_at_3", "hit_at_5", "mean_reciprocal_rank"])]
    plt.figure(figsize=(7, 4))
    plt.bar(scores["metric"], scores["value"], color="#3d6fb6")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Retrieval Quality Metrics")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "retrieval_metrics.png", dpi=180)
    plt.close()


def plot_rank_distribution(per_query: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(per_query["rank"], bins=range(1, per_query["rank"].max() + 3), color="#4a8f5a", edgecolor="white")
    plt.xlabel("Gold document rank")
    plt.ylabel("Number of queries")
    plt.title("Gold Document Rank Distribution")
    plt.tight_layout()
    plt.savefig(RESULTS / "gold_rank_distribution.png", dpi=180)
    plt.close()


def plot_readme_overview() -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    boxes = [
        ("30 abstracts", 0.15),
        ("30 labeled\nqueries", 0.39),
        ("TF-IDF retrieval\nTop-5 ranking", 0.63),
        ("Hit@k + MRR\nmetrics", 0.86),
    ]
    for text, x in boxes:
        ax.text(x, 0.55, text, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef6ff", edgecolor="#336699"))
    for start, end in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(end[1] - 0.11, 0.55), xytext=(start[1] + 0.11, 0.55), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("Tiny RAG retrieval evaluation workflow", fontsize=15)
    fig.tight_layout()
    fig.savefig("assets/readme_project_overview.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
