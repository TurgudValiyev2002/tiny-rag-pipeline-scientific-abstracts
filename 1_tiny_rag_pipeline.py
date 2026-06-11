from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


RESULTS = Path("results")


TOPIC_GROUPS = {
    "federated_learning": [
        ("fedavg", "Federated Averaging trains local client models and averages their weight updates on a server.", "How does FedAvg combine client models?"),
        ("fedprox", "FedProx adds a proximal term to stabilize federated learning under non-IID client data.", "Which federated method adds a proximal term for non-IID clients?"),
        ("secure_aggregation", "Secure aggregation hides individual client updates while allowing the server to recover the aggregate.", "How can a server aggregate updates without seeing each client update?"),
        ("federated_dp", "Differential privacy in federated learning clips and noises updates to reduce gradient leakage.", "What privacy method adds noise to federated client updates?"),
        ("client_drift", "Client drift happens when local training moves models in different directions because client data distributions differ.", "What problem occurs when non-IID client updates move in different directions?"),
    ],
    "edge_ai": [
        ("edge_latency", "Edge AI runs inference near sensors to reduce latency and dependence on cloud connectivity.", "Why run inference near sensors?"),
        ("tinyml", "TinyML deploys compact neural networks on microcontrollers with strict memory and energy limits.", "What is machine learning on microcontrollers called?"),
        ("model_compression", "Model compression reduces model size using pruning, quantization, distillation, or low-rank methods.", "What methods reduce neural network model size?"),
        ("edge_privacy", "Edge inference can improve privacy because raw sensor data may stay on the local device.", "How can edge inference help privacy?"),
        ("hardware_constraints", "Embedded AI must respect memory, compute, battery, and thermal constraints during deployment.", "What constraints matter for embedded AI deployment?"),
    ],
    "compression": [
        ("int8_quantization", "Int8 quantization stores weights or activations with 8-bit integers to reduce memory and bandwidth.", "What stores model values with 8-bit integers?"),
        ("low_bit_quantization", "Low-bit quantization uses very few bits per parameter and may damage accuracy if too aggressive.", "Why can very low-bit quantization hurt accuracy?"),
        ("weight_pruning", "Weight pruning removes small or low-importance weights to create sparse neural networks.", "What compression method removes low-importance weights?"),
        ("structured_pruning", "Structured pruning removes channels, filters, or blocks so hardware can accelerate the sparse model.", "Which pruning type removes channels or filters?"),
        ("knowledge_distillation", "Knowledge distillation trains a smaller student model to match a larger teacher model.", "What compression method trains a student from a teacher model?"),
    ],
    "computer_vision": [
        ("cnn_filters", "Convolutional networks learn local filters that detect edges, textures, parts, and objects.", "What model family learns local image filters?"),
        ("vision_transformer", "Vision Transformers split images into patches and process them as token sequences with self-attention.", "Which vision model uses image patches and self-attention?"),
        ("swin_transformer", "Swin Transformer uses shifted local windows and hierarchy for efficient visual attention.", "Which transformer uses shifted windows for images?"),
        ("image_augmentation", "Image augmentation applies transformations such as crop, flip, color jitter, and rotation during training.", "What technique applies crops and flips during image training?"),
        ("semantic_segmentation", "Semantic segmentation assigns a class label to every pixel in an image.", "Which task predicts a class for every pixel?"),
    ],
    "object_detection": [
        ("object_detection", "Object detection predicts object classes and bounding boxes in an image.", "Which task predicts object labels and bounding boxes?"),
        ("yolo_detector", "YOLO performs object detection in a single forward pass and is often used for real-time detection.", "Which detector is known for single-pass real-time detection?"),
        ("faster_rcnn", "Faster R-CNN uses region proposals before classification and bounding-box refinement.", "Which detector uses region proposals?"),
        ("anchor_boxes", "Anchor boxes are predefined box shapes used by some detectors to predict object locations.", "What predefined shapes help detectors predict object boxes?"),
        ("mean_average_precision", "Mean average precision summarizes object detection precision and recall across classes and IoU thresholds.", "Which metric summarizes detection precision and recall across IoU thresholds?"),
    ],
    "rag": [
        ("rag_pipeline", "Retrieval augmented generation retrieves evidence documents before generating an answer.", "What does RAG retrieve before answering?"),
        ("dense_retrieval", "Dense retrieval maps queries and documents into embedding vectors and ranks by vector similarity.", "What retrieval method compares embedding vectors?"),
        ("bm25_retrieval", "BM25 is a sparse lexical retrieval method based on term frequency, inverse document frequency, and document length.", "Which sparse retriever uses term frequency and IDF?"),
        ("reranking", "Reranking applies a stronger model to reorder initially retrieved documents for relevance.", "What step reorders initially retrieved documents?"),
        ("retrieval_hit_at_k", "Hit at k measures whether the correct document appears within the top k retrieved results.", "Which retrieval metric checks if the gold document is in the top k?"),
    ],
    "llm_evaluation": [
        ("hallucination", "Language model hallucination occurs when a model produces fluent but unsupported or false claims.", "What is a fluent but unsupported LLM answer called?"),
        ("prompt_engineering", "Prompt engineering studies how instructions, examples, and constraints change model behavior.", "What studies the effect of instructions and examples on LLM behavior?"),
        ("calibration", "Calibration measures whether predicted probabilities match observed correctness frequencies.", "What checks whether model confidence matches correctness?"),
        ("toxicity_eval", "Toxicity evaluation measures whether model outputs contain harmful, abusive, or unsafe language.", "What evaluation checks harmful or abusive model outputs?"),
        ("llm_judge", "LLM-as-judge evaluation uses a language model to rate outputs according to a rubric.", "What evaluation method uses a model to grade outputs with a rubric?"),
    ],
    "agentic_ai": [
        ("tool_use_agent", "Tool-using agents call calculators, search systems, or code tools to solve tasks more reliably.", "What kind of agent calls calculators or search tools?"),
        ("planning_agent", "Planning agents decompose a goal into steps before executing actions.", "What agent decomposes a goal into steps?"),
        ("reflection_loop", "Reflection loops let an agent critique its own output and revise the next action.", "What loop lets an agent critique and revise its output?"),
        ("memory_agent", "Agent memory stores useful prior context so later decisions can use earlier information.", "What component stores prior context for an agent?"),
        ("tool_routing", "Tool routing selects the correct external tool based on task intent and input constraints.", "What chooses the correct external tool for a task?"),
    ],
    "graph_learning": [
        ("degree_centrality", "Degree centrality measures how many direct connections a node has in a graph.", "Which centrality counts direct node connections?"),
        ("betweenness_centrality", "Betweenness centrality measures how often a node lies on shortest paths between other nodes.", "Which centrality uses shortest paths through a node?"),
        ("community_detection", "Community detection finds groups of nodes with dense internal connections.", "What graph task finds densely connected groups?"),
        ("graph_convolution", "Graph convolution aggregates neighboring node features to learn graph-aware representations.", "Which graph method aggregates neighboring node features?"),
        ("node_classification", "Node classification predicts labels for graph nodes using features and graph structure.", "What task predicts labels for graph nodes?"),
    ],
    "ml_evaluation": [
        ("cross_validation", "Cross-validation trains and tests models across multiple data splits to estimate generalization.", "What evaluation repeats training across several splits?"),
        ("confusion_matrix", "A confusion matrix shows correct and incorrect predictions for each class.", "What table shows classifier confusions between classes?"),
        ("macro_f1", "Macro F1 averages class F1 scores equally and is useful when class balance matters.", "Which F1 score treats every class equally?"),
        ("roc_auc", "ROC-AUC measures how well a classifier ranks positive examples above negative examples.", "Which metric measures ranking quality across thresholds?"),
        ("early_stopping", "Early stopping halts training when validation performance no longer improves.", "What stops training when validation performance stops improving?"),
    ],
    "responsible_ai": [
        ("fairness_selection_rate", "Selection rate compares how often a model predicts a positive outcome for different groups.", "Which fairness metric compares positive prediction rates between groups?"),
        ("equal_opportunity", "Equal opportunity compares true positive rates across demographic groups.", "Which fairness concept compares true positive rates across groups?"),
        ("privacy_attack", "Privacy attacks can infer sensitive information from model outputs or gradients.", "What attack tries to infer private information from model behavior?"),
        ("model_card", "A model card documents intended use, limitations, training data, and evaluation results.", "What document describes model use, limitations, and evaluation?"),
        ("data_datasheet", "A datasheet for datasets documents data collection, composition, recommended use, and risks.", "What document describes dataset collection and risks?"),
    ],
    "training_methods": [
        ("dropout_regularization", "Dropout randomly disables neurons during training to reduce overfitting.", "What regularization randomly disables neurons during training?"),
        ("weight_decay", "Weight decay penalizes large model weights to improve generalization.", "What regularization penalizes large weights?"),
        ("batch_normalization", "Batch normalization stabilizes activations and can speed up neural network training.", "What layer normalizes activations during training?"),
        ("learning_rate_schedule", "Learning rate schedules change the optimizer step size during training.", "What changes optimizer step size over training?"),
        ("adam_optimizer", "Adam combines momentum and adaptive learning rates for gradient-based optimization.", "Which optimizer uses momentum and adaptive learning rates?"),
    ],
    "datasets": [
        ("mnist_dataset", "MNIST contains handwritten digit images and is often used for basic classification experiments.", "Which dataset contains handwritten digit images?"),
        ("fashionmnist_dataset", "FashionMNIST contains grayscale clothing images across ten fashion categories.", "Which dataset contains grayscale clothing images?"),
        ("cifar10_dataset", "CIFAR-10 contains small color images from ten object classes.", "Which dataset has ten classes of small color images?"),
        ("coco_dataset", "COCO provides images with object detection, segmentation, and caption annotations.", "Which dataset provides object boxes, masks, and captions?"),
        ("cora_dataset", "Cora is a citation network dataset with paper features, citation links, and topic labels.", "Which dataset is a citation network with paper topic labels?"),
    ],
    "deployment": [
        ("onnx_export", "ONNX export converts trained models into an interoperable format for deployment.", "What format helps export models for interoperable deployment?"),
        ("latency_benchmark", "Latency benchmarking measures how long inference takes under deployment conditions.", "What benchmark measures inference time?"),
        ("throughput", "Throughput measures how many inputs a system processes per unit time.", "What measures processed inputs per unit time?"),
        ("monitoring_drift", "Data drift monitoring checks whether production inputs change compared with training data.", "What monitoring checks production inputs against training data?"),
        ("rollback_strategy", "Rollback strategy restores a previous model when a new deployment causes problems.", "What deployment plan restores a previous model after problems?"),
    ],
    "time_series": [
        ("forecasting", "Forecasting predicts future values from historical time-series observations.", "What task predicts future time-series values?"),
        ("seasonality", "Seasonality is a repeated temporal pattern such as daily, weekly, or yearly cycles.", "What is a repeated temporal pattern in a time series?"),
        ("anomaly_detection", "Anomaly detection finds unusual observations that differ from expected behavior.", "What task finds unusual observations?"),
        ("rolling_window", "Rolling-window evaluation trains on past windows and tests on later time periods.", "What evaluation trains on past windows and tests later periods?"),
        ("temporal_leakage", "Temporal leakage happens when future information leaks into time-series training features.", "What problem happens when future information enters training features?"),
    ],
    "nlp": [
        ("tokenization", "Tokenization splits text into units such as words, subwords, or characters.", "What splits text into words or subwords?"),
        ("named_entity_recognition", "Named entity recognition finds entities such as people, organizations, and locations in text.", "What task extracts people, organizations, and locations?"),
        ("text_classification", "Text classification assigns labels such as topic, intent, or sentiment to documents.", "What task assigns labels to text documents?"),
        ("summarization", "Summarization produces a shorter version of a document while preserving important information.", "What task creates a shorter version of a document?"),
        ("machine_translation", "Machine translation converts text from one language into another language.", "What task converts text between languages?"),
    ],
    "security": [
        ("adversarial_examples", "Adversarial examples are inputs modified to fool a model while appearing similar to humans.", "What inputs are modified to fool a model?"),
        ("data_poisoning", "Data poisoning attacks inject harmful training examples to corrupt model behavior.", "What attack corrupts training data?"),
        ("model_extraction", "Model extraction tries to copy a model by querying it and observing outputs.", "What attack tries to copy a model through queries?"),
        ("prompt_injection", "Prompt injection tries to override an LLM application's instructions using malicious input.", "What attack tries to override an LLM application's instructions?"),
        ("membership_inference", "Membership inference predicts whether a sample was part of a model's training data.", "What attack predicts whether an example was in training data?"),
    ],
    "optimization": [
        ("gradient_descent", "Gradient descent updates parameters in the direction that reduces loss.", "What optimization method updates parameters to reduce loss?"),
        ("momentum", "Momentum accumulates past gradients to smooth and accelerate optimization.", "What optimization technique accumulates past gradients?"),
        ("learning_rate", "Learning rate controls the step size of parameter updates during training.", "What controls optimizer step size?"),
        ("loss_landscape", "Loss landscape describes how the objective changes across parameter space.", "What describes objective values across parameter space?"),
        ("hyperparameter_search", "Hyperparameter search tests settings such as learning rate, depth, and regularization strength.", "What process tests settings like learning rate and depth?"),
    ],
    "reinforcement_learning": [
        ("q_learning", "Q-learning estimates action values and learns a policy from rewards without requiring a model of the environment.", "Which reinforcement learning method learns action values from rewards?"),
        ("policy_gradient", "Policy gradient methods directly optimize a parameterized policy using reward-weighted gradient estimates.", "Which methods directly optimize a parameterized policy?"),
        ("reward_function", "A reward function defines the feedback signal that guides a reinforcement learning agent toward desired behavior.", "What signal guides a reinforcement learning agent?"),
        ("exploration_exploitation", "Exploration tries new actions while exploitation uses actions that currently look best.", "What tradeoff balances trying new actions and using known good actions?"),
        ("offline_rl", "Offline reinforcement learning trains policies from fixed logged data without new environment interaction.", "What reinforcement learning setting trains from fixed logged data?"),
    ],
    "multimodal_ai": [
        ("vision_language_model", "Vision-language models connect images and text so a system can reason across both modalities.", "What model type connects images and text?"),
        ("image_captioning", "Image captioning generates natural-language descriptions for visual content.", "What task generates text descriptions of images?"),
        ("clip_contrastive", "CLIP-style contrastive learning aligns image and text embeddings using paired examples.", "What learning approach aligns image and text embeddings?"),
        ("multimodal_embedding", "Multimodal embeddings place different data types in a shared representation space for retrieval or reasoning.", "What representation places different data types in a shared space?"),
        ("visual_question_answering", "Visual question answering answers text questions about the content of an image.", "What task answers questions about an image?"),
    ],
}


def build_corpus_and_queries() -> tuple[pd.DataFrame, pd.DataFrame]:
    docs = []
    queries = []
    for group, items in TOPIC_GROUPS.items():
        for doc_id, abstract, query in items:
            docs.append({"doc_id": doc_id, "topic_group": group, "abstract": abstract})
            queries.append({"query": query, "gold_doc_id": doc_id, "topic_group": group})
    return pd.DataFrame(docs), pd.DataFrame(queries)


def reciprocal_rank(ranked_doc_ids: list[str], gold_doc_id: str) -> float:
    for idx, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id == gold_doc_id:
            return 1.0 / idx
    return 0.0


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    corpus, queries = build_corpus_and_queries()
    corpus.to_csv(RESULTS / "abstract_corpus.csv", index=False)
    queries.to_csv(RESULTS / "query_set.csv", index=False)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    doc_matrix = vectorizer.fit_transform(corpus["abstract"])

    retrieval_rows = []
    metric_rows = []
    for row in queries.itertuples(index=False):
        sims = cosine_similarity(vectorizer.transform([row.query]), doc_matrix).ravel()
        order = sims.argsort()[::-1]
        ranked_doc_ids = corpus.iloc[order]["doc_id"].tolist()
        gold_rank = ranked_doc_ids.index(row.gold_doc_id) + 1
        metric_rows.append(
            {
                "query": row.query,
                "topic_group": row.topic_group,
                "gold_doc_id": row.gold_doc_id,
                "rank": gold_rank,
                "hit_at_1": int(gold_rank <= 1),
                "hit_at_3": int(gold_rank <= 3),
                "hit_at_5": int(gold_rank <= 5),
                "hit_at_10": int(gold_rank <= 10),
                "reciprocal_rank": reciprocal_rank(ranked_doc_ids, row.gold_doc_id),
            }
        )
        for rank, doc_idx in enumerate(order[:10], start=1):
            retrieval_rows.append(
                {
                    "query": row.query,
                    "gold_doc_id": row.gold_doc_id,
                    "rank": rank,
                    "retrieved_doc_id": corpus.loc[doc_idx, "doc_id"],
                    "retrieved_topic_group": corpus.loc[doc_idx, "topic_group"],
                    "score": round(float(sims[doc_idx]), 4),
                    "is_gold": corpus.loc[doc_idx, "doc_id"] == row.gold_doc_id,
                    "retrieved_abstract": corpus.loc[doc_idx, "abstract"],
                }
            )

    retrieval = pd.DataFrame(retrieval_rows)
    per_query = pd.DataFrame(metric_rows)
    summary = pd.DataFrame(
        [
            {"metric": "queries", "value": len(queries)},
            {"metric": "documents", "value": len(corpus)},
            {"metric": "topic_groups", "value": corpus["topic_group"].nunique()},
            {"metric": "hit_at_1", "value": per_query["hit_at_1"].mean()},
            {"metric": "hit_at_3", "value": per_query["hit_at_3"].mean()},
            {"metric": "hit_at_5", "value": per_query["hit_at_5"].mean()},
            {"metric": "hit_at_10", "value": per_query["hit_at_10"].mean()},
            {"metric": "mean_reciprocal_rank", "value": per_query["reciprocal_rank"].mean()},
            {"metric": "mean_gold_rank", "value": per_query["rank"].mean()},
        ]
    )
    group_summary = per_query.groupby("topic_group").agg(
        queries=("query", "count"),
        hit_at_1=("hit_at_1", "mean"),
        hit_at_5=("hit_at_5", "mean"),
        mean_rank=("rank", "mean"),
    ).reset_index()

    retrieval.to_csv(RESULTS / "rag_retrieval_results.csv", index=False)
    per_query.to_csv(RESULTS / "retrieval_metrics_by_query.csv", index=False)
    summary.to_csv(RESULTS / "retrieval_summary.csv", index=False)
    group_summary.to_csv(RESULTS / "retrieval_summary_by_topic_group.csv", index=False)
    plot_summary(summary)
    plot_rank_distribution(per_query)
    plot_group_summary(group_summary)
    plot_readme_overview()
    print(summary.to_string(index=False))


def plot_summary(summary: pd.DataFrame) -> None:
    scores = summary[summary["metric"].isin(["hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mean_reciprocal_rank"])]
    plt.figure(figsize=(8, 4))
    plt.bar(scores["metric"], scores["value"], color="#3d6fb6")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("100-Query Retrieval Quality Metrics")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "retrieval_metrics.png", dpi=180)
    plt.close()


def plot_rank_distribution(per_query: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(per_query["rank"], bins=range(1, min(per_query["rank"].max() + 3, 25)), color="#4a8f5a", edgecolor="white")
    plt.xlabel("Gold document rank")
    plt.ylabel("Number of queries")
    plt.title("Gold Document Rank Distribution")
    plt.tight_layout()
    plt.savefig(RESULTS / "gold_rank_distribution.png", dpi=180)
    plt.close()


def plot_group_summary(group_summary: pd.DataFrame) -> None:
    top = group_summary.sort_values("hit_at_1").head(12)
    plt.figure(figsize=(8, 5))
    plt.barh(top["topic_group"], top["hit_at_1"], color="#b26a3b")
    plt.xlim(0, 1.05)
    plt.xlabel("Hit@1")
    plt.title("Hardest Topic Groups by Hit@1")
    plt.tight_layout()
    plt.savefig(RESULTS / "topic_group_hit_at_1.png", dpi=180)
    plt.close()


def plot_readme_overview() -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    boxes = [
        ("100 abstracts\n20 topic groups", 0.15),
        ("100 labeled\nqueries", 0.39),
        ("TF-IDF retrieval\nTop-10 ranking", 0.63),
        ("Hit@k, MRR,\nrank analysis", 0.86),
    ]
    for text, xpos in boxes:
        ax.text(xpos, 0.55, text, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef6ff", edgecolor="#336699"))
    for start, end in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(end[1] - 0.11, 0.55), xytext=(start[1] + 0.11, 0.55), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("100-document RAG retrieval evaluation workflow", fontsize=15)
    fig.tight_layout()
    fig.savefig("assets/readme_project_overview.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
