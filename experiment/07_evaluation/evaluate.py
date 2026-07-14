import json
import csv
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# ── Load evaluation dataset ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_CSV = os.path.join(BASE_DIR, "evaluation_dataset.csv")

human_labels = []
with open(EVAL_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["doc_a"]:
            human_labels.append({
                "doc_a": row["doc_a"],
                "doc_b": row["doc_b"],
                "human_score": int(row["human_similarity"]),
                "category": row["expected_category"]
            })

print(f"Loaded {len(human_labels)} labeled pairs")

# ── MRR function ──
def mean_reciprocal_rank(ranked_results, relevant_set):
    """
    ranked_results: list of doc_ids in ranked order
    relevant_set: set of doc_ids that are actually relevant
    """
    for rank, doc_id in enumerate(ranked_results, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0

# ── Evaluate function ──
def evaluate(experiment_name, predictions):
    """
    predictions: list of dicts with doc_a, doc_b, predicted_score
    threshold: gap within +-20 = correct
    """
    y_true = []
    y_pred = []

    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*60}")
    print(f"{'Doc A':<10} {'Doc B':<10} {'Human':>7} {'Pred':>7} {'Gap':>7} Flag")
    print("-" * 60)

    for label in human_labels:
        doc_a = label["doc_a"]
        doc_b = label["doc_b"]
        human = label["human_score"]

        # Find prediction
        pred_score = None
        for p in predictions:
            if (doc_a in p["doc_a"] and doc_b in p["doc_b"]) or 
               (doc_b in p["doc_a"] and doc_a in p["doc_b"]):
                pred_score = p["score"]
                break

        if pred_score is None:
            continue

        gap = pred_score - human
        is_correct = abs(gap) <= 20

        y_true.append(1 if human >= 40 else 0)
        y_pred.append(1 if pred_score >= 30 else 0)

        flag = "OK" if is_correct else ("FP" if gap > 20 else "FN")
        print(f"{doc_a:<10} {doc_b:<10} {human:>7} {pred_score:>7.1f} {gap:>+7.1f} {flag}")

    # Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    print(f"\nPrecision : {precision:.2f}")
    print(f"Recall    : {recall:.2f}")
    print(f"F1 Score  : {f1:.2f}")

    return {"precision": precision, "recall": recall, "f1": f1}

# Load TF-IDF results
tfidf_results = []
TFIDF_CSV = os.path.join(BASE_DIR, "TF_IDF_results.csv")

with open(TFIDF_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tfidf_results.append({
            "doc_a": row["Doc A"],
            "doc_b": row["Doc B"],
            "score": float(row["TF-IDF Score"])
        })

# Load SBERT results
sbert_results = []
SBERT_CSV = os.path.join(BASE_DIR, "sbert_results.csv")

with open(SBERT_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sbert_results.append({
            "doc_a": row["Doc A"],
            "doc_b": row["Doc B"],
            "score": float(row["SBERT Score"])
        })

# Evaluate both
tfidf_metrics = evaluate("TF-IDF + Cosine Similarity", tfidf_results)
sbert_metrics = evaluate("Sentence Transformers (SBERT)", sbert_results)

# Final comparison
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"{'Metric':<15} {'TF-IDF':>10} {'SBERT':>10}")
print("-"*40)
print(f"{'Precision':<15} {tfidf_metrics['precision']:>10.2f} {sbert_metrics['precision']:>10.2f}")
print(f"{'Recall':<15} {tfidf_metrics['recall']:>10.2f} {sbert_metrics['recall']:>10.2f}")
print(f"{'F1 Score':<15} {tfidf_metrics['f1']:>10.2f} {sbert_metrics['f1']:>10.2f}")