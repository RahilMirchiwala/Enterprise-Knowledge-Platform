import os
import json
import pandas as pd # type: ignore
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import numpy as np

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS_JSON = os.path.join(BASE_DIR, "documents.json")
EVAL_CSV  = os.path.join(BASE_DIR, "evaluation_dataset.csv")

with open(DOCS_JSON,"r",encoding="utf-8") as f:
  documents = json.load(f)
  doc_ids = list(documents.keys())
  doc_texts = list(documents.values())
  print(f"Loaded {len(doc_ids)} documents")

print("Creating embending...")
embeddings = model.encode(doc_texts,show_progress_bar=True)
print(f"Embending Shape: {embeddings.shape}")

similarity_matrix = cosine_similarity(embeddings)
print(f"Similarity matrix shape: {similarity_matrix.shape}")

results = []
for i,j in combinations(range(len(doc_ids)),2):
  score = similarity_matrix[i][j]
  results.append({
    "Doc A": doc_ids[i],
    "Doc B": doc_ids[j],
    "SBERT Score": round(score* 100, 2)
  })

df = pd.DataFrame(results)
df = df.sort_values("SBERT Score", ascending=False).reset_index(drop=True)

print("\nTop 20 Most Similar Pairs:")
print(df.head(20).to_string(index=False))

import csv

human_labels = []
with open(EVAL_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["doc_a"]:
            human_labels.append((
                row["doc_a"],
                row["doc_b"],
                int(row["human_similarity"]),
                row["expected_category"]
            ))

print("\nHuman vs SBERT Comparison:")
print(f"{'Doc A':<10} {'Doc B':<10} {'Category':<20} {'Human':>7} {'SBERT':>8} {'Gap':>8}  Flag")
print("-" * 80)

for doc_a, doc_b, human_score, category in human_labels:
    match = df[
        ((df["Doc A"].str.contains(doc_a)) & (df["Doc B"].str.contains(doc_b))) |
        ((df["Doc A"].str.contains(doc_b)) & (df["Doc B"].str.contains(doc_a)))
    ]
    if not match.empty:
        sbert_score = match.iloc[0]["SBERT Score"]
        gap = sbert_score - human_score
        flag = "FP" if gap > 20 else ("FN" if gap < -20 else "OK")
        print(f"{doc_a:<10} {doc_b:<10} {category:<20} {human_score:>7} {sbert_score:>8} {gap:>+8.1f}  {flag}")

df.to_csv("sbert_results.csv", index=False)
print("\nResults saved to sbert_results.csv")
