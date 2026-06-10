import os
import json
import pandas as pd # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from itertools import combinations

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS_JSON = os.path.join(BASE_DIR, "documents.json")
EVAL_CSV  = os.path.join(BASE_DIR, "evaluation_dataset.csv")

with open(DOCS_JSON, "r", encoding="utf-8") as f:
  documents = json.load(f)

doc_ids = list(documents.keys())
doc_texts = list(documents.values())

vectorizer = TfidfVectorizer(
  stop_words = "english",
  ngram_range= (1, 2),
  min_df=1
)

tfidf_matrix = vectorizer.fit_transform(doc_texts)
# print(f"Matrix shape: {tfidf_matrix.shape}")

similiarity_metrix = cosine_similarity(tfidf_matrix)
print(f"Similiarity metrix shape: {similiarity_metrix.shape}")

results = []

for i, j in combinations(range(len(doc_ids)),2):
  score = similiarity_metrix[i][j]
  results.append({
    "Doc A": doc_ids[i],
    "Doc B": doc_ids[j],
    "TF-IDF Score": round(score * 100, 2)
  })

df = pd.DataFrame(results)
df = df.sort_values("TF-IDF Score", ascending=False).reset_index(drop=True)
print(df.head(25).to_string(index=False))


import csv

# ── Load Human Labels from CSV ──
human_labels = []
with open(EVAL_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["doc_a"]:  # empty rows skip karo
            human_labels.append((
                row["doc_a"],
                row["doc_b"],
                int(row["human_similarity"]),
                row["expected_category"]
            ))

print(f"Loaded {len(human_labels)} human labeled pairs\n")

print("\nHuman vs TF-IDF Comparison:")
print(f"{'Doc A':<10} {'Doc B':<10} {'Human':>8} {'TF-IDF':>8} {'Gap':>8} {'Flag'}")
print("-" * 60)

for doc_a, doc_b, human_score, category in human_labels:
    match = df[
        ((df["Doc A"].str.contains(doc_a)) & (df["Doc B"].str.contains(doc_b))) |
        ((df["Doc A"].str.contains(doc_b)) & (df["Doc B"].str.contains(doc_a)))
    ]
    if not match.empty:
        tfidf_score = match.iloc[0]["TF-IDF Score"]
        gap = tfidf_score - human_score
        flag = "FP" if gap > 20 else ("FN" if gap < -20 else "TP")
        print(f"{doc_a:<10} {doc_b:<10} {human_score:>8} {tfidf_score:>8} {gap:>+8.1f}  {flag}")

df.to_csv("TF_IDF_results.csv", index=False)
print("\nResults saved to TF_IDF_results.csv")

