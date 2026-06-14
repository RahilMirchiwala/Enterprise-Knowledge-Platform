import os
import json
import csv
import pandas as pd # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS_JSON = os.path.join(BASE_DIR, "documents.json")
EVAL_CSV = os.path.join(BASE_DIR, "evaluation_dataset.csv")

with open(DOCS_JSON,"r",encoding="utf-8") as f:
  documents = json.load(f)

doc_ids = list(documents.keys())
doc_texts = list(documents.values())

# print(f"Loaded {len(doc_ids)} documents")

#TF-IDF
vectorizer = TfidfVectorizer(
  stop_words="english",
  ngram_range=(1,2),
  min_df = 1
)
tfidf_matrix = vectorizer.fit_transform(doc_texts)

#SBERT
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(doc_texts,show_progress_bar=True)

# print(f"TF-IDF matrix: {tfidf_matrix.shape}")
# print(f"SBERT embeddings: {embeddings.shape}")

#TF-IDF Similarity
tfidf_sim = cosine_similarity(tfidf_matrix) 

#SBERT Similarity
sbert_sim = cosine_similarity(embeddings)

# print(f"TF-IDF sim matrix: {tfidf_sim.shape}")
# print(f"SBERT sim matrix: {sbert_sim.shape}")


def normalize(matrix):
    min_val = matrix.min()
    max_val = matrix.max()
    return (matrix - min_val) / (max_val - min_val)

# Dono matrices normalize karo
tfidf_norm = normalize(tfidf_sim)
sbert_norm = normalize(sbert_sim)

print(f"TF-IDF range after normalize: {tfidf_norm.min():.2f} - {tfidf_norm.max():.2f}")
print(f"SBERT range after normalize:  {sbert_norm.min():.2f} - {sbert_norm.max():.2f}")

weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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

# print(f"Loaded {len(human_labels)} human labels")

best_score = 0
best_weight = 0
results_grid = []

for w_tfidf in weights:
    w_sbert = round(1.0 - w_tfidf, 1)
    
    # Hybrid matrix banao
    
    hybrid_sim = (w_tfidf * tfidf_norm) + (w_sbert * sbert_norm)
    
    # Har human label pe check karo
    ok_count = 0
    for doc_a, doc_b, human_score, category in human_labels:
        # Doc indexes dhundho
        if doc_a in " ".join(doc_ids) and doc_b in " ".join(doc_ids):
            i = next((idx for idx, d in enumerate(doc_ids) if doc_a in d), None)
            j = next((idx for idx, d in enumerate(doc_ids) if doc_b in d), None)
            
            if i is not None and j is not None:
                hybrid_score = round(hybrid_sim[i][j] * 100, 2)
                gap = hybrid_score - human_score
                if -20 <= gap <= 20:
                    ok_count += 1
    
    results_grid.append((w_tfidf, w_sbert, ok_count))
    print(f"TF-IDF: {w_tfidf} | SBERT: {w_sbert} | OK pairs: {ok_count}/25")
    
    if ok_count > best_score:
        best_score = ok_count
        best_weight = w_tfidf

print(f"\nBest weight - TF-IDF: {best_weight} | SBERT: {round(1.0 - best_weight, 1)}")
print(f"Best OK pairs: {best_score}/25")


metadata = {}
METADATA_CSV = os.path.join(BASE_DIR, "documents", "document_metadata.csv")

with open(METADATA_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        metadata[row["doc_id"]] = {
            "department": row["department"],
            "business_function": row["business_function"]
        }

print(f"Loaded metadata for {len(metadata)} documents")

def get_multilier(doc_a_id,doc_b_id):
   
   a_id = next((k for k in metadata.keys() if k in doc_a_id),None)
   b_id = next((k for k in metadata.keys() if k in doc_b_id),None)

   if a_id is None or b_id is None:
      return 1.0
   
   a = metadata[a_id]
   b = metadata[b_id]

   # Same dept + Same function
   if a["department"] == b["department"] and a["business_function"] == a["business_function"]:
      return 1.0
   
   #Same dep only
   elif a["department"] == b["department"]:
      return 0.8
   
   #Diff dep
   else:
      return 0.8
   
results_metadata = []

for i, j in combinations(range(len(doc_ids)), 2):
  sbert_score = sbert_sim[i][j]

  #Apply multiplier
  multiplier = get_multilier(doc_ids[i],doc_ids[j])

  final_score = min(sbert_score * multiplier, 1.0)

  results_metadata.append({
    "Doc A": doc_ids[i],
    "Doc B": doc_ids[j],
    "SBERT Score": round(sbert_score * 100, 2),
    "Multiplier": multiplier,
    "Final Score": round(final_score * 100, 2)
  })

df_meta = pd.DataFrame(results_metadata)
df_meta = df_meta.sort_values("Final Score", ascending=False).reset_index(drop=True)

# print("\nTop 10 pairs:")
# print(df_meta.head(10).to_string(index=False))print("\nHuman vs Metadata-Aware Scoring:")
multipliers = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

best_ok = 0
best_mult = 0

for mult in multipliers:
    ok_count = 0
    for doc_a, doc_b, human_score, category in human_labels:
        match = df_meta[
            ((df_meta["Doc A"].str.contains(doc_a)) & (df_meta["Doc B"].str.contains(doc_b))) |
            ((df_meta["Doc A"].str.contains(doc_b)) & (df_meta["Doc B"].str.contains(doc_a)))
        ]
        if not match.empty:
            row = match.iloc[0]
            sbert_score = row["SBERT Score"] / 100
            
            # Multiplier apply karo
            a_id = next((k for k in metadata.keys() if k in row["Doc A"]), None)
            b_id = next((k for k in metadata.keys() if k in row["Doc B"]), None)
            
            if a_id and b_id:
                a = metadata[a_id]
                b = metadata[b_id]
                if a["department"] == b["department"]:
                    m = 1.0
                else:
                    m = mult  # alag dept multiplier test kar rahe hain
                    
                final = min(sbert_score * m * 100, 100)
                gap = final - human_score
                if -20 <= gap <= 20:
                    ok_count += 1
    
    print(f"Alag dept multiplier: {mult} | OK: {ok_count}/25")
    if ok_count > best_ok:
        best_ok = ok_count
        best_mult = mult

print(f"\nBest multiplier: {best_mult} | Best OK: {best_ok}/25")