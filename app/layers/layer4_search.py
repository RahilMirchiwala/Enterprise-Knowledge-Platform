import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_JSON = os.path.join(BASE_DIR, "documents.json")

# Documents load karo
with open(DOCS_JSON, "r", encoding="utf-8") as f:
    documents = json.load(f)

doc_ids = list(documents.keys())
doc_texts = list(documents.values())

# TF-IDF — lightweight! (~20MB vs SBERT 400MB)
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)
tfidf_matrix = vectorizer.fit_transform(doc_texts)

print(f"TF-IDF ready! {len(doc_ids)} documents indexed.")

def search(query: str, top_k: int = 3):
    """Query se relevant documents dhundho"""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "doc_id": doc_ids[idx],
            "score": round(float(scores[idx]) * 100, 2),
            "text": doc_texts[idx][:1500]
        })
    return results