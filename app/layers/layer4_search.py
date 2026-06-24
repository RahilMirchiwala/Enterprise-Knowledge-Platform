import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS_JSON = os.path.join(BASE_DIR, "documents.json")

with open(DOCS_JSON, "r", encoding="utf-8") as f:
    documents = json.load(f)

doc_ids = list(documents.keys())
doc_texts = list(documents.values())

# TF-IDF — lightweight!
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(doc_texts)

def search(query: str, top_k: int = 3):
    m, emb = get_model()
    query_vector = m.encode([query])
    scores = cosine_similarity(query_vector, emb)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "doc_id": doc_ids[idx],
            "score": round(float(scores[idx]) * 100, 2),
            "text": doc_texts[idx][:2000]
        })
    return results