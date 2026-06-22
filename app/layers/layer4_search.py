import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS_JSON = os.path.join(BASE_DIR, "documents.json")

with open(DOCS_JSON, "r", encoding="utf-8") as f:
    documents = json.load(f)

doc_ids = list(documents.keys())
doc_texts = list(documents.values())

print("Loading SBERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(doc_texts)
print("SBERT ready!")

def search(query: str, top_k: int = 3):
    query_vector = model.encode([query])
    scores = cosine_similarity(query_vector, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "doc_id": doc_ids[idx],
            "score": round(float(scores[idx]) * 100, 2),
            "text": doc_texts[idx][:2000]
        })
    return results