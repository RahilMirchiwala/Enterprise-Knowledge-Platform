import os
import json
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS_JSON = os.path.join(BASE_DIR, "documents.json")

with open(DOCS_JSON, "r", encoding="utf-8") as f:
    documents = json.load(f)

doc_ids = list(documents.keys())
doc_texts = list(documents.values())

# print(f"Loaded {len(doc_ids)} documents")

print("Creating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(doc_texts, show_progress_bar=True)

# print(f"Embeddings shape: {embeddings.shape}")

def find_similar_docs(query,top_k=3):
  query_vector = model.encode([query])

  scores = cosine_similarity(query_vector,embeddings)[0]

  top_indices = np.argsort(scores)[::-1][:top_k]

  results = []
  for idx in top_indices:
     results.append({
        "doc_id": doc_ids[idx],
        "score": round(scores[idx]*100,2),
        "text": doc_texts[idx][:2000]
     })

  return results

query = "What is the remote work policy?"
results = find_similar_docs(query)

# print(f"Query: {query}\n")
# for r in results:
#     print(f"Doc: {r['doc_id']}")
#     print(f"Score: {r['score']}%")
#     print()

def ask_groq(query,relevant_docs):
  context = ""
  for doc in relevant_docs:
    context += f"Document: {doc['doc_id']}\n"
    context += f"Document: {doc['text']}\n\n"

  response = client.chat.completions.create(
     model='llama-3.3-70b-versatile',
     messages=[
        {
          "role": "system",
          "content": "You are a helpful assistant for NovaBridge Consulting. Answer questions based only on the provided documents."
        },
        {
          "role": "user",
          "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
     ],
     max_tokens=500
  )

  return response.choices[0].message.content

# Test 
# answer = ask_groq(query, results)
# print(f"Answer:\n{answer}")

queries = [
    "What are the expense reimbursement limits?",
    "How to handle a P1 incident?"
]

for q in queries:
    print(f"\nQuery: {q}")
    print("-" * 50)
    results = find_similar_docs(q)
    print(f"Found docs: {[r['doc_id'] for r in results]}")
    answer = ask_groq(q, results)
    print(f"Answer: {answer}")
