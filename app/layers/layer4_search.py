import chromadb
from sentence_transformers import SentenceTransformer
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR,"chroma_db")

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection("documents")
print(f"Loaded {collection.count()} documents from ChromaDB")

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready!")

def search(query: str, top_k: int = 3):
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "doc_id": results["ids"][0][i],
            "score": round((1 - results["distances"][0][i]) * 100, 2),
            "text": results["documents"][0][i][:2000]
        })
    
    return output