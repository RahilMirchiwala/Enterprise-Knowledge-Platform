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

def find_similar(doc_id: str, threshold: float = 0.4, top_k: int = 5):
    source_results = collection.get(
        ids=[doc_id],
        include=["embeddings"]
    )

    if not source_results["ids"]:
        raise ValueError(f"Document {doc_id} not found in index")
    
    source_embedding = source_results["embeddings"][0]

    results = collection.query(
        query_embeddings=[source_embedding],
        n_results=top_k + 1
    )

    output = []
    for i in range(len(results["ids"][0])):
        similar_doc_id = results["ids"][0][i]

        if similar_doc_id == doc_id:
            continue
        
        score = round((1 - results["distances"][0][i]) * 100, 2)

        if score < threshold * 100:
            continue

        if score >= 90:
            category = "Near Duplicate"
        elif score >= 70:
            category = "Highly Similar"
        elif score >= 40:
            category = "Moderately Similar"
        elif score >= 10:
            category = "Slightly Related"
        else:
            category = "Unrelated"
        
        output.append({
            "doc_id": similar_doc_id,
            "similarity_score": score,
            "category": category
        })
    
    return output