import chromadb
from sentence_transformers import SentenceTransformer
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_JSON = os.path.join(BASE_DIR, "documents.json")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

with open(DOCS_JSON, "r", encoding="utf-8") as f:
    documents = json.load(f)

doc_ids = list(documents.keys())
doc_texts = list(documents.values())
print(f"Loaded {len(doc_ids)} documents")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(doc_texts, show_progress_bar=True)
print(f"Embeddings shape: {embeddings.shape}")

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

collection.add(
    ids=doc_ids,
    embeddings=embeddings.tolist(),
    documents=doc_texts,
)

print(f"\nDone! {collection.count()} documents indexed in ChromaDB")
print(f"Index saved at: {CHROMA_DIR}")