# Phase 3 — Replace Toy Search with a Real Vector Store

**Timeline:** 2–3 weeks  
**Goal:** Swap TF-IDF retrieval for a vector database backed by SBERT embeddings. Understand why every production RAG system is built this way.

---

## Why This Phase Exists

After Phase 1, your app uses SBERT for search. That's correct and already a big improvement. But there's still a hidden problem: **the embeddings are recomputed from scratch every time the server starts**.

Look at `layer4_search.py` after Phase 1. It will look something like:

```python
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_texts = list(documents.values())
doc_embeddings = model.encode(doc_texts)  # runs every server start
```

On 30 documents this takes 1–2 seconds. Fine. On 30,000 documents it takes 20 minutes. On 3 million documents — a realistic enterprise knowledge base — it takes hours. And it re-runs every time you restart the server.

This is called **recomputing work you've already done** and it's one of the most common architectural mistakes in ML systems.

The solution is a **vector database**: a system that stores pre-computed embeddings and retrieves them efficiently. You compute embeddings once, store them, and query them forever.

---

## Concept 1 — What a Vector Database Actually Is

You already understand that SBERT converts text into a 384-dimensional vector. A vector database is simply a system optimised for answering one question:

> "Given this query vector, find me the K stored vectors that are most similar to it."

This operation is called **Approximate Nearest Neighbour (ANN) search**. It's "approximate" because finding the exact nearest neighbour in 384 dimensions requires comparing against every stored vector — which is too slow at scale. ANN algorithms (like HNSW, the algorithm FAISS uses) build an index structure that lets you find "close enough" neighbours in milliseconds instead of seconds.

The tradeoff: ANN might occasionally miss the absolute best match in favour of finding 99% of the best matches in 1% of the time. For document search, this is always the right tradeoff.

```
Without vector DB (your current system):
  Start server → load 30 docs → encode all 30 → hold in RAM → search on query

With vector DB:
  One-time: encode 30 docs → store embeddings on disk
  Start server → load index from disk (milliseconds) → search on query

At scale (30,000 docs):
  Without: 20+ minutes every restart
  With:    < 1 second every restart
```

---

## Concept 2 — FAISS vs ChromaDB

There are two good options for your project. Choose one and understand why you chose it.

### FAISS (Facebook AI Similarity Search)
- Built by Meta, used in production at massive scale (billions of vectors)
- Extremely fast, very efficient memory usage
- Lower-level API — you manage the index yourself (add, save, load)
- No metadata storage — you store text and IDs separately
- Best for: when you care most about speed and control

### ChromaDB
- Newer, built specifically for LLM/RAG applications
- Higher-level API — handles embeddings, metadata, and documents together
- Slower than FAISS but much easier to use
- Built-in filtering by metadata (e.g., "find similar documents in the HR department only")
- Best for: when you want to ship fast and need metadata filtering

**Recommendation for your project:** Use **ChromaDB** for Phase 3. The metadata filtering (department, region, doc type) aligns directly with the NER work you did in Experiment 05, and the simpler API means you'll understand what's happening rather than fighting FAISS's index management.

After you've shipped with ChromaDB, read the FAISS documentation and understand the performance difference. Knowing both is interview-grade knowledge.

---

## Concept 3 — Why This Matters for RAG

Your RAG pipeline is only as good as its retrieval step. The pipeline is:

```
User question → retrieve relevant documents → pass to LLM → answer
```

If retrieval returns the wrong documents, the LLM has bad context. It will either:
1. Answer incorrectly based on irrelevant information ("hallucinating with confidence")
2. Say "I don't have enough information" — which is honest, but useless

The quality of a RAG system is roughly: **50% retrieval, 50% generation**. Most beginners focus all their attention on the LLM (the exciting part). Senior ML engineers know that improving retrieval is usually where the biggest quality gains are.

By putting your SBERT embeddings in ChromaDB, you enable:
- **Persistent storage** — embeddings survive server restarts
- **Dynamic ingestion** — add new documents without recomputing everything
- **Metadata filtering** — "only search Finance documents" (this is huge for enterprise use cases)
- **Scalability** — works the same way whether you have 30 or 30,000 documents

---

## Tasks

---

### 1. Set up ChromaDB and create a persistent collection

**What to do:**  
Install `chromadb`. Create a script `app/scripts/build_index.py` that:
1. Reads all documents from `documents.json`
2. Encodes them with SBERT
3. Stores them in a ChromaDB collection with metadata (doc_id, department, doc_type, region from your NER layer)
4. Saves the collection to disk at `app/chroma_db/`

```python
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="app/chroma_db")
collection = client.get_or_create_collection("documents")

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(doc_texts).tolist()

collection.add(
    ids=doc_ids,
    embeddings=embeddings,
    documents=doc_texts,
    metadatas=[{"department": "HR", "region": "India"} for ...]  # from NER
)
```

**Why this matters:**  
This separates the "build the index" step (done once) from the "query the index" step (done on every request). This is the correct architecture for any ML system that has expensive setup work.

---

### 2. Rewrite `layer4_search.py` to query ChromaDB

**What to do:**  
Replace the TF-IDF/SBERT + in-memory compute with a ChromaDB query:

```python
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="app/chroma_db")
collection = client.get_collection("documents")
model = SentenceTransformer('all-MiniLM-L6-v2')

def search(query: str, top_k: int = 3, department: str = None):
    query_embedding = model.encode([query]).tolist()
    
    where_filter = {"department": department} if department else None
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter
    )
    return results
```

Notice the `department` filter. This is only possible because you stored metadata. This is a feature your TF-IDF implementation could never support.

**Why this matters:**  
You've just made your search endpoint stateless and fast. The model and index load once when the server starts. Each query is a millisecond database lookup, not a full encode-and-compare pass over all documents.

---

### 3. Add a `POST /ingest` endpoint

**What to do:**  
Add a new endpoint to `main.py` that accepts a `.docx` file upload and adds it to the ChromaDB index.

```python
from fastapi import UploadFile, File

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    # extract text → classify → NER → encode → add to ChromaDB
    ...
    return {"message": f"Document {file.filename} ingested successfully", "doc_id": doc_id}
```

**Why this matters:**  
Right now your platform has 30 hardcoded documents. That's a demo, not a product. A real knowledge platform needs to accept new documents. The `/ingest` endpoint is what transforms your project from "a system that works on 30 fixed files" to "a system that can grow".

This is also a good test of your layered architecture. If the layers are well-designed, `/ingest` should just call `layer1_extract` → `layer2_classify` → `layer3_ner` → `layer4_search.add()` in sequence. If you struggle to wire them together, it means the layer boundaries need work.

---

### 4. Benchmark: ChromaDB vs old TF-IDF

**What to do:**  
Write a simple benchmark script that times:
1. Server startup time (old TF-IDF vs ChromaDB)
2. Query time for a single search request
3. Query time for 100 concurrent requests (use Python's `threading` or `asyncio`)

Report the numbers in your `RESULTS.md`.

**Why this matters:**  
Numbers tell a story that words can't. "ChromaDB is faster" is a claim. "ChromaDB serves 100 queries in 0.8 seconds vs TF-IDF's 12 seconds at startup" is evidence. This is the difference between asserting something and proving it.

When you present this to Jeavio, having benchmarks shows engineering rigour — you didn't just assume the new approach was better, you measured it.

---

## How You Know Phase 3 Is Complete

- [ ] `app/scripts/build_index.py` exists and builds a persistent ChromaDB collection
- [ ] `app/layers/layer4_search.py` queries ChromaDB, not TF-IDF or raw SBERT
- [ ] `POST /ingest` endpoint works — you can upload a new `.docx` and query it immediately
- [ ] `GET /search` accepts an optional `department` parameter and filters correctly
- [ ] Benchmark numbers documented in `RESULTS.md`

---

## What You Will Have Learned

By the end of Phase 3, you will understand the architectural difference between **research code** (compute what you need when you need it) and **production code** (pre-compute and index, then serve fast). This is one of the most important conceptual transitions you'll make as an ML engineer. Nearly every scalability problem in ML systems comes down to this distinction.
