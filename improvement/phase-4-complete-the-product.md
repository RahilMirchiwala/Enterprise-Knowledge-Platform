# Phase 4 — Complete the Product

**Timeline:** 2–3 weeks  
**Goal:** Build the feature the project was always supposed to have — duplicate detection. Turn the experiment series into a coherent, working product with a real use case story.

---

## Why This Phase Exists

Read your own README introduction:

> "NovaBridge Consulting has 30+ business documents across 5 departments — many of which are duplicates, regional variants, or updated versions. Finding similar or duplicate documents manually is time-consuming and error-prone. The Goal: Build and evaluate different algorithms to automatically detect document similarity."

Now look at your API endpoints:
- `GET /search` — search for relevant documents given a query
- `GET /ask` — answer a question using RAG

Neither of these is **duplicate detection**. You spent 6 experiments proving which algorithm best identifies similar documents — then you built an app that doesn't use that capability. The original stated goal was never delivered.

This matters for two reasons:

1. **Portfolio credibility.** If someone reads your README, then uses your app, and asks "where's the duplicate detection?" — there's no good answer. You promised it and didn't ship it.

2. **Product thinking.** The ability to trace a business problem all the way through research to a working feature is exactly what Jeavio looks for. Their clients don't care about F1 scores. They care about "can this system find my duplicate contracts?"

This phase closes that loop.

---

## Concept 1 — What "Duplicate Detection" Actually Means at an Enterprise Level

In the NovaBridge scenario, "duplicate" is not binary. It exists on a spectrum — which is why you created 5 similarity categories in your evaluation dataset:

| Category | Meaning | Example |
|---|---|---|
| Near Duplicate (90–100%) | Same document, minor edits | HR-001 v1.0 vs HR-001 v1.1 |
| Highly Similar (70–89%) | Same purpose, different region | India Onboarding vs UK Onboarding |
| Moderately Similar (40–69%) | Related topic, different scope | Expense Policy vs Travel Policy |
| Slightly Related (10–39%) | Same department, different function | HR Onboarding vs HR Performance Review |
| Unrelated (0–9%) | Different department/function | HR Onboarding vs Engineering Deployment SOP |

A useful duplicate detection system needs to:
1. Accept a document as input
2. Return ranked matches with their similarity score
3. Categorise the match (Near Duplicate, Highly Similar, etc.)
4. Explain *why* they're similar (shared entities, shared terms, shared structure)

Point 4 is what separates a useful tool from a black box. A lawyer reviewing duplicate contracts doesn't just want to know "these two are 87% similar" — they want to know *what's the same* and *what's different*.

---

## Concept 2 — The Difference Between Search and Duplicate Detection

These two features sound similar but have different designs:

**Search:** User provides a *query string* → system returns relevant *documents*  
**Duplicate detection:** User provides a *document* → system returns similar *documents*

The mechanics overlap (both use vector similarity) but the input, threshold, and output format are different. Duplicate detection should:
- Accept a full document (not a short query)
- Apply a similarity *threshold* (only return results above 40%, for example)
- Return a category label alongside the score
- Ideally highlight *what* makes them similar

Building both correctly shows you understand the difference between retrieval tasks — that's interview-level knowledge.

---

## Concept 3 — Chunking (Why It Matters for Long Documents)

Your current system encodes entire documents as single 384-dimensional vectors. For 30 short business documents, this works. For longer documents (10-page contracts, 50-page policy manuals), it fails.

Why? A single embedding vector has to represent the *entire* document. If a 50-page contract has 5 relevant paragraphs buried in 200 irrelevant ones, the embedding "averages out" the signal and the relevant parts get diluted.

The solution is **chunking**: split documents into smaller pieces (paragraphs, or fixed-size overlapping windows), embed each chunk separately, and match at the chunk level.

```
Without chunking:
  50-page contract → 1 embedding → rough match

With chunking:
  50-page contract → 200 paragraph embeddings → precise match at paragraph level
```

For your 30 short documents, chunking may not change results much. But implementing it shows you understand *why* production RAG systems chunk — and demonstrates forward-thinking architecture.

---

## Tasks

---

### 1. Add a `POST /detect-duplicates` endpoint

**What to do:**  
Add a new endpoint that accepts a document (either by `doc_id` from the existing corpus, or as a file upload) and returns ranked similar documents with similarity scores and category labels.

```python
@app.post("/detect-duplicates")
def detect_duplicates(doc_id: str, threshold: float = 0.4, top_k: int = 5):
    """
    Given a document ID, find all similar documents above the threshold.
    Returns ranked results with similarity score and category label.
    """
    results = find_similar(doc_id, threshold, top_k)
    return {
        "source_document": doc_id,
        "duplicates_found": len(results),
        "results": results
    }
```

The response should include the category label:
```json
{
  "source_document": "HR-001",
  "duplicates_found": 2,
  "results": [
    {
      "doc_id": "HR-002",
      "similarity_score": 84.2,
      "category": "Highly Similar",
      "shared_entities": ["HR", "Onboarding", "Policy"]
    }
  ]
}
```

**Why this matters:**  
This is the endpoint that delivers on the original promise of the project. Without it, all the experiments are academic exercises. With it, you have a working product that solves the stated business problem.

---

### 2. Add chunk-level similarity (document chunking)

**What to do:**  
Modify `layer1_extract.py` to also produce a chunked version of each document — split by paragraph or by 200-word windows with 50-word overlap. Store chunk embeddings in ChromaDB alongside the full-document embeddings, in a separate collection.

Add a `chunked=True` parameter to `/detect-duplicates` that switches between full-document and chunk-level matching.

**Why this matters:**  
This shows you understand the tradeoffs in embedding design. Full-document embeddings are faster and simpler. Chunk-level embeddings are more precise for longer documents. Giving the caller control over this shows API design maturity.

The 50-word overlap between chunks is called a **sliding window** and it's important: it ensures that a sentence at the boundary of a chunk isn't cut off and missed. Understanding why this overlap exists is a common interview question for RAG engineering roles.

---

### 3. Build a simple web UI for duplicate detection

**What to do:**  
Your `static/` folder already exists. Build a simple HTML + JavaScript page (`static/index.html`) that lets a user:
1. Select a document from a dropdown (populated from `GET /`)
2. Click "Find Duplicates"
3. See the results as a ranked list with score and category

No frameworks needed — plain HTML, CSS, and `fetch()` calls to your API. Keep it simple and functional.

**Why this matters:**  
A UI makes your project understandable to non-technical stakeholders in 10 seconds. A Jeavio hiring manager can open a browser and immediately understand what the system does. Without a UI, they have to read API docs and make curl commands — and most won't bother.

This is a business communication skill, not just a technical one. The best ML systems in the world fail to get adopted because their builders never made them accessible to the people who would use them.

---

### 4. Write a case study in the README

**What to do:**  
Replace the current README "My Approach" section with a structured case study:

```
## Case Study: NovaBridge Document Intelligence

### The Business Problem
[2-3 sentences on the problem and its cost]

### The Approach
[The experiment progression — why you started with TF-IDF, what each step added]

### Results
[Link to RESULTS.md — the comparison table with proper metrics]

### What's Running in Production
[Describe the final architecture: ChromaDB + SBERT + Groq RAG + duplicate detection]

### Limitations and Next Steps
[Be honest: 30 documents is a small dataset, model isn't fine-tuned, etc.]
```

**Why this matters:**  
This format is exactly how you'd present this project in a Jeavio interview, a client presentation, or a technical blog post. The ability to frame technical work as a business solution — problem → approach → evidence → result — is a senior communication skill. Start building it now.

The "Limitations" section is especially important. Professionals who know the limits of their own systems are trusted more than those who oversell. A candidate who says "here's what it does well and here's where it breaks down" demonstrates maturity.

---

## How You Know Phase 4 Is Complete

- [ ] `POST /detect-duplicates` endpoint works and returns similarity scores with category labels
- [ ] Chunk-level similarity is implemented and accessible via an API parameter
- [ ] `static/index.html` provides a working UI for the duplicate detection feature
- [ ] README has a proper case study section with Problem → Approach → Results → Limitations structure

---

## What You Will Have Learned

By the end of Phase 4, you will understand what it means to **close a product loop** — to take a stated business problem, research a solution, build it, and deliver something a non-technical person can use. This is the full stack of a data scientist's job. Most people can do one or two of those steps. Doing all four is what makes you valuable.
