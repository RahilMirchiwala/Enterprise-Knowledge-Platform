# Phase 1 — Fix & Professionalize the Repository

**Timeline:** 1–2 weeks  
**Goal:** Make the repo production-honest. Every fix here is a professional habit you are building for life.

---

## Why This Phase Exists

You built something real. Six experiments, a FastAPI app, a layered architecture — that is more than most CS students ever do. But right now there is a gap between what the repo *claims* and what it actually *does*. That gap is the first thing a senior engineer or hiring manager will notice.

In professional software, **trust is built through consistency**. If your README says "Coming Soon" for experiments you finished, or your production app runs the algorithm your own experiments proved was worse, it signals one thing: you built this fast and didn't go back to clean it up. That's fine for a personal weekend project. It's not fine for a portfolio you're using to get hired at a company that ships production AI.

Every task in this phase is about closing the gap between what you say and what you do.

---

## Tasks

---

### 1. Wire SBERT into `layer4_search.py` instead of TF-IDF

**What to do:**  
Replace the TF-IDF vectorizer in `app/layers/layer4_search.py` with `SentenceTransformer('all-MiniLM-L6-v2')` and cosine similarity over SBERT embeddings.

**Why this matters:**  
This is the single most important fix in the repo. Here is the problem in plain terms:

- Your Experiment 01 proved TF-IDF gets 5/25 pairs correct.
- Your Experiment 02 proved SBERT gets 14/25 pairs correct — nearly 3× better.
- Your production app (`layer4_search.py`) uses TF-IDF anyway.

This means the RAG pipeline in `/ask` — your most impressive feature — is retrieving documents with the worst algorithm you tested. The LLM gets bad context and gives worse answers. You proved this yourself. Then you ignored it.1

This is called **not closing the loop** and it's one of the most common junior mistakes in ML engineering. Research and production have to agree. If your experiment proves X is better, X goes into production. Always.

The deeper lesson: experiments are only valuable if you act on them. Otherwise 3 they're just graphs.

---

### 2. Fill in `layer1_extract.py`

**What to do:**  
Move the document text extraction logic from `experiment/01_TF-IDF/extract_text.py` into `app/layers/layer1_extract.py`. This layer should expose a function like `extract_text(file_path: str) -> str`.

**Why this matters:**  
You have a layered architecture (`layer1` → `layer2` → ... → `layer5`). That design communicates that each layer has a clear, single responsibility. But `layer1_extract.py` is empty. That means Layer 1 — the foundation of your entire pipeline — does nothing.

Empty files in a codebase are confusing to anyone who reads it. They look like unfinished work. A reader has to wonder: is this intentional? Is it a placeholder? Did the author forget? You never want a reader to ask those questions.

The deeper lesson: Architecture is a promise. If you define five layers, all five must exist and do something. Otherwise don't define them.

---

### 3. Add a `.env.example` file

**What to do:**  
Create a file called `.env.example` at the root of the repo with this content:

```
GROQ_API_KEY=your_groq_api_key_here
```

**Why this matters:**  
Your app requires `GROQ_API_KEY` to run. Without it, `layer5_llm.py` will throw an error. But nowhere in the repo does it tell a new developer this. They have to read the source code, find the `os.getenv("GROQ_API_KEY")` call, then figure out they need a Groq account and API key.

This is called a **poor developer experience (DX)**. In professional teams, if someone clones your repo and can't run it in under 10 minutes, that's a failure. `.env.example` is the standard convention for documenting required environment variables. It costs you 2 minutes to write and saves every future collaborator (or interviewer) from confusion.

The `.env` file itself (with real secrets) must never be committed to git — that's already in your `.gitignore`, which is good. `.env.example` has no real secrets, just the variable names and placeholder values.

The deeper lesson: Code is read more often than it is written. Always optimise for the reader's experience.

---

### 4. Fix the main README

**What to do:**  
- Remove "Coming Soon" from Experiments 02 and 03 in the results summary table — they are complete.
- Fix the directory name in the project structure diagram: it shows `experiments/` but the real directory is `experiment/` (no s).
- Update the results summary table to include the actual results from all 6 experiments.

**Why this matters:**  
Documentation that is wrong is often worse than no documentation. A reader following your README will look for `experiments/01_tfidf/` and find nothing because the real path is `experiment/01_TF-IDF/`. They'll assume the project is broken. You'll lose them before they even read a line of code.

The "Coming Soon" entries are a smaller version of the same problem. They signal that the work is incomplete, when it isn't. You've done 6 experiments. Let the README say so.

Stale documentation is one of the most consistent complaints senior engineers have about junior work. It's not about the writing quality — it's about the habit of updating docs when you update code. Every time you change something, the docs change with it.

The deeper lesson: Documentation is code. It rots the same way. Treat it with the same care.

---

### 5. Add error handling to FastAPI endpoints

**What to do:**  
Wrap the logic in `/search` and `/ask` in try/except blocks. Return a proper HTTP 422 or 500 response with a meaningful message instead of crashing.

Example:
```python
from fastapi import HTTPException

@app.get("/search")
def search_docs(query: str, top_k: int = 3):
    try:
        results = search(query, top_k)
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
```

**Why this matters:**  
Right now, if `search()` throws an exception — bad input, missing model file, memory error — the whole server crashes with a raw Python traceback. In production, that means:

1. The user sees a confusing 500 error with no context.
2. The error leaks internal implementation details (file paths, variable names) which is a security risk.
3. Depending on the deployment, the server may stay down until someone manually restarts it.

Error handling is not about covering edge cases you think are unlikely. It's about designing for the reality that **systems fail in ways you don't predict**. A good engineer assumes failure and handles it gracefully. A great engineer surfaces enough information in the error to make it debuggable.

The deeper lesson: Defensive programming is not pessimism. It's respect for the complexity of real systems.

---

### 6. Remove Hindi/mixed-language comments from code

**What to do:**  
Go through `layer2_classify.py`, `layer4_search.py`, `layer5_llm.py`, and the experiment files. Replace comments like `"Documents load karo"` and `"query se relevant documents dhundho"` with English equivalents or remove them entirely if the code is self-explanatory.

**Why this matters:**  
This is about professional context, not culture. Hindi comments in your personal projects are completely fine — write however you think most clearly. But a portfolio repo is not a personal project. It is a professional artefact you are showing to an international company.

Jeavio works with private equity firms across multiple geographies. Their engineering team likely includes people from many different backgrounds. A codebase that only some readers can fully understand creates an invisible barrier. Professional codebases are written so that any qualified engineer can read them, regardless of language background.

There's also a subtler issue: mixed-language comments make it harder to search your own code (`grep "load"` won't find `"load karo"`), and they can confuse automated tools like linters and documentation generators.

The deeper lesson: Code comments are communication. Ask yourself: who is the audience? Write for them.

---

## How You Know Phase 1 Is Complete

- [ ] `app/layers/layer4_search.py` uses SBERT, not TF-IDF
- [ ] `app/layers/layer1_extract.py` has working extraction logic
- [ ] `.env.example` exists at the repo root
- [ ] Main README results table reflects all 6 completed experiments
- [ ] No directory name mismatches between the README and the actual repo
- [ ] `/search` and `/ask` endpoints return proper HTTP errors instead of crashing
- [ ] No non-English comments in any file under `app/`

---

## What You Will Have Learned

By the end of Phase 1, you will have practised the most important non-technical skill in professional engineering: **closing the loop**. Building something is 60% of the job. Going back, checking it against your own claims, and fixing the inconsistencies is the other 40% — and it's what separates junior engineers from ones people trust to work independently.
