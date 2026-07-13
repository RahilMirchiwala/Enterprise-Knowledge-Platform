# Phase 6 — Interview Preparation for Jeavio

**Timeline:** Ongoing alongside other phases  
**Goal:** Be able to walk into a Jeavio interview and answer both "what did you build" and "what would you do differently at scale" with equal confidence.

---

## Why This Phase Is Different

The previous five phases are about improving your project. This phase is about improving *you* — specifically, your ability to articulate what you know, what you built, and how you think.

Jeavio is not a typical product company. They are a technology *partner* for private equity firms. That means their engineers:
- Talk to non-technical clients regularly (PE partners, portfolio company executives)
- Move between projects and domains frequently
- Are expected to learn new domains fast (fintech one quarter, healthcare the next)
- Need to clearly communicate tradeoffs, not just implement solutions

An interview at Jeavio is therefore not just a technical test. It is a test of whether you can think on your feet, explain your reasoning clearly, and be honest about what you don't know.

The most important thing to understand going in: **your project is your biggest asset and your biggest liability**. It's your asset because you built something real and can speak about it deeply. It's your liability because an interviewer will probe every corner of it — including the corners where it's weak.

Be ready.

---

## Part 1 — Understand Your Own Project Deeply

Before you can answer questions about Jeavio's AI work, you need to be able to answer any question about yours. Here are the questions you should be able to answer cold, without hesitation:

### Architecture Questions

**"Walk me through your system architecture end to end."**  
Practise a 3-minute explanation that covers: documents → text extraction → classification → NER → embedding → vector store → retrieval → LLM generation. Draw it on a whiteboard if needed. Know which layer does what.

**"Why did you use a layered architecture instead of one big file?"**  
Answer: each layer has a single, testable responsibility. Changing the LLM provider only touches `layer5_llm.py`. Changing the vector store only touches `layer4_search.py`. This is called **separation of concerns** and it's why the system is maintainable.

**"What happens if the Groq API goes down?"**  
Your current code will crash with an unhandled exception. Know this. The correct answer is: "Right now it fails ungracefully. In production I'd add a retry with exponential backoff, a fallback to a local model, and an error message to the user that explains the service is temporarily unavailable."

---

### ML/Algorithm Questions

**"Why did TF-IDF fail on regional document variants?"**  
TF-IDF represents documents as bags of words. It has no concept of meaning — only word frequency. "Aadhaar" and "National Insurance Number" have zero word overlap but represent the same concept (government-issued identity verification). TF-IDF assigns 0% similarity. SBERT represents both phrases as nearby vectors in embedding space because it was trained on semantic similarity.

**"Why is SBERT better than TF-IDF for your use case?"**  
SBERT produces fixed-size embeddings (384 dimensions) that capture meaning, not vocabulary. It generalises across regional terminology, synonyms, and paraphrases. TF-IDF is vocabulary-dependent and fails whenever the same concept is expressed with different words.

**"What are the limitations of SBERT for your use case?"**  
`all-MiniLM-L6-v2` is a general-purpose model — not fine-tuned on business documents. It doesn't understand domain-specific terminology particularly well (e.g., it might not know that "ESIC" and "Employee State Insurance" are the same thing). Fine-tuning on a domain-specific corpus would improve results.

**"Your hybrid search experiment (Experiment 03) performed worse than SBERT alone. Why?"**  
Because TF-IDF introduced noise. The best weight grid search found that 0% TF-IDF + 100% SBERT was optimal — meaning TF-IDF contributed nothing and adding it only diluted the signal. The lesson: combining weak signals doesn't improve results; it degrades them.

---

### Evaluation Questions

**"How do you know your system is good enough?"**  
This is a trap question for anyone who only reports "X/25 correct". After Phase 2, your answer should be: "I evaluate against 25 human-labeled document pairs using Precision@3, Recall@3, F1, and Mean Reciprocal Rank. SBERT achieves [your numbers]. I know the limitations: 25 pairs is a small evaluation set, and the ground truth is single-annotator, so there's no inter-rater reliability measure."

**"What would you do differently if you had more time/data?"**  
Good answer: fine-tune SBERT on a larger corpus of business documents with human-labeled similarity pairs. Use a larger evaluation set (250+ pairs) with multiple annotators and measure inter-annotator agreement (Cohen's Kappa). Implement A/B testing on the live system to measure user satisfaction as the true north-star metric.

---

## Part 2 — Core Data Science Concepts to Study

These are topics Jeavio will expect you to know. Study each one deeply enough to explain it to a non-technical person AND to discuss implementation tradeoffs.

---

### RAG (Retrieval Augmented Generation)

**What to know:**  
- The full pipeline: question → retrieval → context assembly → generation → answer
- Why RAG exists: LLMs have a knowledge cutoff and can't know proprietary information. RAG grounds the LLM in real, up-to-date documents.
- Failure modes: retrieval fails (wrong documents retrieved) → LLM hallucinates with false confidence. Generation fails (LLM ignores context) → answer doesn't reflect the documents.
- Key design decisions: chunk size, overlap, top-K, re-ranking, prompt template

**Why Jeavio cares:** They build RAG systems for PE firm knowledge bases. This is their core product.

---

### Vector Databases and Embeddings

**What to know:**  
- Why vector DBs exist: traditional databases can't do "find me the most similar thing to X"
- Approximate Nearest Neighbour (ANN) vs exact search: ANN is faster but not perfect; for document search, ANN is always the right tradeoff
- Common vector DBs: FAISS (Meta, fastest), ChromaDB (easiest for RAG), Pinecone (managed, expensive), Weaviate (open source, feature-rich)
- Embedding models: `all-MiniLM-L6-v2` (fast, 80MB, general), `all-mpnet-base-v2` (slower, more accurate), OpenAI `text-embedding-3-small` (API, very good, costs money)

**Why Jeavio cares:** Every production AI system they build uses a vector store. You need to know which one to pick and why.

---

### LLM Fundamentals

**What to know:**  
- **Temperature**: controls randomness. 0 = deterministic (same input, same output every time). 1 = creative/unpredictable. For document Q&A, use low temperature (0.1–0.3) because you want accuracy, not creativity.
- **Context window**: the maximum number of tokens (words, roughly) an LLM can read at once. GPT-4 = 128K tokens. Llama 3.3 70B = 128K tokens. This limits how much document context you can pass in a single RAG call.
- **Hallucination**: when an LLM generates confident-sounding text that is factually wrong. RAG reduces hallucination by grounding the model in real documents. It does not eliminate it.
- **Prompt engineering**: the craft of writing instructions that guide LLM behaviour. Your system prompt (`"Answer only based on the provided documents"`) is a basic example. More advanced: few-shot examples, chain-of-thought prompting, structured output instructions.
- **Token cost**: LLM APIs charge per token. A naive RAG system that passes 10 full documents in every request will be expensive at scale. Production systems use summarisation, re-ranking, or shorter chunks to keep context lean.

**Why Jeavio cares:** They use LLMs as the generation layer in their AI products. You need to understand the tradeoffs, not just call the API.

---

### Evaluation and Metrics

Review everything from Phase 2 until you can explain it without notes:
- Precision vs Recall and when each matters more
- F1 as the harmonic mean (not arithmetic mean — know why harmonic)
- MRR for ranked retrieval systems
- NDCG (Normalised Discounted Cumulative Gain) — the next step beyond MRR; graded relevance instead of binary

---

### SQL

Jeavio works with enterprise clients who have their data in databases. You will be expected to query data.

Study:
- `SELECT`, `WHERE`, `GROUP BY`, `ORDER BY`, `LIMIT`
- `JOIN` types (INNER, LEFT, RIGHT) and when to use each
- Window functions (`ROW_NUMBER()`, `RANK()`, `LAG()`, `LEAD()`)
- Aggregations (`COUNT`, `SUM`, `AVG`, `MAX`, `MIN`)
- Subqueries and CTEs (`WITH cte AS (...)`)

Practice resource: [Mode SQL Tutorial](https://mode.com/sql-tutorial/) and [LeetCode SQL problems](https://leetcode.com/problemset/database/) (start with Easy, work to Medium).

---

## Part 3 — System Design for AI Systems

At some point in a Jeavio interview, you may be asked to design an AI system from scratch. This is where your project experience becomes directly useful. Here's how to approach it:

### The Question You're Likely to Get

*"Design a document Q&A system for a PE firm with 50,000 documents across 20 portfolio companies. Users should be able to ask natural language questions and get answers grounded in the actual documents."*

You have built exactly this. Here's how to answer:

**1. Clarify requirements first** (2–3 minutes)  
- How many users? Concurrent? Peak load?
- Latency requirements? (< 2 seconds? < 500ms?)
- Document types? (PDFs, Word, HTML?)
- Multi-tenant? (Each portfolio company sees only their own docs?)
- Languages? (English only?)

**2. Sketch the architecture** (5–10 minutes)  
```
Ingestion Pipeline:
  New document → text extraction → chunking → embedding → vector store

Query Pipeline:
  User question → embed query → vector search (top K chunks) 
  → re-rank → assemble context → LLM → answer + sources
```

**3. Call out the key design decisions**  
- Chunk size: 200–500 words with 50-word overlap (balances precision and context)
- Vector store: Pinecone or Weaviate for multi-tenant managed scale
- Embedding model: OpenAI `text-embedding-3-small` for quality/cost tradeoff
- LLM: GPT-4o or Llama 3.3 70B via Groq depending on latency/cost constraints
- Multi-tenancy: metadata filtering in the vector store by `company_id`

**4. Discuss limitations honestly**  
- Hallucination is not eliminated, just reduced
- RAG doesn't handle multi-hop reasoning well ("which portfolio company had the highest revenue growth?")
- Cold start latency for new document ingestion

You should be able to do this in 15–20 minutes because you've built a version of it.

---

## Part 4 — Soft Skills That Jeavio Explicitly Values

Their website says: "human-first culture", "co-ideation with clients", "psychological safety", "purpose-driven work". This is not marketing fluff — it describes what they look for in interviews.

**Be honest about what you don't know.**  
"I haven't worked with this at scale, but here's how I'd approach learning it" is a better answer than a confident guess that turns out to be wrong.

**Show curiosity, not just completion.**  
"I got 14/25 with SBERT. I'm not satisfied with that — here's what I want to try next" is more interesting than "I finished all 6 experiments".

**Ask good questions.**  
At the end of any interview you'll be asked "do you have questions?" Have 3 ready. Good ones:
- "What does the first 90 days look like for someone in this role?"
- "What's a recent AI project you shipped that you're most proud of, and what made it hard?"
- "How do you approach the gap between what an experiment proves and what actually ships?"

That last question is something you've lived through in this project. Asking it shows you're thinking at a level beyond "I built a thing and it worked".

---

## How You Know Phase 6 Is Ongoing

There's no checklist here. Instead: schedule one 45-minute mock interview with a peer, a mentor, or even yourself (record it and watch it back). Answer each question in this document out loud, as if you're in the interview. The ones you struggle to answer are the ones to study.

---

## Final Note

You built something genuinely impressive for a CS student. The gap between where you are and where Jeavio needs you to be is not large — it's mostly about finishing what you started, measuring it properly, and being able to explain it clearly.

The engineers who succeed at companies like Jeavio are not the ones who know the most. They're the ones who think clearly, communicate honestly, and keep learning. You're already showing all three of those things. This plan just makes it visible.
