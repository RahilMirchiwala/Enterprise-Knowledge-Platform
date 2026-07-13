# Experiment 06 — Groq LLM + SBERT RAG Pipeline (Layer 6)

**Status:** Complete
**Result:** RAG pipeline working — SBERT retrieves relevant documents, Groq generates accurate answers

---

## What is RAG?

**RAG = Retrieval Augmented Generation**

```
R → Retrieval   = SBERT finds the most relevant documents
A → Augmented   = those documents are passed as context to the LLM
G → Generation  = Groq generates an answer grounded in that context
```

Without RAG:
```
User: "What are NovaBridge's expense reimbursement limits?"
LLM:  "I don't have access to NovaBridge's internal policies." ❌
```

With RAG:
```
User:  "What are the expense reimbursement limits?"
SBERT: retrieves FIN-001, FIN-005, FIN-007
Groq:  reads those documents → accurate, specific answer ✅
```

---

## Pipeline Architecture

```
User Question
      ↓
SentenceTransformer (all-MiniLM-L6-v2)
      ↓  query vector — 384 dimensions
ChromaDB vector search (top 3 most similar documents)
      ↓  retrieved document text
Groq API (llama-3.3-70b-versatile)
      ↓  reads document context + user question
Accurate Answer + Source Citations
```

**Note:** This experiment used direct SBERT cosine similarity.
The production app uses ChromaDB for persistent storage and faster lookup.

---

## Model

```
Groq API → llama-3.3-70b-versatile
         → 70 billion parameters
         → fast inference via Groq's custom hardware
         → free tier available at console.groq.com
```

---

## Results

### Query 1 — Expense Reimbursement Limits

```
Query: "What are the expense reimbursement limits?"

Retrieved:
→ FIN-001 Expense Reimbursement Policy (v1.0)
→ FIN-005 Expense Reimbursement Policy v2.0
→ FIN-007 Travel and Entertainment Policy

Answer:
FIN-001 (v1.0):
- Meals & Entertainment: INR 1,500 / GBP 40
- Local Travel: INR 800 / GBP 25
- Accommodation: INR 6,000/night / GBP 150/night

FIN-005 (v2.0, effective July 2026):
- Meals & Entertainment: INR 2,000/day (+33%)
- Local Travel: INR 1,200/day (+50%)
- Accommodation: INR 7,500/night (+25%)
- Communication: INR 750/month (+50%)
```

Groq automatically compared both versions and highlighted the differences.

---

### Query 2 — P1 Incident Handling

```
Query: "How to handle a P1 incident?"

Retrieved:
→ ENG-002 Incident Response Procedure
→ ENG-008 Disaster Recovery Plan
→ OPS-002 Escalation Matrix

Answer:
1. Detection & Alerting  → PagerDuty alert; acknowledge within 5 minutes
2. Triage & Escalation   → notify Engineering Manager and VP Engineering immediately
3. Mitigation            → isolate, apply fix or rollback; post updates every 15 minutes
4. Resolution & Review   → close ticket; blameless post-mortem within 48 hours
```

Groq synthesised information from 3 separate documents into a single structured answer.

---

### Query 3 — Remote Work Policy

```
Query: "What is the remote work policy?"

Retrieved:
→ HR-003 Remote Work Policy (Global)
→ HR-006 US Remote Work Policy

Answer:
Global policy (HR-003):
- 90-day probation required before eligibility
- VPN mandatory for all remote work
- Core hours: 10:00–16:00 local time

US-specific additions (HR-006):
- Cross-state remote work requires Finance + HR written approval
- California employees: full business expense reimbursement (CA Labor Code 2802)
- ADA accommodation: route through HR only, not line manager
```

---

## Key Findings

### What Worked

- SBERT retrieved the correct documents for every test query
- Groq produced accurate, structured answers grounded in document content
- Multi-document synthesis: P1 incident answer drew from 3 separate sources
- Version comparison: expense limits from FIN-001 and FIN-005 compared automatically
- No hallucination: answers stayed within the provided document context

### What Failed Initially

```
text[:500]  → table data was cut off → expense limits were missing from context
text[:1500] → full table content included → accurate limits returned
```

**Lesson:** Context window size directly affects answer quality. More context = better answers, but at higher token cost.

---

## Context Window Tradeoff

| Context Size | Speed | Table Data | Token Cost |
|---|---|---|---|
| text[:500] | Fast | Missing | Low |
| text[:1500] | Moderate | Complete | Moderate |
| text[:5000] | Slow | Complete | High |

**Chosen for production: 1500 characters per document** — balances completeness and cost.

---

## Observation

> SBERT retrieves the right documents. Groq generates accurate answers
> from those documents. The combination works well for enterprise Q&A.
>
> The most important engineering decision in this pipeline is context size.
> At 500 characters, structured data (tables, lists) was frequently cut off
> and Groq could not answer questions about specific limits or thresholds.
> At 1500 characters, all critical content was included.
>
> This is a production concern that does not appear in toy demos:
> real enterprise documents store important information in tables,
> not in prose paragraphs.

---

## Security

```
API keys must never be hardcoded in source code.

Correct approach:
- Store in .env file (excluded from git via .gitignore)
- Load at runtime: os.getenv("GROQ_API_KEY")
- Set as environment variable in the hosting platform dashboard

A leaked API key in a public GitHub repo can be scraped by
automated bots within minutes.
```

---

## Files

| File | Purpose |
|---|---|
| `llm_query.py` | SBERT + Groq RAG pipeline experiment |

---

## How To Run

```bash
pip install groq sentence-transformers python-dotenv

# Create .env file
echo GROQ_API_KEY=your_key_here > .env

python llm_query.py
```

---

## All 6 Layers — Complete

| Layer | Technology | Status |
|---|---|---|
| Layer 1 | Text Extraction (python-docx) | Complete |
| Layer 2 | Classification (TF-IDF + XGBoost) | Complete |
| Layer 3 | NER (spaCy EntityRuler) | Complete |
| Layer 4 | Similarity Search (SBERT + ChromaDB) | Complete |
| Layer 5 | Hybrid Search | Abandoned — performed worse than Layer 4 alone |
| Layer 6 | LLM Reasoning (Groq + LLaMA 3.3 70B) | Complete |
