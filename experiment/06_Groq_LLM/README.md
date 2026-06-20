# Experiment 06 — Groq LLM + SBERT RAG Pipeline (Layer 6)

**Status:** Complete
**Result:** RAG pipeline working — SBERT finds relevant docs, Groq gives accurate answers

---

## What is RAG?

**RAG = Retrieval Augmented Generation**

```
R → Retrieval   = SBERT ne relevant documents dhundhe
A → Augmented   = un documents ka context LLM ko diya
G → Generation  = Groq ne intelligent answer generate kiya
```

Without RAG:
```
User: "What are expense limits?"
LLM:  "I don't know NovaBridge's specific limits" ❌
```

With RAG:
```
User:  "What are expense limits?"
SBERT: finds FIN-001, FIN-005, FIN-007
Groq:  reads those docs → accurate answer ✅
```

---

## Pipeline Architecture

```
User Question
      ↓
SentenceTransformer ("all-MiniLM-L6-v2")
      ↓ (query vector — 384 dims)
Cosine Similarity vs 30 document embeddings
      ↓ (top 3 most similar docs)
Groq API (llama-3.3-70b-versatile)
      ↓ (reads doc context + question)
Accurate Answer ✅
```

---

## Model Used

```
Groq API → llama-3.3-70b-versatile
         → 70 billion parameters
         → fast inference (Groq hardware)
         → free tier available
```

---

## Results

### Query 1 — Expense Reimbursement Limits

```
Query: "What are the expense reimbursement limits?"

Found docs:
→ FIN-001 Expense Reimbursement Policy      (v1.0)
→ FIN-005 Expense Reimbursement Policy v2.0
→ FIN-007 Travel and Entertainment Policy

Groq Answer:
FIN-001 (v1.0) limits:
- Meals & Entertainment: INR 1,500 / GBP 40
- Local Travel: INR 800 / GBP 25
- Accommodation: INR 6,000/night / GBP 150/night

FIN-005 (v2.0) updated limits (effective July 2026):
- Meals & Entertainment: INR 2,000/day (+33%)
- Local Travel: INR 1,200/day (+50%)
- Accommodation: INR 7,500/night (+25%)
- Communication: INR 750/month (+50%)
```

**Groq ne dono versions compare karke bataya!** ✅

---

### Query 2 — P1 Incident Handling

```
Query: "How to handle a P1 incident?"

Found docs:
→ ENG-002 Incident Response Procedure
→ ENG-008 Disaster Recovery Plan
→ OPS-002 Escalation Matrix

Groq Answer:
1. Detection & Alerting → PagerDuty alert, acknowledge in 5 min
2. Triage & Escalation  → notify Engineering Manager + VP Engineering
3. Mitigation           → isolate, fix/rollback, update #incidents every 15 min
4. Resolution & Review  → close ticket, post-mortem within 48 hours
```

**Groq ne 3 documents combine karke complete answer diya!** ✅

---

### Query 3 — Remote Work Policy

```
Query: "What is the remote work policy?"

Found docs:
→ HR-003 Remote Work Policy (Global)
→ HR-006 US Remote Work Policy

Groq Answer:
Global policy (HR-003):
- 90-day probation required
- VPN mandatory
- Core hours 10:00-16:00

US-specific (HR-006):
- Cross-state work needs Finance + HR approval
- California: full expense reimbursement required (CA Labor Code)
```

---

## Key Findings

### What Worked Well
```
✅ SBERT correctly found relevant documents for all queries
✅ Groq gave accurate, structured answers
✅ Version comparison — FIN-001 vs FIN-005 automatically handled
✅ Multi-document synthesis — combined 3 docs for P1 answer
✅ No hallucination — answers based only on provided context
```

### What Failed Initially
```
❌ text[:500] → table data cut off → expense limits missing

Fix:
✅ text[:1500] → full table data included → accurate limits!
```

---

## Important Lesson — Context Window

```
text[:500]  → fast but misses tables ❌
text[:1500] → slightly slower but complete data ✅
text[:5000] → most complete but expensive (more tokens)
```

**Balance: 1500 chars per document = sweet spot for our use case**

---

## My Observation

> Maine Groq + SBERT RAG pipeline try kiya aur observe kiya ki
> SBERT sahi se relevant documents nikal raha hai aur Groq un
> documents se accurate answer de raha hai.
>
> Groq ne automatically version comparison bhi kiya —
> FIN-001 (v1.0) aur FIN-005 (v2.0) dono ke limits compare karke bataye.
>
> Ek important learning: text truncation se table data miss hoti thi.
> Context size 500 se 1500 karne par accurate answers aane lage.

---

## Security Note

```
API key NEVER hardcode karo code mein!

✅ .env file mein rakho
✅ .gitignore mein .env add karo
✅ os.getenv("GROQ_API_KEY") se load karo

Reason: GitHub pe push hone se API key public ho jaati hai!
```

---

## Files

| File | Purpose |
|---|---|
| `llm_query.py` | SBERT + Groq RAG pipeline |

---

## How To Run

```bash
pip install groq sentence-transformers python-dotenv

# .env file banao
echo GROQ_API_KEY=your_key_here > .env

python llm_query.py
```

---

## All 6 Layers — Complete!

```
Layer 1 ✅ → Text Extraction      (python-docx)
Layer 2 ✅ → Classification       (TF-IDF + XGBoost)
Layer 3 ✅ → NER                  (spaCy EntityRuler)
Layer 4 ✅ → Similarity Search    (SBERT + Cosine)
Layer 5 ❌ → Hybrid Search        (proved less effective than Layer 4)
Layer 6 ✅ → LLM Reasoning        (Groq + Llama 3.3)
```

## Next Step — FastAPI App

Integrate all layers into a production FastAPI backend:
```
POST /search  → Layer 4 + Layer 6
POST /classify → Layer 2
POST /extract  → Layer 3
```
