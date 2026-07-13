# Enterprise Knowledge Intelligence Platform

> **Live Demo:** [enterprise-knowledge-platform.onrender.com/ui](https://enterprise-knowledge-platform.onrender.com/ui)  
> **API Docs:** [enterprise-knowledge-platform.onrender.com/docs](https://enterprise-knowledge-platform.onrender.com/docs)

---

## Case Study: NovaBridge Document Intelligence

### The Business Problem

NovaBridge Consulting operates across India, UK, and US offices with 30+ business documents spanning 5 departments — HR, Finance, Engineering, Legal, and Operations. Many of these documents are duplicates, regional variants, or updated versions of the same policy. Finding similar or duplicate documents manually was time-consuming and error-prone, and employees had no way to quickly query the knowledge base for answers to policy questions.

**The core challenge:** Documents covering the same workflow often use completely different terminology depending on the region. India onboarding documents reference "Aadhaar", "PAN", and "ESIC". UK onboarding documents reference "HMRC", "PAYE", and "Right to Work". A keyword-based system sees zero overlap and reports 0% similarity — even though both documents describe the same process.

---

### The Approach

Rather than jumping to the most advanced model, I started simple and worked upward — treating each step as a hypothesis to test.

| Step | Hypothesis | Result |
|---|---|---|
| TF-IDF + Cosine | Word overlap is enough to find similar documents | 5/25 correct — failed on regional terminology |
| Sentence Transformers (SBERT) | Semantic meaning matters more than word overlap | 14/25 correct — best result |
| Hybrid Search | Combining TF-IDF + SBERT would improve results | 13/25 — worse than SBERT alone |
| XGBoost Classification | Documents can be classified by department and type | 96.67% training accuracy — overfits on 30 docs |
| spaCy Custom NER | Structured metadata can be extracted automatically | Entities extracted from all 30 documents |
| Groq RAG Pipeline | LLM can answer questions grounded in retrieved documents | Accurate answers with source citations |

Each experiment used the same 25 manually labeled document pairs as ground truth, making results directly comparable.

---

### Results

**Best similarity algorithm: SBERT (Experiment 02)**

```
TF-IDF → 5/25 pairs correct  (20 False Negatives)
SBERT  → 14/25 pairs correct (8 False Negatives, 3 False Positives)
Hybrid → 13/25 pairs correct (worse than SBERT alone)
```

**Key finding:** TF-IDF failed because it only measures word overlap.
SBERT succeeded because it understands meaning — it recognised that
"Aadhaar verification" and "Right to Work check" are both employee
identity verification steps, even though they share zero keywords.

**Why Hybrid performed worse than SBERT alone:**
Combining a weak signal (TF-IDF) with a strong signal (SBERT) degrades
the strong signal. The optimal grid search weight was 0% TF-IDF + 100% SBERT —
confirming TF-IDF added only noise.

---

### What Is Running in Production

```
User Query / Document Upload
        ↓
Layer 1 — Text Extraction     (python-docx)
        ↓
Layer 3 — NER                 (spaCy Custom EntityRuler)
        ↓
Layer 4 — Semantic Search     (SBERT + ChromaDB vector store)
        ↓
Layer 5 — LLM Reasoning       (Groq + LLaMA 3.3 70B)
        ↓
Answer + Sources / Similar Documents
```

**Why ChromaDB instead of in-memory SBERT:**
Embeddings are computed once and stored on disk. Every server restart loads
from disk in milliseconds instead of recomputing 30 embeddings from scratch.
At scale (30,000 documents), recomputing on every restart would take hours.

**Live API Endpoints:**

| Endpoint | Method | Function |
|---|---|---|
| `GET /search` | GET | Semantic search across all indexed documents |
| `GET /ask` | GET | RAG-powered Q&A — retrieve + generate |
| `GET /detect-duplicates` | GET | Find similar documents for any indexed document |
| `POST /ingest` | POST | Add a new .docx document to the knowledge base |
| `GET /health` | GET | System health — model loaded, documents indexed |
| `GET /documents` | GET | List all indexed documents |

---

### Limitations and Next Steps

**Current limitations:**

1. **Small training dataset** — 30 documents is too few for XGBoost to generalise. Department classification achieves 96.67% on training data but fails on unseen text. Minimum 100+ documents per class needed for production.

2. **SBERT not fine-tuned** — `all-MiniLM-L6-v2` is a general-purpose model. It correctly identifies that "Employee Onboarding" and "Client Onboarding" are both "onboarding" — but misses that they serve completely different business purposes. Fine-tuning on domain-specific labeled pairs would close this gap.

3. **No chunking** — Documents are embedded as single vectors. For longer documents (50+ pages), chunking into paragraphs with overlapping windows would improve precision significantly.

4. **25-pair evaluation set** — Ground truth is small and single-annotator. A robust evaluation would use 250+ pairs with multiple annotators and inter-annotator agreement measurement (Cohen's Kappa).

**What I would do next:**
- Fine-tune SBERT on 500+ labeled business document pairs
- Add chunk-level embeddings for longer document support
- Implement Precision@K, Recall@K, F1, and MRR evaluation metrics
- Add MLflow experiment tracking for reproducible comparison

---

## Experiment Results Summary

| Experiment | Algorithm | Result | Key Finding |
|---|---|---|---|
| 01 | TF-IDF + Cosine Similarity | 5/25 correct | Fails when same workflow uses different regional terminology |
| 02 | Sentence Transformers (SBERT) | 14/25 correct | Best performer — understands meaning, not just words |
| 03 | Hybrid Search (TF-IDF + SBERT) | 13/25 correct | Worse than SBERT alone — TF-IDF noise drags scores down |
| 04 | XGBoost Classification | 96.67% dept, 86.67% type | Overfits on 30 documents — needs 100+ per class |
| 05 | spaCy Custom NER | 30/30 docs extracted | Default model fails on business terms; custom EntityRuler works |
| 06 | Groq RAG Pipeline | Working | SBERT retrieves, Groq generates accurate answers |

**Winner: SBERT (Experiment 02)** — used in production via ChromaDB vector store.

---

## Dataset

- **30 documents** across 5 departments (HR, Finance, Engineering, Legal, Operations)
- **5 similarity categories** — Near Duplicate, Highly Similar, Moderately Similar, Slightly Related, Unrelated
- **25 human-labeled pairs** with similarity scores (0-100%)

### Departments

| Dept | Doc IDs | Topics |
|---|---|---|
| HR | HR-001 to HR-007 | Onboarding, Remote Work, Performance Review |
| Finance | FIN-001 to FIN-007 | Expense, Payroll, Budget, Travel |
| Engineering | ENG-001 to ENG-008 | Deployment, Incidents, Security, DR, Code Review |
| Legal | LEG-001 to LEG-006 | NDA, Privacy, Service Agreement, IP Assignment |
| Operations | OPS-001 to OPS-008 | Client Onboarding, Escalation, Audit, BCP, Vendor |

---

## Project Structure

```
Enterprise Knowledge Platform/
│
├── app/                              # Production FastAPI app
│   ├── chroma_db/                    # ChromaDB vector store (auto-generated)
│   ├── layers/
│   │   ├── layer1_extract.py         # Text extraction from .docx
│   │   ├── layer2_classify.py        # XGBoost department + type classifier
│   │   ├── layer3_ner.py             # spaCy custom NER
│   │   ├── layer4_search.py          # SBERT + ChromaDB semantic search
│   │   └── layer5_llm.py             # Groq RAG pipeline
│   ├── scripts/
│   │   └── build_index.py            # One-time ChromaDB index builder
│   ├── static/
│   │   └── index.html                # Simple web UI
│   ├── main.py                       # FastAPI entry point
│   └── requirements.txt
│
├── documents/                        # 30 source .docx files
├── experiment/                       # 6 ML/NLP experiments
│   ├── 01_TF-IDF/
│   ├── 02_Sentence_Transformers/
│   ├── 03_Hybrid/
│   ├── 04_XGBoost_Classification/
│   ├── 05_spaCy_NER/
│   └── 06_Groq_LLM/
│
├── evaluation_dataset.csv            # 25 human-labeled pairs (ground truth)
├── .env.example                      # Required environment variables
├── Procfile                          # Render deployment config
└── README.md
```

---

## Setup & Run Locally

```bash
# 1. Clone repo
git clone https://github.com/RahilMirchiwala/Enterprise-Knowledge-Platform.git
cd Enterprise-Knowledge-Platform

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r app/requirements.txt
pip install spacy
python -m spacy download en_core_web_sm

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Build ChromaDB index (run once)
cd app
python scripts/build_index.py

# 6. Start the server
uvicorn main:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the API or [http://localhost:8000/ui](http://localhost:8000/ui) for the web interface.

---

## Tech Stack

- **FastAPI** — API framework
- **Sentence Transformers** (all-MiniLM-L6-v2) — semantic embeddings
- **ChromaDB** — persistent vector store
- **Groq + LLaMA 3.3 70B** — LLM reasoning layer
- **scikit-learn + XGBoost** — document classification
- **spaCy** — named entity recognition
- **python-docx** — document text extraction

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

---

## Author

**Rahil Mirchiwala**  
CS Student — Parul University  
GitHub: [RahilMirchiwala](https://github.com/RahilMirchiwala)