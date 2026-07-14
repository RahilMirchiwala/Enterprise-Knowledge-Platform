# Results — Enterprise Knowledge Intelligence Platform

A single-page summary of all 6 experiments, evaluated against the same
25 human-labeled document pairs.

---

## Evaluation Setup

- **25 document pairs** manually labeled with similarity scores (0–100%)
- **5 categories:** Near Duplicate, Highly Similar, Moderately Similar, Slightly Related, Unrelated
- **Classification threshold:** score >= 40% = "Similar" (1), else "Not Similar" (0)
- **Correct prediction:** gap within ±20% of human score

---

## Similarity Algorithm Comparison

| Experiment | Algorithm | Correct | FN | FP | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 01 | TF-IDF + Cosine | 5/25 | 20 | 0 | 1.00 | 0.17 | 0.29 |
| 02 | Sentence Transformers (SBERT) | 14/25 | 8 | 3 | 0.76 | 0.89 | 0.82 |
| 03 | Hybrid (TF-IDF + SBERT) | 13/25 | 9 | 3 | 0.74 | 0.86 | 0.80 |

**Winner: SBERT — Experiment 02** (highest F1: 0.82)

> Evaluated using `experiment/07_evaluation/evaluate.py` with fixed threshold = 40%

---

## Key Findings

### Finding 1 — TF-IDF Has Perfect Precision But Terrible Recall

```
TF-IDF Precision = 1.00 — every pair it called "similar" was correct
TF-IDF Recall    = 0.17 — but it missed 83% of actually similar pairs
TF-IDF F1        = 0.29 — poor overall performance

Strategy: "say nothing rather than be wrong"
Problem:  A duplicate detection system that misses 83% of duplicates is useless
```

### Finding 2 — SBERT Understands Meaning, TF-IDF Does Not

```
HR-001 (India Onboarding) vs HR-002 (UK Onboarding)
Human Score  : 95%
TF-IDF Score : 18.41%  ← zero shared keywords → predicted "not similar"
SBERT Score  : 72.37%  ← understood same workflow → predicted "similar"

India uses: "Aadhaar, PAN, ESIC, INR"
UK uses:    "HMRC, PAYE, Right to Work, GBP"
```

### Finding 3 — Hybrid Performed Worse Than SBERT Alone

```
Best hybrid weight by grid search: 0% TF-IDF + 100% SBERT
TF-IDF contributed zero value — only noise.

SBERT F1  = 0.82
Hybrid F1 = 0.80  ← worse!

Lesson: Adding a weak signal to a strong signal degrades the strong signal.
```

### Finding 4 — Why F1 Is The Right Metric Here

```
Looking at Precision alone → TF-IDF appears perfect (1.00) ❌ misleading
Looking at F1             → TF-IDF (0.29) vs SBERT (0.82) ✅ honest

F1 penalises systems that achieve high precision by predicting
"not similar" for everything — which is exactly what TF-IDF does.
```

---

## Experiment 04 — XGBoost Classification

| Task | Metric | Score | Note |
|---|---|---|---|
| Department Classification | CV Accuracy | 96.67% | 5-fold cross-validation |
| Document Type Classification | Training Accuracy | 86.67% | CV failed — insufficient data |

**Limitation:** Strong bias toward HR department (majority class).
Real-world inputs like "contract between vendor" were classified as HR/SOP.
Root cause: 30 documents is far too small — minimum 100+ per class required.

---

## Experiment 05 — spaCy Custom NER

| Metric | Result |
|---|---|
| Documents with entities extracted | 30/30 |
| Default spaCy accuracy | Poor — "Rollback" → PERSON, "CI" → city |
| Custom EntityRuler accuracy | Correct for all defined entity types |
| Training data required | Zero — rule-based approach |

**Key finding:** Rule-based NER outperformed ML-based NER for this narrow domain.

---

## Experiment 06 — Groq RAG Pipeline

| Query | Documents Retrieved | Answer Quality |
|---|---|---|
| Remote work policy | HR-003, HR-006 | Accurate — global + US policies covered |
| Expense reimbursement limits | FIN-001, FIN-005, FIN-007 | Accurate — v1.0 vs v2.0 comparison |
| P1 incident handling | ENG-002, ENG-008, OPS-002 | Accurate — 4-phase response |

**Key finding:** Context size directly affects answer quality.
- `text[:500]` → table data cut off → limits missing
- `text[:1500]` → complete content → accurate answers

---

## Production Decision

```
Retrieval : SBERT (all-MiniLM-L6-v2) + ChromaDB vector store
Generation: Groq API (llama-3.3-70b-versatile)
```

| Decision | Reason |
|---|---|
| SBERT over TF-IDF | F1: 0.82 vs 0.29 |
| ChromaDB over in-memory | Persistent storage — no recompute on restart |
| Hybrid rejected | F1: 0.80 — worse than SBERT alone |

---

## Limitations

| Limitation | Impact | Fix |
|---|---|---|
| 30 document training set | XGBoost overfits | 100+ docs per class |
| SBERT not fine-tuned | False positives on business context | Domain fine-tuning |
| No chunking | Long docs embedded as single vector | Sliding window chunks |
| 25-pair evaluation set | Small ground truth | 250+ pairs, multiple annotators |
| Single annotator | No inter-rater reliability | Cohen's Kappa measurement |

---

## What Would Improve Results

| Improvement | Expected Impact |
|---|---|
| Fine-tune SBERT on domain data | Higher F1 — model learns business context |
| 500+ labeled pairs | More reliable Precision/Recall/F1 |
| Chunk-level embeddings | Better precision on long documents |
| Cross-encoder re-ranking | Reduce False Positives |
| MLflow experiment tracking | Reproducible comparison across runs |
