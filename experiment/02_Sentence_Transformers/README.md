# Experiment 02 — Sentence Transformers (SBERT)

**Status:** Complete
**Result:** 14/25 pairs correct — major improvement over TF-IDF but fails on business context

---

## Why This Experiment

Experiment 01 showed that TF-IDF only looks at word frequency.
It scored HR-001 (India Onboarding) vs HR-002 (UK Onboarding) at just 18.41% — even though a human gives it 95%.

The root cause: same workflow, different regional terminology.

So the question became:
> Can a model that understands **meaning** do better?

---

## How Sentence Transformers Work

### The Problem With TF-IDF
```
"Employee joining process"      }
"New hire onboarding procedure" }  → TF-IDF: ~0% (no shared words)
                                    → Human:  ~95%
```

### What SBERT Does Differently

SBERT converts every sentence into a **384-dimensional vector** (embedding).
These vectors capture meaning — not just words.

```
"I am hungry"           → [0.21, -0.45, 0.87, ...]
"I need food"           → [0.19, -0.41, 0.83, ...]  ← similar vector!
"The stock market fell" → [0.91,  0.12, 0.34, ...]  ← different vector!
```

Same meaning = vectors point in same direction = high cosine similarity ✅

### Model Used
```
all-MiniLM-L6-v2

MiniLM → Mini Language Model (lightweight BERT)
L6     → 6 layers (BERT full = 12 layers)
v2     → Version 2 (improved accuracy)

Output: 384-dimensional embedding per document
```

### Why 384 Dimensions (Not 6655 Like TF-IDF)?
```
TF-IDF:
  30 documents   → 6655 dimensions
  1000 documents → 50,000+ dimensions  ← grows with vocabulary!

SBERT:
  30 documents   → 384 dimensions
  1M documents   → 384 dimensions  ← always fixed! Fast + efficient!
```

---

## How To Run

```bash
pip install sentence-transformers scikit-learn pandas

python sentence_similarity.py
# Output: sbert_results.csv + comparison table
```

---

## Results

### Embeddings Matrix
```
30 documents x 384 dimensions
(vs TF-IDF: 30 x 6655)
```

### Human vs SBERT Comparison

| Doc A | Doc B | Category | Human % | SBERT % | Gap | Flag |
|---|---|---|---|---|---|---|
| HR-001 | HR-002 | Near Duplicate | 95 | 72.37 | -22.6 | FN |
| HR-001 | HR-004 | Near Duplicate | 90 | 83.70 | -6.3 | OK |
| HR-002 | HR-005 | Near Duplicate | 90 | 77.56 | -12.4 | OK |
| HR-004 | HR-005 | Near Duplicate | 95 | 70.91 | -24.1 | FN |
| FIN-001 | FIN-005 | Near Duplicate | 95 | 72.58 | -22.4 | FN |
| FIN-002 | FIN-006 | Near Duplicate | 90 | 65.69 | -24.3 | FN |
| ENG-001 | ENG-006 | Near Duplicate | 95 | 73.01 | -22.0 | FN |
| OPS-001 | OPS-006 | Near Duplicate | 95 | 72.83 | -22.2 | FN |
| HR-003 | HR-006 | Highly Similar | 85 | 76.37 | -8.6 | OK |
| ENG-008 | OPS-008 | Highly Similar | 80 | 56.81 | -23.2 | FN |
| ENG-002 | OPS-002 | Moderately Similar | 65 | 53.83 | -11.2 | OK |
| LEG-002 | OPS-003 | Moderately Similar | 60 | 40.25 | -19.8 | OK |
| LEG-002 | ENG-003 | Moderately Similar | 55 | 25.70 | -29.3 | FN |
| LEG-001 | OPS-007 | Moderately Similar | 50 | 49.93 | -0.1 | OK |
| LEG-005 | OPS-007 | Moderately Similar | 50 | 46.11 | -3.9 | OK |
| LEG-006 | HR-007 | Moderately Similar | 50 | 14.46 | -35.5 | FN |
| FIN-001 | FIN-007 | Moderately Similar | 40 | 59.63 | +19.6 | OK |
| LEG-003 | OPS-001 | Moderately Similar | 40 | 39.59 | -0.4 | OK |
| HR-001 | OPS-001 | Slightly Related | 30 | 62.25 | +32.3 | FP |
| HR-002 | OPS-006 | Slightly Related | 30 | 41.95 | +12.0 | OK |
| ENG-007 | ENG-001 | Slightly Related | 35 | 46.20 | +11.2 | OK |
| OPS-003 | FIN-003 | Unrelated | 15 | 53.45 | +38.5 | FP |
| HR-003 | ENG-003 | Unrelated | 10 | 32.61 | +22.6 | FP |
| LEG-001 | ENG-002 | Unrelated | 5 | 18.88 | +13.9 | OK |
| FIN-003 | ENG-001 | Unrelated | 5 | 21.04 | +16.0 | OK |

### Score Summary

```
           TF-IDF    SBERT    Improvement
OK           5        14        +9 pairs
FN          20         8        -12 pairs
FP           0         3        +3 new FP
```

---

## Key Findings

### Where SBERT Improved Over TF-IDF

**Same workflow, different regional terminology:**
```
HR-001 (India) → "Aadhaar, PAN, ESIC, INR"
HR-002 (UK)    → "HMRC, PAYE, Right to Work, GBP"

TF-IDF: 18.41%  ❌
SBERT:  72.37%  ✅  (+53.96%)
```

SBERT understood both are employee onboarding documents — even with zero shared keywords.

**Version upgrades:**
```
HR-001 ↔ HR-004 (v1.0 vs v2.0)
TF-IDF: 28.41%  ❌
SBERT:  83.70%  ✅  (+55.29%)
```

SBERT understood updated content still describes the same process.

---

### Where SBERT Failed

**False Positives — general concept samjha, business context nahi:**
```
HR-001 ↔ OPS-001 → Human: 30% | SBERT: 62.25% ⚠️

HR-001  = Employee Onboarding  (new hire joining)
OPS-001 = Client Onboarding    (new customer setup)

SBERT ne "onboarding" concept dekha — same!
But business purpose bilkul alag tha — SBERT confuse ho gaya!
```

**False Negatives — vocabulary too different:**
```
LEG-006 ↔ HR-007 → Human: 50% | SBERT: 14.46% ⚠️

LEG-006 = IP Assignment → "assign, invention, rights, intellectual property"
HR-007  = Performance Review → "rating, bonus, calibration, feedback"

Koi semantic overlap nahi — SBERT bhi fail!
```

---

## My Observation

> SBERT ne general concept samjha — but business-specific context nahi samjha.
> Isliye "Employee Onboarding" aur "Client Onboarding" ko same bol diya.
> Aur jo documents bilkul alag vocabulary use karte hain — wahan TF-IDF ki tarah fail hua.

---

## TF-IDF vs SBERT — Final Comparison

| Metric | TF-IDF | SBERT | Winner |
|---|---|---|---|
| Correct pairs | 5/25 | 14/25 | SBERT ✅ |
| Near duplicates | ❌ mostly fail | ✅ mostly ok | SBERT ✅ |
| Unrelated docs | ✅ correct | ⚠️ some FP | TF-IDF ✅ |
| Speed | Fast | Slow (44 sec) | TF-IDF ✅ |
| Vocabulary size | 6655 dims | 384 dims | SBERT ✅ |
| Business context | ❌ | ❌ | Neither |

---

## Next Experiment

**Experiment 03 — Hybrid Search (TF-IDF + SBERT)**

Idea:
```
TF-IDF  → exact keyword matching  → good for unrelated docs
SBERT   → semantic understanding  → good for near duplicates

Hybrid  → combine both scores     → best of both worlds?
```

Expected: fewer False Positives than SBERT, fewer False Negatives than TF-IDF.
