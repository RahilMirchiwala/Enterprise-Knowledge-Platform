# Experiment 01 — TF-IDF + Cosine Similarity

**Status:** Complete
**Result:** 5/25 pairs correct — fails on near duplicates with different terminology

---

## Goal

Test whether TF-IDF + Cosine Similarity can detect document similarity across 30 business documents — including near duplicates, regional variants, and version upgrades.

---

## How TF-IDF Works

### TF — Term Frequency
How often a word appears in a single document.

```
TF = word count in document / total words in document

Example:
"onboarding" appears 1 time in a 3-word document
TF = 1/3 = 0.33
```

### IDF — Inverse Document Frequency
How rare a word is across ALL documents.

```
IDF = log(Total Documents / Documents containing the word)

Example:
"onboarding" appears in 2 out of 3 documents
IDF = log(3/2) = 0.176

Rare word → high IDF → more important
Common word → low IDF → less important
```

### TF-IDF Score
```
TF-IDF = TF x IDF = 0.33 x 0.176 = 0.058
```

Each document becomes a **vector** — one number per unique word across all documents.

### Cosine Similarity
Measures the angle between two document vectors.

```
Same direction  (angle = 0°)  → score = 1.0 → identical
Perpendicular   (angle = 90°) → score = 0.0 → completely different
```

---

## Files

| File | Purpose |
|---|---|
| `extract_text.py` | Extract text from all 30 .docx files → saves to documents.json |
| `tfidf_similarity.py` | Load documents.json → TF-IDF → cosine similarity → compare vs human labels |

---

## How To Run

```bash
pip install python-docx scikit-learn pandas

# Step 1 — Extract text
python extract_text.py
# Output: documents.json (30 documents)

# Step 2 — Run TF-IDF
python tfidf_similarity.py
# Output: tfidf_results.csv (435 pairs) + comparison table
```

---

## Results

### TF-IDF Matrix
```
30 documents x 6655 unique terms
```

### Human vs TF-IDF Comparison

| Doc A | Doc B | Category | Human % | TF-IDF % | Gap | Flag |
|---|---|---|---|---|---|---|
| HR-001 | HR-002 | Near Duplicate | 95 | 18.41 | -76.6 | FN |
| HR-001 | HR-004 | Near Duplicate | 90 | 28.41 | -61.6 | FN |
| HR-002 | HR-005 | Near Duplicate | 90 | 21.87 | -68.1 | FN |
| HR-004 | HR-005 | Near Duplicate | 95 | 20.55 | -74.5 | FN |
| FIN-001 | FIN-005 | Near Duplicate | 95 | 31.01 | -64.0 | FN |
| FIN-002 | FIN-006 | Near Duplicate | 90 | 21.36 | -68.6 | FN |
| ENG-001 | ENG-006 | Near Duplicate | 95 | 39.70 | -55.3 | FN |
| OPS-001 | OPS-006 | Near Duplicate | 95 | 31.76 | -63.2 | FN |
| HR-003 | HR-006 | Highly Similar | 85 | 29.70 | -55.3 | FN |
| ENG-008 | OPS-008 | Highly Similar | 80 | 14.53 | -65.5 | FN |
| ENG-002 | OPS-002 | Moderately Similar | 65 | 7.71 | -57.3 | FN |
| LEG-002 | OPS-003 | Moderately Similar | 60 | 2.25 | -57.8 | FN |
| LEG-002 | ENG-003 | Moderately Similar | 55 | 3.15 | -51.9 | FN |
| LEG-001 | OPS-007 | Moderately Similar | 50 | 5.47 | -44.5 | FN |
| LEG-005 | OPS-007 | Moderately Similar | 50 | 7.66 | -42.3 | FN |
| LEG-006 | HR-007 | Moderately Similar | 50 | 3.27 | -46.7 | FN |
| FIN-001 | FIN-007 | Moderately Similar | 40 | 28.27 | -11.7 | OK |
| LEG-003 | OPS-001 | Moderately Similar | 40 | 3.21 | -36.8 | FN |
| HR-001 | OPS-001 | Slightly Related | 30 | 5.68 | -24.3 | FN |
| HR-002 | OPS-006 | Slightly Related | 30 | 4.12 | -25.9 | FN |
| ENG-007 | ENG-001 | Slightly Related | 35 | 6.56 | -28.4 | FN |
| OPS-003 | FIN-003 | Unrelated | 15 | 7.37 | -7.6 | OK |
| HR-003 | ENG-003 | Unrelated | 10 | 2.56 | -7.4 | OK |
| LEG-001 | ENG-002 | Unrelated | 5 | 0.44 | -4.6 | OK |
| FIN-003 | ENG-001 | Unrelated | 5 | 1.21 | -3.8 | OK |

### Score Summary
```
Total Pairs    : 25
Correct (OK)   : 5  (all were unrelated / low similarity documents)
False Negative : 20 (TF-IDF underestimated similarity)
False Positive : 0
```

---

## Key Findings

**TF-IDF works when:**
- Documents are completely unrelated — no shared words, no shared topic
- Low similarity pairs (human score below 20%)

**TF-IDF fails when:**
- Same workflow, different regional terminology

```
HR-001 India: "Aadhaar", "PAN", "PF", "ESIC", "INR", "Bangalore"
HR-002 UK:    "Right to Work", "HMRC", "PAYE", "GBP", "pension"

Human Score  : 95%
TF-IDF Score : 18.41%
```

- Version upgrades (v2.0 adds new content → different word distribution)
- Same concept, different regulations across regions

---

## Root Cause

TF-IDF only measures **word overlap**.
It has zero understanding of meaning or context.

```
"Employee joining process"       }
"New hire onboarding procedure"  }  → TF-IDF score: ~0%
                                    → Human score:  ~95%
```

Same meaning. Different words. TF-IDF is blind to this.

---

## What This Means

> I tested TF-IDF against 25 manually labeled document pairs and observed
> that it performs well on unrelated documents but consistently fails when
> the same workflow is described using different regional terminology.
> This is a fundamental limitation of frequency-based approaches.

---

## Next Experiment

**Experiment 02 — Sentence Transformers (all-MiniLM-L6-v2)**

Converts text into semantic embeddings — captures meaning, not just words.

Expected: HR-001 vs HR-002 score should jump from 18% to 80%+
Because the model understands "Aadhaar verification" and "Right to Work check"
are both employee identity verification steps.
