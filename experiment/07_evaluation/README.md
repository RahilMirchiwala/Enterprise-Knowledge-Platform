# Experiment 07 — Evaluation (Precision, Recall, F1)

**Status:** Complete
**Result:** SBERT F1=0.82 vs TF-IDF F1=0.29 — SBERT is the clear winner

---

## Why This Experiment Exists

Previous experiments reported results as "X/25 correct" — which is an outcome,
not a metric. A system that predicts "not similar" for every pair would score
17/25 correct on this dataset (the 17 pairs that are genuinely not similar)
and appear better than SBERT. That would be wrong.

Proper evaluation metrics answer different questions:

| Metric | Question |
|---|---|
| Precision | Of everything the model called "similar", how much actually was? |
| Recall | Of everything that was actually similar, how much did the model find? |
| F1 | What is the harmonic balance between precision and recall? |

---

## Evaluation Setup

```
Dataset    : 25 human-labeled document pairs
Threshold  : score >= 40% = "Similar" (1), else "Not Similar" (0)
             40% = start of "Moderately Similar" category
Ground truth: human_similarity column in evaluation_dataset.csv
```

---

## How To Run

```bash
cd experiment/07_evaluation
python evaluate.py
```

---

## Results

### TF-IDF + Cosine Similarity

```
Precision : 1.00  — every pair it called "similar" was actually similar
Recall    : 0.17  — but it only found 17% of actually similar pairs
F1        : 0.29  — poor overall
```

### Sentence Transformers (SBERT)

```
Precision : 0.76  — 76% of "similar" predictions were correct
Recall    : 0.89  — found 89% of actually similar pairs
F1        : 0.82  — strong overall performance
```

### Final Comparison

| Metric | TF-IDF | SBERT | Winner |
|---|---|---|---|
| Precision | 1.00 | 0.76 | TF-IDF |
| Recall | 0.17 | 0.89 | SBERT |
| F1 | 0.29 | 0.82 | SBERT |

**Winner: SBERT** — F1 score 0.82 vs 0.29

---

## Why TF-IDF Has Perfect Precision But Fails Overall

```
TF-IDF maximum score on this dataset = 39.7% (ENG-001 vs ENG-006)

With threshold = 40%:
→ TF-IDF never predicted "similar" for any pair
→ All 5 "correct" predictions were pairs the model called "not similar"
   that were also labeled "not similar" by humans (True Negatives)
→ Precision = 1.00 is misleading — the model never took a risk

A system that never predicts "similar" has perfect precision
and zero recall. It is useless for duplicate detection.

F1 = 0.29 correctly captures this failure.
```

---

## Why F1 Is The Right Metric For This Use Case

```
Precision alone → TF-IDF looks perfect (1.00) ❌
Recall alone    → both models look reasonable
F1              → TF-IDF (0.29) vs SBERT (0.82) ✅ honest comparison

F1 is the harmonic mean — it penalises extreme imbalances.
A model with 100% precision and 0% recall gets F1 = 0.
```

---

## Precision-Recall Tradeoff

For duplicate document detection, **Recall matters more than Precision**:

```
Missing a duplicate (low recall):
→ Employee creates a conflicting policy version
→ Legal team misses a duplicate contract
→ Finance team applies outdated expense limits
→ Business impact: HIGH ❌

False positive (low precision):
→ System flags two documents as similar that aren't
→ User reviews them and dismisses the suggestion
→ Business impact: LOW (minor inconvenience) ✅
```

SBERT's Recall = 0.89 makes it the right choice for this use case.

---

## Key Findings

**1. Word overlap is not enough**
TF-IDF sees zero overlap between "Aadhaar, PAN, ESIC" and "HMRC, PAYE, Right to Work"
even though both describe employee identity verification. Recall = 0.17 proves this.

**2. Semantic understanding significantly improves recall**
SBERT Recall = 0.89 vs TF-IDF Recall = 0.17 — a 5x improvement.
SBERT understood that documents describing the same workflow in different
regional terminology are semantically similar.

**3. Hybrid search added noise, not signal**
The optimal hybrid weight was 0% TF-IDF + 100% SBERT.
TF-IDF's low-quality scores contaminated SBERT's accurate scores.
F1 dropped from 0.82 (SBERT alone) to 0.80 (Hybrid).

---

## Limitations of This Evaluation

```
1. Small dataset — 25 pairs is too few for statistical significance
   Standard NLP benchmarks use 1000+ evaluation examples

2. Single annotator — no inter-rater reliability measurement
   Production evaluation would use Cohen's Kappa across multiple annotators

3. Binary threshold — similarity is continuous, not binary
   A ranking metric like MRR or NDCG would capture this better

4. No held-out test set — same 25 pairs used for both development and evaluation
   Risk of overfitting the threshold to this specific dataset
```

---

## Files

| File | Purpose |
|---|---|
| `evaluate.py` | Loads TF-IDF and SBERT results, computes Precision/Recall/F1 |

---

## Next Steps

To make this evaluation production-grade:

1. Expand to 250+ labeled pairs with multiple annotators
2. Add MRR (Mean Reciprocal Rank) for ranked retrieval evaluation
3. Add NDCG for graded relevance evaluation
4. Integrate MLflow to track metrics across experiment runs
5. Add a held-out test set separate from the development set
