# Experiment 04 — XGBoost Document Classification (Layer 2)

**Status:** Complete
**Result:** 96.67% training accuracy (Department) | 86.67% (Doc Type) — overfits on small dataset

---

## Goal

Automatically classify every document into:
1. **Department** → HR, Finance, Engineering, Legal, Operations
2. **Document Type** → SOP, Policy, Contract, Report, Guidelines, Plan, Procedure, Process, Checklist, Agreement

This enables the platform to filter search results by department and document type.

---

## Pipeline

```
Document Text
     ↓
TF-IDF Vectorizer → numerical features (6655 dimensions)
     ↓
XGBoost Classifier → Department + Document Type
```

### Why TF-IDF for XGBoost?
XGBoost cannot process raw text — it requires numerical input.
TF-IDF converts text into a vector of word importance scores.

### Why XGBoost?
```
Single Decision Tree → weak, prone to errors
XGBoost (100 trees)  → each tree corrects the previous tree's mistakes
                     → strong ensemble model
```

---

## Key Parameters

```python
XGBClassifier(
    n_estimators=100,       # number of trees
    max_depth=3,            # tree depth — limits overfitting
    random_state=42,        # reproducibility
    eval_metric="mlogloss"  # multi-class log loss
)
```

| Parameter | Value | Reason |
|---|---|---|
| n_estimators | 100 | 100 trees → strong ensemble |
| max_depth | 3 | shallow trees → less overfitting |
| random_state | 42 | consistent results across runs |
| eval_metric | mlogloss | appropriate for multi-class classification |

---

## Training Data

```
30 documents × 2 labels each

Department distribution:
HR          → 7 documents
Finance     → 6 documents
Engineering → 6 documents
Legal       → 5 documents
Operations  → 6 documents

Doc Type distribution:
SOP         → 8 documents
Policy      → 8 documents
Contract    → 2 documents
Guidelines  → 2 documents
Plan        → 1 document  ← insufficient
Report      → 1 document  ← insufficient
Checklist   → 1 document  ← insufficient
Process     → 1 document  ← insufficient
Procedure   → 2 documents
Agreement   → 1 document  ← insufficient
```

---

## Results

### Department Classification
```
Cross Validation (5-fold):
Round 1: 100%
Round 2: 83%   ← one fold had a missing class
Round 3: 100%
Round 4: 100%
Round 5: 100%

Average: 96.67%
```

### Document Type Classification
```
Cross Validation failed — some classes had only 1 document.
StratifiedKFold also failed — not enough data to stratify.

Solution: Trained on full dataset.
Training Accuracy: 86.67%
```

---

## Real World Test

| Input Text | Expected | Predicted | Result |
|---|---|---|---|
| "contract between vendor and company" | Legal / Contract | HR / SOP | Fail |
| "quarterly budget variance report" | Finance / Report | HR / SOP | Fail |
| "API security guidelines for authentication" | Engineering / Guidelines | HR / Guidelines | Partial |

---

## Key Findings

### Finding 1 — Overfitting
```
Training accuracy → 96.67%
Real world test   → incorrect predictions

The model memorised the training data.
It did not generalise to unseen text.

Root cause: 30 documents is far too small.
Production-grade classification requires 100+ examples per class.
```

### Finding 2 — Class Imbalance Bias
```
HR → 7 documents (largest class)

The model learned: "when uncertain, predict HR."
This is a classic symptom of majority class bias.

Fix: Balanced training data — equal documents per class.
```

### Finding 3 — Cross Validation Failure on Doc Type
```
Several doc types had only 1 document:
Agreement, Checklist, Plan, Process, Report → 1 each

Splitting into folds removes the only example of that class.
XGBoost cannot train on classes it has never seen.

Fix: Minimum 5–10 documents per class for cross validation to work.
```

---

## Observation

> XGBoost achieved 96.67% training accuracy but failed on real-world inputs.
> With only 30 training examples, the model developed a strong bias toward
> the HR department — the largest class in the training set.
>
> This is an acceptable baseline result. In production, this classifier
> would require at least 100 labeled documents per class to generalise reliably.

---

## Saved Models

| File | Purpose |
|---|---|
| `dept_model.pkl` | Department classifier |
| `type_model.pkl` | Document type classifier |
| `dept_encoder.pkl` | Label encoder for departments |
| `type_encoder.pkl` | Label encoder for doc types |
| `vectorizer.pkl` | TF-IDF vectorizer |

Note: `.pkl` files are excluded from git — regenerate by running `classify.py`.

---

## How To Run

```bash
pip install xgboost scikit-learn pandas

python classify.py
# Output: trained models saved as .pkl files
```

---

## Next Experiment

**Experiment 05 — spaCy NER (Layer 3)**

Extract structured metadata directly from document text:
- Department mentions
- Region and location entities
- Document type signals
- Compliance terms (ESIC, PF, HMRC, GDPR)
