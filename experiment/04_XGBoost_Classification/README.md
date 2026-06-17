# Experiment 04 — XGBoost Document Classification (Layer 2)

**Status:** Complete
**Result:** 96.67% training accuracy (Department) | 86.67% (Doc Type) — but overfits on small data

---

## Goal

Automatically classify every document into:
1. **Department** → HR, Finance, Engineering, Legal, Operations
2. **Document Type** → SOP, Policy, Contract, Report, Guidelines, Plan, Procedure, Process, Checklist, Agreement

Why? So the platform can filter search results by department and type.

---

## How It Works

```
Document Text
     ↓
TF-IDF Vectorizer → numbers (6655 features)
     ↓
XGBoost Classifier → Department + Doc Type
```

### Why TF-IDF for XGBoost?
XGBoost cannot process raw text — it needs numbers.
TF-IDF converts text into a vector of word importance scores.

### Why XGBoost?
```
Single Decision Tree → weak, prone to errors
XGBoost (100 trees)  → each tree fixes previous tree's mistakes
                     → strong, accurate model! ✅
```

---

## Key Parameters

```python
XGBClassifier(
    n_estimators=100,  # 100 trees
    max_depth=3,       # tree depth — prevents overfitting
    random_state=42,   # reproducibility
    eval_metric="mlogloss"  # multi-class log loss
)
```

| Parameter | Value | Why |
|---|---|---|
| n_estimators | 100 | 100 trees → strong ensemble |
| max_depth | 3 | shallow trees → less overfitting |
| random_state | 42 | same results every run |
| eval_metric | mlogloss | best for multi-class problems |

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
Plan        → 1 document  ← very few!
Report      → 1 document  ← very few!
Checklist   → 1 document  ← very few!
Process     → 1 document  ← very few!
Procedure   → 2 documents
Agreement   → 1 document  ← very few!
```

---

## Results

### Department Classification
```
Cross Validation (5-fold):
Round 1: 100%
Round 2: 83%   ← one group had missing class
Round 3: 100%
Round 4: 100%
Round 5: 100%

Average: 96.67% ✅
```

### Document Type Classification
```
Cross Validation failed! → kuch classes mein sirf 1 document
StratifiedKFold bhi fail → data bahut kam!

Solution: Poore data pe train kiya
Training Accuracy: 86.67%
```

---

## Real World Test

```python
tests = [
    "This contract is entered between vendor and company",
    "Quarterly budget variance report for Q2 2026",
    "API security guidelines for authentication"
]
```

| Text | Expected | Predicted | Result |
|---|---|---|---|
| "contract between vendor..." | Legal / Contract | HR / SOP | ❌ |
| "quarterly budget report..." | Finance / Report | HR / SOP | ❌ |
| "API security guidelines..." | Engineering / Guidelines | HR / Guidelines | Partial ✅ |

---

## Key Findings

### Finding 1 — Overfitting
```
Training accuracy → 96.67% ✅
Real world test   → fail   ❌

Model ne training data yaad kar liya
Naye documents pe generalize nahi kar paya!

Reason: 30 documents bahut kam hain
ML models ke liye minimum 100+ per class chahiye
```

### Finding 2 — Class Imbalance
```
HR → 7 documents (sabse zyada)
Model ne seekha → "kuch bhi ho → HR bol do!"

Isko kehte hain: Bias towards majority class

Fix: Balanced data chahiye
     Har class mein equal documents
```

### Finding 3 — Doc Type Cross Validation Failed
```
Kuch types mein sirf 1 document:
Agreement, Checklist, Plan, Process, Report → 1 each

Cross Validation split karo → class missing ho jaati hai
XGBoost crash kar deta hai!

Fix: Zyada documents per type chahiye
```

---

## My Observation

> Maine XGBoost try kiya aur observe kiya ki jab model test karo
> apne hi documents pe toh accuracy badhiya aati hai (96.67%) —
> but jab naya document aata hai toh fail hota hai.
>
> Kyunki 30 documents bahut kam hain. Model ne HR department ki
> taraf bias seekh liya kyunki HR mein sabse zyada documents the.
>
> Abhi ke liye ye acceptable hai — production mein 100+ documents
> per class chahiye honge accurate classification ke liye.

---

## Saved Models

| File | Purpose |
|---|---|
| `dept_model.pkl` | Department classifier |
| `type_model.pkl` | Document type classifier |
| `dept_encoder.pkl` | Label encoder for departments |
| `type_encoder.pkl` | Label encoder for doc types |
| `vectorizer.pkl` | TF-IDF vectorizer |

Note: `.pkl` files are in `.gitignore` — regenerate by running `classify.py`

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
- Region/location entities
- Document type signals
- Key dates and deadlines
