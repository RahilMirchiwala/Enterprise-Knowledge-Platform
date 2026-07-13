# Phase 2 — Learn to Measure Like a Professional

**Timeline:** 2–3 weeks  
**Goal:** Replace ad-hoc accuracy counts with proper evaluation metrics. Learn to run and track experiments the way industry does.

---

## Why This Phase Exists

Right now your experiments report results like "5/25 correct" or "14/25 correct". That tells a reader something — but not enough. Here is the problem:

**"Correct" is not a metric. It's an outcome.** 

A metric tells you *how* something failed, not just *that* it failed. Without proper metrics:
- You can't compare two models objectively.
- You can't tell whether an improvement is real or just luck.
- You can't explain to a client or stakeholder *why* one approach is better.

Jeavio builds AI for private equity firms. PE firms ask hard questions: "How accurate is this system? How do we know it won't miss critical documents?" You need to be able to answer those questions with numbers that mean something.

This phase teaches you how.

---

## Concept 1 — Why "X/25 Correct" Is Not Enough

Imagine a document similarity system that always says "not similar" for every pair. On your dataset, where most pairs are genuinely not similar, it might get 17/25 "correct" — better than your SBERT model. But it's useless. It never finds anything.

This is the core problem with raw accuracy: **it rewards systems that are cautiously wrong**.

Real evaluation needs to answer four questions separately:
1. Of all the pairs it said were similar, how many actually were? (**Precision**)
2. Of all the pairs that actually were similar, how many did it find? (**Recall**)
3. What's the balance between those two? (**F1 Score**)
4. When it ranked similar documents, were the best ones at the top? (**Mean Reciprocal Rank**)

---

## Concept 2 — Precision, Recall, and F1

### Precision
> Of everything the model said was similar, what fraction was actually similar?

```
Precision = True Positives / (True Positives + False Positives)
```

A model with high precision rarely cries wolf. When it says "these two documents are similar", it's usually right.

**Low precision cost:** You show users irrelevant results. They lose trust in the system.

### Recall
> Of everything that was actually similar, what fraction did the model find?

```
Recall = True Positives / (True Positives + False Negatives)
```

A model with high recall rarely misses things. It finds most of the similar documents.

**Low recall cost:** The system misses important matches. A legal team misses a duplicate contract. A finance team misses a regional policy variant.

### The Precision-Recall Tradeoff
You almost always have to trade one against the other. Lowering your similarity threshold catches more matches (higher recall) but includes more false alarms (lower precision). The right tradeoff depends on the use case:

- **Duplicate detection (your project):** Recall matters more. Missing a duplicate is worse than flagging a false positive.
- **Legal contract search:** Precision matters more. Returning an irrelevant contract is worse than missing one.

Understanding this tradeoff and being able to articulate it is a sign of a mature data scientist.

### F1 Score
The harmonic mean of precision and recall. Penalises extreme imbalances — a model with 100% precision but 0% recall gets an F1 of 0.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Use F1 as your headline metric when you want a single number that respects both sides.

---

## Concept 3 — Mean Reciprocal Rank (MRR)

Your search system returns a *ranked list* of similar documents, not just a yes/no answer. "Correct" doesn't capture rank quality. MRR does.

```
MRR = (1/N) × Σ (1 / rank of first relevant result)
```

Example:
- Query 1: the first relevant document is at rank 2 → score = 1/2
- Query 2: the first relevant document is at rank 1 → score = 1/1
- Query 3: the first relevant document is at rank 3 → score = 1/3

```
MRR = (1/3) × (0.5 + 1.0 + 0.33) = 0.61
```

A perfect system (always returns the best match first) has MRR = 1.0.

MRR is the standard metric for search and retrieval systems. Jeavio builds document search systems. If you walk into an interview and know MRR, you stand out.

---

## Tasks

---

### 1. Add a proper evaluation module to each experiment

**What to do:**  
In each experiment folder, add an `evaluate.py` file that computes Precision@K, Recall@K, F1, and MRR against your `evaluation_dataset.csv`.

Use `sklearn.metrics` for precision/recall/F1. Write MRR by hand — it's 5 lines and understanding it by implementing it is more valuable than importing it.

```python
def mean_reciprocal_rank(ranked_results, relevant_set):
    """
    ranked_results: list of doc_ids in ranked order
    relevant_set: set of doc_ids that are actually relevant
    """
    for rank, doc_id in enumerate(ranked_results, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0
```

**Why this matters:**  
You already have the ground truth (your 25 labeled pairs). You have the model outputs. The only thing missing is the code that connects them correctly. This is a core data science skill — evaluating models rigorously — and right now it's absent from your repo.

---

### 2. Build a unified evaluation notebook

**What to do:**  
Create `experiment/07_evaluation/compare_all.ipynb` — a Jupyter notebook that:
- Loads all experiment results (or re-runs each model on the 25 test pairs)
- Computes Precision@3, Recall@3, F1, MRR for each approach
- Produces a single comparison table and a bar chart

| Experiment | Precision@3 | Recall@3 | F1 | MRR |
|---|---|---|---|---|
| TF-IDF | ? | ? | ? | ? |
| SBERT | ? | ? | ? | ? |
| Hybrid | ? | ? | ? | ? |

**Why this matters:**  
Right now your README compares experiments using "X/25 correct" — different experiments, different counting, no shared baseline. The notebook forces you to put every model on the exact same footing.

This is called **controlled comparison**, and it's a fundamental scientific discipline. If you change the evaluation metric between experiments, you don't know if the improvement is real or just an artefact of how you measured it. The notebook eliminates that doubt.

---

### 3. Introduce MLflow for experiment tracking

**What to do:**  
Install `mlflow` and add 10 lines to each experiment script that log:
- The model name and parameters
- The evaluation metrics (Precision, Recall, F1, MRR)
- The runtime

```python
import mlflow

with mlflow.start_run(run_name="TF-IDF Baseline"):
    mlflow.log_param("model", "TF-IDF + Cosine")
    mlflow.log_param("ngram_range", "(1,2)")
    mlflow.log_metric("precision_at_3", 0.21)
    mlflow.log_metric("recall_at_3", 0.18)
    mlflow.log_metric("f1", 0.19)
    mlflow.log_metric("mrr", 0.24)
```

Run `mlflow ui` to see a dashboard comparing all your runs.

**Why this matters:**  
Right now your experiments live in separate folders with separate READMEs. If you want to compare them, you have to manually read each README and copy numbers into a table. That's fine for 6 experiments. It doesn't scale.

MLflow is the industry standard for experiment tracking. Every serious ML team — including the ones Jeavio works with — uses some form of it. Knowing how to log experiments and compare runs is a baseline expectation for any ML role. Adding it to your repo signals you understand how ML is done at scale, not just in notebooks.

The deeper lesson: An experiment you can't reproduce or compare is not an experiment — it's a one-off.

---

### 4. Write a `RESULTS.md` at the repo root

**What to do:**  
Create a `RESULTS.md` file that summarises all 6 experiments in one place using proper metrics. One table, all models, side by side. Add a "Key Takeaway" paragraph below it in plain English.

**Why this matters:**  
This is the document a hiring manager, a senior engineer, or a Jeavio client would read first if they wanted to understand what you built and whether it works. It's your executive summary.

It also forces you to synthesise your own work. The ability to take 6 weeks of experiments and distill them into one coherent narrative is a high-value professional skill. Most engineers can execute. Fewer can communicate what they executed and why it matters.

---

## How You Know Phase 2 Is Complete

- [ ] Each experiment folder has an `evaluate.py` that computes Precision@K, Recall@K, F1, and MRR
- [ ] `experiment/07_evaluation/compare_all.ipynb` exists and produces a comparison table and chart
- [ ] MLflow is integrated — you can run `mlflow ui` and see all 6 experiments as separate runs
- [ ] `RESULTS.md` exists at the repo root with a single comparison table and a plain-English summary

---

## What You Will Have Learned

By the end of Phase 2, you will understand evaluation as a discipline — not just "did it work" but "how well did it work, in what ways did it fail, and how does it compare to the alternative?" This is the difference between a data scientist who runs models and one who understands them.
