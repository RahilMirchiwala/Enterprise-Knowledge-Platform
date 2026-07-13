# Experiment 03 — Hybrid Search (TF-IDF + SBERT + Metadata)

**Status:** Complete
**Result:** 13/25 — did not outperform SBERT alone (14/25)

---

## Goal

Experiment 02 showed SBERT getting 14/25 pairs correct but producing 3 False Positives:

```
HR-001 ↔ OPS-001  → Human: 30% | SBERT: 62% ⚠️ FP
OPS-003 ↔ FIN-003 → Human: 15% | SBERT: 53% ⚠️ FP
HR-003  ↔ ENG-003 → Human: 10% | SBERT: 32% ⚠️ FP
```

Hypothesis: Combining TF-IDF and metadata with SBERT would reduce False Positives.

---

## Attempts

### Attempt 1 — Simple Weighted Average

```
Hybrid = (w_tfidf × TF-IDF) + (w_sbert × SBERT)
```

Grid Search result:
```
TF-IDF: 0.0 | SBERT: 1.0 → 13/25  ← best
TF-IDF: 0.1 | SBERT: 0.9 → 13/25
TF-IDF: 0.2 | SBERT: 0.8 → 12/25
...
TF-IDF: 1.0 | SBERT: 0.0 →  5/25  ← worst
```

**Finding:** Best weight = 0.0 TF-IDF — TF-IDF adds no value.

---

### Attempt 2 — Normalized Weighted Average

Problem: TF-IDF scores (0–40%) and SBERT scores (40–80%) were on different scales.

Fix: Normalized both to 0–1 range.

```
Formula: normalized = (score - min) / (max - min)

TF-IDF after normalize: 0.00 - 1.00
SBERT after normalize:  0.00 - 1.00
```

Grid Search result:
```
Best weight → TF-IDF: 0.0 | SBERT: 1.0 → 13/25
```

**Finding:** Normalization fixed the scale — but not the information quality.
TF-IDF's low-quality scores still drag SBERT scores down.

---

### Attempt 3 — Metadata-Aware Scoring

Idea: Use department and business function metadata to penalize or boost similarity scores.

```
Same dept + Same function → multiplier 1.0  (no change)
Same dept only            → multiplier 0.8  (slight penalty)
Different dept            → multiplier 0.7  (penalty)

Final Score = SBERT Score × Multiplier
```

Grid Search on multipliers:
```
Different dept multiplier: 0.5 → 11/25
Different dept multiplier: 0.6 → 12/25
Different dept multiplier: 0.7 → 13/25  ← best
Different dept multiplier: 0.8 → 13/25
Different dept multiplier: 0.9 → 14/25
Different dept multiplier: 1.0 → 14/25  ← same as SBERT alone!
```

**Finding:** Multiplier 1.0 = no change = SBERT alone (14/25).
Every penalty applied either fixed one pair while breaking another.

---

## Why Hybrid Failed

### Problem 1 — TF-IDF Drags SBERT Down
```
TF-IDF → word frequency → low quality signal
SBERT  → semantic meaning → high quality signal

Mixing them contaminates SBERT's accurate scores with TF-IDF's noise.
Normalization fixes scale — it does not fix information quality.
```

### Problem 2 — Fixed Multiplier Is Not Flexible Enough
```
ENG-008 ↔ OPS-008
Disaster Recovery vs Business Continuity
Different dept — but concepts are closely related!
Human: 80% | Multiplier 0.7 → 39% ❌ too low!

HR-001 ↔ OPS-001
Employee Onboarding vs Client Onboarding
Different dept — concepts are genuinely different!
Human: 30% | Multiplier 0.7 → 43% ⚠️ still too high

One multiplier cannot satisfy both cases.
```

---

## Final Comparison — All 3 Experiments

| Experiment | Algorithm | OK | FN | FP |
|---|---|---|---|---|
| 01 | TF-IDF | 5/25 | 20 | 0 |
| 02 | SBERT | 14/25 | 8 | 3 |
| 03 | Hybrid (best attempt) | 13/25 | 9 | 3 |

**Winner: SBERT alone — Experiment 02**

---

## Key Finding

> Combining a weak signal (TF-IDF) with a strong signal (SBERT) degrades
> the strong signal. The best grid search weight was 0% TF-IDF + 100% SBERT —
> meaning TF-IDF contributed nothing. Mixing inferior signals does not
> improve results; it introduces noise.

---

## What Would Actually Help

```
Option 1 — Fine-tune SBERT on domain-specific data
           Train on labeled business document pairs
           Requires 500+ labeled pairs minimum

Option 2 — Cross-encoder model
           Processes both documents together (more context-aware)
           Slower but significantly more accurate

Option 3 — LLM-based reasoning
           Ask GPT/Llama: "Are these two documents similar?"
           Most accurate — but expensive and slow at scale
```

---

## Files

| File | Purpose |
|---|---|
| `hybrid_similarity.py` | TF-IDF + SBERT + Metadata scoring experiments |

---

## How To Run

```bash
python hybrid_similarity.py
```

---

## Key Lesson

> Simple rule-based hybrid approaches cannot handle the complexity of
> enterprise document similarity. Meaningful improvement requires
> fine-tuning or more sophisticated models (cross-encoder, LLM reasoning).
