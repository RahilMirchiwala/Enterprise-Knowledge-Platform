# Experiment 03 — Hybrid Search (TF-IDF + SBERT + Metadata)

**Status:** Complete
**Result:** 13/25 — SBERT alone (14/25) se better nahi hua

---

## Goal

Experiment 02 mein SBERT ne 14/25 pairs sahi kiye — but 3 False Positives aaye:

```
HR-001 ↔ OPS-001  → Human: 30% | SBERT: 62% ⚠️ FP
OPS-003 ↔ FIN-003 → Human: 15% | SBERT: 53% ⚠️ FP
HR-003  ↔ ENG-003 → Human: 10% | SBERT: 32% ⚠️ FP
```

Hypothesis: TF-IDF aur metadata combine karne se False Positives kam honge.

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

**Finding:** Best weight = 0.0 TF-IDF — matlab TF-IDF ka koi faayda nahi!

---

### Attempt 2 — Normalized Weighted Average

Problem: TF-IDF scores (0-40%) aur SBERT scores (40-80%) alag ranges mein the.

Fix: Normalize kiya dono ko 0-1 range mein.

```
Formula: normalized = (score - min) / (max - min)

TF-IDF after normalize: 0.00 - 1.00
SBERT after normalize:  0.00 - 1.00
```

Grid Search result:
```
Best weight → TF-IDF: 0.0 | SBERT: 1.0 → 13/25
```

**Finding:** Normalization ne scale fix kiya — but information quality nahi badli!
TF-IDF ki low-quality scores SBERT ko drag down karti hain.

---

### Attempt 3 — Metadata-Aware Scoring

Idea: Department aur business function information use karo similarity penalize/boost karne ke liye.

```
Same dept + Same function → multiplier 1.0  (no change)
Same dept only            → multiplier 0.8  (slight penalize)
Alag dept                 → multiplier 0.7  (penalize)

Final Score = SBERT Score × Multiplier
```

Grid Search on multipliers:
```
Alag dept multiplier: 0.5 → 11/25
Alag dept multiplier: 0.6 → 12/25
Alag dept multiplier: 0.7 → 13/25  ← best
Alag dept multiplier: 0.8 → 13/25
Alag dept multiplier: 0.9 → 14/25
Alag dept multiplier: 1.0 → 14/25  ← same as SBERT alone!
```

**Finding:** Multiplier 1.0 = no change = SBERT alone (14/25)
Koi bhi penalty lagao — ek pair fix hota hai, doosra toot jaata hai!

---

## Why Hybrid Failed

### Problem 1 — TF-IDF drags SBERT down
```
TF-IDF → word frequency → low quality information
SBERT  → semantic meaning → high quality information

Mix karne se SBERT ka accurate score
TF-IDF ke galat score se contaminate ho jaata hai!

Normalization sirf scale fix karta hai
Underlying quality nahi badlti.
```

### Problem 2 — Fixed multiplier flexible nahi hai
```
ENG-008 ↔ OPS-008
Disaster Recovery vs Business Continuity
Alag dept — but concepts closely related!
Human: 80% | Multiplier 0.7 → 39% ❌ bahut kam!

HR-001 ↔ OPS-001
Employee Onboarding vs Client Onboarding
Alag dept — concepts alag!
Human: 30% | Multiplier 0.7 → 43% ⚠️ abhi bhi zyada

Ek multiplier dono ko satisfy nahi kar sakta!
```

---

## Final Comparison — All 3 Experiments

| Experiment | Algorithm | OK | FN | FP |
|---|---|---|---|---|
| 01 | TF-IDF | 5/25 | 20 | 0 |
| 02 | SBERT | 14/25 | 8 | 3 |
| 03 | Hybrid (best attempt) | 13/25 | 9 | 3 |

**Winner: SBERT alone — Experiment 02** 🏆

---

## My Observation

> Maine Hybrid try kiya — normalize bhi kiya, weights bhi tune kiye, metadata multipliers bhi lagaye — phir bhi SBERT alone se better result nahi aaya.
>
> Root cause: TF-IDF ki information quality hi kam hai. Use kisi bhi tarah mix karo — SBERT ka score girata hai, badhta nahi.
>
> Metadata multipliers ka problem ye tha ki ek fixed number alag alag document pairs ke liye alag behave karta tha — ENG-008 vs OPS-008 (related concepts, alag dept) aur HR-001 vs OPS-001 (unrelated concepts, alag dept) dono pe same multiplier kaam nahi karta.

---

## What Would Actually Help

```
Option 1 — Fine-tuning SBERT
25 labeled pairs pe train karo
"HR Onboarding" vs "Client Onboarding" alag hain — model ko sikhao
Zyada data chahiye hoga — 25 pairs kam hain

Option 2 — Cross-encoder model
Bi-encoder (current SBERT) → fast but less accurate
Cross-encoder → dono documents saath process karta hai → more accurate
Slower but better at understanding business context

Option 3 — LLM-based reasoning (Layer 6)
GPT/Llama se poochho — "Are these two documents similar?"
Context + reasoning → most accurate
But expensive aur slow
```

---

## Files

| File | Purpose |
|---|---|
| `hybrid_similarity.py` | TF-IDF + SBERT + Metadata scoring |

---

## How To Run

```bash
python hybrid_similarity.py
```

---

## Key Lesson

> Simple rule-based hybrid approaches complex document similarity problems
> handle nahi kar sakte. Real improvement ke liye fine-tuning ya
> more sophisticated models (cross-encoder, LLM) ki zaroorat hai.
