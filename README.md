# Enterprise Knowledge Intelligence Platform
### NovaBridge Consulting — Document Similarity Experiments

---

## What Is This Project?

NovaBridge Consulting operates across India, UK, and US offices.
They have 30+ business documents across 5 departments — many of which are duplicates, regional variants, or updated versions.

**The Problem:** Finding similar or duplicate documents manually is time-consuming and error-prone.

**The Goal:** Build and evaluate different algorithms to automatically detect document similarity — from simple keyword matching to semantic understanding.

---

## My Approach

Instead of directly using the most advanced model, I deliberately started simple and worked upward.

Each experiment answers one question:
> *"Does this algorithm understand that two documents are similar — even when written differently?"*

I tested every algorithm against the same 25 manually labeled document pairs, so results are directly comparable.

---

## Experiment Results Summary

| Experiment | Algorithm | Correct Pairs | False Negatives | Key Finding |
|---|---|---|---|---|
| 01 | TF-IDF + Cosine | 5 / 25 | 20 / 25 | Fails on same meaning, different words |
| 02 | Sentence Transformers | Coming Soon | — | — |
| 03 | Hybrid Search | Coming Soon | — | — |

---

## Dataset

- **30 documents** across 5 departments (HR, Finance, Engineering, Legal, Operations)
- **5 similarity categories** — Near Duplicate, Highly Similar, Moderately Similar, Slightly Related, Unrelated
- **25 human-labeled pairs** with similarity scores (0–100%)
- Documents include: regional variants (India/UK/US), version upgrades (v1.0 → v2.0), and completely unrelated pairs

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
├── documents/                    # 30 .docx source documents
├── evaluation_dataset.csv        # 25 human-labeled pairs (ground truth)
│
├── experiments/
│   ├── 01_tfidf/
│   │   ├── extract_text.py       # Extract text from .docx files
│   │   ├── tfidf_similarity.py   # TF-IDF + Cosine Similarity
│   │   └── README.md             # Experiment 1 findings
│   │
│   ├── 02_sentence_transformers/
│   │   └── README.md             # Coming Soon
│   │
│   └── 03_hybrid/
│       └── README.md             # Coming Soon
│
└── README.md                     # This file
```

---

## Key Insight So Far

> TF-IDF correctly identifies unrelated documents but consistently underestimates similarity between documents that share the same workflow but use different regional terminology.
>
> Example: HR-001 (India Onboarding) vs HR-002 (UK Onboarding)
> Human Score: 95% — TF-IDF Score: 18.41%
>
> The algorithm sees "Aadhaar, PAN, ESIC" vs "HMRC, PAYE, Right to Work" as completely different — because they are different words. But a human immediately recognizes both as employee identity verification steps.

This is why semantic models are needed for real-world document intelligence.

---

## Tech Stack

- Python 3.13
- scikit-learn
- sentence-transformers (Experiment 2)
- pandas
- python-docx

---

## Author

**Rahil Mirchiwala**
CS Student — Parul University
GitHub: [RahilMirchiwala](https://github.com/RahilMirchiwala)
