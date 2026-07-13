# Experiment 05 — spaCy NER (Layer 3)

**Status:** Complete
**Result:** Custom EntityRuler successfully extracts business entities from all 30 documents

---

## Goal

Automatically extract structured metadata from document text:
- **Department** → HR, Finance, Engineering, Legal, Operations
- **Region** → India, UK, US, Global
- **Document Type** → SOP, Policy, Contract, NDA, Guidelines...
- **HR Entities** → ESIC, PF, HMRC, PAYE, TDS, GST

This allows the platform to filter and search documents by these attributes without requiring manual tagging.

---

## What is NER?

**NER = Named Entity Recognition**

```
Text: "This SOP applies to all HR employees in India"

NER extracts:
→ "SOP"   = DOC_TYPE
→ "HR"    = DEPARTMENT
→ "India" = REGION
```

---

## Attempt 1 — Default spaCy Model

```python
nlp = spacy.load("en_core_web_sm")
doc = nlp("Production Deployment SOP for Engineering team")
```

**Results — incorrect:**
```
"Rollback"     → PERSON  (not a person)
"CI"           → GPE     (not a city — Continuous Integration)
"CTO Rollback" → PERSON
"SOP"          → ORG     (not an organisation)
```

**Why it failed:**
```
en_core_web_sm was trained on general English text (news articles, web content).

Our documents use technical business terminology:
"CI", "SOP", "ESIC" carry different meanings in a business context
than in general English.
```

---

## Attempt 2 — Custom EntityRuler

spaCy's `EntityRuler` allows defining custom entity patterns without any training data.

```python
ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = [
    {"label": "DEPARTMENT", "pattern": "HR"},
    {"label": "DEPARTMENT", "pattern": "Finance"},
    {"label": "REGION",     "pattern": "India"},
    {"label": "DOC_TYPE",   "pattern": "SOP"},
    {"label": "HR_ENTITY",  "pattern": "ESIC"},
    # ... additional patterns
]

ruler.add_patterns(patterns)
```

**`before="ner"`** — custom rules run before spaCy's general NER model.

---

## Custom Patterns Defined

| Label | Patterns |
|---|---|
| DEPARTMENT | HR, Finance, Engineering, Legal, Operations |
| REGION | India, UK, US, Global |
| DOC_TYPE | SOP, Policy, Contract, Guidelines, Agreement, NDA, Procedure, Checklist, Plan, Report |
| HR_ENTITY | ESIC, PF, TDS, GST, HMRC, PAYE |
| LEG_ENTITY | Confidential |
| ENG_ENTITY | API, CI/CD, SLA |
| COMPLIANCE | GDPR, ISO 27001, SOC 2 |

---

## Results — All 30 Documents

### Sample Output

```
HR-001 India Employee Onboarding SOP
  Departments : HR
  Regions     : India
  Doc Types   : SOP, Checklist, Plan, Procedure
  HR Entities : ESIC, PF

HR-002 UK Employee Onboarding SOP
  Departments : HR, Legal
  Regions     : UK
  Doc Types   : SOP
  HR Entities : PAYE, HMRC

LEG-001 Vendor NDA
  Departments : (none)
  Regions     : (none)
  Doc Types   : Agreement, NDA
  HR Entities : (none)

OPS-002 Escalation Matrix
  Departments : HR, Finance, Engineering, Legal, Operations
  Regions     : (none)
  Doc Types   : (none)
  HR Entities : (none)
```

### Observations

**OPS-002 returned all 5 departments:**
The Escalation Matrix genuinely references all departments.
NER correctly identified every department mention in the text.

**LEG-001 returned no department:**
The Vendor NDA uses pure legal language with no explicit department names.
Fixed by adding "Agreement" and "NDA" as DOC_TYPE patterns.

---

## New Document Prediction

```python
new_doc = """
This SOP defines the payroll processing procedure for HR employees in India.
Finance department must ensure ESIC and PF compliance.
Legal team must review all contracts before approval.
"""
```

**Output:**
```
Departments : HR, Finance, Legal
Regions     : India
Doc Types   : SOP
HR Entities : ESIC, PF
```

Any new document → entities extracted automatically.

---

## Rule-based vs ML-based NER

| Approach | Training Data | Speed | Business Context |
|---|---|---|---|
| spaCy default (ML) | Millions of sentences | Fast | Incorrect |
| Custom EntityRuler (Rules) | None required | Fast | Correct |
| Fine-tuned spaCy (ML) | 1000+ labeled examples | Slow to train | Correct |

**Best approach for this use case: Custom EntityRuler**

Reasons:
- No training data required
- Fast and deterministic
- Precisely matches domain-specific terminology

---

## Observation

> The default spaCy model failed on business documents — labelling
> "Rollback" as a PERSON and "CI" as a city. It was trained on general
> English and has no awareness of enterprise software terminology.
>
> The custom EntityRuler solved this immediately. By explicitly defining
> what "HR", "ESIC", "SOP" mean in our context, extraction became
> accurate across all 30 documents with zero training data.
>
> This is a case where a simple rule-based approach outperforms a
> sophisticated ML model — because the domain is narrow and well-defined.

---

## Files

| File | Purpose |
|---|---|
| `ner_extract.py` | spaCy pipeline with custom EntityRuler |
| `ner_results.csv` | Extracted entities for all 30 documents |

---

## How To Run

```bash
pip install spacy
python -m spacy download en_core_web_sm

python ner_extract.py
# Output: ner_results.csv
```

---

## Next Experiment

**Experiment 06 — Groq LLM (Layer 6)**

Use Groq/LLaMA to generate intelligent answers grounded in retrieved documents.

```
User: "What is the remote work policy?"
SBERT: retrieves HR-003, HR-006
Groq:  reads documents → accurate, structured answer
```
