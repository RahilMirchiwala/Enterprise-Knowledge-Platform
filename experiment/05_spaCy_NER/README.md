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
Why? So the platform can filter and search documents by these attributes — without user manually tagging every document.
 
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
 
**Results — completely wrong:**
```
"Rollback"     → PERSON     ❌ (not a person!)
"CI"           → GPE        ❌ (not a city — it's Continuous Integration!)
"CTO Rollback" → PERSON     ❌
"SOP"          → ORG        ❌ (not an organization!)
```
 
**Why failed:**
```
en_core_web_sm → trained on general English
                 news articles, web text
 
Our documents → technical business documents
                "CI", "SOP", "ESIC" have different
                meaning in business context!
```
 
---
 
## Attempt 2 — Custom EntityRuler
 
spaCy allows adding custom rules via `EntityRuler` — no training data needed!
 
```python
ruler = nlp.add_pipe("entity_ruler", before="ner")
 
patterns = [
    {"label": "DEPARTMENT", "pattern": "HR"},
    {"label": "DEPARTMENT", "pattern": "Finance"},
    {"label": "REGION",     "pattern": "India"},
    {"label": "DOC_TYPE",   "pattern": "SOP"},
    {"label": "HR_ENTITY",  "pattern": "ESIC"},
    # ... more patterns
]
 
ruler.add_patterns(patterns)
```
 
**`before="ner"`** → custom rules run first, then spaCy's general NER
 
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
 
---
 
## Results — All 30 Documents
 
### Sample Results
 
```
HR-001_India_Employee_Onboarding_SOP
  Departments : HR
  Regions     : India
  Doc Types   : SOP, Checklist, Plan, Procedure
  HR Entities : ESIC, PF
 
HR-002_UK_Employee_Onboarding_SOP
  Departments : HR, Legal
  Regions     : UK
  Doc Types   : SOP
  HR Entities : PAYE, HMRC
 
LEG-001_Vendor_NDA
  Departments : (none)
  Regions     : (none)
  Doc Types   : Agreement, NDA
  HR Entities : (none)
 
OPS-002_Escalation_Matrix
  Departments : HR, Finance, Engineering, Legal, Operations
  Regions     : (none)
  Doc Types   : (none)
  HR Entities : (none)
```
 
### Interesting Observations
 
**OPS-002 Escalation Matrix → all 5 departments:**
```
Correct! Escalation Matrix genuinely covers all departments.
NER correctly identified all department mentions in the text. ✅
```
 
**LEG-001 Vendor NDA → no department:**
```
NDA uses pure legal language — no department names mentioned.
Fix: Added "Agreement" and "NDA" as DOC_TYPE patterns. ✅
```
 
---
 
## New Document Prediction
 
```python
new_doc = """
This SOP defines the payroll processing procedure for HR employees in India.
Finance department must ensure ESIC and PF compliance.
Legal team must review all contracts.
"""
 
predict_new_document(new_doc)
```
 
**Output:**
```
Departments : HR, Finance, Legal
Regions     : India
Doc Types   : SOP
HR Entities : ESIC, PF
```
 
Any new document → automatically extract entities! ✅
 
---
 
## My Observation
 
> spaCy ka general model business documents ke liye kaam nahi kiya.
> "Rollback" ko PERSON bol diya, "CI" ko city samjha, "SOP" ko organization!
>
> Isliye custom EntityRuler banaya jo hamare specific business entities
> samajhta hai — HR, Finance, ESIC, PF, HMRC etc.
>
> Rule-based approach ML-based se better tha kyunki:
> - Training data ki zaroorat nahi
> - Fast aur accurate
> - Business-specific terms exactly match karta hai
 
---
 
## Rule-based vs ML-based NER
 
| Approach | Data Needed | Speed | Business Context |
|---|---|---|---|
| spaCy default (ML) | Crores sentences | Fast | ❌ Wrong |
| Custom EntityRuler (Rules) | 0 training data | Fast | ✅ Correct |
| Fine-tuned spaCy (ML) | 1000+ examples | Slow to train | ✅ Correct |
 
**Winner for our use case: Custom EntityRuler** ✅
 
---
 
## Files
 
| File | Purpose |
|---|---|
| `ner_extract.py` | spaCy NER pipeline + custom EntityRuler |
| `ner_results.csv` | Extracted entities for all 30 documents |
 
---
 
## How To Run
 
```bash
pip install spacy
python -m spacy download en_core_web_sm
 
python ner_extract.py
# Output: ner_results.csv (entities for all 30 docs)
```
 
---
 
## Next Experiment
 
**Experiment 06 — Groq LLM (Layer 6)**
 
Use Groq/Llama to generate intelligent summaries of similar documents.
 
```
User: "Remote work policy kya hai?"
SBERT: finds HR-003, HR-006
Groq:  "NovaBridge ki remote work policy ke mutabik:
        VPN mandatory hai, core hours 10-4 hain..."
```