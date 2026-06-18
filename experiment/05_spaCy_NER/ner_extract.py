import spacy
import json
import csv
import os

nlp = spacy.load("en_core_web_sm")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS_JSON = os.path.join(BASE_DIR, "documents.json")

with open(DOCS_JSON, "r", encoding="utf-8") as f:
    documents = json.load(f)

ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = [
    # Departments
    {"label": "DEPARTMENT", "pattern": "HR"},
    {"label": "DEPARTMENT", "pattern": "Finance"},
    {"label": "DEPARTMENT", "pattern": "Engineering"},
    {"label": "DEPARTMENT", "pattern": "Legal"},
    {"label": "DEPARTMENT", "pattern": "Operations"},

    # Regions
    {"label": "REGION", "pattern": "India"},
    {"label": "REGION", "pattern": "UK"},
    {"label": "REGION", "pattern": "US"},
    {"label": "REGION", "pattern": "Global"},

    # Document Types
    {"label": "DOC_TYPE", "pattern": "SOP"},
    {"label": "DOC_TYPE", "pattern": "Policy"},
    {"label": "DOC_TYPE", "pattern": "Contract"},
    {"label": "DOC_TYPE", "pattern": "Guidelines"},
    {"label": "DOC_TYPE", "pattern": "Agreement"},
    {"label": "DOC_TYPE", "pattern": "NDA"},
    {"label": "DOC_TYPE", "pattern": "Procedure"},
    {"label": "DOC_TYPE", "pattern": "Checklist"},
    {"label": "DOC_TYPE", "pattern": "Plan"},
    {"label": "DOC_TYPE", "pattern": "Report"},

    # HR / Compliance Entities
    {"label": "HR_ENTITY", "pattern": "ESIC"},
    {"label": "HR_ENTITY", "pattern": "PF"},
    {"label": "HR_ENTITY", "pattern": "TDS"},
    {"label": "HR_ENTITY", "pattern": "GST"},
    {"label": "HR_ENTITY", "pattern": "HMRC"},
    {"label": "HR_ENTITY", "pattern": "PAYE"},

    # Legal Entities
    {"label": "LEG_ENTITY", "pattern": "Confidential"},

    # Engineering Entities
    {"label": "ENG_ENTITY", "pattern": "API"},
    {"label": "ENG_ENTITY", "pattern": "CI/CD"},
    {"label": "ENG_ENTITY", "pattern": "SLA"},
]

ruler.add_patterns(patterns)

def extract_entities(text):
    doc = nlp(text[:2000])

    departments = set()
    regions = set()
    doc_types = set()
    hr_entities = set()

    for ent in doc.ents:
        if ent.label_ == "DEPARTMENT":
            departments.add(ent.text)
        elif ent.label_ == "REGION":
            regions.add(ent.text)
        elif ent.label_ == "DOC_TYPE":
            doc_types.add(ent.text)
        elif ent.label_ == "HR_ENTITY":
            hr_entities.add(ent.text)

    return {
        "departments": departments,
        "regions": regions,
        "doc_types": doc_types,
        "hr_entities": hr_entities
    }

print("\nAll Documents - NER Extraction:")
print("-" * 60)

output = []
for doc_id, text in documents.items():
    entities = extract_entities(text)

    print(f"\n{doc_id}")
    print(f"  Departments : {entities['departments']}")
    print(f"  Regions     : {entities['regions']}")
    print(f"  Doc Types   : {entities['doc_types']}")
    print(f"  HR Entities : {entities['hr_entities']}")

    output.append({
        "doc_id": doc_id,
        "departments": ", ".join(entities["departments"]),
        "regions":     ", ".join(entities["regions"]),
        "doc_types":   ", ".join(entities["doc_types"]),
        "hr_entities": ", ".join(entities["hr_entities"])
    })

with open("ner_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "doc_id", "departments", "regions", "doc_types", "hr_entities"
    ])
    writer.writeheader()
    writer.writerows(output)

print("\nNER results saved to ner_results.csv!")

def predict_new_document(text):
    entities = extract_entities(text)
    print("\nNew Document Entities:")
    print(f"  Departments : {entities['departments']}")
    print(f"  Regions     : {entities['regions']}")
    print(f"  Doc Types   : {entities['doc_types']}")
    print(f"  HR Entities : {entities['hr_entities']}")

# Test
new_doc = """
This SOP defines the payroll processing procedure for all HR employees in India.
Finance department must ensure ESIC and PF compliance by 15th of each month.
Legal team must review all contracts before approval.
"""
predict_new_document(new_doc)