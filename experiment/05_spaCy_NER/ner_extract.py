import spacy
import json
import csv
import os

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS_JSON = os.path.join(BASE_DIR, "documents.json")

# Load documents
with open(DOCS_JSON, "r", encoding="utf-8") as f:
    documents = json.load(f)

# Add EntityRuler
ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = [

    # =====================
    # DEPARTMENTS
    # =====================
    {"label": "DEPARTMENT", "pattern": "HR"},
    {"label": "DEPARTMENT", "pattern": "Finance"},
    {"label": "DEPARTMENT", "pattern": "Engineering"},
    {"label": "DEPARTMENT", "pattern": "Legal"},
    {"label": "DEPARTMENT", "pattern": "Operations"},
    {"label": "DEPARTMENT", "pattern": "Sales"},
    {"label": "DEPARTMENT", "pattern": "Marketing"},

    # =====================
    # REGIONS
    # =====================
    {"label": "REGION", "pattern": "India"},
    {"label": "REGION", "pattern": "UK"},
    {"label": "REGION", "pattern": "US"},
    {"label": "REGION", "pattern": "Global"},

    # =====================
    # DOCUMENT TYPES
    # =====================
    {"label": "DOC_TYPE", "pattern": "SOP"},
    {"label": "DOC_TYPE", "pattern": "Policy"},
    {"label": "DOC_TYPE", "pattern": "Procedure"},
    {"label": "DOC_TYPE", "pattern": "Guidelines"},
    {"label": "DOC_TYPE", "pattern": "Agreement"},
    {"label": "DOC_TYPE", "pattern": "Checklist"},
    {"label": "DOC_TYPE", "pattern": "Plan"},
    {"label": "DOC_TYPE", "pattern": "Report"},
    {"label": "DOC_TYPE", "pattern": "Template"},

    # =====================
    # BUSINESS PROCESSES
    # =====================
    {"label": "PROCESS", "pattern": [{"LOWER": "onboarding"}]},
    {"label": "PROCESS", "pattern": [{"LOWER": "offboarding"}]},
    {"label": "PROCESS", "pattern": [{"LOWER": "payroll"}]},
    {"label": "PROCESS", "pattern": [{"LOWER": "deployment"}]},
    {"label": "PROCESS", "pattern": [{"LOWER": "audit"}]},
    {"label": "PROCESS", "pattern": [{"LOWER": "reimbursement"}]},
    {"label": "PROCESS", "pattern": [{"LOWER": "procurement"}]},

    {
        "label": "PROCESS",
        "pattern": [{"LOWER": "incident"}, {"LOWER": "response"}]
    },

    {
        "label": "PROCESS",
        "pattern": [{"LOWER": "vendor"}, {"LOWER": "management"}]
    },

    {
        "label": "PROCESS",
        "pattern": [{"LOWER": "change"}, {"LOWER": "management"}]
    },

    {
        "label": "PROCESS",
        "pattern": [{"LOWER": "risk"}, {"LOWER": "management"}]
    },

    {
        "label": "PROCESS",
        "pattern": [{"LOWER": "client"}, {"LOWER": "onboarding"}]
    },

    # =====================
    # HR
    # =====================
    {"label": "HR_ENTITY", "pattern": "ESIC"},
    {"label": "HR_ENTITY", "pattern": "PF"},
    {"label": "HR_ENTITY", "pattern": "PAYE"},
    {"label": "HR_ENTITY", "pattern": "HMRC"},

    # =====================
    # FINANCE
    # =====================
    {"label": "FIN_ENTITY", "pattern": "Budget"},
    {"label": "FIN_ENTITY", "pattern": "Expense"},
    {"label": "FIN_ENTITY", "pattern": "Revenue"},
    {"label": "FIN_ENTITY", "pattern": "Profit"},
    {"label": "FIN_ENTITY", "pattern": "Tax"},
    {"label": "FIN_ENTITY", "pattern": "Invoice"},

    # =====================
    # ENGINEERING
    # =====================
    {"label": "ENG_ENTITY", "pattern": "API"},
    {"label": "ENG_ENTITY", "pattern": "CI/CD"},
    {"label": "ENG_ENTITY", "pattern": "Docker"},
    {"label": "ENG_ENTITY", "pattern": "Kubernetes"},
    {"label": "ENG_ENTITY", "pattern": "Microservices"},
    {"label": "ENG_ENTITY", "pattern": "SLA"},

    # =====================
    # LEGAL
    # =====================
    {"label": "LEGAL_ENTITY", "pattern": "NDA"},
    {"label": "LEGAL_ENTITY", "pattern": "Confidential"},
    {"label": "LEGAL_ENTITY", "pattern": "Liability"},
    {"label": "LEGAL_ENTITY", "pattern": "Intellectual Property"},

    # =====================
    # COMPLIANCE
    # =====================
    {"label": "COMPLIANCE", "pattern": "GDPR"},
    {"label": "COMPLIANCE", "pattern": "ISO 27001"},
    {"label": "COMPLIANCE", "pattern": "SOC 2"},
    {"label": "COMPLIANCE", "pattern": "HIPAA"},
]
if "entity_ruler" in nlp.pipe_names:
    ruler = nlp.get_pipe("entity_ruler")
else:
    ruler = nlp.add_pipe(
        "entity_ruler",
        before="ner",
        config={"phrase_matcher_attr": "LOWER"}
    )

ruler.add_patterns(patterns)


def extract_entities(text):
    doc = nlp(text)

    entities = {
        "departments": set(),
        "regions": set(),
        "doc_types": set(),
        "processes": set(),
        "hr_entities": set(),
        "fin_entities": set(),
        "eng_entities": set(),
        "legal_entities": set(),
        "compliance": set()
    }

    for ent in doc.ents:

        if ent.label_ == "DEPARTMENT":
            entities["departments"].add(ent.text)

        elif ent.label_ == "REGION":
            entities["regions"].add(ent.text)

        elif ent.label_ == "DOC_TYPE":
            entities["doc_types"].add(ent.text)

        elif ent.label_ == "PROCESS":
            entities["processes"].add(ent.text)

        elif ent.label_ == "HR_ENTITY":
            entities["hr_entities"].add(ent.text)

        elif ent.label_ == "FIN_ENTITY":
            entities["fin_entities"].add(ent.text)

        elif ent.label_ == "ENG_ENTITY":
            entities["eng_entities"].add(ent.text)

        elif ent.label_ == "LEGAL_ENTITY":
            entities["legal_entities"].add(ent.text)

        elif ent.label_ == "COMPLIANCE":
            entities["compliance"].add(ent.text)

    return entities


print("\nAll Documents - Metadata Extraction")
print("=" * 70)

output = []

for doc_id, text in documents.items():

    entities = extract_entities(text)

    print(f"\n{doc_id}")
    print(entities)

    output.append({
        "doc_id": doc_id,
        "departments": ", ".join(sorted(entities["departments"])),
        "regions": ", ".join(sorted(entities["regions"])),
        "doc_types": ", ".join(sorted(entities["doc_types"])),
        "processes": ", ".join(sorted(entities["processes"])),
        "hr_entities": ", ".join(sorted(entities["hr_entities"])),
        "fin_entities": ", ".join(sorted(entities["fin_entities"])),
        "eng_entities": ", ".join(sorted(entities["eng_entities"])),
        "legal_entities": ", ".join(sorted(entities["legal_entities"])),
        "compliance": ", ".join(sorted(entities["compliance"]))
    })

# Save CSV
csv_file = "ner_results.csv"

with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "doc_id",
            "departments",
            "regions",
            "doc_types",
            "processes",
            "hr_entities",
            "fin_entities",
            "eng_entities",
            "legal_entities",
            "compliance"
        ]
    )

    writer.writeheader()
    writer.writerows(output)

print(f"\nNER results saved to {csv_file}")


# Test New Document
new_doc = """
This SOP defines the payroll processing procedure for all HR employees in India.
Finance department must ensure ESIC and PF compliance.
Legal team must review all NDA contracts before approval.
API security guidelines must follow ISO 27001 requirements.
"""

print("\n" + "=" * 70)
print("NEW DOCUMENT TEST")
print("=" * 70)

print(extract_entities(new_doc))