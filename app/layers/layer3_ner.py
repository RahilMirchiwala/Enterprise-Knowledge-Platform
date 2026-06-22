import spacy

nlp = spacy.load("en_core_web_sm")

if "entity_ruler" in nlp.pipe_names:
    ruler = nlp.get_pipe("entity_ruler")
else:
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

    # Business Processes
    {"label": "PROCESS", "pattern": [{"LOWER": "onboarding"}]},
    {"label": "PROCESS", "pattern": [{"LOWER": "payroll"}]},
    {"label": "PROCESS", "pattern": [{"LOWER": "deployment"}]},
    {"label": "PROCESS", "pattern": [{"LOWER": "audit"}]},
    {"label": "PROCESS", "pattern": [{"LOWER": "reimbursement"}]},
    {"label": "PROCESS", "pattern": [{"LOWER": "incident"}, {"LOWER": "response"}]},
    {"label": "PROCESS", "pattern": [{"LOWER": "vendor"}, {"LOWER": "management"}]},

    # HR Entities
    {"label": "HR_ENTITY", "pattern": "ESIC"},
    {"label": "HR_ENTITY", "pattern": "PF"},
    {"label": "HR_ENTITY", "pattern": "HMRC"},
    {"label": "HR_ENTITY", "pattern": "PAYE"},
    {"label": "HR_ENTITY", "pattern": "TDS"},

    # Engineering Entities
    {"label": "ENG_ENTITY", "pattern": "API"},
    {"label": "ENG_ENTITY", "pattern": "SLA"},
    {"label": "ENG_ENTITY", "pattern": "CI/CD"},

    # Legal Entities
    {"label": "LEGAL_ENTITY", "pattern": "NDA"},
    {"label": "LEGAL_ENTITY", "pattern": "Confidential"},

    # Compliance
    {"label": "COMPLIANCE", "pattern": "GDPR"},
    {"label": "COMPLIANCE", "pattern": "ISO 27001"},
    {"label": "COMPLIANCE", "pattern": "SOC 2"},
]

ruler.add_patterns(patterns)


def extract(text: str):
    doc = nlp(text[:2000])

    departments = set()
    regions = set()
    doc_types = set()
    processes = set()
    hr_entities = set()
    eng_entities = set()
    legal_entities = set()
    compliance = set()

    for ent in doc.ents:
        if ent.label_ == "DEPARTMENT":
            departments.add(ent.text)
        elif ent.label_ == "REGION":
            regions.add(ent.text)
        elif ent.label_ == "DOC_TYPE":
            doc_types.add(ent.text)
        elif ent.label_ == "PROCESS":
            processes.add(ent.text)
        elif ent.label_ == "HR_ENTITY":
            hr_entities.add(ent.text)
        elif ent.label_ == "ENG_ENTITY":
            eng_entities.add(ent.text)
        elif ent.label_ == "LEGAL_ENTITY":
            legal_entities.add(ent.text)
        elif ent.label_ == "COMPLIANCE":
            compliance.add(ent.text)

    # set → list (JSON serializable)
    return {
        "departments": list(departments),
        "regions": list(regions),
        "doc_types": list(doc_types),
        "processes": list(processes),
        "hr_entities": list(hr_entities),
        "eng_entities": list(eng_entities),
        "legal_entities": list(legal_entities),
        "compliance": list(compliance)
    }