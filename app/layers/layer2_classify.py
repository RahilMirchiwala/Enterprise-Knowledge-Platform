import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "dept_model.pkl"), "rb") as f:
    dept_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "type_model.pkl"), "rb") as f:
    type_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "dept_encoder.pkl"), "rb") as f:
    dept_encoder = pickle.load(f)

with open(os.path.join(BASE_DIR, "type_encoder.pkl"), "rb") as f:
    type_encoder = pickle.load(f)

with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

def classify(text: str):
    vec = vectorizer.transform([text])
    
    dept_pred = dept_model.predict(vec)
    dept = dept_encoder.classes_[dept_pred[0]]
    
    type_pred = type_model.predict(vec)
    doc_type = type_encoder.classes_[type_pred[0]]
    
    return {
        "department": dept,
        "document_type": doc_type
    }