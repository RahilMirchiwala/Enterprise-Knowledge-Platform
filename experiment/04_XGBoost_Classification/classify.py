import os
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCS_JSON = os.path.join(BASE_DIR, "documents.json")
META_CSV = os.path.join(BASE_DIR,"documents","document_metadata.csv")

with open(DOCS_JSON,"r",encoding="utf-8") as f:
  documents = json.load(f)

metadata = pd.read_csv(META_CSV)

# print(len(documents))
# print(metadata.head(3))

texts = []
dept_labels = []
type_labels = []

for _,row in metadata.iterrows():
  doc_id = row["doc_id"]

  matching = [text for key, text in documents.items() if doc_id in key]

  if matching:
    texts.append(matching[0])
    dept_labels.append(row["department"])
    type_labels.append(row["document_type"])

# print(f"Training samples: {len(texts)}")
# print(f"Departments: {set(dept_labels)}")
# print(f"Doc types: {set(type_labels)}")

dept_encoder = LabelEncoder()
dept_encoded = dept_encoder.fit_transform(dept_labels)

type_encoder = LabelEncoder()
type_encoded = type_encoder.fit_transform(type_labels)

# print(f"Department classes: {dept_encoder.classes_}")
# print(f"Type classes: {type_encoder.classes_}")
# print(f"\nSample encoding:")
# print(f"'{dept_labels[0]}' = {dept_encoded[0]}")
# print(f"'{type_labels[0]}' = {type_encoded[0]}")

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1
)

X = vectorizer.fit_transform(texts)

# print(f"Feature matrix shape: {X.shape}")

dept_model = XGBClassifier(
  n_estimators=100,
  max_depth = 3,
  random_state = 42,
  eval_metric="mlogloss"
)

dept_model.fit(X, dept_encoded)

# dept_scores = cross_val_score(dept_model, X, dept_encoded, cv=5, scoring="accuracy")

# print(f"Department Classification:")
# print(f"Scores: {dept_scores}")
# print(f"Average Accuracy: {dept_scores.mean()*100:.2f}%")

type_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42,
    eval_metric="mlogloss"
)

type_model.fit(X, type_encoded)

type_pred = type_model.predict(X)
type_acc = accuracy_score(type_encoded, type_pred)

# print(f"Document Type Classification:")
# print(f"Training Accuracy: {type_acc*100:.2f}%")

# # Models save karo
# with open("dept_model.pkl", "wb") as f:
#     pickle.dump(dept_model, f)

# with open("type_model.pkl", "wb") as f:
#     pickle.dump(type_model, f)

# # Encoders bhi save karo
# with open("dept_encoder.pkl", "wb") as f:
#     pickle.dump(dept_encoder, f)

# with open("type_encoder.pkl", "wb") as f:
#     pickle.dump(type_encoder, f)

# # Vectorizer bhi save karo
# with open("vectorizer.pkl", "wb") as f:
#     pickle.dump(vectorizer, f)

# print("Models saved!")

def predict_document(text):
    vec = vectorizer.transform([text])
    
    dept_pred = dept_model.predict(vec)
    dept = dept_encoder.classes_[dept_pred[0]]
    
    type_pred = type_model.predict(vec)
    doc_type = type_encoder.classes_[type_pred[0]]
    
    return dept, doc_type

# Test karo
tests = [
    "This contract is entered between vendor and company",
    "Quarterly budget variance report for Q2 2026",
    "API security guidelines for authentication and authorization"
]

for text in tests:
    dept, doc_type = predict_document(text)
    print(f"Text: {text[:40]}...")
    print(f"Department: {dept} | Type: {doc_type}\n")