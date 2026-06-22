from fastapi import FastAPI
from dotenv import load_dotenv
from layers.layer5_llm import ask
from layers.layer4_search import search
from layers.layer2_classify import classify
from layers.layer3_ner import extract

load_dotenv()

app = FastAPI(
    title="Enterprise Knowledge Intelligence Platform",
    description="NovaBridge Consulting Document Search API",
    version="1.0.0"
)

@app.get("/")
def home():
    return {"message": "Welcome to NovaBridge Knowledge Platform!"}

@app.get("/search")
def search_docs(query: str, top_k: int = 3):
    results = search(query, top_k)
    return {"query": query, "results": results}

@app.get("/ask")
def ask_question(query: str):
    result = ask(query)
    return result

@app.get("/classify")
def classify_doc(text: str):
    result = classify(text)
    return result

@app.get("/extract")
def extract_entities(text: str):
    result = extract(text)
    return result