from fastapi import FastAPI
from dotenv import load_dotenv
from layers.layer5_llm import ask
from layers.layer4_search import search

load_dotenv()

app = FastAPI(
    title="Enterprise Knowledge Intelligence Platform",
    description="NovaBridge Consulting Document Search API",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "message": "Welcome to NovaBridge Knowledge Platform!",
        "endpoints": {
            "/search": "SBERT Document Search",
            "/ask": "Groq RAG Q&A",
            "/docs": "API Documentation"
        }
    }

@app.get("/search")
def search_docs(query: str, top_k: int = 3):
    results = search(query, top_k)
    return {"query": query, "results": results}

@app.get("/ask")
def ask_question(query: str):
    result = ask(query)
    return result