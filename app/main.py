from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from layers.layer5_llm import ask
from layers.layer4_search import search
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()

app = FastAPI(
    title="Enterprise Knowledge Intelligence Platform",
    description="NovaBridge Consulting Document Search API",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return {
        "message": "Welcome to NovaBridge Knowledge Platform!",
        "endpoints": {
            "/search": "SBERT Document Search",
            "/ask": "Groq RAG Q&A",
            "/ui": "Web Interface",
            "/docs": "API Documentation"
        }
    }

@app.get("/search")
def search_docs(query: str, top_k: int = 3):
    try:
        results = search(query, top_k)
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@app.get("/ask")
def ask_question(query: str):
    try:
        result = ask(query)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Question answering failed: {str(e)}"
        )

@app.get("/ui")
def ui():
    return FileResponse("static/index.html")