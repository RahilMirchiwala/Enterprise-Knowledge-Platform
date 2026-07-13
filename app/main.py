from fastapi import FastAPI, HTTPException, UploadFile, File
import tempfile
import os
from dotenv import load_dotenv
from layers.layer5_llm import ask
from layers.layer4_search import search, find_similar, collection
from layers.layer1_extract import extract_text
from layers.layer4_search import search, find_similar, collection, model
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
    
@app.get("/detect-duplicates")
def detect_duplicates(doc_id: str, threshold: float = 0.4, top_k: int = 5):
    try:
        results = find_similar(doc_id, threshold, top_k)
        return {
            "source_document": doc_id,
            "threshold": threshold,
            "duplicates_found": len(results),
            "results": results
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Duplicate detection failed: {str(e)}")

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
    
@app.get("/documents")
def list_documents():
    try:
        results = collection.get(include=[])  # sirf IDs chahiye
        return {
            "total": len(results["ids"]),
            "documents": results["ids"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ui")
def ui():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    try:
        doc_count = collection.count()
        return {
            "status": "ok",
            "model_loaded": True,
            "documents_indexed": doc_count,
            "version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )
    
@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".docx"):
            raise HTTPException(
                status_code=400,
                detail="Only .docx files are supported"
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        text = extract_text(tmp_path)

        doc_id = file.filename.replace(".docx", "").replace(" ", "_")

        embedding = model.encode([text]).tolist()

        collection.add(
            ids=[doc_id],
            embeddings=embedding,
            documents=[text]
        )

        os.unlink(tmp_path)
        
        return {
            "message": f"Document ingested successfully",
            "doc_id": doc_id,
            "text_length": len(text)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )