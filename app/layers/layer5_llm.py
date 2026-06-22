import os
from groq import Groq
from dotenv import load_dotenv
from layers.layer4_search import search

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask(query: str, top_k: int = 3):
    relevant_docs = search(query, top_k)

    context = ""
    for doc in relevant_docs:
        context += f"Document: {doc['doc_id']}\n"
        context += f"Content: {doc['text']}\n\n"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant for NovaBridge Consulting. Answer questions based only on the provided documents. Be concise and accurate."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ],
        max_tokens=500
    )
    
    return {
        "query": query,
        "answer": response.choices[0].message.content,
        "sources": [doc["doc_id"] for doc in relevant_docs]
    }