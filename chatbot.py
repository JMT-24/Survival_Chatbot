from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import chromadb
import json
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Initialize FastAPI
app = FastAPI()

# Load ChromaDB
chroma_client = chromadb.PersistentClient(path="./chromadb_data")
collection = chroma_client.get_collection(name="knowledge_base")

# Load fine-tuned transformer model
model_path = "./trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Set up HTML rendering
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Path for missing knowledge storage
missing_knowledge_path = Path("missing-knowledge.json")

# Ensure the missing knowledge file exists
if not missing_knowledge_path.exists():
    missing_knowledge_path.write_text(json.dumps([]))

def log_missing_knowledge(question):
    """Store unanswered questions in missing-knowledge.json"""
    with missing_knowledge_path.open("r+") as f:
        data = json.load(f)
        if question not in data:
            data.append(question)
            f.seek(0)
            json.dump(data, f, indent=4)

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def ask_question(request: Request):
    user_input = await request.json()
    query = user_input.get("message", "")

    # Search ChromaDB for the best match with similarity scores
    results = collection.query(query_texts=[query], n_results=1, include=["documents", "distances"])

    if results["documents"] and results["documents"][0]:
        best_answer = results["documents"][0][0]  # Most relevant answer
        similarity_score = results["distances"][0][0]  # Get similarity score
        
        threshold = 1.5  # Set a strict similarity threshold (lower is more strict)
        print(f"Query: {query}")
        print(f"Best match: {best_answer} (Score: {similarity_score})")

        if similarity_score > threshold:
            best_answer = "I don't know the answer to that yet."
            log_missing_knowledge(query)
        elif similarity_score < 0.5:  # Extra check for very strong matches
            best_answer = results["documents"][0][0]
        else:
            best_answer = f"Maybe this helps: {results['documents'][0][0]}"


    else:
        best_answer = "I don't know the answer to that yet."
        log_missing_knowledge(query)  # Log unknown queries

    return {"response": best_answer}
