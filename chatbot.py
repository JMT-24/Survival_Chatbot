from fastapi import FastAPI
import chromadb
import spacy
import json
import os

# Load ChromaDB
chroma_client = chromadb.PersistentClient(path="./chromadb_data")
collection = chroma_client.get_collection(name="knowledge_base")

# Load trained spaCy model
nlp = spacy.load("survival_model")

def log_missing_knowledge(query):
    """Logs missing questions to a JSON file."""
    file_path = "missing_knowledge.json"

    # Load existing missing knowledge if file exists
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                missing_data = json.load(f)
            except json.JSONDecodeError:
                missing_data = []
    else:
        missing_data = []

    # Avoid duplicate logging
    if query not in missing_data:
        missing_data.append(query)
        with open(file_path, "w") as f:
            json.dump(missing_data, f, indent=4)

app = FastAPI()

@app.get("/ask")
def ask_question(query: str):
    doc = nlp(query)
    predicted_label = max(doc.cats, key=doc.cats.get)  # Get highest scoring category
    confidence = doc.cats[predicted_label]  # Get confidence score
    
    # Set a confidence threshold (e.g., 0.7)
    if confidence < 0.7:
        predicted_label = "UNKNOWN"

    # Search in ChromaDB using predicted category or fallback to query
    search_query = query if predicted_label == "UNKNOWN" else predicted_label
    results = collection.query(query_texts=[search_query], n_results=1)
    
    if results["documents"] and results["documents"][0]:
        return {
            "query": query,
            "category": predicted_label,
            "confidence": confidence,
            "retrieved_answer": results["documents"][0][0]
        }

    # Log missing knowledge
    log_missing_knowledge(query)
    
    return {
        "query": query,
        "category": predicted_label,
        "confidence": confidence,
        "retrieved_answer": "I don't know."
    }