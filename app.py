from fastapi import FastAPI
import chromadb
import spacy
import json
import os

# Load ChromaDB
chroma_client = chromadb.PersistentClient(path="./chromadb_data")
collection = chroma_client.get_collection(name="knowledge_base")

# Load spaCy
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    """Extract important words from user input."""
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"]]

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
    keywords = extract_keywords(query)
    search_query = " ".join(keywords)

    # Search using extracted keywords
    results = collection.query(query_texts=[search_query], n_results=1)

    if results["documents"] and results["documents"][0]:  
        best_answer = results["documents"][0][0]

        # Check if the answer contains at least one main keyword
        if any(keyword.lower() in best_answer.lower() for keyword in keywords):
            return {
                "query": query,
                "keywords_used": keywords,
                "retrieved_answer": best_answer
            }

    # If no good answer is found, log it as missing knowledge
    log_missing_knowledge(query)
    
    return {
        "query": query,
        "keywords_used": keywords,
        "retrieved_answer": "I don't know."
    }
