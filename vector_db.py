import chromadb
import json
from sentence_transformers import SentenceTransformer

# Load embedding model for text similarity
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB (persistent storage)
chroma_client = chromadb.PersistentClient(path="chroma_db")  
collection = chroma_client.get_or_create_collection(name="chat_knowledge")

MISSING_KNOWLEDGE_FILE = "missing_questions.json"

def add_knowledge(question, answer):
    """Store Q&A in the database with embeddings."""
    embedding = embedder.encode(question).tolist()
    collection.add(ids=[question], embeddings=[embedding], metadatas=[{"answer": answer}])

def log_missing_question(question):
    """Log unanswered questions for later review."""
    try:
        with open(MISSING_KNOWLEDGE_FILE, "r") as file:
            missing_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        missing_data = []

    missing_data.append(question)

    with open(MISSING_KNOWLEDGE_FILE, "w") as file:
        json.dump(missing_data, file, indent=4)

def retrieve_best_match(user_input):
    """Find the most relevant stored response. If not found, log it."""
    query_embedding = embedder.encode(user_input).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=1)

    if results["metadatas"] and results["metadatas"][0]:
        return results["metadatas"][0][0]["answer"]

    log_missing_question(user_input)
    return "I don't have an answer for that yet. I'll pass this to the admin for review."
