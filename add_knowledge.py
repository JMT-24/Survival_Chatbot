import chromadb
import json
import os

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chromadb_data")
collection_name = "knowledge_base"

# Create or get the collection
try:
    collection = chroma_client.get_collection(name=collection_name)
except ValueError:
    collection = chroma_client.create_collection(name=collection_name)

# Load knowledge base from JSON file
knowledge_file = "knowledge_base.json"
if not os.path.exists(knowledge_file):
    print(f"{knowledge_file} not found.")
    exit()

with open(knowledge_file, "r", encoding="utf-8") as f:
    try:
        knowledge_data = json.load(f)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in knowledge_base.json")
        exit()

# Add knowledge to ChromaDB
for i, entry in enumerate(knowledge_data):
    if "text" in entry and "category" in entry:
        collection.add(
            ids=[f"kb_{i}"],  # Unique ID
            documents=[entry["text"]],
            metadatas=[{"category": entry["category"]}]
        )
        print(f"Added: {entry['text']}")
    else:
        print(f"Skipping invalid entry at index {i}: {entry}")

print("Knowledge base updated successfully!")
