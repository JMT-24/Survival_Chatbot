import chromadb
import json

# Load knowledge base from JSON
with open("data.json", "r") as f:
    data = json.load(f)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chromadb_data")
collection = chroma_client.get_or_create_collection(name="knowledge_base")

# Add data to ChromaDB
for i, item in enumerate(data):
    collection.add(
        ids=[str(i)],
        documents=[item["answer"]],
        metadatas=[{"question": item["question"]}]
    )

print("✅ Knowledge Base Loaded into ChromaDB!")
