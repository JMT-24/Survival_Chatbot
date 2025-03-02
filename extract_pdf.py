import fitz  # PyMuPDF for extracting text from PDFs
import json
import chromadb
import os

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chromadb_data")
collection = chroma_client.get_or_create_collection(name="knowledge_base")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

def store_knowledge_from_pdf(pdf_path, category):
    """Processes and stores knowledge from a PDF file."""
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Split text into chunks (ChromaDB works better with smaller text chunks)
    chunks = extracted_text.split("\n\n")  # Split by double newlines (paragraphs)
    
    # Load existing knowledge base JSON
    json_file = "knowledge_base.json"
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            knowledge_base = json.load(f)
    else:
        knowledge_base = []
    
    new_entries = []
    
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) > 10:  # Ignore very short or empty lines
            entry = {"text": chunk.strip(), "label": category}
            knowledge_base.append(entry)
            new_entries.append(entry)
            
    # Save updated knowledge to JSON
    with open(json_file, "w") as f:
        json.dump(knowledge_base, f, indent=4)
    
    # Add new data to ChromaDB
    for i, entry in enumerate(new_entries):
        collection.add(documents=[entry["text"]], metadatas=[{"label": entry["label"]}], ids=[f"pdf_{category}_{i}"])
    
    print(f"Added {len(new_entries)} new knowledge entries to ChromaDB and JSON.")

# Example usage:
pdf_path = "./documents/13_ways_toMakeFire.pdf"  # Change this to your PDF file name
category = "FIRE_STARTING"  # Define a category
store_knowledge_from_pdf(pdf_path, category)
