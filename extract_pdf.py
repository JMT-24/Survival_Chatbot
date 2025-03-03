import fitz  # PyMuPDF
import spacy
import numpy as np

# Load spaCy's pre-trained model (medium-sized with vectors)
nlp = spacy.load("en_core_web_md")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text.strip()

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=200):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) > chunk_size:
            chunks.append(chunk.strip())
            chunk = ""
        chunk += sentence + " "
    
    if chunk:  # Add the last chunk
        chunks.append(chunk.strip())
    
    return chunks

# Function to convert chunks into vector embeddings
def convert_chunks_to_vectors(chunks):
    vectors = {chunk: nlp(chunk).vector for chunk in chunks}
    return vectors

# Main execution
pdf_path = "./documents/13_ways_toMakeFire.pdf"  # Update this with your file path

# Extract, Transform, and Load
text = extract_text_from_pdf(pdf_path)
chunks = split_text_into_chunks(text)
vector_storage = convert_chunks_to_vectors(chunks)

# Display stored data
for i, (chunk, vector) in enumerate(vector_storage.items()):
    print(f"Chunk {i+1}: {chunk[:100]}...")  # Show first 100 chars
    print(f"Vector: {vector[:5]}...")  # Show first 5 elements
    print("-" * 50)
