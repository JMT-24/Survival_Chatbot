import spacy

nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    """Extracts important words from a user query."""
    doc = nlp(text)
    
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"]]
    named_entities = [ent.text for ent in doc.ents]
    
    return {
        "keywords": keywords,
        "named_entities": named_entities
    }

# Example queries
queries = [
    "How do I purify water in an emergency?",
    "What should I do during a power outage?",
    "How can I find food in the wild?"
]

for query in queries:
    result = extract_keywords(query)
    print(f"Query: {query}")
    print("Extracted Keywords:", result["keywords"])
    print("Named Entities:", result["named_entities"])
    print("-" * 40)