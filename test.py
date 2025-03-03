import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model with word vectors
nlp = spacy.load("en_core_web_md")

# Example survival knowledge base (expandable)
knowledge_base = {
    "purify_water": "You can purify water by boiling it for at least one minute.",
    "start_fire": "Use dry wood and a fire starter or friction to create fire.",
    "find_food": "You can forage for edible plants, hunt small animals, or fish.",
    "build_shelter": "You can build a shelter using branches, leaves, and any available materials."
}

# Function to preprocess text (remove stopwords, lowercase, etc.)
def preprocess(text):
    doc = nlp(text.lower())  # Lowercase text
    tokens = [token.lemma_ for token in doc if not token.is_stop]  # Remove stopwords, use lemmas
    return " ".join(tokens)

# Convert knowledge base into vector embeddings after preprocessing
knowledge_vectors = {key: nlp(preprocess(text)).vector for key, text in knowledge_base.items()}

def get_best_answer(user_query):
    query_vector = nlp(preprocess(user_query)).vector  # Preprocess and vectorize query

    best_match = None
    best_score = -1

    print(f"User query: {user_query}")
    for key, vector in knowledge_vectors.items():
        score = cosine_similarity([query_vector], [vector])[0][0]  # Compute similarity
        print(f"Comparing with '{key}': Similarity Score = {score:.4f}")  # Debug info

        if score > best_score:
            best_score = score
            best_match = key

    print(f"\nBest Match: {best_match} (Confidence: {best_score:.4f})")  # Debug info

    if best_score > 0.6:  
        return f"AI: {knowledge_base[best_match]} (Confidence: {best_score:.4f})"
    else:
        return f"AI: I don't know the answer to that yet. (Confidence: {best_score:.4f})"

# Test the chatbot
user_input = "How do I stay warm?"
response = get_best_answer(user_input)
print(response)
