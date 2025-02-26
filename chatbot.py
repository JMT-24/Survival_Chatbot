from flask import Flask, request, jsonify, render_template  
from flask_cors import CORS
import json
import difflib  

app = Flask(__name__)  
CORS(app)

# Load knowledge base from JSON
with open("knowledge.json", "r") as file:
    knowledge_base = json.load(file)

# Function to find the best match
def get_best_match(user_input):
    questions = list(knowledge_base.keys())
    closest_matches = difflib.get_close_matches(user_input, questions, n=1, cutoff=0.5)
    if closest_matches:
        return knowledge_base[closest_matches[0]]
    return "I don't know that yet. Try asking in a different way."

# Route to serve the HTML page
@app.route("/")
def home():
    return render_template("index.html")

# API Route for chatbot responses
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").lower()
    response = get_best_match(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
