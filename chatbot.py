from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from vector_db import add_knowledge, retrieve_best_match
import json

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handles chat requests."""
    user_input = request.json.get("message", "").lower()
    response = retrieve_best_match(user_input)
    return jsonify({"response": response})

@app.route("/train", methods=["POST"])
def train():
    """Admins can add new knowledge."""
    data = request.json
    question, answer = data.get("question"), data.get("answer")

    if question and answer:
        add_knowledge(question, answer)
        return jsonify({"message": "Knowledge added successfully!"})
    return jsonify({"error": "Invalid input"}), 400

@app.route("/missing-questions", methods=["GET"])
def get_missing_questions():
    """Retrieve unanswered questions."""
    try:
        with open("missing_questions.json", "r") as file:
            missing_data = json.load(file)
        return jsonify({"missing_questions": missing_data})
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify({"missing_questions": []})

if __name__ == '__main__':
    app.run(debug=True)
