from flask import Flask, request, jsonify
import difflib

app = Flask(__name__)

knowledge_base = {
    "how to purify water": "You can boil water for at least 1 minute, use water purification tablets, or filter it using a cloth and activated charcoal.",
    "how to start a fire": "Use dry leaves and small sticks. If you have no lighter, try the friction method with wood or use flint and steel.",
    "how to find food in the wild": "Look for edible plants, catch fish, or trap small animals. Avoid unknown berries unless you are sure they are safe."
}

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message", "").lower()
    response = knowledge_base.get(user_input, "Dunno yet")
    return jsonify({"response": response})

if __name__=='__main__': 
    app.run(debug=True)