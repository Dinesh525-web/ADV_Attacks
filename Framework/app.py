from flask import Flask, request, jsonify
import sys
import os

# Add the root directory of your project to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.chatbot.chatbot import get_answer

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    """
    Endpoint to handle questions sent via POST request.
    Returns chatbot's answer and confidence score.
    """
    try:
        data = request.get_json()
        question = data.get('question') if data else None

        if not question:
            return jsonify({"error": "No question provided"}), 400

        # Get answer from the chatbot model
        answer, score = get_answer(question)

        return jsonify({
            "answer": answer,
            "score": score
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
