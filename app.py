from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

@app.route("/generate_questions", methods=["POST"])
def generate_questions():
    data = request.get_json()
    job_role = data.get("job_role", "Data Scientist")

    prompt = (
        f"Generate 3 behavioral and 2 technical interview questions for a {job_role} role. "
        "Please list only the questions, numbered."
    )

    # إرسال الطلب إلى API البعيد (Vercel)
    response = requests.post(
        "https://mock-interview-omega-three.vercel.app/generate_questions", 
        json={"job_role": job_role}
    )

    if response.status_code == 200:
        return jsonify(response.json())  # إرجاع البيانات المستلمة من الـ API البعيد
    else:
        return jsonify({"error": "Failed to get questions from external API"}), 500


@app.route("/evaluate_answer", methods=["POST"])
def evaluate_answer():
    data = request.get_json()
    question = data.get("question")
    answer = data.get("answer")

    sentiment = analyzer.polarity_scores(answer)

    # إرسال الطلب إلى API البعيد (Vercel)
    response = requests.post(
        "https://mock-interview-omega-three.vercel.app/evaluate_answer",
        json={"question": question, "answer": answer}
    )

    if response.status_code == 200:
        return jsonify(response.json())  # إرجاع البيانات المستلمة من الـ API البعيد
    else:
        return jsonify({"error": "Failed to evaluate answer from external API"}), 500

@app.route("/")
def home():
    return "API is running. Use /generate_questions or /evaluate_answer"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
