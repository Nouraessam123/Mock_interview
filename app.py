from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
openai.api_key =os.environ.get("OPENAI_API_KEY")

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

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=1000,
        )
        content = response.choices[0].message.content.strip()
        lines = content.split("\n")
        questions = [line.strip() for line in lines if line.strip() and "?" in line]
        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate_answer", methods=["POST"])
def evaluate_answer():
    data = request.get_json()
    question = data.get("question")
    answer = data.get("answer")

    sentiment = analyzer.polarity_scores(answer)

    feedback_prompt = (
        f"Evaluate how well the following answer responds to the interview question "
        f"in terms of relevance, completeness, and clarity.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Provide detailed feedback, then add a score out of 10 using this format:\n"
        f"Rating: X/10"
    )

    try:
        feedback_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": feedback_prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        feedback_text = feedback_response.choices[0].message.content.strip()

        # Extract rating
        rating = None
        if "Rating:" in feedback_text:
            rating_line = [line for line in feedback_text.split('\n') if "Rating:" in line]
            if rating_line:
                match = re.search(r'\d+', rating_line[0])
                if match:
                    extracted = int(match.group())
                    if 0 <= extracted <= 10:
                        rating = extracted

        return jsonify({
            "feedback": feedback_text,
            "rating": rating,
            "sentiment": sentiment
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "API is running. Use /generate_questions or /evaluate_answer"


if __name__ == "__main__":
    app.run(debug=True, port=5000)

