from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import pipeline
import re

# ---------------- Env / Keys ----------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env")

# ---------------- Flask ----------------
app = Flask(__name__, template_folder="templates")

# ---------------- Models ----------------
genai.configure(api_key=API_KEY)

# Set up the model with a generation configuration
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
}

# Using Gemini 1.5 Flash
gemini = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# HuggingFace BERT for sentiment
bert_sa = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=-1
)

conversation_history = []


def stars_from_label(label: str) -> int:
    try:
        return int(label.strip().split()[0])
    except Exception:
        return 3


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"reply": "Please type something so I can respond."})

    conversation_history.append({"role": "user", "parts": [message]})
    
    try:
        # Use the chat history for context
        chat_session = gemini.start_chat(history=conversation_history[:-1])
        response = chat_session.send_message(message)
        reply = response.text.strip() or "I'm here with you."

    except Exception as e:
        reply = f"I'm here with you. (Gemini issue: {e})"

    conversation_history.append({"role": "model", "parts": [reply]})
    return jsonify({"reply": reply})


@app.route("/analyse", methods=["POST"])
def analyse():
    if not conversation_history:
        return jsonify({"analysis": "No conversation yet. Say something first."})

    convo_text = "\n".join(f"{m['role'].capitalize()}: {m['parts'][0]}" for m in conversation_history)

    try:
        result = bert_sa(convo_text[:512])[0]
        stars = stars_from_label(result["label"])
        conf = float(result["score"])
        
        mood_text = f"{stars} star mood (confidence {conf:.2f})"
        
        # Simple severity mapping
        severity_map = {1: 10, 2: 8, 3: 5, 4: 3, 5: 1}
        severity = severity_map.get(stars, 5)

    except Exception as e:
        return jsonify({"analysis": f"Sentiment model error: {e}"})

    # Use Gemini to generate suggestion and explanation
    try:
        prompt = f"""
Analyze the following conversation and provide a mental wellness summary.

Conversation:
{convo_text}

Detected mood: {mood_text}.

1.  **Suggestion:** Provide one short, kind, practical self-care suggestion (max 2 sentences).
2.  **Why:** Explain in 1-2 sentences (empathetic, plain language) why this suggestion fits the user's feelings.
"""
        analysis_response = gemini.generate_content(prompt)
        analysis_text = analysis_response.text

        # Extract suggestion and why using regex
        suggestion_match = re.search(r"Suggestion:\s*(.*)", analysis_text, re.IGNORECASE | re.DOTALL)
        why_match = re.search(r"Why:\s*(.*)", analysis_text, re.IGNORECASE | re.DOTALL)

        suggestion = suggestion_match.group(1).strip() if suggestion_match else "Take a moment for yourself."
        why = why_match.group(1).strip() if why_match else "It seems like a gentle step could be helpful right now."

    except Exception as e:
        suggestion = "Try a brief breathing break."
        why = f"Could not generate explanation: {e}"


    pretty = (
        "SOMBOT Analysis:\n"
        f"🧠 Mood detected: {mood_text}\n"
        f"📊 Severity: {severity} / 10\n"
        f"💡 Suggestion: {suggestion}\n\n"
        f"🤔 Why this suggestion?\n{why}"
    )
    return jsonify({"analysis": pretty})


# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)
