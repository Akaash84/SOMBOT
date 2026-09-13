# 🧠 SOMBOT — State of Mind Bot

<p align="center">

**AI-Powered Conversational Wellness & Explainable Sentiment Analysis**

A conversational AI system designed to provide a friendly space for users to express how they feel, analyze their conversation using NLP, and generate personalized wellness-oriented suggestions.

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask)
![Gemini](https://img.shields.io/badge/Google-Gemini%201.5%20Flash-4285F4?logo=google)
![BERT](https://img.shields.io/badge/NLP-BERT-orange)
![License](https://img.shields.io/badge/Project-Academic%20%2F%20Research-green)

</p>

---

## 📌 Overview

**SOMBOT (State of Mind Bot)** is an AI + ML + NLP based conversational wellness application that allows users to talk naturally about their thoughts and feelings.

The system uses **Google Gemini** to provide a friendly, contextual conversation. Once the user finishes sharing their thoughts, they can trigger an **Analyze** operation that processes the conversation using a BERT-based sentiment analysis model.

The resulting sentiment information is then combined with Gemini to generate:

* 🧠 Detected mood
* 📊 Estimated severity
* 💡 Personalized wellness-oriented suggestion
* 🤔 Explanation of why the suggestion may be relevant

The project is designed with **explainable AI** in mind, with LIME-based interpretability planned/being incorporated to make model decisions easier to understand.

> **Important:** SOMBOT is a wellness-oriented AI project for educational and research purposes. It is not a medical diagnostic system and should not be used as a substitute for professional mental-health care.

---

# 🎯 Problem Statement

People often have difficulty expressing their emotions or identifying what may be affecting their state of mind.

Existing conversational systems can provide generic responses, but they may not provide:

* Contextual conversation
* Sentiment analysis
* Severity estimation
* Personalized suggestions
* An explanation for why a suggestion was generated

SOMBOT attempts to combine these capabilities into a single conversational system.

---

# 💡 Solution

SOMBOT follows a simple but meaningful workflow:

```text
User expresses their thoughts
            ↓
AI-powered conversation
            ↓
Conversation history collected
            ↓
User clicks "Analyze"
            ↓
BERT sentiment analysis
            ↓
Mood & severity estimation
            ↓
Gemini analyzes the conversation
            ↓
Personalized suggestion
            ↓
Explanation of the suggestion
```

The goal is to move from a simple chatbot toward a system that can **converse → analyze → explain → suggest**.

---

# ✨ Key Features

## 💬 1. AI Conversational Chatbot

SOMBOT provides a natural conversational interface powered by **Google Gemini**.

Users can freely talk about:

* Their current feelings
* Stressful situations
* Daily experiences
* Personal concerns
* Thoughts affecting their mood

The chatbot maintains conversation context during the session, allowing the interaction to feel more natural rather than treating every message as an isolated question.

---

## 🧠 2. BERT-Based Sentiment Analysis

When the user selects **Analyze**, SOMBOT processes the conversation using a Hugging Face BERT-based sentiment model:

```text
nlptown/bert-base-multilingual-uncased-sentiment
```

The model produces a **1–5 star sentiment rating** along with a confidence score.

The result is converted into a human-readable mood representation.

Example:

```text
Detected Mood:
3 star mood

Confidence:
0.82
```

---

## 📊 3. Severity Estimation

SOMBOT maps the detected sentiment into a simple severity scale.

Current mapping:

| BERT Rating | Severity |
| ----------: | -------: |
|         ⭐ 1 |  10 / 10 |
|         ⭐ 2 |   8 / 10 |
|         ⭐ 3 |   5 / 10 |
|         ⭐ 4 |   3 / 10 |
|         ⭐ 5 |   1 / 10 |

This provides an easy-to-understand representation of the detected emotional state.

> The severity score is an application-level interpretation of the sentiment model output and should not be considered a clinical measurement.

---

## 🤖 4. Gemini-Powered Personalized Suggestions

After sentiment analysis, the conversation and detected mood are passed to Gemini.

Gemini generates:

### 💡 Suggestion

A short, practical and empathetic wellness-oriented suggestion.

### 🤔 Why?

An explanation describing why the suggestion is relevant to the user's expressed feelings.

Example:

```text
💡 Suggestion:
Take a short break and spend a few minutes doing
something relaxing before returning to the situation.

🤔 Why this suggestion?
Your conversation indicates that you may be feeling
overwhelmed, so a short break may help you regain focus.
```

---

# 🔍 Explainable AI

A major design goal of SOMBOT is to make AI-assisted analysis more understandable.

Instead of:

```text
User Input
    ↓
AI Model
    ↓
Prediction
```

the intended explainable pipeline is:

```text
User Input
    ↓
NLP Analysis
    ↓
Sentiment Prediction
    ↓
Explanation
    ↓
Personalized Recommendation
```

## LIME

**LIME (Local Interpretable Model-Agnostic Explanations)** can be used to identify which portions of a user's text contribute most strongly to a local prediction.

Conceptually:

```text
User Conversation
       ↓
Sentiment Model
       ↓
Prediction
       ↓
LIME
       ↓
Important Words / Features
       ↓
Human-Readable Explanation
```

This helps reduce the "black-box" nature of machine-learning predictions.

---

# 🧪 NLP Analysis Pipeline

The complete intended analysis pipeline is:

```text
                  USER
                   │
                   ▼
        ┌─────────────────────┐
        │  Gemini Chatbot     │
        │                     │
        │ Natural Conversation│
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Conversation History│
        └──────────┬──────────┘
                   │
                   │ Analyze
                   ▼
        ┌─────────────────────┐
        │    NLP Analysis     │
        └──────────┬──────────┘
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
     ┌─────────┐       ┌─────────┐
     │  BERT   │       │  VADER  │
     │Sentiment│       │Sentiment│
     └────┬────┘       └────┬────┘
          │                 │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Mood / Severity │
          │    Analysis     │
          └────────┬────────┘
                   │
                   ▼
             ┌───────────┐
             │   LIME    │
             │Explainable│
             │    AI     │
             └─────┬─────┘
                   │
                   ▼
          ┌─────────────────┐
          │     Gemini      │
          │ Recommendation  │
          └────────┬────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
    Personalized       Explanation
     Suggestion         "Why?"
```

> **Implementation note:** The current public `app.py` implements Gemini + BERT + severity mapping + Gemini suggestions. VADER/LIME are part of the project's explainable-AI architecture and can be integrated as the analysis layer evolves.

---

# 🏗️ System Architecture

```text
┌───────────────────────────────────────────────────────────┐
│                       SOMBOT SYSTEM                       │
└───────────────────────────────────────────────────────────┘

                         USER
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│                    WEB INTERFACE                          │
│                                                           │
│              HTML / CSS / JavaScript                      │
│                                                           │
│        ┌──────────────┐     ┌──────────────┐             │
│        │ Chat Screen  │     │ Analyze      │             │
│        │              │     │ Button       │             │
│        └──────┬───────┘     └──────┬───────┘             │
└───────────────┼─────────────────────┼─────────────────────┘
                │                     │
                ▼                     ▼
┌───────────────────────────────────────────────────────────┐
│                     FLASK BACKEND                         │
│                                                           │
│  /chat                              /analyse              │
│    │                                   │                  │
│    ▼                                   ▼                  │
│ Conversation History             NLP Analysis             │
└────────────┬──────────────────────────────┬───────────────┘
             │                              │
             ▼                              ▼
┌───────────────────────┐       ┌───────────────────────────┐
│    GOOGLE GEMINI      │       │       NLP LAYER           │
│                       │       │                           │
│ Gemini 1.5 Flash      │       │ BERT / VADER              │
│                       │       │                           │
│ • Conversation        │       │ • Sentiment               │
│ • Suggestions         │       │ • Mood                    │
│ • Explanation         │       │ • Severity                │
└───────────┬───────────┘       └────────────┬──────────────┘
            │                                │
            │                                ▼
            │                     ┌────────────────────────┐
            │                     │   EXPLAINABLE AI       │
            │                     │                        │
            │                     │         LIME           │
            │                     └────────────┬───────────┘
            │                                  │
            └────────────────┬─────────────────┘
                             ▼
                ┌──────────────────────────┐
                │     FINAL ANALYSIS       │
                │                          │
                │ 🧠 Mood                  │
                │ 📊 Severity              │
                │ 💡 Suggestion            │
                │ 🤔 Explanation           │
                └──────────────────────────┘
```

---

# 🔄 End-to-End Application Flow

### Step 1 — User starts a conversation

The user opens SOMBOT and starts talking about how they feel.

### Step 2 — Gemini responds

Gemini generates a contextual and conversational response.

### Step 3 — Conversation is stored

The application maintains the conversation history during the current session.

### Step 4 — User selects Analyze

The user clicks the **Analyze** button when they want an assessment of the conversation.

### Step 5 — BERT analyzes the conversation

The conversation is passed through the sentiment-analysis model.

### Step 6 — Mood is determined

The BERT output is converted into a star-based mood representation.

### Step 7 — Severity is estimated

The star rating is mapped to an application-level severity score.

### Step 8 — Gemini generates a recommendation

Gemini receives the conversation and detected mood and generates a practical suggestion.

### Step 9 — Explanation is generated

Gemini also explains why the suggestion fits the conversation.

### Step 10 — Results are displayed

The user receives a consolidated result:

```text
SOMBOT Analysis

🧠 Mood detected
📊 Severity
💡 Suggestion

🤔 Why this suggestion?
```

---

# 🛠️ Technology Stack

| Technology                    | Purpose                                         |
| ----------------------------- | ----------------------------------------------- |
| **Python**                    | Core programming language                       |
| **Flask**                     | Web application backend                         |
| **Google Gemini**             | Conversational AI and recommendation generation |
| **Hugging Face Transformers** | NLP model integration                           |
| **BERT**                      | Sentiment analysis                              |
| **VADER**                     | Additional sentiment analysis layer             |
| **LIME**                      | Explainable AI                                  |
| **PyTorch**                   | Machine-learning model execution                |
| **HTML / CSS / JavaScript**   | Frontend interface                              |
| **python-dotenv**             | Environment variable management                 |
| **Git / GitHub**              | Version control                                 |

---

# 📁 Project Structure

```text
SOMBOT/
│
├── templates/
│   ├── entry.html
│   └── index.html
│
├── .gitignore
├── app.py
└── README.md
```

### `app.py`

Main Flask application containing:

* Gemini configuration
* BERT sentiment model
* Conversation handling
* Chat endpoint
* Analysis endpoint
* Severity calculation
* Recommendation generation

### `templates/entry.html`

Entry page for the application.

### `templates/index.html`

Main chatbot interface.

---

# ⚙️ Installation

## Prerequisites

Make sure you have:

* Python 3.10+
* pip
* Google Gemini API key
* Git

---

## 1. Clone the Repository

```bash
git clone https://github.com/Akaash84/SOMBOT.git
cd SOMBOT
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```powershell
pip install flask python-dotenv google-generativeai transformers torch vaderSentiment lime scikit-learn
```

---

# 🔑 Environment Configuration

Create a `.env` file in the project root:

```text
SOMBOT/
│
├── .env
├── app.py
└── templates/
```

Add:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

The application reads the API key using:

```python
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
```

**Never commit your `.env` file or API key to GitHub.**

---

# ▶️ Running the Application

Start the Flask server:

```powershell
python app.py
```

The application runs locally at:

```text
http://127.0.0.1:5000
```

Open the URL in a browser.

---

# 🧪 Example Interaction

### User

```text
I've been feeling really stressed about my exams.
I don't feel prepared and I keep worrying about failing.
```

### SOMBOT

```text
I'm here with you. It sounds like the upcoming exams
are creating a lot of pressure for you.
```

The user continues the conversation.

After clicking **Analyze**:

```text
SOMBOT Analysis

🧠 Mood detected:
3 star mood

📊 Severity:
5 / 10

💡 Suggestion:
Take a short break and divide your preparation
into smaller achievable tasks.

🤔 Why this suggestion?

Your conversation suggests that exam pressure
may be making the situation feel overwhelming.
Breaking the workload into smaller steps may make
it feel more manageable.
```

---

# 🔬 Machine Learning Components

## BERT

SOMBOT uses a pretrained BERT-based sentiment model:

```text
nlptown/bert-base-multilingual-uncased-sentiment
```

The model is used to classify the emotional sentiment of the conversation.

---

## VADER

VADER can provide an additional rule-based sentiment signal.

This creates an opportunity to compare:

```text
BERT-based sentiment
        +
Lexicon-based sentiment
        ↓
More informative sentiment analysis
```

---

## LIME

LIME is used for local model interpretability.

Its purpose is to help answer:

> **"Which parts of the user's text influenced the model's prediction?"**

This is particularly valuable when developing AI systems where transparency and interpretability are important.

---

# 🧠 Why Explainability Matters

Mental-wellness applications require particular care because users may interpret AI output as authoritative.

An explainable approach can help users understand that the output is based on patterns in their provided text rather than an unexplained or absolute judgment.

SOMBOT therefore aims to move toward:

```text
Prediction
    +
Explanation
    +
Context
    =
More Transparent AI
```

---

# 🔐 Security & Privacy

SOMBOT processes potentially sensitive user conversations.

Recommended practices include:

* Keep API keys in environment variables
* Never commit `.env`
* Avoid logging private conversations
* Avoid storing personal conversations unnecessarily
* Use HTTPS when deploying
* Apply appropriate access controls for production deployments

For a production version, additional privacy and security controls should be implemented before handling real sensitive user data.

---

# ⚠️ Responsible AI Disclaimer

SOMBOT is an **academic/research-oriented AI wellness project**.

It does **not**:

* Diagnose mental-health conditions
* Replace psychologists, psychiatrists, or other healthcare professionals
* Provide medical treatment
* Guarantee the accuracy of sentiment predictions
* Provide emergency intervention

The sentiment and severity scores are computational estimates and should not be interpreted as clinical assessments.

If someone is experiencing an immediate mental-health crisis or is in danger, they should seek help from a qualified professional or appropriate local emergency service.

---

# 🚀 Future Enhancements

The project can be extended with:

### 🧠 Advanced NLP

* Combined BERT + VADER sentiment scoring
* Emotion classification
* Context-aware emotion detection
* Multilingual emotion analysis

### 🔍 Explainable AI

* LIME-based word-level explanations
* SHAP-based explanations
* Visual sentiment contribution charts
* Explainable recommendation generation

### 🎙️ Multimodal Interaction

* Speech-to-Text
* Text-to-Speech
* Voice-based conversations
* Emotion-aware speech analysis

### 📊 Analytics

* Mood history
* Sentiment trends
* Personal progress dashboard
* Conversation analytics
* Long-term emotional trend visualization

### 🔐 Privacy

* Secure user authentication
* Encrypted data storage
* Privacy-focused conversation management
* Data deletion controls

### 📱 Deployment

* Responsive mobile interface
* Progressive Web App
* Cloud deployment
* Scalable backend architecture

---

# 📈 Project Roadmap

```text
                         SOMBOT
                           │
              ┌────────────┴────────────┐
              │                         │
        Current System             Future System
              │                         │
              ▼                         ▼
       Gemini Chatbot             Voice Interaction
       BERT Analysis              Multimodal AI
       Severity Score             Advanced Emotion AI
       Gemini Suggestions         LIME / SHAP
       Web Interface              Mood Dashboard
                                  Secure Accounts
                                  Personalized Trends
```

---

# 👨‍💻 Author

## Akaash Manda

**AI / ML • NLP • Full-Stack Development**

GitHub:
https://github.com/Akaash84

Project:
https://github.com/Akaash84/SOMBOT

---

# ⭐ Acknowledgements

This project makes use of open-source technologies and AI/NLP frameworks including:

* Google Gemini
* Hugging Face Transformers
* BERT
* VADER
* LIME
* Flask
* PyTorch

---

# 📌 Project Status

🚧 **Active Development**

SOMBOT is an evolving academic/research project focused on combining conversational AI, NLP-based sentiment analysis, personalized recommendations, and explainable AI.

---

<p align="center">

### 🧠 Talk. Analyze. Understand. Improve.

**SOMBOT — State of Mind Bot**

</p>
