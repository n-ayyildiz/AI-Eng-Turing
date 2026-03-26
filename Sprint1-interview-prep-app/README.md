## 🌐 Live Demo

👉 [Click here to try the app](https://ai-eng-turing-ba2zwymwtxecturos9vona.streamlit.app/)

---

# 🎯 AI Interview Coach

An AI-powered interview preparation app built with **Streamlit** and **OpenAI**. Practice real interview scenarios, get structured feedback on your answers, and track your progress — all in one place.

---

## ✨ Features

### Interview Practice
- **One question at a time** — the AI asks questions tailored to your role, type, and difficulty
- **Structured feedback** — every answer is scored out of 10 with Strengths, Weaknesses, a Tip, and a model answer to learn from
- **Improve your answer** — if you score below 10, you can try again on the same question
- **Session summary** — average score, best answer, and a colour-coded score breakdown at the end

### Personalised to Your Role
- **Paste a job description** — the AI extracts required skills and tailors questions to the role
- **Upload your CV** — PDF or Word (.docx); the AI identifies your skill gaps against the job
- **Gap-focused questions** — when both are provided, the interview targets your weakest areas

### Five Prompt Styles
Switch between interviewing styles from the sidebar to experience different feedback approaches:
- **Zero-Shot** — straightforward, instruction-based coaching
- **Few-Shot** — anchored on examples for consistent formatting
- **Chain-of-Thought** — step-by-step reasoning for deeper feedback
- **Persona-Based** — choose a Friendly, Neutral, or Strict interviewer
- **Structured Output** — template-driven format, ideal for job-specific sessions

### Tunable Settings
Adjust model behaviour from the sidebar:
- Choose from GPT-4.1, GPT-4.1 mini, GPT-4.1 nano, GPT-4o, GPT-4o mini
- Temperature, Top-p, Max Tokens, Frequency Penalty
- Number of questions per session (1–10)

### Session Export
- Download a full **HTML report** at the end of each session
- Includes all questions, your answers (including improved attempts), AI feedback, and scores
- Open in any browser — or print to PDF

### Cost Tracking
- Live token cost per API call
- Session total displayed in the sidebar

---

## 🚀 Getting Started

### 1. Set up the project
```bash
cd interview-prep-app
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your OpenAI API key
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-key-here
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
interview-prep-app/
├── app.py              # Main Streamlit application
├── prompts.py          # 5 system prompts + prompt builder
├── security.py         # Input validation and security guards
├── pricing.py          # Token cost calculator
├── cv_parser.py        # CV and job description analysis
├── exporter.py         # HTML session report generator
├── requirements.txt    # Python dependencies
├── .env                # API key (never commit this)
└── .gitignore          # Excludes .env and venv from Git
```

---

## ⚠️ Known Limitations

- **Score variability** — the same answer may receive slightly different scores across sessions due to model non-determinism
- **Format sensitivity** — the app relies on the model following structured output markers; occasional deviations can affect feedback parsing
- **Security filters** — input validation catches injection attempts and meaningless content (spam, repeated characters) but does not restrict domain-specific vocabulary, so all legitimate interview answers pass regardless of role or subject area
- **No persistent storage** — session data resets on page refresh; there is no cross-session memory

## 💡 Potential Improvements

- Persistent session storage across page refreshes
- JSON-based structured output for more reliable score parsing
- LLM-as-a-judge for objective, reproducible scoring
- Audio input for verbal answer practice
- Multilingual support
- Cloud deployment (Streamlit Cloud, AWS, or Azure)

---

## 📦 Dependencies

```
streamlit>=1.35.0
openai>=1.30.0
python-dotenv>=1.0.0
pymupdf>=1.24.0
python-docx>=1.1.0
```
