# LLM-Abstain in ReleaseHub  
### A Release-Intelligent Chatbot System

LLM-Abstain is an **abstention-first LLM pipeline** for software update and release-note analysis. It combines **agent orchestration**, **Bronze–Silver–Gold data layers**, and **evidence-grounded reasoning** to ensure the system answers **only when sufficient verified data exists**, otherwise explicitly abstaining to prevent hallucinations. The system is optimized for **low latency**, **traceability**, and **research-grade reliability**, using **DuckDB** for structured evidence storage and **Gemini (via LangChain)** for controlled reasoning.

## Gemini Flash

LLM-Abstain uses **Google Gemini** models (Gemini Flash / Gemini Pro) via **LangChain’s Google GenAI integration** for controlled, evidence-grounded reasoning.

### Pre-requisites
- Python **3.10+**
- Git
- Gemini API Key:  
  https://ai.google.dev/
- LangChain Google GenAI Integration:  
  https://python.langchain.com/docs/integrations/llms/google_genai/


---

### Setup Guide
```bash
pip install langchain-google-genai google-generativeai
```

---

## 🍎 macOS Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/SE4CPS/Research-on-Software-Updates.git
cd Research-on-Software-Updates/llm-abstain
```

2️⃣ Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3️⃣ Install dependencies
```bash
pip install -r requirements.txt
pip install langchain-google-genai google-generativeai
```


4️⃣ Set environment variable

export GOOGLE_API_KEY="your_api_key_here"

5️⃣ Run the application
```bash
python app.py
```

## 🪟Windows Setup

1️⃣ Clone the repository
```bash
git clone https://github.com/SE4CPS/Research-on-Software-Updates.git
cd Research-on-Software-Updates\llm-abstain
```

2️⃣ Create and activate virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

3️⃣ Install dependencies
```bash
pip install -r requirements.txt
pip install langchain-google-genai google-generativeai
```

4️⃣ Set environment variable (file name .venv)
```bash
setx GOOGLE_API_KEY "your_api_key_here"
```

5️⃣ Run the application
```bash
python app.py
```


