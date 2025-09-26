# 📡 Research on Software Updates — LangChain Agent with Gemini + RAG

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)  
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green?logo=chainlink)](https://www.langchain.com/)  
[![Gemini](https://img.shields.io/badge/Google-Gemini-black?logo=google)](https://deepmind.google/technologies/gemini/)  
[![FAISS](https://img.shields.io/badge/FAISS-VectorDB-orange)](https://faiss.ai/) 

This project builds an **AI agent** that answers questions about **software, OS, patch, and security updates** by combining:

- ⚡ **Live API retrieval** (OS & Reddit feeds via [releasetrain.io](https://releasetrain.io))  
- 📚 **Vector store (RAG)** for contextual retrieval  
- 🤖 **LLMs (Google Gemini / HuggingFace)** for natural-language responses  
- 🧠 **Conversational memory** so the agent remembers previous questions  

---

## ✨ Features
- **Flexible Q&A**: Ask about any software company, OS, driver, or patch update.  
- **Natural time parsing**: Queries like *“updates last week”*, *“between March and May 2024”*, *“yesterday”*.  
- **Vendor filters**: Detects keywords (Windows, Linux, Nvidia, Intel, Ubuntu, etc.).  
- **RAG Integration**: If APIs fail or miss info, falls back to local FAISS vector search.  
- **Conversational context**: Follow-up questions use memory (e.g., *“what’s the latest version?”* after asking about Linux kernel).  
- **Streamlit UI**: Clean web app to chat with your agent.  

---

## 🗂 Project Structure
ReleaseNotesRec-RAG/
├── app.py                # Streamlit app entrypoint
├── basic_rag.py          # RAG pipeline + retrieval logic
├── vector_store.py       # FAISS index storage and embeddings
├── enriched_utc.py       # Time/date parsing helpers
├── SoftwareUpdateSurvey.csv  # Example survey dataset
├── requirements.txt      # Python dependencies
├── .session_memory.json  # Stores conversational memory
├── .live_cache/          # API cache for offline fallback
├── .seeds/               # Seed data for bootstrapping
└── release_notes_store/  # Embedded dataset with FAISS index

---

## 🚀 Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-org>/Research-on-Software-Updates.git
   cd Research-on-Software-Updates/ReleaseNotesRec-RAG

2. 	Install dependencies
    pip install -r requirements.txt

3.	Set environment variables
    export GOOGLE_API_KEY="your-gemini-api-key"

4.	Run the Streamlit app
    streamlit run app.py

💡 Example Queries
	•	“What Windows driver updates caused issues last month?”
	•	“Summarize the latest Linux kernel release notes.”
	•	“Any patch problems from Nvidia in July 2024?”
	•	“What’s the latest Ubuntu version?” (after asking about Ubuntu earlier — memory used)

🧩 Tech Stack
	•	LangChain
	•	FAISS for vector search
	•	SentenceTransformers for embeddings
	•	Streamlit for UI
	•	Gemini (via langchain_google_genai)

📌 Notes
	•	By default, the agent fetches live data first.
	•	If the API fails or data is missing, results come from the vector store.
	•	Conversational memory allows seamless follow-up questions.

🛠 Future Work
	•	Expand vendor vocab for more software providers.
	•	Add evaluation metrics for accuracy improvement.
	•	Support multi-API fusion (CVE feeds, vendor APIs).
