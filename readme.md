# ReleaseNotesRec-RAG

## Overview

This project is a **Release-Notes Chatbot** built with a RAG (Retrieval-Augmented Generation) architecture in Streamlit. It ingests:

- **OS release notes** via API  
- **Reddit discussions** via API  
- **User feedback** from a CSV file  

Data is embedded with `sentence-transformers`, indexed using FAISS, and queried via an Azure OpenAI deployment.

---

## Features

- **Data ingestion** from:
  - CSV (`SoftwareUpdateSurvey.csv`)
  - OS API: `https://releasetrain.io/api/component?q=os`
  - Reddit API: `https://releasetrain.io/api/reddit`
- **Embeddings** using `all-mpnet-base-v2`
- **Persistent FAISS** vector store
  - “🔄 Refresh data” button re-indexes and resets cache
- **Streamlit UI**:
  - Top-K slider to control retrieval depth
  - Scrollable chat history
  - Real-time assistant responses
- **Azure OpenAI** integration:
  - Uses new `AzureOpenAI` SDK if available
  - Falls back to legacy `openai` client for compatibility

---

## Requirements

- **Python 3.11+**

Create a file named `.env` in the project root with:

```bash
AZURE_OPENAI_ENDPOINT=<your-endpoint>
AZURE_OPENAI_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT=<deployment-name>  # e.g. gpt-4o
AZURE_OPENAI_API_VERSION=2025-01-01-preview
```

❗**Important:** Do NOT commit your `.env` file to version control.  
Make sure `.env` is listed in your `.gitignore`.

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Usage

1. **Clone** the repo  
   ```bash
   git clone https://github.com/SE4CPS/ReleaseNotesRec-RAG.git
   cd ReleaseNotesRec-RAG
   ```
2. **Add** your Azure credentials to a `.env` file  
3. **Install** required packages  
   ```bash
   pip install -r requirements.txt
   ```
4. **Launch** the Streamlit app  
   ```bash
   streamlit run app.py
   ```

---

## Configuration

- **CSV path**: `SoftwareUpdateSurvey.csv`
- **OS API**: `https://releasetrain.io/api/component?q=os`
- **Reddit API**: `https://releasetrain.io/api/reddit`
- **Embedding model**: change `EMB_MODEL` in `app.py` if needed

---

## System Prompt

We use a chain-of-thought style prompt (not shown to the user) to guide reasoning:

1. Understand the user query  
2. Retrieve relevant documents  
3. Form reasoning internally  
4. Provide a clear answer  
5. Say “I do not know.” if unsure

---

## Example Query

```text
User: What are the known security vulnerabilities in the latest Nvidia drivers?
```

**Assistant** will summarize the most recent CVEs and provide update recommendations based on the retrieved context.

---

## Project Structure

```
.
├── app.py                      # Streamlit app with Azure RAG integration
├── SoftwareUpdateSurvey.csv   # User feedback CSV
├── requirements.txt           # Dependencies
├── .env                       # Azure credentials (NOT committed)
├── release_notes_store/       # Cached vector store (auto-generated)
├── README.md
```

---

## Contributing

Feel free to open issues or submit pull requests to improve this project.

---

## License

This project is licensed under MIT. See [LICENSE](LICENSE) for details.
