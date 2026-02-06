# Agent Prototype

A LangChain-based AI agent that uses an Ollama LLM backend to process natural language queries and invoke tools to retrieve information.

---

## Prerequisites & Environment i used while working on this

### Python Version
- **Python 3.13+** (currently using Python 3.13 in virtual env)

### Key Package Versions
```
langchain==1.2.7
langchain-core==1.2.7
langchain-community==0.4.1
langchain-ollama==1.0.1
ollama==0.6.1
python-dotenv==1.2.1
```

### System Requirements
- **Ollama**: Must be installed and running locally (LLaMA 3.1 model required)
  - Download: [ollama.ai](https://ollama.ai)
  - Run: `ollama run llama3.1`

---

##  Installation & Setup

### 1) Install dependencies
```bash
pip install --upgrade pip
pip install langchain langchain-core langchain-ollama ollama python-dotenv
```

### 2) Verify Ollama is running
```bash
# Check if Ollama service is active (macOS)
brew services list | grep ollama

# Or start manually
ollama serve
```

### 3) Ensure the LLaMA 3.1 model is available
```bash
ollama pull llama3.1
```

---

## 🔧 Configuration

### System Prompt
The agent is configured with a strict system prompt:
```
Always return ONLY valid JSON.
No prefixes.
No commentary.
```

This ensures machine-friendly, parseable outputs.

### LLM Settings
- **Model**: `llama3.1`
- **Temperature**: `0` (deterministic outputs)
- **Backend**: Ollama (local)

---

## 📚 What the Agent Does

1. **Accepts user queries** via natural language input
2. **Invokes tools** (e.g., `get_weather`) when needed
3. **Returns JSON responses** with structured data

### Available Tools
- **`get_weather(city: str)`** — Retrieves weather info for a given city
  - Returns: `{"tool_message": "...", "weather": [...]}`

---

## 🔍 Troubleshooting

### Issue: `ImportError: No module named 'langchain_ollama'`
- **Solution**: Activate venv and reinstall: `pip install langchain-ollama`

### Issue: Connection refused / Ollama not found
- **Solution**: Ensure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull llama3.1`)

### Issue: Agent returns non-JSON or malformed output
- **Solution**: Check the LLM's system prompt and add JSON validation:
  ```python
  import json
  result = get_final_ai_content(result)
  data = json.loads(result)  # Validate before use
  ```

### Issue: venv not recognized
- **Solution**: Create a new one: `python3 -m venv venv_new && source venv_new/bin/activate`

---

##  Example Usage (Multi-Turn Conversation)

```python
from main import agent, get_final_ai_content
import json

questions = ["What is the weather in sf?", "How about in nyc?"]
history = [{"role": "system", "content": "Always return ONLY valid JSON."}]

for q in questions:
    history.append({"role": "user", "content": q})
    result = agent.invoke({"messages": history})
    ai = get_final_ai_content(result)
    
    # Validate JSON
    data = json.loads(ai)
    print(json.dumps(data, indent=2))
    
    # Add to history for context
    history.append({"role": "assistant", "content": ai})
```

---

##  Work in progress to follow Best Practices 

- [ ] Add more tools (web search, file read, etc.)
- [ ] Implement persistent conversation history (database)
- [ ] Add retry logic and rate limiting
- [ ] Create unit tests for tools and parsers
- [ ] Deploy as a REST API (FastAPI/Flask)
- [ ] Add logging and monitoring
- [ ] Support streaming responses

---

Contents:
```
langchain==1.2.7
langchain-core==1.2.7
langchain-community==0.4.1
langchain-ollama==1.0.1
ollama==0.6.1
python-dotenv==1.2.1
# ... (other dependencies)
```


