# ReleaseTrain.io — Software Ecosystem Framework

Architecture overview, data flow, frontend modules, and a complete API reference for **releasetrain.io**.  
The platform ingests ecosystem signals (GitHub, CVE/NVD, Reddit), enriches and classifies updates, and exposes them via a clean API and triage UI.

---

## Quick Links

- 🌐 **Live:** <https://releasetrain.io/>
- 💻 **Client:** <https://github.com/SE4CPS/releasetrain-client>
- 📋 **Project Board:** <https://github.com/orgs/SE4CPS/projects/24>

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Research-on-Software-Updates.git
   cd Research-on-Software-Updates
   ```

2. **Set up the Release Notes Generator (codesnip):**
   ```bash
   cd research-release-notes-generator
   pip install -e .
   ```

3. **Set up the Sentiment Analysis module:**
   ```bash
   cd ../research-open-text-sentimental
   pip install -r requirements.txt
   ```

### Individual Component Setup

#### Release Notes Generator (codesnip)
```bash
cd research-release-notes-generator
pip install -e .

# Test installation
codesnip --help
```

#### Sentiment Analysis Research
```bash
cd research-open-text-sentimental
pip install -r requirements.txt

# Run sentiment analysis scripts
python scripts/reddit_fetch.py
python scripts/complete_sentiment_trajectory_visualization.py
```

#### Dashboard Quality Research
```bash
cd research-dashboard-quality
# See specific README in the directory for setup instructions
```

#### RAG for Software Update Data
```bash
cd research-rag-for-software-update-data/gemeni
# See specific README in the directory for setup instructions
```

### Environment Variables

For the Release Notes Generator, you may need to set up API keys:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export GITHUB_TOKEN="your-github-token"
```

### Verification

To verify your installation:
```bash
# Test codesnip installation
codesnip --version

# Test Python environment for sentiment analysis
python -c "import pandas, matplotlib, seaborn; print('All packages installed successfully')"
```

---

## System Architecture

```mermaid
graph LR
  A["a) Bot1 (GitHub) [online]"] --> D["d) MongoDB Data Lake [atlas]"]
  B["b) Bot2 (CVE/NVD) [online]"] --> D
  C["c) Bot3 (Reddit) [online]"] --> D

  Note1["Improve data quality [ongoing]"] --> D

  subgraph E_SG["e) AI / ML"]
    direction LR
    Type["Type [ok]"]
    Security["Security [ok]"]
    Breaking["Breaking [ok]"]
    Sentimental["Sentiment [todo]"]
    RAG["LLM / RAG [pilot]"]
    Type --> E["Classified Output"]
    Security --> E
    Breaking --> E
    Sentimental --> E
    RAG --> E
  end

  subgraph Azure["g) Azure Cloud"]
    AZ["gpt-4o"] <--> RAG
  end

  E <--> Agent["🤖 Agentic AI (LLM+API+DB+Rules)"]
  Agent <--> D
  Agent <--> F["f) Backend API (Express)"]

  F <--> FEEntry["Frontend Router"]

  subgraph H["h) Frontend"]
    direction LR
    FE1["1) Ecosystem Graph Architecture [ok]"]
    FE2["2) Ecosystem UML Architecture [ok]"]
    FE3["3) Ecosystem Software Updates [ok]"]
    FE4["4) Triage Buttons [ok]"]
    FE5["5) Release Notes Generator [todo]"]
    FEEntry --> FE1 & FE2 & FE3 & FE4 & FE5
  end
