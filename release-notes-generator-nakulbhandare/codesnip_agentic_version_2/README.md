# Codesnip

AI-powered GitHub PR analyzer and release notes generator, running fully local via **Ollama**.

---

## Install

```bash
pip install .
```

---

## Setup

```bash
# That's it — only needed for private repos:
codesnip config github_token ghp_xxxxxxxxxxxx
```

`ollama_url` defaults to `http://localhost:11434` and `ollama_model` defaults to `llama3`.
You never need to set them unless you're using a different host or model.

---

## Usage

```bash
# Analyze a PR
codesnip analyze <username/reponame> <PR number>
codesnip analyze torvalds/linux 1234

# Generate release notes
codesnip release-notes <username/reponame> <PR number>
codesnip release-notes torvalds/linux 1234
```

### Analysis covers:
- 🚀 Features
- 🐛 Bug Fixes
- ⚡ Performance & Profiling
- 🔍 Linting & Static Analysis
- 🧹 Code Quality
- 🎨 Formatting
- 🏗️ Structural / Architecture
- 💥 Breaking Changes
- ⚠️ Risk Assessment

---

## Optional overrides

Only needed if Ollama is on a different machine or you want a different model:

```bash
codesnip config ollama_url    http://192.168.1.10:11434
codesnip config ollama_model  mistral
```

Config is stored at `~/.codesnip/config.json`.
