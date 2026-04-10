"""
llm_service.py — Ollama-backed LLM with detailed, instruction-heavy prompts.

Key fix: generate_release_notes now receives the FULL DIFF (not just commits)
so the LLM can actually see what changed.
"""

import time
import requests
from codesnip.infrastructure.config_service import get_ollama_url, get_ollama_model
from codesnip.shared import logger


def _chat(prompt: str, temperature: float = 0.2) -> str:
    url = f"{get_ollama_url()}/api/chat"
    model = get_ollama_model()

    logger.info(f"Model: {model}  ·  Endpoint: {url}")
    logger.start("llm", "Waiting for Ollama response…")

    t0 = time.time()
    try:
        response = requests.post(
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": temperature},
                "stream": False,
            },
            timeout=240,
        )
    except requests.exceptions.ConnectionError:
        logger.fail("llm", "Cannot connect to Ollama")
        raise ValueError(
            f"Cannot connect to Ollama at {get_ollama_url()}. "
            "Ensure it is running:  ollama serve"
        )

    if response.status_code == 404:
        model = get_ollama_model()
        logger.fail("llm", f"Model '{model}' not found in Ollama")
        raise ValueError(f"Model '{model}' not found. Run:  ollama pull {model}")

    response.raise_for_status()
    elapsed = time.time() - t0
    logger.ok("llm", f"Ollama responded in {elapsed:.1f}s")

    content = response.json()["message"]["content"].strip()
    logger.detail("Response length", f"{len(content)} characters", "dim")
    return content


# ─────────────────────────────────────────────────────────────────────────────

class LLMService:

    def analyze_pr(
        self,
        diff: str,
        commits: str,
        meta: dict,
        check_report: str = "",
    ) -> str:
        """
        Full 8-category analysis.
        check_report: pre-computed static analysis text from code_checker.run().
        """
        static_block = ""
        if check_report:
            static_block = f"""
{check_report}

The issues listed above were detected by automated static analysis BEFORE this prompt.
You MUST reference them in the appropriate sections below.
"""

        prompt = f"""You are a senior software engineer performing a thorough code review.
Analyse every section of the diff carefully. Your response MUST cover ALL 8 categories.
If a section has nothing to report, write exactly: "• None detected"
— but only after genuinely checking the diff.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PR INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Title:   {meta.get('title', 'N/A')}
Author:  {meta.get('author', 'N/A')}
Branch:  {meta.get('head', '?')} → {meta.get('base', '?')}
URL:     {meta.get('url', '')}
State:   {meta.get('state', '')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMITS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{commits}
{static_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL DIFF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{diff[:14000]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUCTIONS: Write your review using EXACTLY the 8 headings below.
Under each heading use bullet points. Reference file names and line
content from the diff. Do not invent issues not present in the diff.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🚀 Features
What new capability, endpoint, config option, CLI flag, class, or function was added?
Even small additions count. Look for new `def`, new `class`, new config keys, new routes.

## 🐛 Bug Fixes
Were any null checks, guard clauses, try/except blocks, or incorrect conditions fixed?
Look for: added `if x is None`, added exception handling, corrected logic.

## ⚡ Performance & Profiling
Look for: loops replaced by comprehensions, blocking calls replaced with async,
caching added, redundant reads removed, repeated computations eliminated.

## 🔍 Linting & Static Analysis
Incorporate the automated check results above.
Also look for: unused variables visible in diff, missing type hints on new public
functions, bare `except:` clauses, shadowed names, undefined references.

## 🧹 Code Quality
Magic numbers that should be constants, functions > 50 lines, duplicated logic,
poor variable names (e.g. `x`, `tmp`, `data`), missing docstrings on public methods,
deeply nested `if/for` blocks (> 3 levels).

## 🎨 Formatting & Style
Lines > 88 chars, trailing whitespace, inconsistent indentation, mixed quote styles,
missing blank lines between top-level definitions, non-standard docstring format.

## 🏗️ Structural / Architecture
Business logic in wrong layer (e.g. DB call in CLI), direct API calls outside
infrastructure, tight coupling, missing abstractions, wrong file/module placement.

## 💥 Breaking Changes
Removed public functions, renamed parameters, changed return types, altered CLI
argument order, removed config keys, changed defaults that callers depend on.

## ⚠️ Risk Assessment
State: LOW / MEDIUM / HIGH
Then 2-3 sentences explaining the biggest risks or why there are none.
"""
        return _chat(prompt, temperature=0.2)

    # ─────────────────────────────────────────────────────────────────────────

    def generate_release_notes(
        self,
        diff: str,
        commits: str,
        meta: dict,
        check_report: str = "",
    ) -> str:
        """
        Generate structured release notes.
        Receives the FULL DIFF so the model can see actual changes.
        """
        static_block = ""
        if check_report:
            static_block = f"\n{check_report}\n"

        prompt = f"""You are a technical writer generating release notes for a software change.
Read the full diff carefully. Your output MUST cover ALL sections below.
If a section is empty, write "• None" — but verify first.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PR INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Title:   {meta.get('title', 'N/A')}
Author:  {meta.get('author', 'N/A')}
Branch:  {meta.get('head', '?')} → {meta.get('base', '?')}
URL:     {meta.get('url', '')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMITS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{commits}
{static_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL DIFF (read carefully — this is the source of truth)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{diff[:12000]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generate release notes using EXACTLY these headings:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🚀 Features
New functions, classes, endpoints, CLI commands, config options added.

## 🐛 Bug Fixes
Defects, crashes, incorrect logic corrected.

## ⚡ Performance & Profiling
Speed, memory, or efficiency improvements.

## 🔍 Linting & Static Analysis
Code cleanliness improvements: unused imports removed, type hints added, etc.

## 🧹 Code Quality
Refactoring, naming improvements, constant extraction, complexity reduction.

## 🎨 Formatting
Style and formatting fixes.

## 🏗️ Structural / Architecture
Module reorganisation, layer separation improvements, file moves.

## 💥 Breaking Changes
Changes that require callers or users to update their code or config.

## 📦 Deployment Notes
Dependencies changed, env vars added/removed, database migrations, restart required.
"""
        return _chat(prompt, temperature=0.3)
