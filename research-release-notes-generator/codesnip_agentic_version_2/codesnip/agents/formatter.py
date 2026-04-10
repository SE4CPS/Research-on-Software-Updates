"""
formatter.py
============
OllamaFormatter — final Ollama call that receives ALL 8 agent results
and produces polished, cohesive release notes.

This is the ONLY place that sees the combined output of every agent.
Its job is purely formatting, not analysis — it trusts the findings.

Flow:
  [8 × AgentResult]  →  build_aggregated_prompt()
                      →  single Ollama call
                      →  clean release notes markdown
"""
from __future__ import annotations

import time
import requests
from typing import Sequence

from codesnip.agents import AgentResult
from codesnip.infrastructure.config_service import get_ollama_url, get_ollama_model
from codesnip.shared import logger


def _severity_badge(findings) -> str:
    """Return overall risk level from finding severities."""
    if not findings:
        return "LOW"
    sevs = {f.severity for f in findings}
    if "critical" in sevs:
        return "CRITICAL"
    if "high" in sevs:
        return "HIGH"
    if "medium" in sevs:
        return "MEDIUM"
    return "LOW"


class OllamaFormatter:
    """
    Takes all agent results and makes one final Ollama call to produce
    polished, human-readable release notes from the raw findings.
    """

    def format_release_notes(
        self,
        results: Sequence[AgentResult],
        meta: dict,
    ) -> str:
        logger.section("FORMATTER  —  Generating release notes from agent results")

        # Build the aggregated findings block
        sections_text = self._build_sections_block(results)
        total_findings = sum(len(r.findings) for r in results)
        all_findings   = [f for r in results for f in r.findings]
        risk           = _severity_badge(all_findings)

        logger.detail("Total findings across all agents", str(total_findings), "cyan")
        logger.detail("Overall risk level", risk,
                      "red" if risk in ("CRITICAL", "HIGH") else
                      "yellow" if risk == "MEDIUM" else "green")

        prompt = f"""You are a senior technical writer producing official release notes
for a software pull request.

You have received structured analysis from 8 specialist agents that each examined
the PR from a different angle. Your job is to FORMAT their findings into clean,
professional release notes. Do NOT add new analysis — only format what is given.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PULL REQUEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Title:   {meta.get('title', 'N/A')}
Author:  {meta.get('author', 'N/A')}
Branch:  {meta.get('head', '?')} → {meta.get('base', '?')}
URL:     {meta.get('url', '')}
Risk:    {risk}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGENT FINDINGS  (8 categories, {total_findings} total findings)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{sections_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATTING INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Produce release notes using EXACTLY these 9 sections in this order.
For each section:
  - Use the agent's findings as your source of truth
  - Write clean, professional bullet points (not the raw agent output)
  - If the agent reported nothing, write "• None" for that section
  - Do NOT invent findings not present in the agent output

## 🚀 Features
## 🐛 Bug Fixes
## ⚡ Performance & Profiling
## 🔍 Linting & Static Analysis
## 🧹 Code Quality
## 🎨 Formatting & Style
## 🏗️ Structural / Architecture
## 💥 Breaking Changes
## ⚠️ Risk Assessment
  State the risk level ({risk}) and write 2–3 sentences summarising
  the most important findings across all categories.
"""

        logger.start("formatter_llm", f"Sending {total_findings} findings to Ollama for formatting…")
        logger.detail("Model", get_ollama_model(), "cyan")
        t0 = time.time()

        try:
            resp = requests.post(
                f"{get_ollama_url()}/api/chat",
                json={
                    "model": get_ollama_model(),
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {"temperature": 0.1},   # low temp = consistent format
                    "stream": False,
                },
                timeout=300,
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            raise ValueError(
                f"Cannot connect to Ollama at {get_ollama_url()}. Run: ollama serve"
            )
        except Exception as exc:
            raise ValueError(f"Formatter Ollama call failed: {exc}")

        elapsed = time.time() - t0
        logger.ok("formatter_llm",
                  f"Release notes generated — {len(content)} chars  ({elapsed:.1f}s)")
        return content

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_sections_block(self, results: Sequence[AgentResult]) -> str:
        """
        Build the aggregated findings text that goes into the formatter prompt.
        For each agent: heading + deterministic findings + LLM narrative.
        """
        blocks: list[str] = []
        for r in results:
            block_lines = [f"### {r.emoji} {r.category}"]

            if r.error:
                block_lines.append(f"  [ERROR: agent failed — {r.error}]")
            elif r.skipped:
                block_lines.append("  [Nothing to report]")
            else:
                # Deterministic findings first (hard facts)
                if r.findings:
                    block_lines.append("  Deterministic findings:")
                    for f in r.findings:
                        loc = f" [{f.file}:{f.line}]" if f.file and f.line \
                              else (f" [{f.file}]" if f.file else "")
                        block_lines.append(
                            f"    [{f.severity.upper()}]{loc} {f.title} — {f.detail}"
                        )
                # Agent's LLM narrative (contextual reasoning)
                if r.llm_narrative.strip():
                    block_lines.append("  Agent narrative:")
                    for line in r.llm_narrative.splitlines():
                        block_lines.append(f"    {line}")

            blocks.append("\n".join(block_lines))

        return "\n\n".join(blocks)
