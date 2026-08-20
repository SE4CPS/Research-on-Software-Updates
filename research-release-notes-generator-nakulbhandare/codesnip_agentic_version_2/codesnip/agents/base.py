"""
base.py — Intelligent BaseAgent with FULL visibility logging.

Every agent shows:
  - Memory loaded (past PRs, recurring patterns)
  - Sandbox execution results per language (crashes, leaks, timing)
  - What the detector found (severity, file, line)
  - Predictions from memory (with confidence %)
  - What was written to memory (learning)
  - Ollama call status + response length

The user can SEE the agent learning and the sandbox actually executing.
"""
from __future__ import annotations

import time
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from codesnip.agents import AgentResult, Finding, PRContext
from codesnip.agents.memory.store import get_memory
from codesnip.infrastructure.config_service import get_ollama_url, get_ollama_model
from codesnip.shared import logger


# ── Severity colours (ANSI) ──────────────────────────────────────────────────

_SEV_ICON = {
    "critical": "🔴", "high": "🟠", "medium": "🟡",
    "low":  "🔵", "info": "⚪",
}
_SEV_LABEL = {
    "critical": "CRIT", "high": "HIGH",
    "medium": "MED ", "low": "LOW ", "info": "INFO",
}


@dataclass
class Prediction:
    type:        str
    description: str
    confidence:  float
    evidence:    list[str]
    severity:    str


class IntelligentBaseAgent(ABC):
    CATEGORY: str = ""
    EMOJI:    str = ""

    @abstractmethod
    def detect(self, ctx: PRContext) -> list[Finding]: ...

    @abstractmethod
    def build_prompt(self, ctx, findings, predictions, memory_context) -> str: ...

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, ctx: PRContext, findings: list[Finding]) -> list[Prediction]:
        predictions: list[Prediction] = []
        mem = get_memory()
        for f in findings:
            key     = mem._make_key(ctx.repo, self.CATEGORY, f.title)
            conf    = mem.get_pattern_confidence(ctx.repo, key)
            similar = mem.get_similar_past_findings(ctx.repo, f.title)
            if len(similar) >= 2:
                predictions.append(Prediction(
                    type        = "recurring_pattern",
                    description = (
                        f"`{f.title}` seen {len(similar)}x in this repo before. "
                        f"Reliability: {conf:.0%}."
                    ),
                    confidence  = min(0.5 + len(similar) * 0.1, 0.95),
                    evidence    = [s["detail"] for s in similar[:3]],
                    severity    = f.severity,
                ))
        return predictions

    # ── Memory helpers ────────────────────────────────────────────────────────

    def _load_memory_context(self, ctx: PRContext) -> str:
        mem      = get_memory()
        summary  = mem.get_repo_summary(ctx.repo)
        patterns = mem.get_recurring_patterns(ctx.repo, self.CATEGORY, min_occurrences=2)
        lines    = [f"=== MEMORY: {self.CATEGORY} | {ctx.repo} ==="]
        lines.append(f"PRs analysed: {summary['total_prs_analysed']}")
        if patterns:
            lines.append("Recurring patterns (seen 2+ times):")
            for p in patterns[:6]:
                lines.append(
                    f"  [{p['occurrences']}x] {p['example_detail'][:80]}"
                    + (f" (confirmed {p['confirmed_count']}x)" if p['confirmed_count'] else "")
                )
        error_files = summary.get("error_prone_files", [])
        if error_files:
            lines.append("Historically error-prone files:")
            for ef in error_files[:4]:
                lines.append(f"  {ef['file']}  ({ef['cnt']} past issues)")
        if not patterns and summary['total_prs_analysed'] == 0:
            lines.append("  First time analysing this repo.")
        return "\n".join(lines)

    def _record_to_memory(self, ctx: PRContext,
                          findings: list[Finding],
                          predictions: list[Prediction]) -> int:
        mem   = get_memory()
        count = 0
        for f in findings:
            mem.record_finding(
                repo=ctx.repo, pr_number=ctx.pr_number, agent=self.CATEGORY,
                category=self.CATEGORY, file=f.file or "", line=f.line or 0,
                severity=f.severity, title=f.title, detail=f.detail,
                tool_source=getattr(f, "tool_source", "unknown"),
            )
            count += 1
        for p in predictions:
            mem.record_prediction(
                repo=ctx.repo, pr_number=ctx.pr_number, agent=self.CATEGORY,
                prediction_type=p.type, description=p.description,
                confidence=p.confidence, evidence=p.evidence,
            )
        return count

    # ── Ollama helper ─────────────────────────────────────────────────────────

    def _ollama(self, prompt: str, temperature: float = 0.15) -> str:
        url   = f"{get_ollama_url()}/api/chat"
        model = get_ollama_model()
        try:
            resp = requests.post(
                url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {"temperature": temperature},
                    "stream": False,
                },
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            raise ValueError(
                f"Cannot connect to Ollama at {get_ollama_url()}. Run: ollama serve"
            )
        except Exception as exc:
            return f"[LLM error in {self.CATEGORY}: {exc}]"

    # ── Diff helpers ──────────────────────────────────────────────────────────

    def _added_lines(self, ctx):
        out = []
        for fname, data in ctx.diff_files.items():
            for t, line in data.get("ordered", []):
                if t == "+": out.append((fname, line))
        return out

    def _removed_lines(self, ctx):
        out = []
        for fname, data in ctx.diff_files.items():
            for t, line in data.get("ordered", []):
                if t == "-": out.append((fname, line))
        return out

    def _exec(self, ctx):    return ctx.execution_report
    def _sandbox(self, ctx): return getattr(ctx, "sandbox_report", None)

    @staticmethod
    def _fmt_findings(findings):
        if not findings: return "  None."
        rows = []
        for f in findings:
            loc = f" [{f.file}:{f.line}]" if f.file and f.line else (f" [{f.file}]" if f.file else "")
            rows.append(f"  [{f.severity.upper()}]{loc} {f.title} — {f.detail}")
        return "\n".join(rows)

    @staticmethod
    def _fmt_predictions(predictions):
        if not predictions: return "  None."
        rows = []
        for p in predictions:
            rows.append(f"  [{p.severity.upper()}] {p.description}  (confidence: {p.confidence:.0%})")
        return "\n".join(rows)

    # ── Main orchestrator — FULL VISIBILITY ───────────────────────────────────

    def run(self, ctx: PRContext) -> AgentResult:
        logger.section(f"AGENT  {self.EMOJI}  {self.CATEGORY}")
        t0 = time.time()

        # ── 1. MEMORY ─────────────────────────────────────────────────────────
        mem_context   = self._load_memory_context(ctx)
        mem           = get_memory()
        mem_summary   = mem.get_repo_summary(ctx.repo)
        past_prs      = mem_summary["total_prs_analysed"]
        top_patterns  = mem_summary.get("top_patterns", [])
        patterns_this = mem.get_recurring_patterns(ctx.repo, self.CATEGORY, min_occurrences=2)

        logger.info(f"  📚 MEMORY  ─────────────────────────────────")
        if past_prs == 0:
            logger.info(f"     First analysis for this repo — no prior knowledge")
        else:
            logger.info(f"     Past PRs analysed:  {past_prs}")
            logger.info(f"     Patterns learned:   {len(top_patterns)} total  |  "
                        f"{len(patterns_this)} for {self.CATEGORY}")
            if patterns_this:
                logger.info(f"     Top patterns for {self.CATEGORY}:")
                for p in patterns_this[:4]:
                    confirmed = f"  ✓ confirmed {p['confirmed_count']}x" if p['confirmed_count'] else ""
                    logger.info(f"       [{p['occurrences']}x] {p['example_detail'][:70]}{confirmed}")
            error_files = mem_summary.get("error_prone_files", [])
            if error_files:
                logger.info(f"     Error-prone files:")
                for ef in error_files[:3]:
                    logger.info(f"       {ef['file']}  ({ef['cnt']} past issues)")

        # ── 2. SANDBOX visibility ─────────────────────────────────────────────
        sandbox = self._sandbox(ctx)
        if sandbox is not None:
            logger.info(f"  🧪 SANDBOX ─────────────────────────────────")
            if not sandbox.executed:
                logger.info(f"     Status:  not executed  ({sandbox.error or 'no supported files'})")
            else:
                langs_str = ", ".join(sandbox.languages_detected) if sandbox.languages_detected else "unknown"
                logger.info(f"     Status:  ✅ executed")
                logger.info(f"     Languages:  {langs_str}")
                logger.info(f"     Functions profiled:  {len(sandbox.functions)}")
                logger.info(f"     Total execution time:  {sandbox.total_exec_ms:.0f}ms")

                if sandbox.functions:
                    logger.info(f"     Function profiles:")
                    for fp in sandbox.functions[:8]:
                        tag  = f"  [{fp.language}]"
                        exc  = f"  💥 {fp.raised_exception[:50]}" if fp.raised_exception else ""
                        slow = "  🐌 SLOW" if fp.is_slow else ""
                        leak = "  💧 LEAK" if fp.is_leak_suspect else ""
                        mem_str = f"  mem={fp.mem_delta_kb:+.0f}KB" if abs(fp.mem_delta_kb) > 0.5 else ""
                        logger.info(
                            f"       {fp.name:<22} {fp.cpu_time_ms:6.1f}ms"
                            f"{mem_str}{tag}{exc}{slow}{leak}"
                        )

                if sandbox.crash_functions:
                    logger.info(f"     💥 Crashes detected ({len(sandbox.crash_functions)}):")
                    for fp in sandbox.crash_functions:
                        logger.info(f"       {fp.name}() → {fp.raised_exception}")

                if sandbox.leak_suspects:
                    logger.info(f"     💧 Memory leak suspects ({len(sandbox.leak_suspects)}):")
                    for fp in sandbox.leak_suspects:
                        logger.info(f"       {fp.name}()  object_delta={fp.object_delta}  mem_delta={fp.mem_delta_kb:+.1f}KB")

                if sandbox.slow_functions:
                    logger.info(f"     🐌 Slow functions ({len(sandbox.slow_functions)}):")
                    for fp in sandbox.slow_functions:
                        logger.info(f"       {fp.name}()  {fp.cpu_time_ms:.0f}ms")

                if sandbox.raw_output and len(sandbox.raw_output.strip()) > 0:
                    preview = sandbox.raw_output.strip()[:120].replace('\n', ' ')
                    logger.info(f"     Raw output preview: {preview}…")

        # ── 3. DETECT ─────────────────────────────────────────────────────────
        logger.info(f"  🔍 DETECT  ─────────────────────────────────")
        logger.start(f"det_{self.CATEGORY}", "Running detector…")
        t_det = time.time()
        try:
            findings = self.detect(ctx)
        except Exception as exc:
            logger.fail(f"det_{self.CATEGORY}", f"Detector crashed: {exc}")
            return AgentResult(category=self.CATEGORY, emoji=self.EMOJI, error=str(exc))

        det_ms = (time.time() - t_det) * 1000
        logger.ok(f"det_{self.CATEGORY}",
                  f"{len(findings)} finding(s) in {det_ms:.0f}ms")

        if findings:
            logger.info(f"     Findings:")
            for f in findings:
                icon  = _SEV_ICON.get(f.severity, "⚪")
                label = _SEV_LABEL.get(f.severity, "???")
                loc   = f"  [{f.file}:{f.line}]" if f.file and f.line else \
                        (f"  [{f.file}]" if f.file else "")
                logger.info(f"       {icon} {label}  {f.title}{loc}")
                if f.detail and f.detail != f.title:
                    logger.info(f"              ↳ {f.detail[:90]}")
        else:
            logger.info(f"     No findings for {self.CATEGORY} in this PR")

        # ── 4. PREDICT (from memory) ──────────────────────────────────────────
        logger.info(f"  🧠 PREDICT (memory-based) ───────────────────")
        logger.start(f"pred_{self.CATEGORY}", "Generating predictions from past PRs…")
        try:
            predictions = self.predict(ctx, findings)
        except Exception as exc:
            logger.warn(f"pred_{self.CATEGORY}", f"Prediction skipped: {exc}")
            predictions = []

        if predictions:
            logger.ok(f"pred_{self.CATEGORY}", f"{len(predictions)} prediction(s) from memory")
            for p in predictions:
                icon = _SEV_ICON.get(p.severity, "⚪")
                logger.info(f"       {icon} [{p.confidence:.0%} confidence]  {p.description[:80]}")
                for ev in p.evidence[:2]:
                    logger.info(f"              evidence: {ev[:70]}")
        else:
            logger.ok(f"pred_{self.CATEGORY}",
                      "No memory-based predictions" +
                      (" (not enough history yet)" if past_prs < 3 else ""))

        # ── 5. LEARN — write to memory ────────────────────────────────────────
        logger.info(f"  💾 LEARNING ────────────────────────────────")
        stored = self._record_to_memory(ctx, findings, predictions)

        # Show what was just learned
        if findings or predictions:
            logger.ok(f"mem_{self.CATEGORY}",
                      f"Wrote {stored} finding(s) + {len(predictions)} prediction(s) to memory")
            logger.info(f"     DB: ~/.codesnip/memory/agent_memory.db")
            logger.info(f"     After this PR: {past_prs + 1} PRs analysed for {ctx.repo}")

            # Show how memory is growing
            new_patterns = mem.get_recurring_patterns(ctx.repo, self.CATEGORY, min_occurrences=2)
            if len(new_patterns) > len(patterns_this):
                newly = len(new_patterns) - len(patterns_this)
                logger.info(f"     🆕 {newly} new pattern(s) emerged from this PR!")
                for p in new_patterns[len(patterns_this):len(patterns_this)+2]:
                    logger.info(f"        Pattern: [{p['occurrences']}x] {p['example_detail'][:60]}")
        else:
            logger.info(f"     Nothing new to learn for {self.CATEGORY} this PR")

        # ── 6. OLLAMA ─────────────────────────────────────────────────────────
        logger.info(f"  🤖 OLLAMA ──────────────────────────────────")
        model = get_ollama_model()
        prompt = self.build_prompt(ctx, findings, predictions, mem_context)
        logger.start(f"llm_{self.CATEGORY}",
                     f"Sending to {model}  ({len(prompt)} prompt chars)…")
        t_llm = time.time()
        narrative = self._ollama(prompt)
        llm_ms = (time.time() - t_llm) * 1000

        if narrative.startswith("[LLM error"):
            logger.fail(f"llm_{self.CATEGORY}", narrative)
        else:
            logger.ok(f"llm_{self.CATEGORY}",
                      f"Response: {len(narrative)} chars in {llm_ms:.0f}ms")
            # Show a brief preview of the LLM output
            preview = narrative.replace('\n', ' ').strip()[:120]
            logger.info(f"     Preview: {preview}…" if len(narrative) > 120 else f"     Output: {narrative.strip()}")

        total_ms = (time.time() - t0) * 1000
        logger.info(f"  ─────────────────────────────────────────────")
        logger.ok(f"agent_{self.CATEGORY}",
                  f"{self.EMOJI} {self.CATEGORY} done  "
                  f"findings={len(findings)}  predictions={len(predictions)}  "
                  f"total={total_ms:.0f}ms")

        return AgentResult(
            category=self.CATEGORY, emoji=self.EMOJI,
            findings=findings, llm_narrative=narrative,
            skipped=(not findings and not predictions and not narrative.strip()),
        )
