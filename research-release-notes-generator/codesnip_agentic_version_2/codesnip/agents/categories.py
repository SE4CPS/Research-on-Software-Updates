"""
categories.py — 8 Intelligent Agents

Each agent:
  detect()   → reads real tool output from ExecutionReport (ruff/mypy/radon/bandit/vulture)
               + reads sandbox execution results (actual runtime behaviour)
  predict()  → uses SQLite memory to find patterns across past PRs
  build_prompt() → sends real tool output + sandbox data + memory context to Ollama

NO guessing. NO regex on diff text for correctness checks.
Every finding comes from a real tool or real execution.
"""
from __future__ import annotations

import ast
import os
import re

from codesnip.agents import Finding, PRContext
from codesnip.agents.base import IntelligentBaseAgent, Prediction
from codesnip.agents.memory.store import get_memory


# ── Shared filter ─────────────────────────────────────────────────────────────

def _in_pr(report, pr_files: set, issue_file: str) -> bool:
    """Only surface issues in files that were actually changed in this PR."""
    if not pr_files:
        return True
    norm = issue_file.replace("\\", "/")
    return any(norm.endswith(f.replace("\\", "/")) for f in pr_files)


# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURES
# ══════════════════════════════════════════════════════════════════════════════
class FeaturesAgent(IntelligentBaseAgent):
    CATEGORY = "Features"
    EMOJI    = "🚀"

    def detect(self, ctx: PRContext) -> list[Finding]:
        findings: list[Finding] = []
        report = self._exec(ctx)

        if not report.cloned:
            return findings

        seen = set()
        for rel in ctx.diff_files:
            if not rel.endswith(".py"):
                continue
            full = os.path.join(report.repo_path, rel)
            if not os.path.exists(full):
                continue
            try:
                tree = ast.parse(open(full, encoding="utf-8", errors="replace").read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith("_") and (rel, node.name) not in seen:
                        seen.add((rel, node.name))
                        findings.append(Finding("info", f"New/modified `{node.name}()`",
                            f"{rel}:{node.lineno}", rel, node.lineno))
                elif isinstance(node, ast.ClassDef) and (rel, node.name) not in seen:
                    seen.add((rel, node.name))
                    findings.append(Finding("info", f"New/modified class `{node.name}`",
                        f"{rel}:{node.lineno}", rel, node.lineno))

        return findings[:20]

    def build_prompt(self, ctx, findings, predictions, memory_context) -> str:
        return f"""{memory_context}

You are writing the FEATURES section of a pull-request changelog.
Identify new capabilities: functions, classes, endpoints, CLI commands, config options.

PR: {ctx.meta.get('title')}  |  Author: {ctx.meta.get('author')}
Commits:
{ctx.commits}

Confirmed new/modified symbols (from AST of cloned code):
{self._fmt_findings(findings)}

Predictions from memory:
{self._fmt_predictions(predictions)}

Diff:
{ctx.diff[:3500]}

Write 1–6 bullet points. Use past tense. Name exact function/class and file.
If nothing new: "• No new features."
Bullet points ONLY — no heading."""


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUG FIXES
# ══════════════════════════════════════════════════════════════════════════════
class BugFixAgent(IntelligentBaseAgent):
    CATEGORY = "Bug Fixes"
    EMOJI    = "🐛"

    def detect(self, ctx: PRContext) -> list[Finding]:
        findings: list[Finding] = []
        report   = self._exec(ctx)
        pr_files = set(ctx.diff_files.keys())
        sandbox  = self._sandbox(ctx)

        # Real crash data from sandbox execution
        if sandbox and sandbox.executed:
            for fp in sandbox.crash_functions:
                if _in_pr(report, pr_files, fp.file):
                    findings.append(Finding(
                        "critical",
                        f"`{fp.name}()` raises {fp.raised_exception.split(':')[0]} at runtime",
                        f"Actual exception when called: {fp.raised_exception[:120]}",
                        fp.file, fp.line,
                    ))

        # Bandit HIGH/MEDIUM — real security bugs
        for issue in report.security:
            if issue.severity in ("HIGH", "MEDIUM") and _in_pr(report, pr_files, issue.file):
                findings.append(Finding(
                    "high" if issue.severity == "HIGH" else "medium",
                    f"[{issue.test_id}] {issue.test_name}",
                    issue.message, issue.file, issue.line,
                ))

        # Ruff B-codes (bugbear) — real bug patterns
        for issue in report.ruff:
            if issue.code.startswith("B") and _in_pr(report, pr_files, issue.file):
                findings.append(Finding("high", f"[{issue.code}] {issue.message}",
                    "flake8-bugbear: actual footgun", issue.file, issue.line))

        # Syntax errors — code is broken
        for err in report.syntax_errors:
            if _in_pr(report, pr_files, err.file):
                findings.append(Finding("critical", f"Syntax error: {err.message}",
                    "Module will fail to import", err.file, err.line))

        # AST: bare except in changed files
        for rel in ctx.diff_files:
            if not rel.endswith(".py"):
                continue
            full = os.path.join(report.repo_path, rel) if report.cloned else ""
            if not full or not os.path.exists(full):
                continue
            try:
                tree = ast.parse(open(full, encoding="utf-8", errors="replace").read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    findings.append(Finding("high",
                        "Bare `except:` swallows BaseException, KeyboardInterrupt, SystemExit",
                        "Use `except Exception:` or specific type",
                        rel, node.lineno))

        return findings

    def predict(self, ctx: PRContext, findings: list[Finding]) -> list[Prediction]:
        predictions = super().predict(ctx, findings)
        sandbox = self._sandbox(ctx)

        # If sandbox showed crashes, predict this will fail in production
        if sandbox and sandbox.crash_functions:
            predictions.append(Prediction(
                type        = "crash",
                description = (
                    f"{len(sandbox.crash_functions)} function(s) crashed during sandbox execution. "
                    f"These WILL raise exceptions in production."
                ),
                confidence  = 0.95,
                evidence    = [f"{fp.name}: {fp.raised_exception}" for fp in sandbox.crash_functions[:3]],
                severity    = "critical",
            ))
        return predictions

    def build_prompt(self, ctx, findings, predictions, memory_context) -> str:
        report  = self._exec(ctx)
        sandbox = self._sandbox(ctx)
        bandit_out = report.bandit_raw[:1500] if report.bandit_raw else "bandit not installed"
        sandbox_out = ""
        if sandbox and sandbox.crash_functions:
            sandbox_out = "\nSANDBOX CRASHES (actual runtime execution):\n"
            for fp in sandbox.crash_functions:
                sandbox_out += f"  {fp.name}() → {fp.raised_exception}\n"

        return f"""{memory_context}

You are writing the BUG FIXES section of a pull-request changelog.
Use ONLY the tool output below — do not invent issues.

PR: {ctx.meta.get('title')}
Commits:
{ctx.commits}

Real bugs found by tools:
{self._fmt_findings(findings)}

Predictions from past PR history:
{self._fmt_predictions(predictions)}

=== BANDIT SECURITY SCAN ===
{bandit_out}
{sandbox_out}

Diff:
{ctx.diff[:3000]}

Write 1–6 bullet points. For each bug: what was wrong, file:line, what fixed it.
If sandbox showed crashes, mention them explicitly with the exception type.
If no bugs fixed: "• No bug fixes."
Bullet points ONLY."""


# ══════════════════════════════════════════════════════════════════════════════
# 3. PERFORMANCE + MEMORY LEAK DETECTION
# ══════════════════════════════════════════════════════════════════════════════
class PerformanceAgent(IntelligentBaseAgent):
    CATEGORY = "Performance & Profiling"
    EMOJI    = "⚡"

    def detect(self, ctx: PRContext) -> list[Finding]:
        findings: list[Finding] = []
        report   = self._exec(ctx)
        pr_files = set(ctx.diff_files.keys())
        sandbox  = self._sandbox(ctx)

        # Real execution data — actual slow functions
        if sandbox and sandbox.executed:
            for fp in sandbox.slow_functions:
                if _in_pr(report, pr_files, fp.file):
                    findings.append(Finding(
                        "high",
                        f"`{fp.name}()` took {fp.cpu_time_ms:.0f}ms in sandbox",
                        f"cProfile hotspots: {', '.join(fp.top_calls[:2]) or 'N/A'}",
                        fp.file, fp.line,
                    ))

            # Real memory leak suspects from tracemalloc
            for fp in sandbox.leak_suspects:
                if _in_pr(report, pr_files, fp.file):
                    findings.append(Finding(
                        "critical",
                        f"`{fp.name}()` leaked {fp.object_delta:+d} objects (tracemalloc)",
                        f"Memory grew {fp.mem_delta_kb:+.1f}KB — objects not freed after GC",
                        fp.file, fp.line,
                    ))

            # High memory usage functions
            for fp in sandbox.functions:
                if fp.mem_delta_kb > 50000 and _in_pr(report, pr_files, fp.file):  # >50MB peak
                    findings.append(Finding(
                        "high",
                        f"`{fp.name}()` peak memory {fp.mem_delta_kb/1024:.1f}MB",
                        f"Consider streaming or chunking",
                        fp.file, fp.line,
                    ))

        # Radon high-complexity = performance risk
        for c in report.complex_functions():
            if _in_pr(report, pr_files, c.file):
                findings.append(Finding(
                    "high" if c.complexity >= 15 else "medium",
                    f"`{c.function}()` cyclomatic complexity CC={c.complexity} (rank {c.rank})",
                    f"High CC = more paths = harder to optimise",
                    c.file, c.line,
                ))

        # AST: blocking I/O patterns in changed files
        for rel in ctx.diff_files:
            if not rel.endswith(".py"):
                continue
            full = os.path.join(report.repo_path, rel) if report.cloned else ""
            if not full or not os.path.exists(full):
                continue
            try:
                tree = ast.parse(open(full, encoding="utf-8", errors="replace").read())
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                # time.sleep()
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "sleep"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "time"):
                    findings.append(Finding("high", "`time.sleep()` blocks event loop",
                        "Use `await asyncio.sleep()` in async context", rel, node.lineno))

                # requests without timeout
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("get","post","put","delete","patch")
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "requests"):
                    if not any(kw.arg == "timeout" for kw in node.keywords):
                        findings.append(Finding("high",
                            f"requests.{node.func.attr}() has no timeout= — can hang forever",
                            "Add timeout=30 or configurable timeout", rel, node.lineno))

                # Nested loops
                if isinstance(node, (ast.For, ast.While)):
                    for child in ast.walk(node):
                        if child is not node and isinstance(child, (ast.For, ast.While)):
                            findings.append(Finding("medium",
                                f"Nested loop — O(n²) risk",
                                "Consider dict lookup or vectorisation",
                                rel, getattr(node, "lineno", 0)))
                            break

        return findings

    def predict(self, ctx: PRContext, findings: list[Finding]) -> list[Prediction]:
        predictions = super().predict(ctx, findings)
        sandbox     = self._sandbox(ctx)
        mem         = get_memory()

        # Memory leak prediction from sandbox data + history
        if sandbox and sandbox.leak_suspects:
            total_leak = sum(fp.object_delta for fp in sandbox.leak_suspects)
            predictions.append(Prediction(
                type        = "memory_leak",
                description = (
                    f"MEMORY LEAK DETECTED: {len(sandbox.leak_suspects)} function(s) "
                    f"did not free {total_leak:,} objects after GC. "
                    f"Under sustained load this will cause OOM."
                ),
                confidence  = 0.90,
                evidence    = [
                    f"{fp.name}: +{fp.object_delta} objects, +{fp.mem_delta_kb:.1f}KB"
                    for fp in sandbox.leak_suspects[:3]
                ],
                severity    = "critical",
            ))

        # Performance regression prediction from history
        recurring = mem.get_recurring_patterns(ctx.repo, self.CATEGORY, min_occurrences=2)
        if recurring:
            predictions.append(Prediction(
                type        = "performance_regression",
                description = (
                    f"This repo has had {len(recurring)} recurring performance "
                    f"pattern(s). Current PR may reintroduce known issues."
                ),
                confidence  = 0.7,
                evidence    = [p["example_detail"][:60] for p in recurring[:3]],
                severity    = "medium",
            ))

        return predictions

    def build_prompt(self, ctx, findings, predictions, memory_context) -> str:
        sandbox = self._sandbox(ctx)
        report  = self._exec(ctx)

        sandbox_section = ""
        if sandbox and sandbox.executed:
            sandbox_section = f"""
=== SANDBOX EXECUTION RESULTS (actual runtime) ===
Total execution time: {sandbox.total_exec_ms:.0f}ms
Functions executed: {len(sandbox.functions)}
Slow functions (>1s): {len(sandbox.slow_functions)}
Memory leak suspects (object_delta>500): {len(sandbox.leak_suspects)}
"""
            if sandbox.leak_suspects:
                sandbox_section += "LEAK SUSPECTS:\n"
                for fp in sandbox.leak_suspects:
                    sandbox_section += (
                        f"  {fp.name}(): objects {fp.objects_before}→{fp.objects_after} "
                        f"(delta={fp.object_delta:+d}), memory {fp.mem_delta_kb:+.1f}KB\n"
                    )
            if sandbox.slow_functions:
                sandbox_section += "SLOW FUNCTIONS:\n"
                for fp in sandbox.slow_functions:
                    sandbox_section += f"  {fp.name}(): {fp.cpu_time_ms:.0f}ms, top calls: {fp.top_calls[:2]}\n"

        radon_out = report.radon_raw[:1000] if report.radon_raw else "radon not installed"

        return f"""{memory_context}

You are writing the PERFORMANCE & MEMORY section of a pull-request changelog.
You have ACTUAL runtime data from sandbox execution. Use it.

PR: {ctx.meta.get('title')}

Tool-based findings:
{self._fmt_findings(findings)}

Memory-based predictions (from {get_memory().get_repo_summary(ctx.repo)['total_prs_analysed']} past PRs):
{self._fmt_predictions(predictions)}

=== RADON COMPLEXITY ===
{radon_out}
{sandbox_section}

Write 1–6 bullet points. For memory leaks, state exact object deltas.
For slow functions, state exact ms. Predict what will happen under production load.
If no issues: "• No performance issues detected."
Bullet points ONLY."""


# ══════════════════════════════════════════════════════════════════════════════
# 4. LINTING
# ══════════════════════════════════════════════════════════════════════════════
class LintingAgent(IntelligentBaseAgent):
    CATEGORY = "Linting & Static Analysis"
    EMOJI    = "🔍"

    def detect(self, ctx: PRContext) -> list[Finding]:
        findings: list[Finding] = []
        report   = self._exec(ctx)
        pr_files = set(ctx.diff_files.keys())

        for issue in report.ruff:
            if not _in_pr(report, pr_files, issue.file):
                continue
            sev = (
                "high"   if issue.code.startswith(("F8","E9")) else
                "medium" if issue.code.startswith(("F4","B"))  else
                "low"
            )
            findings.append(Finding(sev, f"[{issue.code}] {issue.message}",
                f"{issue.file}:{issue.line}:{issue.col}", issue.file, issue.line))

        for issue in report.mypy:
            if not _in_pr(report, pr_files, issue.file):
                continue
            if issue.severity == "error":
                findings.append(Finding("high",
                    f"[mypy:{issue.code}] {issue.message}",
                    f"Type error at {issue.file}:{issue.line}",
                    issue.file, issue.line))

        return findings

    def build_prompt(self, ctx, findings, predictions, memory_context) -> str:
        report   = self._exec(ctx)
        ruff_out = report.ruff_raw[:2000] if report.ruff_raw else "ruff not installed"
        mypy_out = report.mypy_raw[:1500] if report.mypy_raw else "mypy not installed"

        return f"""{memory_context}

You are writing the LINTING & STATIC ANALYSIS section of a pull-request changelog.
Use ONLY the actual tool output below. Do not invent issues.

PR: {ctx.meta.get('title')}

Structured findings (PR files only):
{self._fmt_findings(findings)}

Memory-based predictions:
{self._fmt_predictions(predictions)}

=== RUFF OUTPUT ===
{ruff_out}

=== MYPY OUTPUT ===
{mypy_out}

Write bullet points for each real issue. Group by: type errors, unused imports,
undefined names, style violations. Reference exact file:line.
If no issues or tools unavailable: say so explicitly.
Bullet points ONLY."""


# ══════════════════════════════════════════════════════════════════════════════
# 5. CODE QUALITY
# ══════════════════════════════════════════════════════════════════════════════
class CodeQualityAgent(IntelligentBaseAgent):
    CATEGORY = "Code Quality"
    EMOJI    = "🧹"

    def detect(self, ctx: PRContext) -> list[Finding]:
        findings: list[Finding] = []
        report   = self._exec(ctx)
        pr_files = set(ctx.diff_files.keys())

        # Vulture dead code
        for item in report.dead_code:
            if _in_pr(report, pr_files, item.file):
                findings.append(Finding("medium",
                    f"Dead {item.kind} `{item.name}` ({item.confidence}% confidence)",
                    "Vulture: unused — safe to remove",
                    item.file, item.line))

        # High-complexity functions
        for c in report.complex_functions():
            if _in_pr(report, pr_files, c.file):
                findings.append(Finding("medium",
                    f"`{c.function}()` CC={c.complexity} rank={c.rank}",
                    "Aim for CC<10. Split into smaller functions.",
                    c.file, c.line))

        # Long functions from AST
        for rel in ctx.diff_files:
            if not rel.endswith(".py"): continue
            full = os.path.join(report.repo_path, rel) if report.cloned else ""
            if not full or not os.path.exists(full): continue
            try:
                tree = ast.parse(open(full, encoding="utf-8", errors="replace").read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    end = getattr(node, "end_lineno", None)
                    if end and (end - node.lineno) > 50:
                        findings.append(Finding("medium",
                            f"`{node.name}()` is {end - node.lineno} lines",
                            "Functions >50 lines are hard to test",
                            rel, node.lineno))

        return findings

    def build_prompt(self, ctx, findings, predictions, memory_context) -> str:
        report     = self._exec(ctx)
        vulture_out = report.vulture_raw[:1500] if report.vulture_raw else "vulture not installed"
        radon_out   = report.radon_raw[:800]    if report.radon_raw   else "radon not installed"

        return f"""{memory_context}

You are writing the CODE QUALITY section of a pull-request changelog.

PR: {ctx.meta.get('title')}

Tool findings:
{self._fmt_findings(findings)}

Memory predictions:
{self._fmt_predictions(predictions)}

=== VULTURE (dead code) ===
{vulture_out}

=== RADON (complexity) ===
{radon_out}

Write 1–5 bullet points: dead code, complex functions, overly long functions,
TODO markers introduced. Name the function and file.
If quality is good: "• Code quality looks good."
Bullet points ONLY."""


# ══════════════════════════════════════════════════════════════════════════════
# 6. FORMATTING
# ══════════════════════════════════════════════════════════════════════════════
class FormattingAgent(IntelligentBaseAgent):
    CATEGORY = "Formatting & Style"
    EMOJI    = "🎨"

    def detect(self, ctx: PRContext) -> list[Finding]:
        findings: list[Finding] = []
        report   = self._exec(ctx)
        pr_files = set(ctx.diff_files.keys())

        for issue in report.ruff:
            if not _in_pr(report, pr_files, issue.file):
                continue
            if any(issue.code.startswith(p) for p in ("E1","E2","E3","E4","E5","W1","W2","W3","W5","W6")):
                findings.append(Finding("low",
                    f"[{issue.code}] {issue.message}",
                    f"{issue.file}:{issue.line}",
                    issue.file, issue.line))
        return findings

    def build_prompt(self, ctx, findings, predictions, memory_context) -> str:
        report = self._exec(ctx)
        style_lines = "\n".join(
            l for l in (report.ruff_raw or "").splitlines()
            if re.search(r"E[1-5]|W[1-6]", l)
        )[:1000] or "ruff not installed or no style issues"

        return f"""{memory_context}

You are writing the FORMATTING & STYLE section of a pull-request changelog.

PR: {ctx.meta.get('title')}

Ruff style findings (E/W codes, PR files only):
{style_lines}

Structured:
{self._fmt_findings(findings)}

Write 1–4 bullet points: line length, whitespace, blank lines, quote style.
If formatting is clean: "• Formatting looks clean."
Bullet points ONLY."""


# ══════════════════════════════════════════════════════════════════════════════
# 7. ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
class ArchitectureAgent(IntelligentBaseAgent):
    CATEGORY = "Structural / Architecture"
    EMOJI    = "🏗️"

    def detect(self, ctx: PRContext) -> list[Finding]:
        findings: list[Finding] = []
        report   = self._exec(ctx)

        for rel in ctx.diff_files:
            if not rel.endswith(".py"): continue
            full = os.path.join(report.repo_path, rel) if report.cloned else ""
            if not full or not os.path.exists(full): continue
            try:
                tree = ast.parse(open(full, encoding="utf-8", errors="replace").read())
            except SyntaxError:
                continue

            # Infrastructure call in presentation layer
            if any(p in rel for p in ("cli","presentation","view","controller","routes")):
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        names = [a.name for a in getattr(node, "names", [])]
                        mod   = getattr(node, "module", "") or ""
                        for n in names + [mod]:
                            if any(x in n for x in ("requests","pymongo","sqlite3","sqlalchemy","psycopg")):
                                findings.append(Finding("high",
                                    f"Infrastructure import `{n}` in presentation layer `{rel}`",
                                    "Move data access to infrastructure layer",
                                    rel, node.lineno))

            # Star imports
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if any(a.name == "*" for a in node.names):
                        findings.append(Finding("high",
                            f"Star import `from {node.module} import *`",
                            "Pollutes namespace, breaks refactoring tools",
                            rel, node.lineno))

            # God classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = sum(1 for n in ast.walk(node)
                                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
                    if methods > 12:
                        findings.append(Finding("medium",
                            f"Class `{node.name}` has {methods} methods — possible God class",
                            "Split by single responsibility principle",
                            rel, node.lineno))

        return findings

    def build_prompt(self, ctx, findings, predictions, memory_context) -> str:
        return f"""{memory_context}

You are writing the ARCHITECTURE section of a pull-request changelog.

PR: {ctx.meta.get('title')}
Files changed: {', '.join(ctx.diff_files.keys())}

Real architecture issues from AST analysis:
{self._fmt_findings(findings)}

Memory predictions:
{self._fmt_predictions(predictions)}

Diff:
{ctx.diff[:3000]}

Write 1–4 bullet points: layer violations, God classes, star imports,
wrong file placement. If clean: "• No architectural concerns."
Bullet points ONLY."""


# ══════════════════════════════════════════════════════════════════════════════
# 8. BREAKING CHANGES
# ══════════════════════════════════════════════════════════════════════════════
class BreakingChangesAgent(IntelligentBaseAgent):
    CATEGORY = "Breaking Changes"
    EMOJI    = "💥"

    def detect(self, ctx: PRContext) -> list[Finding]:
        findings: list[Finding] = []

        for rel, data in ctx.diff_files.items():
            if not rel.endswith(".py"): continue

            old_sigs: dict[str, str] = {}
            new_sigs: dict[str, str] = {}

            for t, line in data.get("ordered", []):
                m = re.match(r"\s*(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)", line)
                if m:
                    (old_sigs if t == "-" else new_sigs)[m.group(1)] = m.group(2).strip()

            # Removed public functions
            for name, sig in old_sigs.items():
                if name not in new_sigs and not name.startswith("_"):
                    findings.append(Finding("critical",
                        f"Public `{name}()` removed",
                        f"Was: ({sig[:60]}) — all callers will break",
                        rel))

            # Signature changes
            for name in set(old_sigs) & set(new_sigs):
                if old_sigs[name] != new_sigs[name] and not name.startswith("_"):
                    findings.append(Finding("high",
                        f"Signature changed: `{name}()`",
                        f"({old_sigs[name][:50]}) → ({new_sigs[name][:50]})",
                        rel))

            # Removed classes
            old_cls = {m.group(1) for t, l in data.get("ordered",[]) if t=="-"
                       for m in [re.match(r"\s*class\s+(\w+)", l)] if m}
            new_cls = {m.group(1) for t, l in data.get("ordered",[]) if t=="+"
                       for m in [re.match(r"\s*class\s+(\w+)", l)] if m}
            for cls in old_cls - new_cls:
                findings.append(Finding("critical", f"Class `{cls}` removed",
                    "All importers will get ImportError", rel))

        return findings

    def predict(self, ctx: PRContext, findings: list[Finding]) -> list[Prediction]:
        predictions = super().predict(ctx, findings)
        if any(f.severity == "critical" for f in findings):
            predictions.append(Prediction(
                type        = "crash",
                description = (
                    "Critical breaking change detected. Any deployed service importing "
                    "these symbols will crash on next import."
                ),
                confidence  = 0.99,
                evidence    = [f.title for f in findings if f.severity == "critical"],
                severity    = "critical",
            ))
        return predictions

    def build_prompt(self, ctx, findings, predictions, memory_context) -> str:
        return f"""{memory_context}

You are writing the BREAKING CHANGES section of a pull-request changelog.

PR: {ctx.meta.get('title')}
Commits:
{ctx.commits}

Confirmed breaking changes (from diff signature analysis):
{self._fmt_findings(findings)}

Predictions:
{self._fmt_predictions(predictions)}

For each breaking change: what changed, which file, what callers must do.
If no breaking changes: "• No breaking changes."
Bullet points ONLY."""
