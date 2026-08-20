"""
executor.py — Ground truth engine.

What it does:
  1. Clone the PR's head branch into a temp dir
  2. pip install the project + analysis tools (ruff, mypy, radon, bandit, vulture)
  3. Run every tool against the REAL code, capture structured output
  4. Return ExecutionReport — zero guessing, zero regex on diff text

Every agent reads ONLY from ExecutionReport. If a tool is missing,
that section says "tool not installed" rather than fabricating results.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from codesnip.shared import logger


# ── Structured output types ───────────────────────────────────────────────────

@dataclass
class RuffIssue:
    file: str
    line: int
    col: int
    code: str       # E501, F401, B006, etc.
    message: str


@dataclass
class MypyIssue:
    file: str
    line: int
    severity: str   # error | warning | note
    code: str       # [no-untyped-def] etc.
    message: str


@dataclass
class ComplexityResult:
    file: str
    function: str
    line: int
    complexity: int  # McCabe number
    rank: str        # A=1-5  B=6-10  C=11-15  D=16+


@dataclass
class SecurityIssue:
    file: str
    line: int
    severity: str    # LOW | MEDIUM | HIGH
    confidence: str
    test_id: str     # B105, B602, etc.
    test_name: str
    message: str


@dataclass
class DeadCodeItem:
    file: str
    line: int
    kind: str        # function | variable | import | attribute
    name: str
    confidence: int  # percent


@dataclass
class SyntaxErr:
    file: str
    line: int
    message: str


@dataclass
class TestRun:
    passed: int = 0
    failed: int = 0
    errors: int = 0
    output: str = ""
    success: bool = False


@dataclass
class ExecutionReport:
    """Everything agents read from. All real, nothing guessed."""
    cloned: bool = False
    repo_path: str = ""
    py_files: list[str] = field(default_factory=list)

    tools_installed: dict[str, bool] = field(default_factory=dict)

    syntax_errors:   list[SyntaxErr]        = field(default_factory=list)
    ruff:            list[RuffIssue]         = field(default_factory=list)
    mypy:            list[MypyIssue]         = field(default_factory=list)
    complexity:      list[ComplexityResult]  = field(default_factory=list)
    security:        list[SecurityIssue]     = field(default_factory=list)
    dead_code:       list[DeadCodeItem]      = field(default_factory=list)
    tests:           Optional[TestRun]       = None

    # Raw text output — sent verbatim to the LLM so it sees exactly what tools said
    ruff_raw:    str = ""
    mypy_raw:    str = ""
    radon_raw:   str = ""
    bandit_raw:  str = ""
    vulture_raw: str = ""

    elapsed: float = 0.0
    errors:  list[str] = field(default_factory=list)   # execution errors

    def tool_ok(self, name: str) -> bool:
        return self.tools_installed.get(name, False)

    def high_severity_security(self) -> list[SecurityIssue]:
        return [s for s in self.security if s.severity == "HIGH"]

    def complex_functions(self) -> list[ComplexityResult]:
        return [c for c in self.complexity if c.complexity >= 10]


# ── Executor ──────────────────────────────────────────────────────────────────

class PRExecutor:

    ANALYSIS_TOOLS = ["ruff", "mypy", "radon", "bandit", "vulture"]

    def __init__(self, github_token: Optional[str] = None):
        self.token = github_token

    def run(self, repo: str, pr_meta: dict) -> ExecutionReport:
        report  = ExecutionReport()
        tmpdir  = tempfile.mkdtemp(prefix="codesnip_")
        t0      = time.time()

        try:
            self._clone(repo, pr_meta, tmpdir, report)
            if not report.cloned:
                return report

            report.py_files = self._find_py(report.repo_path)
            logger.detail("Python files found", str(len(report.py_files)), "cyan")

            self._install_tools(report)
            self._install_project_deps(report)

            self._syntax_check(report)
            self._run_ruff(report)
            self._run_mypy(report)
            self._run_radon(report)
            self._run_bandit(report)
            self._run_vulture(report)
            self._run_tests(report)

        except Exception as exc:
            report.errors.append(f"Executor error: {exc}")
            logger.fail("executor", str(exc))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            report.elapsed = time.time() - t0
            logger.detail("Total execution time", f"{report.elapsed:.1f}s", "dim")

        return report

    # ── Clone ─────────────────────────────────────────────────────────────────

    def _clone(self, repo: str, pr_meta: dict, tmpdir: str, report: ExecutionReport):
        dest   = os.path.join(tmpdir, "repo")
        branch = pr_meta.get("head", "")
        token  = self.token or ""
        auth   = f"{token}@" if token else ""
        url    = f"https://{auth}github.com/{repo}.git"

        logger.start("clone", f"git clone {repo}  (branch: {branch or 'default'})…")

        def _clone_cmd(extra: list) -> bool:
            r = subprocess.run(
                ["git", "clone", "--depth", "1"] + extra + [url, dest],
                capture_output=True, text=True, timeout=120,
            )
            return r.returncode == 0

        ok = (_clone_cmd(["--branch", branch]) if branch else False) or _clone_cmd([])

        if not ok:
            logger.fail("clone", "Could not clone repo — check token/network")
            report.errors.append("Clone failed")
            return

        report.cloned    = True
        report.repo_path = dest
        logger.ok("clone", f"Cloned to {dest}")

    # ── Install tools ─────────────────────────────────────────────────────────

    def _install_tools(self, report: ExecutionReport):
        logger.start("tools", "Installing analysis tools (ruff mypy radon bandit vulture)…")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "ruff", "mypy", "radon", "bandit", "vulture",
             "-q", "--break-system-packages"],
            capture_output=True, text=True, timeout=180,
        )
        # Check what actually ended up available
        for tool in self.ANALYSIS_TOOLS:
            probe = subprocess.run(
                [sys.executable, "-m", tool, "--version"],
                capture_output=True, text=True, timeout=10,
            )
            report.tools_installed[tool] = (probe.returncode == 0)

        ok  = [t for t, v in report.tools_installed.items() if v]
        bad = [t for t, v in report.tools_installed.items() if not v]
        if ok:  logger.ok("tools",  f"Ready: {', '.join(ok)}")
        if bad: logger.warn("tools", f"Unavailable (no network?): {', '.join(bad)}")

    # ── Project deps ──────────────────────────────────────────────────────────

    def _install_project_deps(self, report: ExecutionReport):
        path = report.repo_path
        # Try in priority order
        for cmd in [
            [sys.executable, "-m", "pip", "install", "-e", ".",
             "--quiet", "--break-system-packages"],
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
             "--quiet", "--break-system-packages"],
        ]:
            marker = "pyproject.toml/setup.py" if cmd[4] == "." else "requirements.txt"
            check  = os.path.join(path, "." if cmd[4] == "." else "requirements.txt")
            # For ".", check if any installable file exists
            has = any(
                os.path.exists(os.path.join(path, f))
                for f in ["pyproject.toml", "setup.py", "setup.cfg"]
            ) if cmd[4] == "." else os.path.exists(os.path.join(path, "requirements.txt"))
            if not has:
                continue
            logger.start("deps", f"pip install from {marker}…")
            r = subprocess.run(cmd, cwd=path, capture_output=True, text=True, timeout=180)
            if r.returncode == 0:
                logger.ok("deps", "Project dependencies installed")
                return
        logger.info("No installable project deps found — continuing")

    # ── Syntax ────────────────────────────────────────────────────────────────

    def _syntax_check(self, report: ExecutionReport):
        logger.start("syntax", f"AST syntax check ({len(report.py_files)} files)…")
        errs = 0
        for rel in report.py_files:
            full = os.path.join(report.repo_path, rel)
            try:
                src = open(full, encoding="utf-8", errors="replace").read()
                ast.parse(src)
            except SyntaxError as e:
                report.syntax_errors.append(SyntaxErr(rel, e.lineno or 0, str(e.msg)))
                errs += 1
        if errs:
            logger.fail("syntax", f"{errs} syntax error(s)")
        else:
            logger.ok("syntax", "All files parse cleanly")

    # ── Ruff ─────────────────────────────────────────────────────────────────

    def _run_ruff(self, report: ExecutionReport):
        if not report.tool_ok("ruff"):
            logger.warn("ruff", "Skipped — not installed")
            return
        logger.start("ruff", "ruff check . --select ALL…")
        r = subprocess.run(
            [sys.executable, "-m", "ruff", "check", ".",
             "--output-format", "json", "--no-cache",
             "--select", "E,W,F,B,C,S,UP,N",
             "--ignore", "D,ANN,ERA,T20"],
            cwd=report.repo_path,
            capture_output=True, text=True, timeout=60,
        )
        raw = r.stdout.strip()
        report.ruff_raw = raw
        try:
            for item in json.loads(raw or "[]"):
                loc = item.get("location", {})
                report.ruff.append(RuffIssue(
                    file    = item.get("filename", "").replace(report.repo_path + os.sep, ""),
                    line    = loc.get("row", 0),
                    col     = loc.get("column", 0),
                    code    = item.get("code", ""),
                    message = item.get("message", ""),
                ))
        except json.JSONDecodeError:
            report.ruff_raw = r.stdout + r.stderr
        logger.ok("ruff", f"{len(report.ruff)} issue(s)")

    # ── Mypy ─────────────────────────────────────────────────────────────────

    def _run_mypy(self, report: ExecutionReport):
        if not report.tool_ok("mypy"):
            logger.warn("mypy", "Skipped — not installed")
            return
        logger.start("mypy", "mypy . --ignore-missing-imports…")
        r = subprocess.run(
            [sys.executable, "-m", "mypy", ".",
             "--ignore-missing-imports",
             "--no-error-summary",
             "--show-error-codes",
             "--no-color-output"],
            cwd=report.repo_path,
            capture_output=True, text=True, timeout=120,
        )
        report.mypy_raw = r.stdout
        pattern = re.compile(r"^(.+?):(\d+):\s*(error|warning|note):\s*(.+?)(?:\s*\[(.+?)\])?$")
        for line in r.stdout.splitlines():
            m = pattern.match(line)
            if m:
                report.mypy.append(MypyIssue(
                    file     = m.group(1).replace(report.repo_path + os.sep, ""),
                    line     = int(m.group(2)),
                    severity = m.group(3),
                    code     = m.group(5) or "",
                    message  = m.group(4),
                ))
        logger.ok("mypy", f"{len(report.mypy)} issue(s)")

    # ── Radon ─────────────────────────────────────────────────────────────────

    def _run_radon(self, report: ExecutionReport):
        if not report.tool_ok("radon"):
            logger.warn("radon", "Skipped — not installed")
            return
        logger.start("radon", "radon cc . -s -j (cyclomatic complexity)…")
        r = subprocess.run(
            [sys.executable, "-m", "radon", "cc", ".", "-s", "-j"],
            cwd=report.repo_path,
            capture_output=True, text=True, timeout=60,
        )
        report.radon_raw = r.stdout
        try:
            for filepath, items in json.loads(r.stdout or "{}").items():
                rel = filepath.replace(report.repo_path + os.sep, "")
                for fn in items:
                    report.complexity.append(ComplexityResult(
                        file       = rel,
                        function   = fn.get("name", ""),
                        line       = fn.get("lineno", 0),
                        complexity = fn.get("complexity", 0),
                        rank       = fn.get("rank", "A"),
                    ))
        except (json.JSONDecodeError, AttributeError):
            pass
        complex_count = len(report.complex_functions())
        logger.ok("radon", f"{len(report.complexity)} functions — {complex_count} with CC≥10")

    # ── Bandit ────────────────────────────────────────────────────────────────

    def _run_bandit(self, report: ExecutionReport):
        if not report.tool_ok("bandit"):
            logger.warn("bandit", "Skipped — not installed")
            return
        logger.start("bandit", "bandit -r . (security scan)…")
        r = subprocess.run(
            [sys.executable, "-m", "bandit", "-r", ".",
             "-f", "json", "-q", "--skip", "B101"],
            cwd=report.repo_path,
            capture_output=True, text=True, timeout=60,
        )
        report.bandit_raw = r.stdout
        try:
            for issue in json.loads(r.stdout or "{}").get("results", []):
                report.security.append(SecurityIssue(
                    file       = issue.get("filename", "").replace(report.repo_path + os.sep, ""),
                    line       = issue.get("line_number", 0),
                    severity   = issue.get("issue_severity", ""),
                    confidence = issue.get("issue_confidence", ""),
                    test_id    = issue.get("test_id", ""),
                    test_name  = issue.get("test_name", ""),
                    message    = issue.get("issue_text", ""),
                ))
        except json.JSONDecodeError:
            pass
        high = len(report.high_severity_security())
        logger.ok("bandit", f"{len(report.security)} issue(s) — {high} HIGH")

    # ── Vulture ───────────────────────────────────────────────────────────────

    def _run_vulture(self, report: ExecutionReport):
        if not report.tool_ok("vulture"):
            logger.warn("vulture", "Skipped — not installed")
            return
        logger.start("vulture", "vulture . --min-confidence 80 (dead code)…")
        r = subprocess.run(
            [sys.executable, "-m", "vulture", ".", "--min-confidence", "80"],
            cwd=report.repo_path,
            capture_output=True, text=True, timeout=60,
        )
        report.vulture_raw = r.stdout
        pattern = re.compile(r"^(.+?):(\d+):\s+unused\s+(\w+)\s+'(.+?)'\s+\((\d+)%")
        for line in r.stdout.splitlines():
            m = pattern.match(line)
            if m:
                report.dead_code.append(DeadCodeItem(
                    file       = m.group(1).replace(report.repo_path + os.sep, ""),
                    line       = int(m.group(2)),
                    kind       = m.group(3),
                    name       = m.group(4),
                    confidence = int(m.group(5)),
                ))
        logger.ok("vulture", f"{len(report.dead_code)} unused item(s)")

    # ── Tests ─────────────────────────────────────────────────────────────────

    def _run_tests(self, report: ExecutionReport):
        path = report.repo_path
        has_tests = (
            any(os.path.isdir(os.path.join(path, d)) for d in ["tests", "test"])
            or bool(list(Path(path).glob("test_*.py")))
            or bool(list(Path(path).glob("*_test.py")))
        )
        if not has_tests:
            logger.info("No test directory found — skipping")
            return

        logger.start("tests", "Running tests (pytest)…")
        for cmd in [
            [sys.executable, "-m", "pytest", "--tb=short", "-q", "--no-header"],
            [sys.executable, "-m", "unittest", "discover", "-q"],
        ]:
            r = subprocess.run(
                cmd, cwd=path, capture_output=True, text=True, timeout=120,
            )
            if r.returncode in (0, 1):
                out = (r.stdout + r.stderr)[:4000]
                tr  = TestRun(output=out, success=(r.returncode == 0))
                for pat, attr in [
                    (r"(\d+) passed", "passed"),
                    (r"(\d+) failed", "failed"),
                    (r"(\d+) error",  "errors"),
                ]:
                    m = re.search(pat, out)
                    if m: setattr(tr, attr, int(m.group(1)))
                report.tests = tr
                status = "all passed ✓" if tr.success else f"{tr.failed} failed"
                logger.ok("tests", f"{tr.passed} passed, {tr.failed} failed — {status}")
                return
        logger.warn("tests", "Could not collect tests")

    # ── Utils ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _find_py(root: str) -> list[str]:
        SKIP = {".git","venv",".venv","env","dist","build","__pycache__",
                ".tox",".mypy_cache",".ruff_cache","node_modules"}
        out = []
        for r, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP]
            for f in files:
                if f.endswith(".py"):
                    out.append(os.path.relpath(os.path.join(r, f), root))
        return out
