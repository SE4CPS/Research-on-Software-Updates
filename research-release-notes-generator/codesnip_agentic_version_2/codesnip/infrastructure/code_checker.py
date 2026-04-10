"""
code_checker.py
===============
Runs real static analysis on the code changed in a PR diff.

Pipeline per Python file:
  1. Parse unified diff  →  reconstruct new file content
  2. ast.parse           →  syntax check
  3. AST walk            →  unused imports, undefined names (light), star imports,
                            duplicate imports, bare `except`, magic numbers,
                            function length, nesting depth
  4. PEP-8 basics        →  line length, trailing whitespace (from raw diff lines)

Results are:
  - Printed live via logger (verbose)
  - Returned as CheckReport  →  injected into the LLM prompt as hard facts
"""

import ast
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from codesnip.shared import logger


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FileResult:
    filename: str
    language: str
    added: int
    removed: int
    syntax_ok: Optional[bool] = None
    syntax_error: Optional[str] = None
    issues: list[str] = field(default_factory=list)   # lint / quality
    warnings: list[str] = field(default_factory=list) # style / minor


@dataclass
class CheckReport:
    files: list[FileResult] = field(default_factory=list)
    total_added: int = 0
    total_removed: int = 0
    py_checked: int = 0
    syntax_errors: int = 0
    total_issues: int = 0
    all_ok: bool = True

    def as_prompt_text(self) -> str:
        """Compact but complete summary for injection into the LLM prompt."""
        lines = [
            "=== AUTOMATED STATIC ANALYSIS RESULTS ===",
            f"Files changed: {len(self.files)}  |  "
            f"+{self.total_added} lines added  |  -{self.total_removed} lines removed",
            f"Python files checked: {self.py_checked}  |  "
            f"Syntax errors: {self.syntax_errors}  |  "
            f"Issues found: {self.total_issues}",
            "",
        ]
        for fr in self.files:
            lines.append(f"── {fr.filename}  ({fr.language})  +{fr.added}/-{fr.removed}")
            if fr.syntax_ok is True:
                lines.append("   [SYNTAX] OK")
            elif fr.syntax_ok is False:
                lines.append(f"   [SYNTAX ERROR] {fr.syntax_error}")
            for iss in fr.issues:
                lines.append(f"   [ISSUE] {iss}")
            for w in fr.warnings:
                lines.append(f"   [WARN]  {w}")
            if not fr.issues and not fr.warnings and fr.syntax_ok is True:
                lines.append("   No issues found.")
            lines.append("")
        lines.append(
            "NOTE: These findings are factual — base your Linting, Code Quality, "
            "and Formatting sections on them."
        )
        return "\n".join(lines)


# ── Language detection ────────────────────────────────────────────────────────

_EXT_LANG = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
    ".jsx": "JSX", ".tsx": "TSX", ".java": "Java", ".go": "Go",
    ".rb": "Ruby", ".rs": "Rust", ".cpp": "C++", ".c": "C",
    ".cs": "C#", ".php": "PHP", ".swift": "Swift", ".kt": "Kotlin",
    ".sh": "Shell", ".bash": "Shell", ".yml": "YAML", ".yaml": "YAML",
    ".json": "JSON", ".toml": "TOML", ".md": "Markdown",
    ".html": "HTML", ".css": "CSS", ".sql": "SQL",
}


def _lang(filename: str) -> str:
    return _EXT_LANG.get(os.path.splitext(filename)[1].lower(), "Unknown")


# ── Diff parser ───────────────────────────────────────────────────────────────

def _parse_diff(diff: str) -> dict[str, dict]:
    """
    Returns { filename: { added: [line,...], removed: [line,...], context: [line,...] } }
    Reconstructing file content = context + added lines (order-preserved).
    """
    files: dict[str, dict] = {}
    cur: Optional[str] = None
    # Track insertion order for reconstruction
    cur_lines: list[tuple[str, str]] = []  # (type, content)  type = +/-/ctx

    def flush():
        if cur and cur_lines:
            a = [c for t, c in cur_lines if t == "+"]
            r = [c for t, c in cur_lines if t == "-"]
            ctx = [c for t, c in cur_lines if t == "ctx"]
            files[cur]["added"] = a
            files[cur]["removed"] = r
            files[cur]["context"] = ctx
            files[cur]["ordered"] = cur_lines  # kept for PEP-8 raw checks

    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            flush()
            fname = line[6:]
            if fname == "/dev/null":
                cur = None
                cur_lines = []
                continue
            cur = fname
            if cur not in files:
                files[cur] = {}
            cur_lines = []

        elif cur is None:
            continue
        elif line.startswith("+") and not line.startswith("+++"):
            cur_lines.append(("+", line[1:]))
        elif line.startswith("-") and not line.startswith("---"):
            cur_lines.append(("-", line[1:]))
        elif line.startswith(" "):
            cur_lines.append(("ctx", line[1:]))

    flush()
    return files


def _reconstruct(data: dict) -> str:
    """Reconstruct the 'new' version of a file from ordered diff lines."""
    lines = []
    for t, c in data.get("ordered", []):
        if t in ("+", "ctx"):
            lines.append(c)
    return "\n".join(lines)


# ── AST-based checks ──────────────────────────────────────────────────────────

def _syntax_check(source: str) -> tuple[bool, Optional[str]]:
    try:
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"line {e.lineno}: {e.msg}"


def _ast_checks(source: str, filename: str) -> tuple[list[str], list[str]]:
    """
    Returns (issues, warnings).
    issues = definite problems; warnings = style/minor concerns.
    """
    issues: list[str] = []
    warnings: list[str] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return issues, warnings

    # ── Collect all names defined / used ─────────────────────────────────────
    imported_names: set[str] = set()
    imported_modules: set[str] = set()
    used_names: set[str] = set()
    star_imports: list[str] = []
    seen_import_keys: set[str] = set()

    for node in ast.walk(tree):

        # ── Imports ──────────────────────────────────────────────────────────
        if isinstance(node, ast.Import):
            key = ast.dump(node)
            if key in seen_import_keys:
                issues.append(f"Duplicate import at line {node.lineno}")
            seen_import_keys.add(key)
            for alias in node.names:
                n = alias.asname or alias.name.split(".")[0]
                imported_names.add(n)
                imported_modules.add(alias.name)

        elif isinstance(node, ast.ImportFrom):
            key = ast.dump(node)
            if key in seen_import_keys:
                issues.append(f"Duplicate import-from at line {node.lineno}")
            seen_import_keys.add(key)
            for alias in node.names:
                if alias.name == "*":
                    star_imports.append(f"from {node.module} import *  (line {node.lineno})")
                else:
                    n = alias.asname or alias.name
                    imported_names.add(n)

        # ── Name usage ───────────────────────────────────────────────────────
        elif isinstance(node, ast.Name):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                used_names.add(node.value.id)

        # ── Bare except ──────────────────────────────────────────────────────
        elif isinstance(node, ast.ExceptHandler):
            if node.type is None:
                issues.append(
                    f"Bare `except:` at line {node.lineno} — "
                    "catches all exceptions including KeyboardInterrupt; use `except Exception:`"
                )

        # ── Function length & missing return type ─────────────────────────────
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", None)
            length = (end - node.lineno) if end else 0
            if length > 50:
                warnings.append(
                    f"Function `{node.name}` is {length} lines long (line {node.lineno})"
                    " — consider splitting into smaller functions"
                )
            # public function missing return annotation
            if not node.name.startswith("_") and node.returns is None:
                warnings.append(
                    f"Public function `{node.name}` (line {node.lineno}) "
                    "has no return type annotation"
                )
            # nested function depth check
            _check_nesting(node, 0, issues)

        # ── Magic numbers ─────────────────────────────────────────────────────
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                if node.value not in (0, 1, -1, 2, True, False, 100):
                    parent_assign = False  # heuristic: skip if it's a default arg
                    warnings.append(
                        f"Magic number `{node.value}` at line {getattr(node, 'lineno', '?')}"
                        " — consider extracting to a named constant"
                    )

    # ── Star imports ─────────────────────────────────────────────────────────
    for si in star_imports:
        issues.append(f"Star import: {si} — avoid wildcard imports")

    # ── Unused imports (light heuristic) ─────────────────────────────────────
    # Only flag if imported name never appears in any Name node in the file
    for name in imported_names:
        if name not in used_names:
            issues.append(f"Possibly unused import: `{name}` (not referenced in this file)")

    # Deduplicate magic number warnings (can be very noisy)
    magic = [w for w in warnings if w.startswith("Magic number")]
    other_warn = [w for w in warnings if not w.startswith("Magic number")]
    if len(magic) > 3:
        warnings = other_warn + [f"Multiple magic numbers found ({len(magic)} occurrences) — extract to constants"]
    else:
        warnings = other_warn + magic

    return issues, warnings


def _check_nesting(node: ast.AST, depth: int, issues: list[str]) -> None:
    """Recursively check if function body has > 4 levels of nesting."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            if depth >= 4:
                issues.append(
                    f"Deep nesting (>{depth} levels) at line {getattr(child, 'lineno', '?')}"
                    " — consider early returns or helper functions"
                )
            _check_nesting(child, depth + 1, issues)


# ── PEP-8 raw-line checks (from the diff lines directly) ─────────────────────

def _pep8_checks(data: dict, filename: str) -> list[str]:
    """Check added lines (+) only for line length and trailing whitespace."""
    warnings: list[str] = []
    long_lines = 0
    trailing_ws = 0

    for t, line in data.get("ordered", []):
        if t != "+":
            continue
        if len(line) > 88:
            long_lines += 1
        if line != line.rstrip():
            trailing_ws += 1

    if long_lines:
        warnings.append(f"{long_lines} added line(s) exceed 88 characters (PEP-8 limit)")
    if trailing_ws:
        warnings.append(f"{trailing_ws} added line(s) have trailing whitespace")

    # Check blank lines between top-level defs (simplified)
    src = "\n".join(c for t, c in data.get("ordered", []) if t in ("+", "ctx"))
    prev_def = False
    for lineno, raw in enumerate(src.splitlines(), 1):
        stripped = raw.strip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            if prev_def:
                warnings.append(
                    f"Missing blank line(s) before definition at line ~{lineno} "
                    "(PEP-8 requires 2 blank lines between top-level definitions)"
                )
            prev_def = True
        elif stripped == "":
            prev_def = False

    return warnings


# ── Main entry ────────────────────────────────────────────────────────────────

def run(diff: str) -> CheckReport:
    """
    Parse diff, run all checks, log everything verbosely.
    Returns CheckReport for prompt injection.
    """
    report = CheckReport()

    logger.section("CODE QUALITY CHECKS")
    logger.start("parse", "Parsing unified diff to extract changed files…")

    parsed = _parse_diff(diff)
    if not parsed:
        logger.warn("parse", "No file changes found in diff")
        return report

    logger.ok("parse", f"Diff parsed — {len(parsed)} file(s) changed")
    logger.detail("Files", ", ".join(parsed.keys()), "cyan")

    for filename, data in parsed.items():
        lang = _lang(filename)
        added = data.get("added", [])
        removed = data.get("removed", [])
        ctx = data.get("context", [])

        fr = FileResult(
            filename=filename, language=lang,
            added=len(added), removed=len(removed),
        )
        report.total_added += len(added)
        report.total_removed += len(removed)

        logger.file_header(filename, lang, len(added), len(removed))

        if lang != "Python":
            logger.check_skip(f"Language={lang} — skipping AST analysis (Python only)")
            report.files.append(fr)
            continue

        report.py_checked += 1
        source = _reconstruct(data)

        if not source.strip():
            logger.check_skip("File appears empty after reconstruction — skipping")
            report.files.append(fr)
            continue

        # 1. Syntax
        logger.start(f"syn_{filename}", "  Syntax check (ast.parse)…")
        ok, err = _syntax_check(source)
        fr.syntax_ok = ok
        fr.syntax_error = err
        if ok:
            logger.check_ok("Syntax: PASS")
        else:
            logger.check_fail(f"Syntax: FAIL — {err}")
            report.syntax_errors += 1
            report.all_ok = False
            report.files.append(fr)
            continue   # no point running further checks on broken syntax

        # 2. AST checks
        logger.start(f"ast_{filename}", "  Running AST analysis (imports, complexity, quality)…")
        issues, warnings = _ast_checks(source, filename)
        fr.issues = issues
        fr.warnings = warnings

        if issues:
            logger.check_fail(f"{len(issues)} issue(s) found:")
            for iss in issues:
                logger.check_fail(f"  {iss}")
            report.total_issues += len(issues)
            report.all_ok = False
        else:
            logger.check_ok("AST analysis: no issues")

        if warnings:
            logger.check_warn(f"{len(warnings)} warning(s):")
            for w in warnings:
                logger.check_warn(f"  {w}")
        else:
            logger.check_ok("Code quality: all checks passed")

        # 3. PEP-8 raw line checks
        logger.start(f"pep_{filename}", "  PEP-8 line checks (length, whitespace, blank lines)…")
        pep_warns = _pep8_checks(data, filename)
        fr.warnings.extend(pep_warns)
        if pep_warns:
            for w in pep_warns:
                logger.check_warn(f"  {w}")
        else:
            logger.check_ok("PEP-8: no line-level issues")

        report.files.append(fr)

    logger.blank()
    logger.ok("checks", (
        f"Checks complete — {report.py_checked} Python file(s) analysed  |  "
        f"+{report.total_added}/-{report.total_removed} lines  |  "
        f"{report.syntax_errors} syntax error(s)  |  "
        f"{report.total_issues} issue(s)"
    ))

    return report
