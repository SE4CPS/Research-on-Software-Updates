"""
use_cases.py — Full 6-step pipeline.

  1. GitHub: PR metadata
  2. GitHub: diff + commits
  3. Parse diff → diff_files + static CheckReport
  4. PRExecutor: clone → ruff/mypy/radon/bandit/vulture/pytest
  5. CodeSandbox: execute changed functions in ALL 16 languages
  6. 8 Intelligent Agents + OllamaFormatter
"""
from __future__ import annotations

from codesnip.agents import PRContext
from codesnip.agents.runner import AgentRunner
from codesnip.agents.formatter import OllamaFormatter
from codesnip.agents.executor import PRExecutor
from codesnip.agents.sandbox.runner import CodeSandbox, detect_language, Language, LanguageExt
from codesnip.infrastructure.github_service import GitHubService
from codesnip.infrastructure.code_checker import run as run_checks, _parse_diff
from codesnip.infrastructure.config_service import get_github_token
from codesnip.shared import logger

# All supported extensions across all 16 languages
_SUPPORTED_EXTENSIONS = {
    '.py', '.js', '.mjs', '.cjs', '.ts', '.tsx',
    '.java', '.c', '.cpp', '.cc', '.cxx',
    '.pl', '.pm', '.sh', '.bash', '.sql', '.s', '.asm',
    '.rb', '.php', '.phtml', '.r', '.R',
    '.kt', '.kts', '.go', '.rs',
}


def _build_context(repo: str, pr_number: int) -> tuple[PRContext, dict]:
    gh = GitHubService()

    # Step 1 — Metadata
    logger.section("STEP 1 / 6  —  PR METADATA")
    logger.start("meta", f"Fetching PR #{pr_number} from {repo}…")
    meta = gh.get_pr_meta(repo, pr_number)
    logger.ok("meta", "Fetched")
    logger.detail("Title",  meta["title"],                      "white")
    logger.detail("Author", meta["author"],                     "cyan")
    logger.detail("Branch", f"{meta['head']} → {meta['base']}", "yellow")
    logger.pr_card(repo, pr_number, meta)

    # Step 2 — Diff + commits
    logger.section("STEP 2 / 6  —  DIFF & COMMITS")
    logger.start("diff", "Fetching unified diff…")
    diff = gh.get_pr_diff(repo, pr_number)
    logger.ok("diff", f"{len(diff):,} bytes  ·  {diff.count(chr(10))} lines")

    logger.start("commits", "Fetching commits…")
    commits = gh.get_pr_commits(repo, pr_number)
    n = commits.count("\n") + 1 if commits.strip() else 0
    logger.ok("commits", f"{n} commit(s)")
    for line in commits.splitlines():
        logger.info(line)

    # Step 3 — Parse diff
    logger.section("STEP 3 / 6  —  PARSE DIFF")
    diff_files   = _parse_diff(diff)
    check_report = run_checks(diff)
    logger.detail("Files changed",
        f"{len(diff_files)}  (+{check_report.total_added}/-{check_report.total_removed} lines)",
        "cyan")

    # Detect languages present in this PR
    supported_files = [f for f in diff_files
                       if any(f.endswith(ext) for ext in _SUPPORTED_EXTENSIONS)]
    from collections import Counter
    lang_counts = Counter()
    for f in supported_files:
        import os
        ext = os.path.splitext(f)[1].lower()
        lang_name = {
            '.py':'Python','.js':'JavaScript','.mjs':'JavaScript','.cjs':'JavaScript',
            '.ts':'TypeScript','.tsx':'TypeScript','.java':'Java',
            '.c':'C','.cpp':'C++','.cc':'C++','.cxx':'C++',
            '.pl':'Perl','.pm':'Perl','.sh':'Bash','.bash':'Bash',
            '.sql':'SQL','.s':'Assembly','.asm':'Assembly',
            '.rb':'Ruby','.php':'PHP','.phtml':'PHP',
            '.r':'R','.R':'R','.kt':'Kotlin','.kts':'Kotlin',
            '.go':'Go','.rs':'Rust',
        }.get(ext, 'Unknown')
        if lang_name != 'Unknown':
            lang_counts[lang_name] += 1

    if lang_counts:
        logger.detail("Languages", ", ".join(f"{l}({n})" for l,n in lang_counts.most_common()), "cyan")

    # Step 4 — Real tool execution
    logger.section("STEP 4 / 6  —  CLONE & STATIC TOOLS  (ruff · mypy · radon · bandit · vulture · pytest)")
    executor    = PRExecutor(github_token=get_github_token())
    exec_report = executor.run(repo, meta)

    if exec_report.cloned:
        logger.ok("executor", _exec_summary(exec_report))
    else:
        logger.warn("executor", "Clone failed — static tool results unavailable")

    # Step 5 — Sandbox: run ALL supported language files
    logger.section("STEP 5 / 6  —  SANDBOX EXECUTION  (16 languages)")

    from codesnip.agents.sandbox.runner import SandboxReport

    if exec_report.cloned and supported_files:
        sandbox     = CodeSandbox(timeout_seconds=30, max_memory_mb=512)
        sandbox_rpt = sandbox.run(exec_report.repo_path, supported_files)

        if sandbox_rpt.executed:
            langs_str = ", ".join(sandbox_rpt.languages_detected) or "unknown"
            logger.ok("sandbox",
                f"{len(sandbox_rpt.functions)} functions  |  "
                f"languages: {langs_str}  |  "
                f"{len(sandbox_rpt.leak_suspects)} leak(s)  |  "
                f"{len(sandbox_rpt.crash_functions)} crash(es)  |  "
                f"{len(sandbox_rpt.slow_functions)} slow")
        else:
            logger.warn("sandbox", f"No execution: {sandbox_rpt.error or 'unknown'}")
    else:
        sandbox_rpt = SandboxReport(
            error="Skipped — repo not cloned or no supported files changed"
        )
        logger.warn("sandbox", "Skipped")

    # Build PRContext
    ctx = PRContext(
        repo=repo, pr_number=pr_number,
        diff=diff, commits=commits, meta=meta,
        check_report=check_report,
        diff_files=diff_files,
        execution_report=exec_report,
        sandbox_report=sandbox_rpt,
    )
    return ctx, meta


def _exec_summary(r) -> str:
    parts = [
        f"ruff: {len(r.ruff)} issues",
        f"mypy: {len(r.mypy)} errors",
        f"CC≥10: {len(r.complex_functions())}",
        f"security: {len(r.security)} ({len(r.high_severity_security())} HIGH)",
        f"dead: {len(r.dead_code)}",
    ]
    if r.tests:
        parts.append(f"tests: {r.tests.passed}✓ {r.tests.failed}✗")
    return "  |  ".join(parts)


class AnalyzePRUseCase:
    def execute(self, repo: str, pr_number: int) -> str:
        ctx, meta = _build_context(repo, pr_number)
        logger.section("STEP 6 / 6  —  8 INTELLIGENT AGENTS")
        results = AgentRunner().run_all(ctx)
        output  = OllamaFormatter().format_release_notes(results, meta)
        logger.ok("pipeline", "Analysis complete ✓")
        return output


class ReleaseNotesUseCase:
    def execute(self, repo: str, pr_number: int) -> str:
        ctx, meta = _build_context(repo, pr_number)
        logger.section("STEP 6 / 6  —  8 INTELLIGENT AGENTS")
        results = AgentRunner().run_all(ctx)
        output  = OllamaFormatter().format_release_notes(results, meta)
        logger.ok("pipeline", "Release notes complete ✓")
        return output
