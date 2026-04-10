from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PRContext:
    repo: str
    pr_number: int
    diff: str
    commits: str
    meta: dict
    check_report: object       # CheckReport  — static AST scan of diff
    diff_files: dict           # filename → {ordered: [(+/-/ctx, line), ...]}
    execution_report: object   # ExecutionReport — ruff/mypy/radon/bandit/vulture
    sandbox_report: object     # SandboxReport  — actual runtime execution


@dataclass
class Finding:
    severity: str   # critical | high | medium | low | info
    title: str
    detail: str
    file: Optional[str] = None
    line: Optional[int] = None


@dataclass
class AgentResult:
    category: str
    emoji: str
    findings: list[Finding] = field(default_factory=list)
    llm_narrative: str = ""
    skipped: bool = False
    error: Optional[str] = None

    def has_content(self) -> bool:
        return bool(self.findings or self.llm_narrative.strip())
