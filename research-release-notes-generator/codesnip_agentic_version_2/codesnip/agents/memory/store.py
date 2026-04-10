"""
memory/store.py
===============
AgentMemory — SQLite-backed persistent memory for every agent.

What agents remember across PRs:
  - Every finding ever produced (by repo, file, pattern)
  - Whether a prediction turned out to be correct (feedback loop)
  - Pattern frequency: "this repo uses bare except in 4 PRs"
  - Code patterns that previously caused crashes/leaks
  - Per-file history: which files are most error-prone

Schema:
  findings     — every finding ever recorded
  patterns     — recurring patterns grouped by repo + signature
  predictions  — things an agent predicted; later marked correct/wrong
  executions   — raw execution results per PR (for trend analysis)
  feedback     — human/automated feedback on agent accuracy
"""
from __future__ import annotations

import json
import sqlite3
import hashlib
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


MEMORY_DIR = Path.home() / ".codesnip" / "memory"
DB_PATH    = MEMORY_DIR / "agent_memory.db"


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class MemoryRecord:
    id: Optional[int]
    repo: str
    pr_number: int
    agent: str
    category: str          # finding category
    file: str
    line: int
    severity: str
    pattern_key: str       # normalised signature used for dedup + learning
    title: str
    detail: str
    tool_source: str       # ruff | mypy | bandit | radon | ast | runtime
    confirmed: bool        # was this finding confirmed accurate?
    ts: float              # unix timestamp


@dataclass
class PatternRecord:
    pattern_key: str
    repo: str
    agent: str
    occurrences: int       # how many times seen in this repo
    first_seen_pr: int
    last_seen_pr: int
    confirmed_count: int   # times marked as real issue
    false_positive_count: int
    example_detail: str


@dataclass
class PredictionRecord:
    id: Optional[int]
    repo: str
    pr_number: int
    agent: str
    prediction_type: str   # memory_leak | crash | performance_regression | security
    description: str
    confidence: float      # 0.0 – 1.0
    evidence: str          # JSON list of supporting pattern_keys
    outcome: Optional[str] = None   # confirmed | false_positive | unknown
    ts: float = 0.0


# ── Store ──────────────────────────────────────────────────────────────────────

class AgentMemory:

    def __init__(self):
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _create_schema(self):
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS findings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            repo            TEXT    NOT NULL,
            pr_number       INTEGER NOT NULL,
            agent           TEXT    NOT NULL,
            category        TEXT    NOT NULL,
            file            TEXT    NOT NULL DEFAULT '',
            line            INTEGER NOT NULL DEFAULT 0,
            severity        TEXT    NOT NULL,
            pattern_key     TEXT    NOT NULL,
            title           TEXT    NOT NULL,
            detail          TEXT    NOT NULL DEFAULT '',
            tool_source     TEXT    NOT NULL DEFAULT 'unknown',
            confirmed       INTEGER NOT NULL DEFAULT 1,
            ts              REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS patterns (
            pattern_key          TEXT NOT NULL,
            repo                 TEXT NOT NULL,
            agent                TEXT NOT NULL,
            occurrences          INTEGER NOT NULL DEFAULT 1,
            first_seen_pr        INTEGER NOT NULL,
            last_seen_pr         INTEGER NOT NULL,
            confirmed_count      INTEGER NOT NULL DEFAULT 0,
            false_positive_count INTEGER NOT NULL DEFAULT 0,
            example_detail       TEXT    NOT NULL DEFAULT '',
            PRIMARY KEY (pattern_key, repo)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            repo             TEXT    NOT NULL,
            pr_number        INTEGER NOT NULL,
            agent            TEXT    NOT NULL,
            prediction_type  TEXT    NOT NULL,
            description      TEXT    NOT NULL,
            confidence       REAL    NOT NULL,
            evidence         TEXT    NOT NULL DEFAULT '[]',
            outcome          TEXT,
            ts               REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS executions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            repo         TEXT    NOT NULL,
            pr_number    INTEGER NOT NULL,
            tool         TEXT    NOT NULL,
            raw_output   TEXT    NOT NULL,
            issue_count  INTEGER NOT NULL DEFAULT 0,
            ts           REAL    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_findings_repo    ON findings(repo, agent);
        CREATE INDEX IF NOT EXISTS idx_findings_pattern ON findings(pattern_key);
        CREATE INDEX IF NOT EXISTS idx_patterns_repo    ON patterns(repo, agent);
        CREATE INDEX IF NOT EXISTS idx_pred_repo        ON predictions(repo, pr_number);
        """)
        self._conn.commit()

    # ── Write ──────────────────────────────────────────────────────────────────

    def record_finding(
        self,
        repo: str, pr_number: int, agent: str, category: str,
        file: str, line: int, severity: str, title: str,
        detail: str, tool_source: str = "unknown",
    ) -> str:
        """Store a finding and update the pattern frequency table. Returns pattern_key."""
        key = self._make_key(repo, agent, title)
        now = time.time()

        self._conn.execute(
            """INSERT INTO findings
               (repo,pr_number,agent,category,file,line,severity,pattern_key,title,detail,tool_source,confirmed,ts)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,1,?)""",
            (repo, pr_number, agent, category, file, line, severity, key, title, detail, tool_source, now),
        )

        # Upsert pattern
        existing = self._conn.execute(
            "SELECT * FROM patterns WHERE pattern_key=? AND repo=?", (key, repo)
        ).fetchone()

        if existing:
            self._conn.execute(
                """UPDATE patterns
                   SET occurrences=occurrences+1, last_seen_pr=?
                   WHERE pattern_key=? AND repo=?""",
                (pr_number, key, repo),
            )
        else:
            self._conn.execute(
                """INSERT INTO patterns
                   (pattern_key,repo,agent,occurrences,first_seen_pr,last_seen_pr,
                    confirmed_count,false_positive_count,example_detail)
                   VALUES (?,?,?,1,?,?,0,0,?)""",
                (key, repo, agent, pr_number, pr_number, detail[:200]),
            )

        self._conn.commit()
        return key

    def record_prediction(
        self,
        repo: str, pr_number: int, agent: str,
        prediction_type: str, description: str,
        confidence: float, evidence: list[str],
    ) -> int:
        now = time.time()
        cur = self._conn.execute(
            """INSERT INTO predictions
               (repo,pr_number,agent,prediction_type,description,confidence,evidence,ts)
               VALUES (?,?,?,?,?,?,?,?)""",
            (repo, pr_number, agent, prediction_type, description,
             confidence, json.dumps(evidence), now),
        )
        self._conn.commit()
        return cur.lastrowid

    def record_execution(
        self, repo: str, pr_number: int, tool: str, raw_output: str, issue_count: int,
    ):
        self._conn.execute(
            "INSERT INTO executions (repo,pr_number,tool,raw_output,issue_count,ts) VALUES (?,?,?,?,?,?)",
            (repo, pr_number, tool, raw_output[:8000], issue_count, time.time()),
        )
        self._conn.commit()

    # ── Read / Query ───────────────────────────────────────────────────────────

    def get_recurring_patterns(self, repo: str, agent: str, min_occurrences: int = 2) -> list[dict]:
        """Patterns seen ≥N times in this repo — used for intelligent prediction."""
        rows = self._conn.execute(
            """SELECT * FROM patterns
               WHERE repo=? AND agent=? AND occurrences>=?
               ORDER BY occurrences DESC""",
            (repo, agent, min_occurrences),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_file_error_history(self, repo: str, file: str, limit: int = 20) -> list[dict]:
        """All past findings in a specific file — tells agents which files are error-prone."""
        rows = self._conn.execute(
            """SELECT * FROM findings
               WHERE repo=? AND file LIKE ?
               ORDER BY ts DESC LIMIT ?""",
            (repo, f"%{file}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_pattern_confidence(self, repo: str, pattern_key: str) -> float:
        """
        0.0 – 1.0: how reliable is this pattern in this repo?
        Based on confirmed / (confirmed + false_positive).
        """
        row = self._conn.execute(
            "SELECT * FROM patterns WHERE pattern_key=? AND repo=?",
            (pattern_key, repo),
        ).fetchone()
        if not row:
            return 0.5   # unknown — neutral prior
        confirmed = row["confirmed_count"]
        fp        = row["false_positive_count"]
        total     = confirmed + fp
        if total == 0:
            return 0.5
        return confirmed / total

    def get_repo_summary(self, repo: str) -> dict:
        """High-level stats for a repo — fed into agent prompts as context."""
        total_prs = self._conn.execute(
            "SELECT COUNT(DISTINCT pr_number) FROM findings WHERE repo=?", (repo,)
        ).fetchone()[0]

        top_patterns = self._conn.execute(
            """SELECT agent, example_detail, occurrences FROM patterns
               WHERE repo=? ORDER BY occurrences DESC LIMIT 10""",
            (repo,),
        ).fetchall()

        error_prone_files = self._conn.execute(
            """SELECT file, COUNT(*) as cnt FROM findings
               WHERE repo=? AND severity IN ('critical','high')
               GROUP BY file ORDER BY cnt DESC LIMIT 5""",
            (repo,),
        ).fetchall()

        recent_predictions = self._conn.execute(
            """SELECT prediction_type, description, confidence, outcome
               FROM predictions WHERE repo=? ORDER BY ts DESC LIMIT 5""",
            (repo,),
        ).fetchall()

        return {
            "total_prs_analysed": total_prs,
            "top_patterns": [dict(r) for r in top_patterns],
            "error_prone_files": [dict(r) for r in error_prone_files],
            "recent_predictions": [dict(r) for r in recent_predictions],
        }

    def get_similar_past_findings(
        self, repo: str, title: str, limit: int = 5,
    ) -> list[dict]:
        """Finds past findings with similar titles — for cross-PR learning."""
        # Simple keyword overlap (no embeddings needed for this use case)
        words = [w for w in title.lower().split() if len(w) > 4]
        if not words:
            return []
        like_clauses = " OR ".join(["title LIKE ?" for _ in words])
        params = [f"%{w}%" for w in words] + [repo, limit]
        rows = self._conn.execute(
            f"""SELECT * FROM findings
                WHERE ({like_clauses}) AND repo=?
                ORDER BY ts DESC LIMIT ?""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_prediction_outcome(self, prediction_id: int, outcome: str):
        """outcome = 'confirmed' | 'false_positive'. Updates pattern confidence."""
        self._conn.execute(
            "UPDATE predictions SET outcome=? WHERE id=?",
            (outcome, prediction_id),
        )
        # Also update pattern confidence for each evidence key
        pred = self._conn.execute(
            "SELECT * FROM predictions WHERE id=?", (prediction_id,)
        ).fetchone()
        if pred:
            evidence_keys = json.loads(pred["evidence"] or "[]")
            col = "confirmed_count" if outcome == "confirmed" else "false_positive_count"
            for key in evidence_keys:
                self._conn.execute(
                    f"UPDATE patterns SET {col}={col}+1 WHERE pattern_key=? AND repo=?",
                    (key, pred["repo"]),
                )
        self._conn.commit()

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_key(repo: str, agent: str, title: str) -> str:
        """
        Normalised pattern key — same error in same repo → same key.
        Strips line numbers and variable names to detect recurring patterns.
        """
        import re
        normalised = re.sub(r"`[^`]+`", "`X`", title.lower())   # `foo` → `X`
        normalised = re.sub(r"\d+", "N", normalised)              # numbers → N
        normalised = re.sub(r"\s+", " ", normalised).strip()
        raw = f"{repo}::{agent}::{normalised}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# Singleton accessor
_instance: Optional[AgentMemory] = None

def get_memory() -> AgentMemory:
    global _instance
    if _instance is None:
        _instance = AgentMemory()
    return _instance
