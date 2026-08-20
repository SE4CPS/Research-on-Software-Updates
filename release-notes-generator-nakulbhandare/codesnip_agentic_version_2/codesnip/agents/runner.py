"""
runner.py — AgentRunner with full progress visibility.
Shows each agent's progress, memory growth, and sandbox intelligence.
"""
from __future__ import annotations
import time

from codesnip.agents import AgentResult, PRContext
from codesnip.agents.categories import (
    FeaturesAgent, BugFixAgent, PerformanceAgent, LintingAgent,
    CodeQualityAgent, FormattingAgent, ArchitectureAgent, BreakingChangesAgent,
)
from codesnip.agents.memory.store import get_memory
from codesnip.shared import logger

ALL_AGENTS = [
    FeaturesAgent, BugFixAgent, PerformanceAgent, LintingAgent,
    CodeQualityAgent, FormattingAgent, ArchitectureAgent, BreakingChangesAgent,
]


class AgentRunner:
    def run_all(self, ctx: PRContext) -> list[AgentResult]:
        results: list[AgentResult] = []
        total   = len(ALL_AGENTS)

        # ── Pre-run summary ──────────────────────────────────────────────────
        mem     = get_memory()
        summary = mem.get_repo_summary(ctx.repo)
        past    = summary["total_prs_analysed"]

        logger.section(f"RUNNING {total} INTELLIGENT AGENTS")
        logger.info(f"")
        logger.info(f"  Repository:  {ctx.repo}  ·  PR #{ctx.pr_number}")
        logger.info(f"  Memory:      {past} prior PR(s) in database")

        sandbox = getattr(ctx, "sandbox_report", None)
        if sandbox and sandbox.executed:
            langs = ", ".join(sandbox.languages_detected) if sandbox.languages_detected else "unknown"
            logger.info(f"  Sandbox:     ✅ {len(sandbox.functions)} functions executed "
                        f"across [{langs}]")
            if sandbox.crash_functions:
                logger.info(f"               💥 {len(sandbox.crash_functions)} crash(es) to investigate")
            if sandbox.leak_suspects:
                logger.info(f"               💧 {len(sandbox.leak_suspects)} memory leak suspect(s)")
            if sandbox.slow_functions:
                logger.info(f"               🐌 {len(sandbox.slow_functions)} slow function(s)")
        elif sandbox:
            logger.info(f"  Sandbox:     ⚠ not executed ({sandbox.error or 'skipped'})")
        else:
            logger.info(f"  Sandbox:     — not available")
        logger.info(f"")

        t_start = time.time()

        for i, AgentClass in enumerate(ALL_AGENTS, 1):
            agent = AgentClass()
            logger.info(f"")
            logger.info(f"  ══════════════════════════════════════════════")
            logger.info(f"  [{i}/{total}]  {agent.EMOJI}  {agent.CATEGORY}")
            logger.info(f"  ══════════════════════════════════════════════")

            result = agent.run(ctx)
            results.append(result)

        # ── Post-run intelligence summary ────────────────────────────────────
        total_elapsed = time.time() - t_start
        total_findings  = sum(len(r.findings) for r in results)
        total_predicted = sum(1 for r in results if not r.skipped)

        logger.info(f"")
        logger.section("INTELLIGENCE SUMMARY")
        logger.info(f"")
        logger.info(f"  Agents completed:   {total}")
        logger.info(f"  Total findings:     {total_findings}")
        logger.info(f"  Agents with output: {total_predicted}")
        logger.info(f"  Total time:         {total_elapsed:.1f}s")
        logger.info(f"")

        # Show what was learned this run
        summary_after = mem.get_repo_summary(ctx.repo)
        prs_after     = summary_after["total_prs_analysed"]
        pats_after    = len(summary_after.get("top_patterns", []))

        logger.info(f"  📈 LEARNING PROGRESS for {ctx.repo}:")
        logger.info(f"     PRs in memory:       {prs_after}  (was {past})")
        logger.info(f"     Patterns learned:    {pats_after}")
        if prs_after >= 3:
            logger.info(f"     Intelligence level:  🧠 Active — predictions enabled")
        elif prs_after >= 1:
            logger.info(f"     Intelligence level:  📖 Learning — need {3 - prs_after} more PR(s) for predictions")
        else:
            logger.info(f"     Intelligence level:  🌱 Starting — first PR analysed")

        top_pats = summary_after.get("top_patterns", [])
        if top_pats:
            logger.info(f"     Top patterns known:")
            for p in top_pats[:5]:
                logger.info(f"       [{p['occurrences']}x] {p.get('example_detail','')[:65]}")
        logger.info(f"")
        logger.ok("runner", f"All {total} agents complete  |  {total_findings} findings  |  {total_elapsed:.1f}s")

        return results
