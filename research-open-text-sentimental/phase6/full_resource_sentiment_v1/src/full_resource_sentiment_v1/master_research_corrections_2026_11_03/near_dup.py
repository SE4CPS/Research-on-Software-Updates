"""Auditable near-duplicate connected components for RQ4 leakage control."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from full_resource_sentiment_v1.master_research_corrections_2026_11_03 import (
    FULLTEXT_JACCARD_AUDIT,
    TITLE_BODY_JACCARD,
    TITLE_MIN_CHARS,
)


def normalize_ws(text: str) -> str:
    return " ".join((text or "").lower().split())


def word_set(text: str) -> frozenset[str]:
    return frozenset(normalize_ws(text).split())


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


@dataclass(frozen=True)
class NearDupPolicy:
    title_min_chars: int = TITLE_MIN_CHARS
    title_body_jaccard: float = TITLE_BODY_JACCARD
    fulltext_jaccard_audit: float = FULLTEXT_JACCARD_AUDIT


def build_components(
    ids: list[str],
    titles: list[str],
    bodies: list[str],
    *,
    policy: NearDupPolicy | None = None,
) -> dict:
    """Return component labels + audit stats. Thresholds are predeclared (no label tuning)."""
    policy = policy or NearDupPolicy()
    n = len(ids)
    assert n == len(titles) == len(bodies)
    uf = UnionFind(n)
    norms_title = [normalize_ws(t) for t in titles]
    norms_body = [normalize_ws(b) for b in bodies]
    norms_full = [f"{nt}\n{nb}".strip() for nt, nb in zip(norms_title, norms_body)]
    body_sets = [word_set(b) for b in bodies]
    full_sets = [word_set(f) for f in norms_full]

    # 1) Exact normalized full text
    by_full: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(norms_full):
        if f:
            by_full[f].append(i)
    exact_full_merges = 0
    for group in by_full.values():
        if len(group) < 2:
            continue
        root = group[0]
        for j in group[1:]:
            uf.union(root, j)
            exact_full_merges += 1

    # 2) Shared nontrivial exact title → pairwise body Jaccard
    by_title: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(norms_title):
        if len(t) >= policy.title_min_chars:
            by_title[t].append(i)
    title_pair_checks = 0
    title_body_merges = 0
    title_body_jaccards: list[float] = []
    for group in by_title.values():
        if len(group) < 2:
            continue
        for a_i in range(len(group)):
            for b_i in range(a_i + 1, len(group)):
                i, j = group[a_i], group[b_i]
                jac = jaccard(body_sets[i], body_sets[j])
                title_pair_checks += 1
                title_body_jaccards.append(jac)
                if jac >= policy.title_body_jaccard:
                    uf.union(i, j)
                    title_body_merges += 1

    # 3) Audit-only: within exact-title buckets also note full-text Jaccard >= audit threshold
    audit_high_full = 0
    for group in by_title.values():
        for a_i in range(len(group)):
            for b_i in range(a_i + 1, len(group)):
                i, j = group[a_i], group[b_i]
                if jaccard(full_sets[i], full_sets[j]) >= policy.fulltext_jaccard_audit:
                    audit_high_full += 1

    roots = [uf.find(i) for i in range(n)]
    # Compact component ids
    root_to_cid: dict[int, int] = {}
    components: list[int] = []
    for r in roots:
        if r not in root_to_cid:
            root_to_cid[r] = len(root_to_cid)
        components.append(root_to_cid[r])

    size_counts = Counter(components)
    jacc_sorted = sorted(title_body_jaccards)
    def pct(p: float) -> float | None:
        if not jacc_sorted:
            return None
        idx = min(len(jacc_sorted) - 1, max(0, int(round(p * (len(jacc_sorted) - 1)))))
        return float(jacc_sorted[idx])

    return {
        "component_ids": components,
        "n_posts": n,
        "n_components": len(root_to_cid),
        "component_size_hist": {str(k): v for k, v in sorted(Counter(size_counts.values()).items())},
        "n_components_size_ge2": sum(1 for s in size_counts.values() if s >= 2),
        "max_component_size": max(size_counts.values()) if size_counts else 0,
        "policy": {
            "title_min_chars": policy.title_min_chars,
            "title_body_jaccard": policy.title_body_jaccard,
            "fulltext_jaccard_audit": policy.fulltext_jaccard_audit,
            "note": "Thresholds predeclared; not tuned on labels or test metrics.",
        },
        "merges": {
            "exact_full_text_edge_count": exact_full_merges,
            "title_body_jaccard_edge_count": title_body_merges,
            "title_pair_checks": title_pair_checks,
            "audit_title_bucket_fulltext_jaccard_ge_audit": audit_high_full,
        },
        "title_body_jaccard_distribution": {
            "n_pairs": len(title_body_jaccards),
            "min": float(jacc_sorted[0]) if jacc_sorted else None,
            "p25": pct(0.25),
            "p50": pct(0.50),
            "p75": pct(0.75),
            "p90": pct(0.90),
            "max": float(jacc_sorted[-1]) if jacc_sorted else None,
            "n_ge_threshold": sum(1 for x in title_body_jaccards if x >= policy.title_body_jaccard),
        },
    }


def assert_no_cross_split_overlap(
    component_ids: list[int],
    train_idx: Iterable[int],
    test_idx: Iterable[int],
) -> int:
    tr = {component_ids[i] for i in train_idx}
    te = {component_ids[i] for i in test_idx}
    return len(tr & te)
