"""Shared Workstream I helpers. New files only; G outputs stay immutable."""

from __future__ import annotations

import json
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from .adapter import extract_post_body
from .broad_sentiment import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    load_local_development_labels,
    verify_capture_pages,
)
from .comments import extract_comment_text
from .hashing import sha256_file
from .inference import (
    bootstrap_mean_median_ci,
    bootstrap_spearman,
    percentile_ci,
    spearman,
    stratified_mean_diff,
    summarize_numeric,
)
from .io_utils import iter_jsonl, read_json
from .paths import (
    BROAD_CAPTURE_DIR,
    BROAD_SENTIMENT_RUN_DIR,
    LOCAL_COMPACT_RECORDS,
    N_BOOTSTRAP,
    PACKAGE_TARBALL,
    RESULTS_DIR,
    WORKSPACE,
)
from .scoring import is_reddit_url, score_observed_text

I_SEED = 20261024
I_N_BOOT = N_BOOTSTRAP
I_RUN_DIR = RESULTS_DIR / "workstream_i_2026-10-22-main"
I_RESTRICTED_DIR = WORKSPACE / "restricted" / "workstream_i_2026-10-22-main"
CAPTURE_ID = "2026-10-22-main"
PACKAGE_VERSION = "0.2.1"
G_HASH_TOLERANCE = 1e-8

G_REFERENCE_POINTS = {
    "A": 0.3237226363,
    "B": -0.1286597395,
    "C": -0.0485610720,
    "D": -0.0259668969,
    "E_spearman": -0.0173658541,
    "F": -0.2901190980,
}

EXPECTED_N = {
    "posts": 37596,
    "comments": 273353,
    "A": 35443,
    "B": 17282,
    "C": 17035,
    "D": 24318,
    "E": 24313,
    "F": 1864,
    "F_troubleshooting": 1628,
    "F_other": 236,
    "count_gap": 18260,
}

G_OUTPUT_NAMES = (
    "post_scores.jsonl",
    "comment_scores.jsonl",
    "thread_metrics.jsonl",
    "analysis_A.json",
    "analysis_B.json",
    "analysis_C.json",
    "analysis_D.json",
    "analysis_E.json",
    "analysis_F.json",
)


def word_tokens(text: str | None) -> list[str]:
    return (text or "").split()


def word_count(text: str | None) -> int:
    return len(word_tokens(text))


def first_k_word_text(text: str | None, k: int) -> str:
    if k <= 0:
        return ""
    return " ".join(word_tokens(text)[:k])


def original_post_text(title: str, body: str) -> str:
    title_ne = bool(str(title or "").strip())
    body_ne = bool(str(body or "").strip())
    if title_ne and body_ne:
        return f"{title} {body}"
    if title_ne:
        return title
    if body_ne:
        return body
    return ""


def ratio_bin_ia(ratio: float) -> str:
    if ratio <= 1:
        return "le_1"
    if ratio <= 2:
        return "(1,2]"
    if ratio <= 5:
        return "(2,5]"
    return "gt_5"


def ratio_bin_ib(ratio: float) -> str:
    if ratio <= 0.25:
        return "le_0.25"
    if ratio <= 0.5:
        return "(0.25,0.5]"
    if ratio <= 1:
        return "(0.5,1]"
    return "gt_1"


def author_reply_count_bin(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 3:
        return "2-3"
    return "ge_4"


def timed_reply_count_bin(n: int) -> str:
    if n <= 2:
        return "2"
    if n == 3:
        return "3"
    if n <= 7:
        return "4-7"
    return "ge_8"


def date_stratum(created_month: str | None) -> str:
    if created_month == "2026-10":
        return "recent_oct2026"
    return "older"


def sign_of(value: float | None) -> str | None:
    if value is None:
        return None
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def reading_aid(
    *,
    reference: float | None,
    diagnostic: float | None,
    strata: dict[str, dict[str, Any]],
    estimand_note: str,
) -> dict[str, Any]:
    populated = {
        name: block
        for name, block in strata.items()
        if int(block.get("n") or 0) >= 200 and block.get("mean") is not None
    }
    ref_sign = sign_of(reference)
    reversals = [
        name
        for name, block in populated.items()
        if sign_of(block.get("mean")) not in {ref_sign, "zero", None} and ref_sign not in {None, "zero"}
    ]
    directionally_stable = bool(populated) and not reversals and ref_sign not in {None, "zero"}
    attenuated = False
    if reference is not None and diagnostic is not None and abs(reference) > 0:
        attenuated = abs(diagnostic) < 0.5 * abs(reference)
    return {
        "directionally_stable": directionally_stable,
        "sign_reversals_n_ge_200": reversals,
        "materially_attenuated": attenuated,
        "rule": (
            "Directionally stable only if sign agrees across N>=200 count-gap and date strata. "
            "Materially attenuated if |diagnostic| < 0.5 |reference|. Reading aid, not a hypothesis test."
        ),
        "estimand_note": estimand_note,
    }


def hash_g_outputs(run_dir: Path = BROAD_SENTIMENT_RUN_DIR) -> dict[str, str]:
    declared = read_json(run_dir / "hashes.json")
    actual = {name: sha256_file(run_dir / name) for name in G_OUTPUT_NAMES}
    mismatches = {name: {"declared": declared[name], "actual": actual[name]} for name in G_OUTPUT_NAMES if declared[name] != actual[name]}
    if mismatches:
        raise ValueError(f"Workstream G output hash mismatch: {mismatches}")
    return actual


def load_jsonl_by_id(path: Path, key: str = "reddit_id") -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        rid = str(row[key])
        if rid in rows:
            raise ValueError(f"duplicate {key} in {path.name}: {rid}")
        rows[rid] = row
    return rows


def load_comments_by_id(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    n = 0
    for row in iter_jsonl(path):
        grouped[str(row["reddit_id"])].append(row)
        n += 1
    if n != EXPECTED_N["comments"]:
        raise ValueError(f"comment_scores rows {n} != {EXPECTED_N['comments']}")
    return grouped


def _finite_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def recompute_g_points(
    posts: dict[str, dict[str, Any]],
    threads: dict[str, dict[str, Any]],
) -> dict[str, float]:
    a_vals = [
        float(row["body_minus_title"])
        for row in posts.values()
        if row.get("title_compound") is not None and row.get("body_compound") is not None
    ]
    b_vals = [
        float(row["author_mean"] - row["combined_compound"])
        for row in threads.values()
        if row.get("author_reply_eligible") and row.get("author_mean") is not None and row.get("combined_compound") is not None
    ]
    c_vals = [
        float(row["community_mean"] - row["author_mean"])
        for row in threads.values()
        if row.get("both_roles_eligible") and row.get("author_mean") is not None and row.get("community_mean") is not None
    ]
    d_vals = [
        float(row["last_minus_first"])
        for row in threads.values()
        if row.get("order_eligible") and row.get("last_minus_first") is not None
    ]
    e_hours = []
    e_diff = []
    for row in threads.values():
        if row.get("elapsed_eligible") and row.get("elapsed_hours") is not None and row.get("last_minus_first") is not None:
            e_hours.append(float(row["elapsed_hours"]))
            e_diff.append(float(row["last_minus_first"]))
    f_tps = [
        float(row["combined_compound"])
        for row in threads.values()
        if row.get("f_eligible") is True and row.get("s2_target") == 1 and row.get("combined_compound") is not None
    ]
    f_other = [
        float(row["combined_compound"])
        for row in threads.values()
        if row.get("f_eligible") is True and row.get("s2_target") == 0 and row.get("combined_compound") is not None
    ]
    e_rho = spearman(e_hours, e_diff)
    if e_rho is None:
        raise ValueError("failed to recompute G E Spearman")
    return {
        "A": float(np.mean(a_vals)),
        "B": float(np.mean(b_vals)),
        "C": float(np.mean(c_vals)),
        "D": float(np.mean(d_vals)),
        "E_spearman": float(e_rho),
        "F": float(np.mean(f_tps) - np.mean(f_other)),
        "n_A": len(a_vals),
        "n_B": len(b_vals),
        "n_C": len(c_vals),
        "n_D": len(d_vals),
        "n_E": len(e_hours),
        "n_F_tps": len(f_tps),
        "n_F_other": len(f_other),
    }


def assert_g_guard(points: dict[str, float]) -> None:
    for key, expected in G_REFERENCE_POINTS.items():
        actual = points[key]
        if abs(actual - expected) > G_HASH_TOLERANCE:
            raise ValueError(f"G reference {key} {actual} differs from {expected} by more than 1e-8")
    if points["n_A"] != EXPECTED_N["A"]:
        raise ValueError(f"A n {points['n_A']} != {EXPECTED_N['A']}")
    if points["n_B"] != EXPECTED_N["B"]:
        raise ValueError(f"B n {points['n_B']} != {EXPECTED_N['B']}")
    if points["n_C"] != EXPECTED_N["C"]:
        raise ValueError(f"C n {points['n_C']} != {EXPECTED_N['C']}")
    if points["n_D"] != EXPECTED_N["D"]:
        raise ValueError(f"D n {points['n_D']} != {EXPECTED_N['D']}")
    if points["n_E"] != EXPECTED_N["E"]:
        raise ValueError(f"E n {points['n_E']} != {EXPECTED_N['E']}")
    if points["n_F_tps"] != EXPECTED_N["F_troubleshooting"] or points["n_F_other"] != EXPECTED_N["F_other"]:
        raise ValueError("F group counts drifted from frozen G")


def verify_f_membership(threads: dict[str, dict[str, Any]], local_path: Path = LOCAL_COMPACT_RECORDS) -> dict[str, Any]:
    local = load_local_development_labels(local_path)
    g_f = {rid for rid, row in threads.items() if row.get("f_eligible") is True}
    independent = set()
    for rid, row in threads.items():
        flags = local.get(rid)
        if flags and flags.get("f_eligible") is True:
            if flags.get("reserved_membership") or flags.get("fresh_membership"):
                raise ValueError(f"protected membership in independent F join: {rid}")
            independent.add(rid)
    if g_f != independent:
        raise ValueError(
            f"F membership mismatch vs development labels: extra={sorted(g_f - independent)[:10]} missing={sorted(independent - g_f)[:10]}"
        )
    if any(threads[rid].get("reserved_membership") or threads[rid].get("fresh_membership") for rid in g_f):
        raise ValueError("protected membership present in G F-eligible rows")
    n_tps = sum(1 for rid in g_f if threads[rid].get("s2_target") == 1)
    n_other = sum(1 for rid in g_f if threads[rid].get("s2_target") == 0)
    if len(g_f) != EXPECTED_N["F"] or n_tps != EXPECTED_N["F_troubleshooting"] or n_other != EXPECTED_N["F_other"]:
        raise ValueError(f"F counts {len(g_f)}/{n_tps}/{n_other} != 1864/1628/236")
    return {"n": len(g_f), "n_troubleshooting": n_tps, "n_other": n_other, "protected_in_f": 0}


def first_author_comment(comments: list[dict[str, Any]]) -> dict[str, Any] | None:
    authors = [row for row in comments if row.get("is_author")]
    if not authors:
        return None
    authors.sort(key=lambda row: (int(row["chronological_index"]), int(row["original_index"])))
    return authors[0]


def timed_comments(comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timed = [row for row in comments if row.get("timestamp_valid")]
    timed.sort(key=lambda row: (float(row["timestamp_epoch_seconds"]), int(row["original_index"])))
    return timed


def ols_coefficient(y: np.ndarray, x: np.ndarray) -> np.ndarray | None:
    if y.size == 0 or x.shape[0] != y.size:
        return None
    try:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(beta)):
        return None
    return beta


def stratified_ols_s2_ci(
    rows: list[dict[str, Any]],
    *,
    seed: int = I_SEED,
    n_boot: int = I_N_BOOT,
) -> dict[str, Any]:
    tps = [row for row in rows if row["s2_target"] == 1]
    other = [row for row in rows if row["s2_target"] == 0]

    def design(sample: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray([row["combined_compound"] for row in sample], dtype=float)
        x = np.column_stack(
            [
                np.ones(len(sample)),
                np.asarray([row["s2_target"] for row in sample], dtype=float),
                np.asarray([row["log1p_chars"] for row in sample], dtype=float),
                np.asarray([row["body_present"] for row in sample], dtype=float),
                np.asarray([row["october"] for row in sample], dtype=float),
                np.asarray([row["count_gap"] for row in sample], dtype=float),
                np.asarray([row["source_is_reddit"] for row in sample], dtype=float),
            ]
        )
        return y, x

    y, x = design(rows)
    beta = ols_coefficient(y, x)
    point = None if beta is None else float(beta[1])
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        t_idx = rng.integers(0, len(tps), size=len(tps))
        o_idx = rng.integers(0, len(other), size=len(other))
        boot = [tps[i] for i in t_idx] + [other[i] for i in o_idx]
        yb, xb = design(boot)
        betab = ols_coefficient(yb, xb)
        if betab is not None:
            samples.append(float(betab[1]))
    return {
        "s2_target_coefficient": point,
        "s2_target_ci95": percentile_ci(samples) if samples else None,
        "n": len(rows),
        "n_troubleshooting": len(tps),
        "n_other": len(other),
        "n_bootstrap": n_boot,
        "seed": seed,
        "bootstrap_unit": "thread_stratified",
        "n_defined_resamples": len(samples),
        "covariates": [
            "intercept",
            "s2_target",
            "log1p(original_post_characters)",
            "body_present",
            "october_vs_older",
            "count_gap",
            "source_confirmed_reddit",
        ],
        "full_beta": None if beta is None else [float(v) for v in beta],
        "missing_covariates": 0,
        "note": "Controls measured covariates only; not selection, topic, or unmeasured confounding.",
    }


def pooled_character_bins(values: list[int], n_bins: int = 5) -> list[float]:
    arr = np.asarray(values, dtype=float)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    cuts = [float(v) for v in np.quantile(arr, qs)]
    unique: list[float] = []
    for cut in cuts:
        if not unique or cut != unique[-1]:
            unique.append(cut)
    return unique


def assign_cut_bin(value: float, cuts: list[float]) -> int:
    if len(cuts) < 2:
        return 0
    for index in range(1, len(cuts)):
        if value <= cuts[index]:
            return index - 1
    return len(cuts) - 2


def collect_text_measures(
    *,
    capture_dir: Path,
    threads: dict[str, dict[str, Any]],
    comments_by_id: dict[str, list[dict[str, Any]]],
    need_prefix_ids: set[str],
) -> dict[str, dict[str, Any]]:
    capture = verify_capture_pages(capture_dir)
    first_author_index = {}
    for rid, comments in comments_by_id.items():
        first = first_author_comment(comments)
        if first is not None:
            first_author_index[rid] = int(first["original_index"])
    measures: dict[str, dict[str, Any]] = {}
    for page in capture["pages"]:
        payload = json.loads((capture_dir / page["file"]).read_text(encoding="utf-8"))
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise ValueError(f"page schema failure {page['file']}")
        for raw in rows:
            if not isinstance(raw, dict) or not raw.get("redditId"):
                continue
            if not is_reddit_url(raw.get("url")):
                continue
            rid = str(raw["redditId"]).strip()
            if rid not in threads:
                continue
            title = "" if raw.get("title") is None else str(raw.get("title") or "")
            body, _field, _present = extract_post_body(raw)
            post_text = original_post_text(title, body)
            title_words = word_count(title)
            body_words = word_count(body)
            post_words = word_count(post_text)
            author_word_counts: list[int] = []
            first_reply_text = None
            raw_comments = raw.get("comments") if isinstance(raw.get("comments"), list) else []
            wanted_index = first_author_index.get(rid)
            g_comments = comments_by_id.get(rid) or []
            author_indices = {int(row["original_index"]) for row in g_comments if row.get("is_author")}
            for index, comment in enumerate(raw_comments):
                if not isinstance(comment, dict) or index not in author_indices:
                    continue
                text, _ = extract_comment_text(comment)
                if not str(text).strip():
                    continue
                author_word_counts.append(word_count(text))
                if wanted_index is not None and index == wanted_index:
                    first_reply_text = text
            row = {
                "reddit_id": rid,
                "title_words": title_words,
                "body_words": body_words,
                "original_post_words": post_words,
                "original_post_chars": len(post_text),
                "title_nonempty": bool(title.strip()),
                "body_nonempty": bool(body.strip()),
                "n_author_reply_texts": len(author_word_counts),
                "mean_author_reply_words": _finite_mean([float(v) for v in author_word_counts]),
                "first_author_reply_words": word_count(first_reply_text) if first_reply_text is not None else None,
                "k_title_body": None,
                "title_prefix_compound": None,
                "body_prefix_compound": None,
                "k_post_first_reply": None,
                "post_prefix_compound": None,
                "first_reply_prefix_compound": None,
            }
            if rid in need_prefix_ids and row["title_nonempty"] and row["body_nonempty"]:
                k = min(title_words, body_words)
                row["k_title_body"] = k
                if k >= 3:
                    title_scored = score_observed_text(first_k_word_text(title, k))
                    body_scored = score_observed_text(first_k_word_text(body, k))
                    row["title_prefix_compound"] = None if title_scored is None else float(title_scored["compound"])
                    row["body_prefix_compound"] = None if body_scored is None else float(body_scored["compound"])
            if rid in need_prefix_ids and first_reply_text is not None and post_words >= 0:
                k2 = min(post_words, word_count(first_reply_text))
                row["k_post_first_reply"] = k2
                if k2 >= 3:
                    post_scored = score_observed_text(first_k_word_text(post_text, k2))
                    reply_scored = score_observed_text(first_k_word_text(first_reply_text, k2))
                    row["post_prefix_compound"] = None if post_scored is None else float(post_scored["compound"])
                    row["first_reply_prefix_compound"] = None if reply_scored is None else float(reply_scored["compound"])
            measures[rid] = row
        print(f"workstream I text pass page {page['page']}/{capture['n_pages']} n={len(measures)}", flush=True)
    if len(measures) != EXPECTED_N["posts"]:
        raise ValueError(f"text measures {len(measures)} != {EXPECTED_N['posts']}")
    return measures


def bootstrap_diff_on_rows(
    rows: list[dict[str, Any]],
    left_key: str,
    right_key: str,
    *,
    difference_name: str,
) -> dict[str, Any]:
    left = [float(row[left_key]) for row in rows if row.get(left_key) is not None and row.get(right_key) is not None]
    right = [float(row[right_key]) for row in rows if row.get(left_key) is not None and row.get(right_key) is not None]
    if len(left) != len(right):
        raise ValueError("paired keys dropped unevenly")
    diff = [r - l for l, r in zip(left, right)]
    out = bootstrap_mean_median_ci(diff, n_boot=I_N_BOOT, seed=I_SEED)
    out["difference_name"] = difference_name
    out["mean_left"] = _finite_mean(left)
    out["mean_right"] = _finite_mean(right)
    return out


def describe_stratum(values: list[float]) -> dict[str, Any]:
    return bootstrap_mean_median_ci(values, n_boot=I_N_BOOT, seed=I_SEED)


def package_and_code_meta() -> dict[str, Any]:
    import hashlib

    files = sorted((WORKSPACE / "src" / "full_resource_sentiment_v1").glob("robustness_i_*.py"))
    files.append(WORKSPACE / "scripts" / "run_workstream_i.py")
    digest_parts = []
    digest = hashlib.sha256()
    for path in files:
        if not path.exists():
            continue
        file_hash = sha256_file(path)
        digest_parts.append(f"{path.name}:{file_hash}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("utf-8"))
        digest.update(b"\n")
    return {
        "package_version": version("releasetrain-sentiment"),
        "vader_version": version("vaderSentiment"),
        "package_tarball_sha256": sha256_file(PACKAGE_TARBALL),
        "code_fingerprint": digest.hexdigest(),
        "code_files": digest_parts,
    }
