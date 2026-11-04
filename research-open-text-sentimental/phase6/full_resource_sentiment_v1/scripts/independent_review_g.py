"""Codex Workstream H: read-only checks, independent of Cursor's analysis code.

Reads immutable capture and derived outputs. Never writes raw post/comment text.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from releasetrain_sentiment import score_post

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "broad_sentiment_2026-10-22-main"
RAW = ROOT / "restricted" / "broad_release_train" / "2026-10-22-main"
DEN = json.loads((ROOT / "results" / "broad_relevance_denominators.json").read_text())
SEED = 20261022
NBOOT = 2000


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def ci_mean(values):
    x = np.asarray(values, float)
    rng = np.random.default_rng(SEED)
    means = np.empty(NBOOT)
    for i in range(NBOOT):
        means[i] = x[rng.integers(0, len(x), len(x))].mean()
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def ci_strat(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(SEED)
    diffs = np.empty(NBOOT)
    draws_a = [rng.integers(0, len(a), len(a)) for _ in range(NBOOT)]
    draws_b = [rng.integers(0, len(b), len(b)) for _ in range(NBOOT)]
    for i in range(NBOOT):
        diffs[i] = a[draws_a[i]].mean() - b[draws_b[i]].mean()
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def mean_and_ci(values):
    x = np.asarray(values, float)
    return {"n": len(x), "mean": float(x.mean()), "ci95": ci_mean(x)}


def close(a, b, tol=1e-6):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


def check_value(a, b, name):
    if not close(a, b):
        raise AssertionError(f"{name}: {a} != {b}")


def utc_epoch(value):
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)) or str(value).replace(".", "", 1).isdigit():
            return float(value)
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, OverflowError):
        return None


def rankdata(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    return ranks


def score_text(value):
    if not str(value or "").strip():
        return None
    return float(score_post({"title": str(value), "description": "", "comments": []})[0]["compound"])


def independent_role(post_author, comment):
    flag = comment.get("is_submitter")
    explicit = flag is True or str(flag).strip().lower() in {"true", "t", "yes", "1"}
    author = str(post_author or "").strip()
    reply_author = str(comment.get("author") or "").strip()
    return explicit or bool(author and reply_author and author == reply_author)


def sample_rescore(posts, comments):
    sample = json.loads((OUT / "handcheck_sample_30.json").read_text())["rows"]
    ids = {r["reddit_id"] for r in sample} | {"1o0nzld", "1wyg8ze"}
    by_post = {r["reddit_id"]: r for r in posts}
    by_comment = defaultdict(list)
    for c in comments:
        if c["reddit_id"] in ids:
            by_comment[c["reddit_id"]].append(c)
    pages = defaultdict(list)
    for rid in ids:
        pages[by_post[rid]["capture_page"]].append(rid)
    n_comments = 0
    n_roles = 0
    n_order = 0
    for page, rids in pages.items():
        raw_rows = json.loads((RAW / f"page_{page:05d}.json").read_text())["data"]
        for rid in rids:
            post = by_post[rid]
            raw = raw_rows[post["capture_row_index"]]
            assert str(raw["redditId"]) == rid
            title = str(raw.get("title") or "")
            body = str(raw.get("author_description") or "")
            check_value(score_text(title), post["title_compound"], f"{rid} title")
            check_value(score_text(body), post["body_compound"], f"{rid} body")
            combined = float(score_post({"title": title, "description": body, "comments": []})[0]["compound"])
            check_value(combined, post["combined_compound"], f"{rid} combined")
            observed = []
            for i, c in enumerate(raw.get("comments") or []):
                if not isinstance(c, dict):
                    continue
                body_text = str(c.get("body") or c.get("text") or "")
                if not body_text.strip():
                    continue
                author = independent_role(raw.get("author"), c)
                t = utc_epoch(c.get("created_utc"))
                observed.append((i, score_text(body_text), author, t))
            saved = {c["original_index"]: c for c in by_comment[rid]}
            assert len(saved) == len(observed)
            for i, score, author, t in observed:
                c = saved[i]
                check_value(score, c["compound"], f"{rid}:{i} reply")
                assert c["is_author"] == author
                assert c["timestamp_valid"] == (t is not None)
                if t is not None:
                    check_value(t, c["timestamp_epoch_seconds"], f"{rid}:{i} timestamp")
                n_comments += 1
                n_roles += 1
            expected = sorted(observed, key=lambda x: (x[3] is None, x[3] if x[3] is not None else 0, x[0]))
            assert [x[0] for x in expected] == [c["original_index"] for c in sorted(by_comment[rid], key=lambda c: c["chronological_index"])]
            n_order += 1
    return {"sample_ids": len(ids), "sample_comment_scores_checked": n_comments, "sample_roles_checked": n_roles, "sample_orders_checked": n_order}


def main():
    capture = json.loads((RAW / "manifest.json").read_text())
    run = json.loads((OUT / "manifest.json").read_text())
    assert sha(RAW / "manifest.json") == run["input_hashes"]["capture_manifest_sha256"]
    assert sha(ROOT / "results" / "broad_capture_compact_2026-10-22-main.jsonl") == run["input_hashes"]["compact_index_sha256"]
    assert sha(ROOT / "results" / "broad_relevance_denominators.json") == run["input_hashes"]["relevance_denominators_sha256"]
    for page in capture["pages"]:
        assert sha(RAW / page["file"]) == page["sha256"]
    declared = json.loads((OUT / "hashes.json").read_text())
    for filename, digest in declared.items():
        assert sha(OUT / filename) == digest
    posts = read_jsonl(OUT / "post_scores.jsonl")
    threads = read_jsonl(OUT / "thread_metrics.jsonl")
    comments = read_jsonl(OUT / "comment_scores.jsonl")
    assert len(posts) == len(threads) == 37596
    assert len(comments) == 273353
    assert len({r["reddit_id"] for r in posts}) == 37596
    assert {r["reddit_id"] for r in posts} == {r["reddit_id"] for r in threads}
    assert len({c["comment_key"] for c in comments}) == len(comments)
    assert len({c["reddit_id"] for c in comments}) == 28122
    assert sum(p["count_gap"] for p in posts) == 18260
    assert sum(p["n_empty_comment_bodies"] for p in posts) == 4
    assert sum(t["body_nonempty"] for t in threads) == 35443
    assert sum(t["author_reply_eligible"] for t in threads) == 17282
    assert sum(t["both_roles_eligible"] for t in threads) == 17035
    assert sum(t["order_eligible"] for t in threads) == 24318
    assert sum(t["elapsed_eligible"] for t in threads) == 24313
    assert sum(t["f_eligible"] for t in threads) == 1864
    assert sum(t["f_eligible"] and t["s2_target"] == 1 for t in threads) == 1628
    assert sum(t["f_eligible"] and t["s2_target"] == 0 for t in threads) == 236
    assert all(not(t["f_eligible"] and (t["fresh_membership"] or t["reserved_membership"])) for t in threads)
    assert all(t["s2_target"] is None for t in threads if t["fresh_membership"] or t["reserved_membership"])
    local_path = ROOT / "results" / "normalized_records_compact.jsonl"
    assert sha(local_path) == run["input_hashes"]["local_compact_sha256"]
    expected_f = {}
    for local in read_jsonl(local_path):
        memberships = set(local["source_memberships"])
        if "development_v2" in memberships and not ({"reserved_test", "fresh_500"} & memberships):
            labels = local.get("labels") or {}
            if labels.get("s2_in_scope") is True:
                expected_f[local["reddit_id"]] = labels["s2_target"]
    actual_f = {t["reddit_id"]: t["s2_target"] for t in threads if t["f_eligible"]}
    thread_ids = {t["reddit_id"] for t in threads}
    assert actual_f == {rid: target for rid, target in expected_f.items() if rid in thread_ids}
    assert all(t["author_mean"] is None for t in threads if not t["author_reply_eligible"])
    assert all(t["community_mean"] is None for t in threads if not t["n_community_replies_nonempty"])
    assert all(t["author_mean"] is not None for t in threads if t["author_reply_eligible"])
    assert any(t["author_mean"] == 0 for t in threads if t["author_reply_eligible"])
    assert any(c["compound"] == 0 for c in comments)
    bad_keys = {"title", "body", "text", "author", "username", "url", "description", "snippet"}
    for path in (OUT / "post_scores.jsonl", OUT / "thread_metrics.jsonl", OUT / "comment_scores.jsonl"):
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                assert not (set(row) & bad_keys), (path, set(row) & bad_keys)
    sample = sample_rescore(posts, comments)
    by_id = {t["reddit_id"]: t for t in threads}
    author_scores = defaultdict(list)
    community_scores = defaultdict(list)
    timed_scores = defaultdict(list)
    for c in comments:
        rid = c["reddit_id"]
        (author_scores if c["is_author"] else community_scores)[rid].append(c["compound"])
        if c["timestamp_valid"]:
            timed_scores[rid].append(c)
    for rid, t in by_id.items():
        a, b = author_scores[rid], community_scores[rid]
        check_value(sum(a)/len(a) if a else None, t["author_mean"], f"{rid} author mean")
        check_value(sum(b)/len(b) if b else None, t["community_mean"], f"{rid} community mean")
        timed = sorted(timed_scores[rid], key=lambda c: (c["timestamp_epoch_seconds"], c["original_index"]))
        if len(timed) >= 2:
            check_value(timed[-1]["compound"]-timed[0]["compound"], t["last_minus_first"], f"{rid} order")
            check_value((timed[-1]["timestamp_epoch_seconds"]-timed[0]["timestamp_epoch_seconds"])/3600 if len({c["timestamp_epoch_seconds"] for c in timed}) >= 2 else None, t["elapsed_hours"], f"{rid} elapsed")
    analyses = {x: json.loads((OUT / f"analysis_{x}.json").read_text()) for x in "ABCDEF"}
    a = [t["body_compound"]-t["title_compound"] for t in threads if t["body_compound"] is not None]
    b = [t["author_mean"]-t["combined_compound"] for t in threads if t["author_mean"] is not None]
    c = [t["community_mean"]-t["author_mean"] for t in threads if t["both_roles_eligible"]]
    d = [t["last_minus_first"] for t in threads if t["order_eligible"]]
    e = [t["last_minus_first"] for t in threads if t["elapsed_eligible"]]
    stats = {k: mean_and_ci(v) for k, v in zip("ABCDE", (a,b,c,d,e))}
    keys = {"A":"primary_body_minus_title", "B":"primary_author_mean_minus_combined", "C":"primary_community_minus_author", "D":"primary_last_minus_first", "E":"primary_last_minus_first"}
    for k in "ABCDE":
        reported = analyses[k][keys[k]]
        assert stats[k]["n"] == reported["n"]
        check_value(stats[k]["mean"], reported["mean"], f"{k} mean")
        for j in (0,1):check_value(stats[k]["ci95"][j], reported["mean_ci95"][j], f"{k} CI{j}")
    frows = [t for t in threads if t["f_eligible"]]
    fa = [t["combined_compound"] for t in frows if t["s2_target"] == 1]
    fb = [t["combined_compound"] for t in frows if t["s2_target"] == 0]
    frep = analyses["F"]["primary_combined_post"]["comparison"]
    fmean = float(np.mean(fa)-np.mean(fb))
    fci = ci_strat(fa,fb)
    check_value(fmean, frep["mean_diff"], "F mean")
    for j in (0,1):check_value(fci[j], frep["mean_diff_ci95"][j], f"F CI{j}")
    # Independent rank-correlation point and a fresh bootstrap using scipy ranks.
    ex = np.array([t["elapsed_hours"] for t in threads if t["elapsed_eligible"]], float)
    ey = np.array([t["last_minus_first"] for t in threads if t["elapsed_eligible"]], float)
    rho = float(np.corrcoef(rankdata(ex),rankdata(ey))[0,1])
    check_value(rho, analyses["E"]["spearman_elapsed_hours_vs_last_minus_first"]["spearman"], "E Spearman")
    rng = np.random.default_rng(SEED)
    rho_draws = []
    for _ in range(NBOOT):
        idx = rng.integers(0, len(ex), len(ex))
        rho_draws.append(float(np.corrcoef(rankdata(ex[idx]), rankdata(ey[idx]))[0,1]))
    rho_ci = [float(np.percentile(rho_draws,2.5)),float(np.percentile(rho_draws,97.5))]
    for j in (0,1):check_value(rho_ci[j], analyses["E"]["spearman_elapsed_hours_vs_last_minus_first"]["spearman_ci95"][j], f"E rho CI{j}")
    fa_arr, fb_arr = np.asarray(fa,float),np.asarray(fb,float)
    pairdiff = fa_arr[:,None]-fb_arr[None,:]
    cliff = float((np.count_nonzero(pairdiff>0)-np.count_nonzero(pairdiff<0))/(len(fa)*len(fb)))
    check_value(cliff, frep["cliffs_delta"], "F Cliff delta")
    result = {"input_pages_hashed":len(capture["pages"]), "output_files_hashed":len(declared), "post_rows":len(posts), "comment_rows":len(comments), "thread_rows":len(threads), "count_gap_retained":18260, "sample":sample, "primary_checks":stats, "F":{"n_a":len(fa),"n_b":len(fb),"mean_diff":fmean,"ci95":fci,"cliffs_delta":cliff}, "E_spearman":rho, "E_spearman_ci95":rho_ci, "zero_observed_comments":sum(c["compound"] == 0 for c in comments), "missing_author_threads":sum(not t["author_reply_eligible"] for t in threads)}
    dest = ROOT / "results" / "codex_independent_review_g_2026-10-24.json"
    dest.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"status":"passed", "sample":sample, "means":{k:v["mean"] for k,v in stats.items()}, "F_mean_diff":fmean, "E_spearman":rho}, indent=2))


if __name__ == "__main__":
    main()
