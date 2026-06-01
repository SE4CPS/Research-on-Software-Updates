"""
Grow the usability vs defect sample by auto-labeling more posts from enhanced results.
Uses keyword-based relevance; saves top N per category to usability_defect_posts_extended.json.
"""

import json
from pathlib import Path
from datetime import datetime


def load_data(data_path: Path) -> dict:
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Slightly expanded keywords to get more posts (original + extra)
USABILITY_KEYWORDS = [
    "usability", "user experience", "ux", "ui", "interface", "design", "intuitive",
    "confusing", "difficult", "hard to use", "user-friendly", "clunky", "awkward",
    "navigation", "workflow", "migration", "optimize", "optimization", "configure",
    "setup", "how to", "best way", "recommend", "alternative", "option",
]
DEFECT_KEYWORDS = [
    "defect", "bug", "error", "issue", "problem", "broken", "crash", "glitch",
    "fault", "failure", "malfunction", "doesn't work", "not working", "fix", "fixing",
    "debug", "troubleshoot", "error message", "exception", "fail", "failed",
    "rto", "bad request", "php error", "fix it", "start from scratch",
]


def relevance_score(post: dict, category: str) -> float:
    """
    Return relevance score for category (usability or defect).
    Uses title and comment context; penalizes strong opposite-category wording in title.
    """
    title = (post.get("title") or "").lower()
    author_replies = post.get("author_replies") or []
    community_comments = post.get("community_comments") or []
    context_parts = [r.get("text", "")[:300].lower() for r in author_replies[:3]]
    context_parts += [c.get("text", "")[:300].lower() for c in community_comments[:5]]
    context = " ".join(context_parts)

    if category == "usability":
        keywords = USABILITY_KEYWORDS
        exclude_keywords = DEFECT_KEYWORDS[:12]
    else:
        keywords = DEFECT_KEYWORDS
        exclude_keywords = USABILITY_KEYWORDS[:12]

    title_score = sum(3 for kw in keywords if kw in title)
    context_score = sum(1 for kw in keywords if kw in context)
    score = title_score * 2 + context_score

    has_exclude = any(kw in title for kw in exclude_keywords)
    if has_exclude and title_score == 0:
        score = max(0, score - 5)

    return score


def get_trajectories(post: dict) -> tuple[list[float], list[float]]:
    """Get author_trajectory and community_trajectory from post (metrics or from replies/comments)."""
    metrics = post.get("metrics") or {}
    if metrics.get("author_trajectory") and metrics.get("community_trajectory"):
        return metrics["author_trajectory"], metrics["community_trajectory"]
    author_replies = post.get("author_replies") or []
    community_comments = post.get("community_comments") or []
    author_traj = [r["sentiment"]["compound"] for r in author_replies]
    community_traj = [c["sentiment"]["compound"] for c in community_comments]
    return author_traj, community_traj


def get_quality_and_counts(post: dict) -> tuple[float, int, int]:
    metrics = post.get("metrics") or {}
    quality = float(metrics.get("overall_quality_score", 0))
    author_count = metrics.get("author_replies_count") or len(post.get("author_replies") or [])
    community_count = metrics.get("community_comments_count") or len(post.get("community_comments") or [])
    return quality, author_count, community_count


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / "data" / "enhanced_automated_sentiment_results.json"
    out_path = project_root / "data" / "usability_defect_posts_extended.json"

    top_n = 30  # aim for up to 30 per category
    min_relevance = 1
    min_quality = 0.3
    min_author_replies = 3
    min_community_comments = 5

    print("Loading enhanced results...")
    data = load_data(data_path)
    posts = data.get("all_analyzed_posts", [])
    print(f"Total posts: {len(posts)}")

    usability_candidates = []
    defect_candidates = []

    for post in posts:
        quality, author_count, community_count = get_quality_and_counts(post)
        if quality < min_quality or author_count < min_author_replies or community_count < min_community_comments:
            continue

        us_score = relevance_score(post, "usability")
        def_score = relevance_score(post, "defect")

        if us_score >= min_relevance and us_score >= def_score:
            usability_candidates.append((us_score, quality, post))
        if def_score >= min_relevance and def_score >= us_score:
            defect_candidates.append((def_score, quality, post))

    # Deduplicate: assign each post to at most one category (usability first, then defect)
    usability_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    defect_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    usability_selected = []
    seen_pids = set()
    for score, quality, post in usability_candidates:
        if len(usability_selected) >= top_n:
            break
        pid = post.get("post_id")
        if pid not in seen_pids:
            seen_pids.add(pid)
            usability_selected.append((score, quality, post))

    defect_selected = []
    for score, quality, post in defect_candidates:
        if len(defect_selected) >= top_n:
            break
        pid = post.get("post_id")
        if pid not in seen_pids:
            seen_pids.add(pid)
            defect_selected.append((score, quality, post))

    def build_post_list(selected: list, category_name: str) -> list:
        out_list = []
        for score, quality, post in selected:
            author_traj, community_traj = get_trajectories(post)
            metrics = post.get("metrics") or {}
            out_list.append({
                "post_id": post.get("post_id"),
                "title": post.get("title"),
                "url": post.get("url"),
                "author": post.get("author"),
                "subreddit": post.get("subreddit"),
                "created_utc": post.get("created_utc"),
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
                "title_sentiment": post.get("title_sentiment", {}),
                "relevance_score": score,
                "quality_score": quality,
                "metrics": metrics,
                "author_trajectory": author_traj,
                "community_trajectory": community_traj,
                "author_replies_count": len(post.get("author_replies") or []),
                "community_comments_count": len(post.get("community_comments") or []),
                "category": category_name,
            })
        return out_list

    usability_posts = build_post_list(usability_selected, "usability")
    defect_posts = build_post_list(defect_selected, "defect")

    output = {
        "analysis_metadata": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "purpose": "Extended usability vs defect sample for trajectory comparison",
            "source_file": str(data_path.name),
            "selection_criteria": {
                "min_relevance_score": min_relevance,
                "min_quality_score": min_quality,
                "min_author_replies": min_author_replies,
                "min_community_comments": min_community_comments,
                "top_n_per_category": top_n,
            },
        },
        "usability_posts": usability_posts,
        "defect_posts": defect_posts,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Usability posts: {len(usability_posts)}")
    print(f"Defect posts: {len(defect_posts)}")
    print(f"Saved: {out_path}")
    print("Next: python3 scripts/compare_usability_defect_extended.py")


if __name__ == "__main__":
    main()
