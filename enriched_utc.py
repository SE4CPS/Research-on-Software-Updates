import json
from datetime import datetime, timezone

# reddit object to run parser
reddit_obj = {
    "_id": "68152b1cb334ed1a291f3a94",
    "redditId": "1kdaxlx",
    "author": "BcuzRacecar",
    "awards": [],
    "created_utc": "2025-05-02T20:26:35",
    "num_comments": 0,
    "score": 1,
    "subreddit": "Android",
    "tag": None,
    "tags": [],
    "title": "High-end smartphone at a mid-range price - Tecno Camon 40 Premier 5G review",
    "updatedAt": "2025-05-02T20:29:16.086Z",
    "upvote_ratio": 1,
    "url": "https://reddit.com/r/Android/comments/1kdaxlx/highend_smartphone_at_a_midrange_price_tecno/",
    "versionList": ["40.0.0"],
    "classification": {
        "securityType": ["UNKNOWN"],
        "componentType": [],
        "breakingType": []
    }
}

def enrich_created_utc(obj):
    try:
        # Parse created_utc string into datetime object
        dt = datetime.strptime(obj["created_utc"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)

        # Calculate date difference
        delta_days = (now.date() - dt.date()).days
        if delta_days == 0:
            relative = "Today"
        elif delta_days == 1:
            relative = "Yesterday"
        elif delta_days > 1:
            relative = f"{delta_days} days ago"
        else:
            relative = f"In {-delta_days} days"

        # Create enriched array
        obj["created_utc_enriched"] = [
            relative,
            dt.strftime("%A"),      # Day of week
            dt.strftime("%B"),      # Month name
            dt.strftime("%-d"),     # Day of month (use %#d on Windows)
            dt.strftime("%Y"),      # Year
            dt.strftime("%H:%M:%S"),# Time
            "UTC"                   # Timezone
        ]
    except Exception as e:
        print(f"Failed to parse date: {e}")
        obj["created_utc_enriched"] = ["Invalid date"]

    return obj

# Enrich the object
enriched_obj = enrich_created_utc(reddit_obj)

# Print result (nicely formatted)
print(json.dumps(enriched_obj, indent=2))
