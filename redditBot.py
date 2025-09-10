import requests
import time
import json
from datetime import datetime
import re

# 🔹 Reddit API Credentials
CLIENT_ID = "d9v_G93sPXd5eYfDe4PR0g"
CLIENT_SECRET = "ZD0Rd39NKYmukKl0UUC-4N_n8XULvA"
USER_AGENT = "script:releasetrain:v1.0 (by u/Cultural_Let_6867)"
API_BASE_URL = "https://releasetrain.io/api"

# 🔹 Global variable to store token
REDDIT_TOKEN = None

# ✅ Get OAuth Token
def get_reddit_token():
    global REDDIT_TOKEN
    auth = requests.auth.HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)
    data = {"grant_type": "client_credentials"}
    headers = {"User-Agent": USER_AGENT}

    response = requests.post("https://www.reddit.com/api/v1/access_token",
                             auth=auth, data=data, headers=headers)
    response.raise_for_status()
    REDDIT_TOKEN = response.json()["access_token"]

# ✅ Fetch components from API
def fetch_components():
    try:
        response = requests.get(f"{API_BASE_URL}/c/frequency")
        response.raise_for_status()
        data = response.json()
        return set(data.get("components", []))  # ✅ Ensure uniqueness using a set
    except requests.exceptions.RequestException as err:
        print(f"[ERROR] Failed to fetch components: {err}")
        return set()  # Return empty set if there's an error

# ✅ Check if a subreddit exists
def subreddit_exists(subreddit):
    if not REDDIT_TOKEN:
        get_reddit_token()  # Get token if not set

    url = f"https://oauth.reddit.com/r/{subreddit}/about"
    headers = {
        "Authorization": f"bearer {REDDIT_TOKEN}",
        "User-Agent": USER_AGENT
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return True  # ✅ Subreddit exists
    elif response.status_code in [403, 404]:
        print(f"[SKIPPING] Subreddit '{subreddit}' does not exist or is private.")
        return False
    else:
        print(f"[ERROR] Unexpected error checking subreddit '{subreddit}': {response.status_code}")
        return False

# ✅ Fetch Reddit posts with OAuth
def fetch_hot_posts(subreddit, limit=10):
    if not REDDIT_TOKEN:
        get_reddit_token()  # Get token if not set

    if not subreddit_exists(subreddit):
        return []  # Skip if subreddit doesn't exist

    url = f"https://oauth.reddit.com/r/{subreddit}/hot?limit={limit}"
    headers = {
        "Authorization": f"bearer {REDDIT_TOKEN}",
        "User-Agent": USER_AGENT
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code == 403:
        print("[ERROR] Access denied. Check your OAuth credentials.")
        return []
    elif response.status_code == 429:
        print("[ERROR] Rate-limited. Try again later.")
        time.sleep(60)  # Wait and retry
        return fetch_hot_posts(subreddit, limit)  # Retry request

    response.raise_for_status()
    return response.json().get("data", {}).get("children", [])

# ✅ Extract version numbers from post titles
def check_for_version_numbers(post_title):
    version_pattern = r'\b(?:v)?(\d+)(?:\.(\d+))?(?:\.(\d+))?\b'
    matches = re.findall(version_pattern, post_title)
    return [f"{major}.{minor or '0'}.{patch or '0'}" for major, minor, patch in matches]

# ✅ Post Reddit data to API
def post_reddit_entry(post_data):
    url = f"{API_BASE_URL}/reddit"

    try:
        response = requests.post(url, json=post_data)
        response.raise_for_status()
        print(f"[SUCCESS] Posted to {url}: {post_data['title']}")
        return response.json()
    except requests.exceptions.RequestException as err:
        print(f"[ERROR] Failed to post to {url}: {err}")
        return {}

# ✅ Process and Post Reddit Data
def process_and_post_reddit_data(posts):
    for post in posts:
        data = post.get('data', {})  # ✅ Safely get 'data' dictionary

        # ✅ Check if required keys exist
        if 'subreddit' not in data or 'title' not in data or 'id' not in data:
            print(f"[WARN] Skipping post with missing data: {json.dumps(data, indent=2)}")
            continue  # Skip this post

        versions = check_for_version_numbers(data['title'])

        reddit_entry = {
            "redditId": data['id'],  
            "title": data['title'],
            "subreddit": data['subreddit'],
            "score": data.get('score', 0),
            "created_utc": datetime.utcfromtimestamp(data['created_utc']).isoformat() if 'created_utc' in data else "N/A",
            "url": f"https://reddit.com{data.get('permalink', '#')}",
            "author": data.get('author', "Unknown"),
            "num_comments": data.get('num_comments', 0),
            "upvote_ratio": data.get('upvote_ratio', 0),
            "awards": data.get('all_awardings', []),
            "versionList": versions,
            "tag": data.get('link_flair_text', None),  # Main flair (e.g., "Privacy")
            "tags": [tag.get("t") for tag in data.get("link_flair_richtext", []) if "t" in tag]  # Extract flair tags
        }

        post_reddit_entry(reddit_entry)

# 🔹 Merge dynamic components from API
all_components = fetch_components()

# ✅ Fetch and process posts for each component
if all_components:
    for component in all_components:
        print(f"[PROCESSING] Checking subreddit: {component}")

        if not subreddit_exists(component):
            continue  # ✅ Skip if subreddit doesn't exist

        print(f"[FETCHING] Fetching posts for: {component}")
        subreddit_posts = fetch_hot_posts(component, limit=10)

        if subreddit_posts:
            process_and_post_reddit_data(subreddit_posts)
        else:
            print(f"[WARN] No posts found in subreddit {component}")
else:
    print("[ERROR] No components to process.")
