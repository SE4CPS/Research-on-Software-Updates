# API Structure and Comments Data Explanation

## API Endpoint

**We are NOT using the official Reddit API directly.**

Instead, we're using a **custom API endpoint**:
```
https://releasetrain.io/api/reddit
```

This is a third-party API that aggregates Reddit posts and provides them in a structured format.

---

## Where We Use the API

### File: `scripts/enhanced_automated_sentiment_analysis.py`

#### 1. Fetching Data from API

```python
def fetch_reddit_posts(self):
    """Fetch all Reddit posts from the API"""
    response = requests.get(self.api_url)  # https://releasetrain.io/api/reddit
    self.posts_data = response.json()
    return True
```

**Location**: Lines 16-27

#### 2. Accessing Comments from API Response

```python
def analyze_post_comments(self, post):
    """Analyze all comments in a post, separating author replies from others"""
    comments = post.get('comments', [])  # ← Comments come from API response
    
    for comment in sorted_comments:
        comment_text = comment.get('body', '')        # Comment text
        comment_author = comment.get('author', '')    # Comment author
        is_author = comment.get('is_submitter', False)  # Is original poster
        score = comment.get('score', 0)              # Comment score
        created_utc = comment.get('created_utc', '') # Timestamp
```

**Location**: Lines 58-96

---

## API Response Structure

The API returns an array of post objects. Each post object has this structure:

```json
{
  "redditId": "1o094i5",
  "title": "How to optimize my wordpress site...",
  "author": "Khan8213",
  "subreddit": "Wordpress",
  "url": "https://reddit.com/r/Wordpress/comments/...",
  "score": 6,
  "num_comments": 60,
  "created_utc": "2025-10-07T08:31:18",
  "author_description": "...",
  
  "comments": [  // ← THIS IS THE KEY: Comments array included in API response
    {
      "body": "No sir it doesn't? As in PSI...",
      "author": "Khan8213",
      "is_submitter": true,  // ← Identifies if commenter is original post author
      "score": 1,
      "created_utc": "2025-10-07T09:09:15"
    },
    {
      "body": "Try using cloudflare...",
      "author": "some_user",
      "is_submitter": false,  // ← Community member comment
      "score": 11,
      "created_utc": "2025-10-07T00:51:48"
    }
    // ... more comments
  ]
}
```

---

## Key Points

### ✅ Yes, Comments Are Included in API Response

The API endpoint (`https://releasetrain.io/api/reddit`) **includes comments** for each post in the response. We don't need to make separate API calls to get comments.

### How We Access Comments

```python
# In enhanced_automated_sentiment_analysis.py, line 60
comments = post.get('comments', [])  # Gets comments array from API response
```

### Comment Structure Expected

Each comment object in the `comments` array has:
- `body`: The comment text
- `author`: Username of commenter
- `is_submitter`: Boolean - true if commenter is the original post author
- `score`: Upvote/downvote score
- `created_utc`: Timestamp

---

## Code Flow

```
1. Fetch from API
   ↓
   requests.get("https://releasetrain.io/api/reddit")
   ↓
   Returns: Array of post objects (each with 'comments' array)

2. Process Each Post
   ↓
   for post in posts_data:
       comments = post.get('comments', [])  # ← Comments already in response
       ↓
       for comment in comments:
           analyze sentiment
           separate author vs community
           create trajectory

3. Save Results
   ↓
   Save to enhanced_automated_sentiment_results.json
```

---

## About API Changes

You mentioned remembering an API that didn't have comments, then it was updated. This suggests:

1. **Earlier Version**: The `releasetrain.io/api/reddit` endpoint may have initially returned posts without comments
2. **Updated Version**: The API was enhanced to include the `comments` array in each post
3. **Current State**: The API now includes full comment data, which is why we can do trajectory analysis

---

## Evidence from Code

### The code expects comments to be present:

```python
# Line 60: Directly accesses 'comments' key
comments = post.get('comments', [])

# Line 70-72: Accesses comment properties
comment_text = comment.get('body', '')
comment_author = comment.get('author', '')
is_author = comment.get('is_submitter', False)
```

If comments weren't in the API response, this code would fail or return empty arrays.

---

## Summary

1. **API Endpoint**: `https://releasetrain.io/api/reddit` (custom API, not official Reddit API)
2. **Comments Included**: Yes, each post object has a `comments` array
3. **Code Location**: `scripts/enhanced_automated_sentiment_analysis.py`
   - Line 20: Fetches from API
   - Line 60: Accesses comments from API response
   - Lines 69-90: Processes each comment
4. **API Update**: Likely the API was updated to include comments, enabling trajectory analysis

---

## Files That Use the API

1. **`scripts/enhanced_automated_sentiment_analysis.py`**
   - Main script that fetches and processes posts with comments
   - Line 11: API URL definition
   - Line 20: API fetch
   - Line 60: Comments access

2. **`scripts/filter_posts_by_comments.py`**
   - Line 8: API URL
   - Line 11: API fetch
   - Note: This script only checks `num_comments`, doesn't process comment content

3. **`scripts/check_and_refresh_data.py`**
   - Uses `EnhancedAutomatedSentimentAnalyzer` which fetches from API
   - Line 23: API URL parameter
