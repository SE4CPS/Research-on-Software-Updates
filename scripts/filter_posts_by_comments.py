from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import json
from datetime import datetime

def fetch_reddit_data():
    """Fetch Reddit posts from the API"""
    url = "https://releasetrain.io/api/reddit"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        reddit_data = response.json()
        print(f"✅ Fetched {len(reddit_data)} Reddit posts from API")
        return reddit_data
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch data from API: {e}")
        return None

def filter_posts_by_comments(reddit_data, min_comments=10):
    """Filter posts that have minimum number of comments"""
    filtered_posts = []
    
    for post in reddit_data:
        if 'num_comments' in post and post['num_comments'] >= min_comments:
            filtered_posts.append(post)
    
    print(f"✅ Found {len(filtered_posts)} posts with {min_comments}+ comments")
    return filtered_posts

def categorize_by_sentiment(posts):
    """Categorize posts by sentiment using VADER"""
    analyzer = SentimentIntensityAnalyzer()
    
    positive_posts = []
    negative_posts = []
    neutral_posts = []
    
    for post in posts:
        title = post['title']
        sentiment_scores = analyzer.polarity_scores(title)
        compound = sentiment_scores['compound']
        
        post['sentiment_scores'] = sentiment_scores
        post['compound_score'] = compound
        
        if compound >= 0.05:
            post['sentiment_label'] = "Positive"
            positive_posts.append(post)
        elif compound <= -0.05:
            post['sentiment_label'] = "Negative"
            negative_posts.append(post)
        else:
            post['sentiment_label'] = "Neutral"
            neutral_posts.append(post)
    
    print(f"📊 Sentiment categorization:")
    print(f"  Positive posts: {len(positive_posts)}")
    print(f"  Negative posts: {len(negative_posts)}")
    print(f"  Neutral posts: {len(neutral_posts)}")
    
    return positive_posts, negative_posts, neutral_posts

def get_top_posts_by_sentiment(posts, category, top_n=5):
    """Get top N posts from a category, ranked by absolute sentiment score"""
    if not posts:
        print(f"❌ No {category} posts found")
        return []
    
    sorted_posts = sorted(posts, key=lambda x: abs(x['compound_score']), reverse=True)
    top_posts = sorted_posts[:top_n]
    
    print(f"\n🏆 Top {len(top_posts)} {category} posts (ranked by sentiment intensity):")
    for i, post in enumerate(top_posts, 1):
        print(f"  {i}. {post['title']}")
        print(f"     Score: {post['compound_score']:.3f} | Comments: {post['num_comments']} | Subreddit: {post['subreddit']}")
        print(f"     URL: {post['url']}")
        print()
    
    return top_posts

def save_to_json(positive_posts, negative_posts, neutral_posts, filename):
    """Save the filtered posts to a JSON file"""
    output_data = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "analysis_purpose": "Sentiment trajectory visualization - posts with 10+ comments",
        "total_posts_analyzed": len(positive_posts) + len(negative_posts) + len(neutral_posts),
        "sentiment_thresholds": {
            "positive": "compound >= 0.05",
            "negative": "compound <= -0.05", 
            "neutral": "-0.05 < compound < 0.05"
        },
        "top_positive_posts": positive_posts,
        "top_negative_posts": negative_posts,
        "top_neutral_posts": neutral_posts
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved results to {filename}")

def main():
    """Main function to execute the filtering process"""
    print("🚀 Starting Reddit posts filtering for sentiment trajectory analysis...\n")
    
    reddit_data = fetch_reddit_data()
    if not reddit_data:
        return
    
    filtered_posts = filter_posts_by_comments(reddit_data, min_comments=10)
    if not filtered_posts:
        print("❌ No posts found with 10+ comments")
        return
    
    positive_posts, negative_posts, neutral_posts = categorize_by_sentiment(filtered_posts)
    
    top_positive = get_top_posts_by_sentiment(positive_posts, "Positive", 10)
    top_negative = get_top_posts_by_sentiment(negative_posts, "Negative", 10)
    top_neutral = get_top_posts_by_sentiment(neutral_posts, "Neutral", 10)
    
    output_filename = "filtered_posts_for_sentiment_analysis.json"
    save_to_json(top_positive, top_negative, top_neutral, output_filename)
    
    print("\n🎯 Summary:")
    print(f"  Total posts with 10+ comments: {len(filtered_posts)}")
    print(f"  Top 10 Positive posts: {len(top_positive)}")
    print(f"  Top 10 Negative posts: {len(top_negative)}")
    print(f"  Top 10 Neutral posts: {len(top_neutral)}")
    print(f"\n📁 Results saved to: {output_filename}")
    print("\n📝 Next step: Manually review the top 10 posts from each category and select the final 3 for visualization.")
    print("   Note: Title sentiment scores may not reflect actual post sentiment - manual verification required!")

if __name__ == "__main__":
    main()
