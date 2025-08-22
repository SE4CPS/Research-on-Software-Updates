from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import json

def fetch_reddit_data():
    """Fetch Reddit data from the API"""
    url = "https://releasetrain.io/api/reddit"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  
        reddit_data = response.json()
        print(f"✅ Fetched {len(reddit_data)} Reddit posts.")
        return reddit_data
    except requests.exceptions.RequestException as e:
        print("❌ Failed to fetch data from API.")
        print(e)
        return None

def analyze_and_sort_posts(reddit_data):
    """Analyze sentiments and sort by negative score"""
    analyzer = SentimentIntensityAnalyzer()
    posts_with_analysis = []
    
    print("\n🔍 Analyzing sentiments for all posts...")
    
    for post in reddit_data:
        title = post['title']
        sentiment_scores = analyzer.polarity_scores(title)
        
        analysis_result = {
            'title': title,
            'url': post.get('url', 'N/A'),
            'subreddit': post.get('subreddit', 'N/A'),
            'author': post.get('author', 'N/A'),
            'existing_tag': post.get('tag', 'N/A'),
            'existing_tags': post.get('tags', []),
            'sentiment_scores': sentiment_scores,
            'compound_score': sentiment_scores['compound'],
            'negative_score': sentiment_scores['neg'],
            'positive_score': sentiment_scores['pos'],
            'neutral_score': sentiment_scores['neu']
        }
        
        if sentiment_scores['compound'] >= 0.05:
            analysis_result['sentiment_label'] = "Positive"
        elif sentiment_scores['compound'] <= -0.05:
            analysis_result['sentiment_label'] = "Negative"
        else:
            analysis_result['sentiment_label'] = "Neutral"
        
        posts_with_analysis.append(analysis_result)
    
    print(f" Analyzed {len(posts_with_analysis)} posts.")
    return posts_with_analysis

def get_top_negative_posts(posts_with_analysis, top_n=20):
    """Get the top N posts with highest negative scores"""
    negative_posts = [post for post in posts_with_analysis if post['sentiment_label'] == 'Negative']
    
    negative_posts.sort(key=lambda x: x['negative_score'], reverse=True)
    
    print(f"\n Found {len(negative_posts)} negative posts out of {len(posts_with_analysis)} total posts.")
    print(f" Selecting top {top_n} most negative posts...")
    
    return negative_posts[:top_n]

def display_top_negative_posts(top_negative_posts):
    """Display the top negative posts in a clear format"""
    print(f"\nTOP 20 NEGATIVE POSTS (Sorted by Negative Score)")
    print("=" * 80)
    
    for i, post in enumerate(top_negative_posts, 1):
        print(f"\n{i}. TITLE: {post['title']}")
        print(f"   SUBREDDIT: r/{post['subreddit']}")
        print(f"   AUTHOR: {post['author']}")
        print(f"   URL: {post['url']}")
        print(f"   SENTIMENT ANALYSIS:")
        print(f"     - Label: {post['sentiment_label']}")
        print(f"     - Negative Score: {post['negative_score']:.3f}")
        print(f"     - Positive Score: {post['positive_score']:.3f}")
        print(f"     - Neutral Score: {post['neutral_score']:.3f}")
        print(f"     - Compound Score: {post['compound_score']:.3f}")
        print(f"   EXISTING CLASSIFICATION:")
        print(f"     - Tag: {post['existing_tag']}")
        print(f"     - Tags: {post['existing_tags']}")
        print("-" * 80)

def save_results_to_file(top_negative_posts, filename="top_20_negative_posts.json"):
    """Save the results to a JSON file"""
    results = {
        "total_posts_analyzed": len(posts_with_analysis),
        "total_negative_posts": len([p for p in posts_with_analysis if p['sentiment_label'] == 'Negative']),
        "top_20_negative_posts": top_negative_posts
    }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Results saved to: {filename}")

def main():
    """Main function to execute the analysis"""
    print("🚀 Starting Top 20 Negative Posts Analysis...")
    print("=" * 50)
    
    reddit_data = fetch_reddit_data()
    if not reddit_data:
        return
    
    global posts_with_analysis
    posts_with_analysis = analyze_and_sort_posts(reddit_data)
    
    top_negative_posts = get_top_negative_posts(posts_with_analysis, 20)
    
    display_top_negative_posts(top_negative_posts)
    
    save_results_to_file(top_negative_posts)
    
    print(f"\n SUMMARY:")
    print(f"   Total posts analyzed: {len(posts_with_analysis)}")
    print(f"   Total negative posts: {len([p for p in posts_with_analysis if p['sentiment_label'] == 'Negative'])}")
    print(f"   Posts with tag='None': {len([p for p in top_negative_posts if p['existing_tag'] == 'None'])}")
    print(f"   Posts with tag='Help Request': {len([p for p in top_negative_posts if p['existing_tag'] == 'Help Request'])}")
    
    print(f"\n Analysis complete! Review the top 20 negative posts above.")
    print("   Next step: Manually review each post by visiting the URLs.")

if __name__ == "__main__":
    main()
