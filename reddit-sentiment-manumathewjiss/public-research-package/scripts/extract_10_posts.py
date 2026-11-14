import json

def extract_10_posts():
    """Extract 5 most negative and 5 least negative posts from top 20"""
    
    with open('top_20_negative_posts.json', 'r') as f:
        data = json.load(f)
    
    top_20_posts = data['top_20_negative_posts']
    
    print("📊 EXTRACTING 10 POSTS FOR EMOTIONAL JOURNEY ANALYSIS")
    print("="*70)
    
    top_5_most_negative = top_20_posts[:5]
    
    bottom_5_least_negative = top_20_posts[-5:]
    
    print("\n🔴 TOP 5 MOST NEGATIVE POSTS:")
    print("-" * 50)
    for i, post in enumerate(top_5_most_negative, 1):
        print(f"\n{i}. TITLE: {post['title']}")
        print(f"   SUBREDDIT: r/{post['subreddit']}")
        print(f"   AUTHOR: {post['author']}")
        print(f"   URL: {post['url']}")
        print(f"   NEGATIVE SCORE: {post['negative_score']:.3f}")
        print(f"   COMPOUND SCORE: {post['compound_score']:.3f}")
    
    print("\n🟢 BOTTOM 5 LEAST NEGATIVE POSTS (from top 20):")
    print("-" * 50)
    for i, post in enumerate(bottom_5_least_negative, 1):
        print(f"\n{i}. TITLE: {post['title']}")
        print(f"   SUBREDDIT: r/{post['subreddit']}")
        print(f"   AUTHOR: {post['author']}")
        print(f"   URL: {post['url']}")
        print(f"   NEGATIVE SCORE: {post['negative_score']:.3f}")
        print(f"   COMPOUND SCORE: {post['compound_score']:.3f}")
    
    selected_posts = {
        "analysis_date": "2025-08-19",
        "source": "Extracted from top_20_negative_posts.json",
        "top_5_most_negative": top_5_most_negative,
        "bottom_5_least_negative": bottom_5_least_negative
    }
    
    with open('selected_10_posts.json', 'w') as f:
        json.dump(selected_posts, f, indent=2)
    
    print(f"\n✅ Saved 10 selected posts to 'selected_10_posts.json'")
    
    print(f"\n📋 MANUAL ANALYSIS CHECKLIST:")
    print(f"   1. Visit each Reddit URL above")
    print(f"   2. Read the post description")
    print(f"   3. Read the comments")
    print(f"   4. Track emotional journey: Title → Description → Comments → Resolution")
    print(f"   5. Note if user got help/resolution")
    
    return selected_posts

if __name__ == "__main__":
    extract_10_posts()
