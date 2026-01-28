"""
Step 1: Statistical Overview of All 324 Posts
Analyze overall patterns and statistics across all posts
"""

import json
import numpy as np
from collections import Counter


def load_data(data_file="data/enhanced_automated_sentiment_results.json"):
    """Load the sentiment analysis results"""
    print("Loading data...")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    posts = data.get('all_analyzed_posts', [])
    print(f"✅ Loaded {len(posts)} posts")
    return posts


def calculate_statistics(posts):
    """Calculate all statistics"""
    print("\n" + "="*80)
    print("STEP 1: STATISTICAL OVERVIEW OF ALL POSTS")
    print("="*80)
    
    total_posts = len(posts)
    print(f"\n📊 Total Posts Analyzed: {total_posts}")
    
    # 1. Sentiment Distribution
    title_sentiments = []
    for post in posts:
        title_sent = post.get('title_sentiment', {}).get('label', 'Unknown')
        title_sentiments.append(title_sent)
    
    sentiment_counts = Counter(title_sentiments)
    print(f"\n📈 Title Sentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total_posts) * 100
        print(f"   {sentiment}: {count} posts ({percentage:.1f}%)")
    
    # 2. Author Sentiment Statistics
    author_sentiments = []
    for post in posts:
        metrics = post.get('metrics', {})
        author_avg = metrics.get('author_avg_sentiment', 0)
        if metrics.get('author_replies_count', 0) > 0:  # Only include posts with author replies
            author_sentiments.append(author_avg)
    
    print(f"\n💬 Author Sentiment Statistics:")
    if author_sentiments:
        print(f"   Mean: {np.mean(author_sentiments):.3f}")
        print(f"   Median: {np.median(author_sentiments):.3f}")
        print(f"   Min: {np.min(author_sentiments):.3f}")
        print(f"   Max: {np.max(author_sentiments):.3f}")
        print(f"   Std Dev: {np.std(author_sentiments):.3f}")
        print(f"   Posts with author replies: {len(author_sentiments)}")
    
    # 3. Community Sentiment Statistics
    community_sentiments = []
    for post in posts:
        metrics = post.get('metrics', {})
        community_avg = metrics.get('community_avg_sentiment', 0)
        if metrics.get('community_comments_count', 0) > 0:  # Only include posts with community comments
            community_sentiments.append(community_avg)
    
    print(f"\n👥 Community Sentiment Statistics:")
    if community_sentiments:
        print(f"   Mean: {np.mean(community_sentiments):.3f}")
        print(f"   Median: {np.median(community_sentiments):.3f}")
        print(f"   Min: {np.min(community_sentiments):.3f}")
        print(f"   Max: {np.max(community_sentiments):.3f}")
        print(f"   Std Dev: {np.std(community_sentiments):.3f}")
        print(f"   Posts with community comments: {len(community_sentiments)}")
    
    # 4. Quality Score Statistics
    quality_scores = []
    for post in posts:
        metrics = post.get('metrics', {})
        quality = metrics.get('overall_quality_score', 0)
        if quality > 0:
            quality_scores.append(quality)
    
    print(f"\n⭐ Quality Score Statistics:")
    if quality_scores:
        print(f"   Mean: {np.mean(quality_scores):.3f}")
        print(f"   Median: {np.median(quality_scores):.3f}")
        print(f"   Min: {np.min(quality_scores):.3f}")
        print(f"   Max: {np.max(quality_scores):.3f}")
        print(f"   Posts with quality scores: {len(quality_scores)}")
    
    # 5. Sentiment Divergence Analysis
    divergences = []
    for post in posts:
        metrics = post.get('metrics', {})
        divergence = metrics.get('sentiment_divergence', 0)
        if divergence > 0:
            divergences.append(divergence)
    
    print(f"\n📏 Sentiment Divergence Statistics:")
    if divergences:
        print(f"   Mean: {np.mean(divergences):.3f}")
        print(f"   Median: {np.median(divergences):.3f}")
        print(f"   Min: {np.min(divergences):.3f}")
        print(f"   Max: {np.max(divergences):.3f}")
        
        # Categorize divergence
        low_div = sum(1 for d in divergences if d < 0.2)
        moderate_div = sum(1 for d in divergences if 0.2 <= d < 0.5)
        high_div = sum(1 for d in divergences if d >= 0.5)
        
        total_div = len(divergences)
        print(f"\n   Divergence Categories:")
        print(f"      Low divergence (<0.2): {low_div} posts ({low_div/total_div*100:.1f}%)")
        print(f"      Moderate divergence (0.2-0.5): {moderate_div} posts ({moderate_div/total_div*100:.1f}%)")
        print(f"      High divergence (>0.5): {high_div} posts ({high_div/total_div*100:.1f}%)")
    
    # 6. Engagement Statistics
    author_reply_counts = []
    community_comment_counts = []
    for post in posts:
        metrics = post.get('metrics', {})
        author_replies = metrics.get('author_replies_count', 0)
        community_comments = metrics.get('community_comments_count', 0)
        author_reply_counts.append(author_replies)
        community_comment_counts.append(community_comments)
    
    print(f"\n💭 Engagement Statistics:")
    print(f"   Author Replies:")
    print(f"      Mean: {np.mean(author_reply_counts):.1f}")
    print(f"      Median: {np.median(author_reply_counts):.1f}")
    print(f"      Min: {np.min(author_reply_counts)}")
    print(f"      Max: {np.max(author_reply_counts)}")
    print(f"   Community Comments:")
    print(f"      Mean: {np.mean(community_comment_counts):.1f}")
    print(f"      Median: {np.median(community_comment_counts):.1f}")
    print(f"      Min: {np.min(community_comment_counts)}")
    print(f"      Max: {np.max(community_comment_counts)}")
    
    # 7. Sentiment Trend Analysis
    improving_authors = 0
    declining_authors = 0
    stable_authors = 0
    
    improving_community = 0
    declining_community = 0
    stable_community = 0
    
    for post in posts:
        metrics = post.get('metrics', {})
        author_shift = metrics.get('author_sentiment_shift', 0)
        community_shift = metrics.get('community_sentiment_shift', 0)
        
        # Author trends (threshold: 0.05)
        if author_shift > 0.05:
            improving_authors += 1
        elif author_shift < -0.05:
            declining_authors += 1
        else:
            stable_authors += 1
        
        # Community trends
        if community_shift > 0.05:
            improving_community += 1
        elif community_shift < -0.05:
            declining_community += 1
        else:
            stable_community += 1
    
    total_with_shifts = total_posts
    print(f"\n📉 Sentiment Trend Analysis:")
    print(f"   Author Sentiment Trends:")
    print(f"      Improving: {improving_authors} ({improving_authors/total_with_shifts*100:.1f}%)")
    print(f"      Declining: {declining_authors} ({declining_authors/total_with_shifts*100:.1f}%)")
    print(f"      Stable: {stable_authors} ({stable_authors/total_with_shifts*100:.1f}%)")
    
    print(f"   Community Sentiment Trends:")
    print(f"      Improving: {improving_community} ({improving_community/total_with_shifts*100:.1f}%)")
    print(f"      Declining: {declining_community} ({declining_community/total_with_shifts*100:.1f}%)")
    print(f"      Stable: {stable_community} ({stable_community/total_with_shifts*100:.1f}%)")
    
    # Summary
    print(f"\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    if author_sentiments:
        print(f"   Average author sentiment: {np.mean(author_sentiments):.3f}")
    if community_sentiments:
        print(f"   Average community sentiment: {np.mean(community_sentiments):.3f}")
    if divergences:
        low_pct = sum(1 for d in divergences if d < 0.2) / len(divergences) * 100
        print(f"   {low_pct:.1f}% show low divergence")
    print(f"   {improving_authors/total_with_shifts*100:.1f}% improving, {declining_authors/total_with_shifts*100:.1f}% declining")
    print("="*80)


def main():
    """Main execution function"""
    try:
        posts = load_data()
        if not posts:
            print("❌ No posts found!")
            return
        
        calculate_statistics(posts)
        
        print("\n✅ Step 1 Complete!")
        print("\nNext: Run step2_subreddit_analysis.py")
        
    except FileNotFoundError:
        print("❌ Error: data/enhanced_automated_sentiment_results.json not found!")
        print("   Make sure you're running from the project root directory.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
