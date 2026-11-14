from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import json
from datetime import datetime
from collections import defaultdict


class EnhancedAutomatedSentimentAnalyzer:
    """Enhanced class with stricter filtering for trajectory reliability"""
    
    def __init__(self, api_url="https://releasetrain.io/api/reddit"):
        self.api_url = api_url
        self.analyzer = SentimentIntensityAnalyzer()
        self.posts_data = []
        
    def fetch_reddit_posts(self):
        """Fetch all Reddit posts from the API"""
        print("🔄 Fetching Reddit posts from API...")
        try:
            response = requests.get(self.api_url)
            response.raise_for_status()
            self.posts_data = response.json()
            print(f"✅ Fetched {len(self.posts_data)} Reddit posts")
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to fetch data from API: {e}")
            return False
    
    def analyze_text_sentiment(self, text):
        """Analyze sentiment of a text using VADER"""
        if not text or text.strip() == "":
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 1.0,
                'neg': 0.0,
                'label': 'Neutral'
            }
        
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            label = 'Positive'
        elif compound <= -0.05:
            label = 'Negative'
        else:
            label = 'Neutral'
        
        return {
            'compound': compound,
            'pos': scores['pos'],
            'neu': scores['neu'],
            'neg': scores['neg'],
            'label': label
        }
    
    def analyze_post_comments(self, post):
        """Analyze all comments in a post, separating author replies from others"""
        comments = post.get('comments', [])
        author = post.get('author', '')
        
        author_replies = []
        community_comments = []
        
        # Sort comments by timestamp to maintain chronological order
        sorted_comments = sorted(comments, key=lambda x: x.get('created_utc', ''))
        
        for comment in sorted_comments:
            comment_text = comment.get('body', '')
            comment_author = comment.get('author', '')
            is_author = comment.get('is_submitter', False)
            
            # Analyze sentiment
            sentiment = self.analyze_text_sentiment(comment_text)
            
            comment_data = {
                'author': comment_author,
                'text': comment_text[:200],
                'sentiment': sentiment,
                'score': comment.get('score', 0),
                'created_utc': comment.get('created_utc', ''),
                'is_author': is_author
            }
            
            # Separate by author vs community
            if is_author or comment_author == author:
                author_replies.append(comment_data)
            else:
                community_comments.append(comment_data)
        
        return {
            'author_replies': author_replies,
            'community_comments': community_comments,
            'total_comments': len(comments)
        }
    
    def calculate_enhanced_metrics(self, analyzed_comments):
        """Calculate enhanced metrics including trajectory reliability"""
        author_replies = analyzed_comments['author_replies']
        community_comments = analyzed_comments['community_comments']
        
        # Basic metrics
        author_avg = sum(r['sentiment']['compound'] for r in author_replies) / len(author_replies) if author_replies else 0.0
        community_avg = sum(c['sentiment']['compound'] for c in community_comments) / len(community_comments) if community_comments else 0.0
        
        # Build sentiment trajectories
        author_trajectory = [r['sentiment']['compound'] for r in author_replies]
        community_trajectory = [c['sentiment']['compound'] for c in community_comments]
        
        # Calculate sentiment shift (early vs late)
        def calculate_shift(trajectory):
            if len(trajectory) < 4:
                return 0.0
            early = sum(trajectory[:len(trajectory)//2]) / (len(trajectory)//2)
            late = sum(trajectory[len(trajectory)//2:]) / (len(trajectory) - len(trajectory)//2)
            return late - early
        
        author_shift = calculate_shift(author_trajectory) if author_trajectory else 0.0
        community_shift = calculate_shift(community_trajectory) if community_comments else 0.0
        
        # NEW: Calculate trajectory reliability metrics
        def calculate_trajectory_reliability(trajectory):
            if len(trajectory) < 2:
                return 0.0  # Not enough data for reliability
            
            # Calculate variance (lower = more consistent)
            mean = sum(trajectory) / len(trajectory)
            variance = sum((x - mean) ** 2 for x in trajectory) / len(trajectory)
            
            # Calculate trend strength (how much sentiment changes over time)
            if len(trajectory) >= 3:
                # Simple linear trend calculation
                x_values = list(range(len(trajectory)))
                n = len(trajectory)
                sum_x = sum(x_values)
                sum_y = sum(trajectory)
                sum_xy = sum(x * y for x, y in zip(x_values, trajectory))
                sum_x2 = sum(x * x for x in x_values)
                
                # Calculate slope
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                trend_strength = abs(slope)
            else:
                trend_strength = 0.0
            
            # Reliability score (0-1, higher is better)
            # Based on: sufficient data points, low variance, clear trend
            data_points_score = min(len(trajectory) / 10, 1.0)  # Optimal at 10+ points
            consistency_score = max(0, 1 - variance)  # Lower variance = higher score
            trend_score = min(trend_strength * 2, 1.0)  # Some trend is good
            
            reliability = (data_points_score * 0.4 + consistency_score * 0.3 + trend_score * 0.3)
            return reliability
        
        author_reliability = calculate_trajectory_reliability(author_trajectory)
        community_reliability = calculate_trajectory_reliability(community_trajectory)
        
        # Overall post quality score
        quality_score = (
            (len(author_replies) / 10) * 0.3 +  # Author engagement
            (len(community_comments) / 20) * 0.3 +  # Community engagement
            author_reliability * 0.2 +  # Author trajectory reliability
            community_reliability * 0.2   # Community trajectory reliability
        )
        
        return {
            'author_avg_sentiment': round(author_avg, 3),
            'community_avg_sentiment': round(community_avg, 3),
            'sentiment_divergence': round(abs(author_avg - community_avg), 3),
            'author_trajectory': author_trajectory,
            'community_trajectory': community_trajectory,
            'author_sentiment_shift': round(author_shift, 3),
            'community_sentiment_shift': round(community_shift, 3),
            'author_replies_count': len(author_replies),
            'community_comments_count': len(community_comments),
            # NEW: Enhanced metrics
            'author_trajectory_reliability': round(author_reliability, 3),
            'community_trajectory_reliability': round(community_reliability, 3),
            'overall_quality_score': round(quality_score, 3)
        }
    
    def analyze_all_posts(self, min_comments=10, min_author_replies=3, min_community_comments=5):
        """
        Analyze all posts with enhanced filtering criteria
        """
        print(f"\n🧠 Analyzing sentiment with enhanced filtering...")
        print(f"   Minimum total comments: {min_comments}")
        print(f"   Minimum author replies: {min_author_replies}")
        print(f"   Minimum community comments: {min_community_comments}")
        
        analyzed_posts = []
        
        for post in self.posts_data:
            num_comments = post.get('num_comments', 0)
            
            # Filter by minimum comments
            if num_comments < min_comments:
                continue
            
            # Analyze title
            title = post.get('title', '')
            title_sentiment = self.analyze_text_sentiment(title)
            
            # Analyze description (if exists)
            description = post.get('author_description', '')
            description_sentiment = self.analyze_text_sentiment(description) if description else None
            
            # Analyze all comments
            analyzed_comments = self.analyze_post_comments(post)
            
            # ENHANCED: Check if post meets minimum criteria
            author_count = len(analyzed_comments['author_replies'])
            community_count = len(analyzed_comments['community_comments'])
            
            if author_count < min_author_replies or community_count < min_community_comments:
                continue  # Skip posts that don't meet enhanced criteria
            
            # Calculate enhanced metrics
            metrics = self.calculate_enhanced_metrics(analyzed_comments)
            
            # Compile results
            post_result = {
                'post_id': post.get('redditId', ''),
                'title': title,
                'url': post.get('url', ''),
                'author': post.get('author', ''),
                'subreddit': post.get('subreddit', ''),
                'created_utc': post.get('created_utc', ''),
                'score': post.get('score', 0),
                'upvote_ratio': post.get('upvote_ratio', 0),
                'num_comments': num_comments,
                
                # Sentiment analysis
                'title_sentiment': title_sentiment,
                'description_sentiment': description_sentiment,
                
                # Comment analysis
                'author_replies': analyzed_comments['author_replies'],
                'community_comments': analyzed_comments['community_comments'],
                
                # Enhanced metrics
                'metrics': metrics
            }
            
            analyzed_posts.append(post_result)
        
        print(f"✅ Analyzed {len(analyzed_posts)} posts meeting enhanced criteria")
        return analyzed_posts
    
    def select_top_posts_enhanced(self, analyzed_posts, top_n=3, min_quality_score=0.3):
        """
        Select top posts using enhanced criteria including quality scoring
        """
        print(f"\n📊 Selecting top {top_n} posts with enhanced criteria...")
        print(f"   Minimum quality score: {min_quality_score}")
        
        positive_posts = []
        negative_posts = []
        neutral_posts = []
        
        # Filter by quality score
        quality_posts = [p for p in analyzed_posts if p['metrics']['overall_quality_score'] >= min_quality_score]
        print(f"   Posts meeting quality threshold: {len(quality_posts)}")
        
        # Categorize by title sentiment
        for post in quality_posts:
            compound = post['title_sentiment']['compound']
            
            if compound >= 0.05:
                positive_posts.append(post)
            elif compound <= -0.05:
                negative_posts.append(post)
            else:
                neutral_posts.append(post)
        
        # Sort by quality score first, then by sentiment intensity
        def sort_key(post):
            return (
                post['metrics']['overall_quality_score'],  # Quality first
                abs(post['title_sentiment']['compound'])   # Then sentiment intensity
            )
        
        positive_top = sorted(positive_posts, key=sort_key, reverse=True)[:top_n]
        negative_top = sorted(negative_posts, key=sort_key, reverse=True)[:top_n]
        neutral_top = sorted(neutral_posts, key=sort_key, reverse=True)[:top_n]
        
        print(f"   Positive posts: {len(positive_posts)} → Selected top {len(positive_top)}")
        print(f"   Negative posts: {len(negative_posts)} → Selected top {len(negative_top)}")
        print(f"   Neutral posts: {len(neutral_posts)} → Selected top {len(neutral_top)}")
        
        # Print quality scores for selected posts
        print(f"\n📈 Quality scores for selected posts:")
        for category, posts in [('Positive', positive_top), ('Negative', negative_top), ('Neutral', neutral_top)]:
            print(f"   {category}:")
            for i, post in enumerate(posts, 1):
                quality = post['metrics']['overall_quality_score']
                author_count = post['metrics']['author_replies_count']
                community_count = post['metrics']['community_comments_count']
                print(f"     {i}. Quality: {quality:.3f} | Author: {author_count} | Community: {community_count}")
        
        return {
            'positive': positive_top,
            'negative': negative_top,
            'neutral': neutral_top
        }
    
    def save_enhanced_results(self, analyzed_posts, top_posts, output_file):
        """Save enhanced analysis results to JSON file"""
        print(f"\n💾 Saving enhanced results to {output_file}...")
        
        results = {
            'analysis_metadata': {
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'api_endpoint': self.api_url,
                'total_posts_fetched': len(self.posts_data),
                'total_posts_analyzed': len(analyzed_posts),
                'sentiment_model': 'VADER (Valence Aware Dictionary and sEntiment Reasoner)',
                'enhanced_filtering': {
                    'min_total_comments': 10,
                    'min_author_replies': 3,
                    'min_community_comments': 5,
                    'min_quality_score': 0.3
                },
                'sentiment_thresholds': {
                    'positive': 'compound >= 0.05',
                    'negative': 'compound <= -0.05',
                    'neutral': '-0.05 < compound < 0.05'
                }
            },
            'all_analyzed_posts': analyzed_posts,
            'top_posts': top_posts,
            'summary': {
                'positive_posts': len(top_posts['positive']),
                'negative_posts': len(top_posts['negative']),
                'neutral_posts': len(top_posts['neutral'])
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Enhanced results saved successfully!")
        return results


def main():
    """Main execution function with enhanced filtering"""
    print("=" * 80)
    print("🚀 ENHANCED AUTOMATED SENTIMENT ANALYSIS")
    print("=" * 80)
    
    # Initialize enhanced analyzer
    analyzer = EnhancedAutomatedSentimentAnalyzer()
    
    # Step 1: Fetch posts
    if not analyzer.fetch_reddit_posts():
        print("❌ Failed to fetch posts. Exiting.")
        return
    
    # Step 2: Analyze with enhanced filtering
    analyzed_posts = analyzer.analyze_all_posts(
        min_comments=10,
        min_author_replies=3,      # NEW: Minimum author replies
        min_community_comments=5   # NEW: Minimum community comments
    )
    
    if not analyzed_posts:
        print("❌ No posts found meeting enhanced criteria. Exiting.")
        return
    
    # Step 3: Select top posts with quality scoring
    top_posts = analyzer.select_top_posts_enhanced(
        analyzed_posts, 
        top_n=3,
        min_quality_score=0.3  # NEW: Minimum quality threshold
    )
    
    # Step 4: Save enhanced results
    output_file = "data/enhanced_automated_sentiment_results.json"
    results = analyzer.save_enhanced_results(analyzed_posts, top_posts, output_file)
    
    # Print enhanced summary
    print("\n" + "=" * 80)
    print("📈 ENHANCED ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total posts analyzed: {len(analyzed_posts)}")
    print(f"\nTop posts selected (with quality scores):")
    
    for category, posts in [('Positive', top_posts['positive']), ('Negative', top_posts['negative']), ('Neutral', top_posts['neutral'])]:
        print(f"\n{category} Posts:")
        for i, post in enumerate(posts, 1):
            quality = post['metrics']['overall_quality_score']
            author_count = post['metrics']['author_replies_count']
            community_count = post['metrics']['community_comments_count']
            title = post['title'][:60] + "..." if len(post['title']) > 60 else post['title']
            print(f"  {i}. Quality: {quality:.3f} | Author: {author_count} | Community: {community_count}")
            print(f"     Title: {title}")
    
    print(f"\n📁 Enhanced results saved to: {output_file}")
    print(f"\n💡 Next step: Run 'python scripts/enhanced_visualization.py' to generate charts")
    print("=" * 80)


if __name__ == "__main__":
    main()
