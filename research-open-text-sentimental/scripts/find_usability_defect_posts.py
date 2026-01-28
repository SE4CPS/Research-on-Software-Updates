"""
Script to find posts related to usability and software defects
from the enhanced sentiment analysis results.
Automatically checks and refreshes data if outdated.
"""

import json
import re
import os
import sys
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from check_and_refresh_data import DataRefreshManager


class UsabilityDefectPostFinder:
    """Find and filter posts related to usability and software defects"""
    
    def __init__(self, data_file="data/enhanced_automated_sentiment_results.json", auto_refresh=True):
        self.data_file = data_file
        self.posts_data = None
        self.auto_refresh = auto_refresh
        
        self.usability_keywords = [
            'usability', 'user experience', 'ux', 'ui', 'interface', 
            'design', 'intuitive', 'confusing', 'difficult', 'hard to use',
            'user-friendly', 'clunky', 'awkward', 'navigation'
        ]
        
        self.defect_keywords = [
            'defect', 'bug', 'error', 'issue', 'problem', 'broken', 
            'crash', 'glitch', 'fault', 'failure', 'malfunction',
            'doesn\'t work', 'not working', 'broken', 'fix', 'fixing',
            'debug', 'troubleshoot', 'issue with', 'problem with',
            'error message', 'exception', 'exception', 'fail', 'failed'
        ]
        
    def load_data(self, auto_refresh=True, max_age_days=7):
        """Load the enhanced sentiment analysis results, checking if refresh is needed"""
        refresh_manager = DataRefreshManager(
            data_file=self.data_file,
            max_age_days=max_age_days
        )
        
        if auto_refresh:
            if refresh_manager.is_data_outdated():
                print("\nData file is outdated. Checking if refresh is needed...")
                user_input = input("Refresh data from API now? This may take several minutes. (y/n): ").strip().lower()
                if user_input in ['y', 'yes']:
                    if not refresh_manager.refresh_data(force=True):
                        print("Failed to refresh data. Using existing file if available.")
                else:
                    print("Skipping refresh. Using existing data file.")
        
        print("Loading sentiment analysis data...")
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.posts_data = data.get('all_analyzed_posts', [])
            print(f"Loaded {len(self.posts_data)} analyzed posts")
            return True
        except FileNotFoundError:
            print(f"File not found: {self.data_file}")
            if auto_refresh:
                print("Attempting to fetch and process data from API...")
                if refresh_manager.refresh_data(force=True):
                    return self.load_data(auto_refresh=False)
            return False
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return False
    
    def calculate_relevance_score(self, post, category='usability'):
        """Calculate relevance score for a specific category (usability or defects)"""
        title = post.get('title', '').lower()
        
        if category == 'usability':
            keywords = self.usability_keywords
            exclude_keywords = self.defect_keywords
        else:
            keywords = self.defect_keywords
            exclude_keywords = self.usability_keywords
        
        title_score = 0
        title_has_exclude = False
        
        for keyword in keywords:
            if keyword in title:
                title_score += 3
        
        for keyword in exclude_keywords:
            if keyword in title:
                title_has_exclude = True
                break
        
        author_replies = post.get('author_replies', [])
        community_comments = post.get('community_comments', [])
        
        context_score = 0
        all_text = ' '.join([r.get('text', '').lower() for r in author_replies[:3]] + 
                           [c.get('text', '').lower() for c in community_comments[:5]])
        
        for keyword in keywords:
            if keyword in all_text:
                context_score += 1
        
        relevance_score = title_score * 3 + context_score
        if title_has_exclude and title_score == 0:
            relevance_score = 0
        
        return relevance_score
    
    def find_relevant_posts(self, category='usability', min_relevance=3):
        """Find posts relevant to a specific category (usability or defects)"""
        print(f"\nSearching for {category}-related posts...")
        print(f"   Minimum relevance score: {min_relevance}")
        
        relevant_posts = []
        
        for post in self.posts_data:
            relevance = self.calculate_relevance_score(post, category=category)
            
            if relevance >= min_relevance:
                metrics = post.get('metrics', {})
                quality_score = metrics.get('overall_quality_score', 0)
                author_count = metrics.get('author_replies_count', 0)
                community_count = metrics.get('community_comments_count', 0)
                
                if quality_score >= 0.3 and author_count >= 3 and community_count >= 5:
                    post_info = {
                        'post': post,
                        'relevance_score': relevance,
                        'quality_score': quality_score,
                        'author_replies_count': author_count,
                        'community_comments_count': community_count,
                        'category': category
                    }
                    relevant_posts.append(post_info)
        
        relevant_posts.sort(key=lambda x: (x['relevance_score'], x['quality_score']), reverse=True)
        
        print(f"Found {len(relevant_posts)} {category} posts meeting quality criteria")
        return relevant_posts
    
    def select_top_posts(self, relevant_posts, category='usability', top_n=3):
        """Select top N posts for a specific category"""
        print(f"\nSelecting top {top_n} {category} posts...")
        
        selected = relevant_posts[:top_n]
        
        print(f"\nSelected {category.upper()} Posts:")
        for i, post_info in enumerate(selected, 1):
            post = post_info['post']
            title = post.get('title', 'N/A')
            relevance = post_info['relevance_score']
            quality = post_info['quality_score']
            author_count = post_info['author_replies_count']
            community_count = post_info['community_comments_count']
            
            print(f"\n  {i}. {title[:70]}...")
            print(f"     Relevance: {relevance} | Quality: {quality:.3f}")
            print(f"     Author replies: {author_count} | Community comments: {community_count}")
            print(f"     URL: {post.get('url', 'N/A')}")
        
        return selected
    
    def save_selected_posts(self, usability_posts, defect_posts, output_file="data/usability_defect_posts.json"):
        """Save selected posts to JSON file, separated by category"""
        print(f"\nSaving selected posts to {output_file}...")
        
        output_data = {
            'analysis_metadata': {
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'purpose': 'Find 3 usability posts and 3 software defect posts for sentiment trajectory comparison',
                'source_file': self.data_file,
                'selection_criteria': {
                    'min_relevance_score': 3,
                    'min_quality_score': 0.3,
                    'min_author_replies': 3,
                    'min_community_comments': 5
                }
            },
            'usability_posts': [],
            'defect_posts': []
        }
        
        def process_posts(selected_posts, category_name):
            posts_list = []
            for post_info in selected_posts:
                post = post_info['post']
                author_trajectory = [
                    reply['sentiment']['compound'] 
                    for reply in post.get('author_replies', [])
                ]
                community_trajectory = [
                    comment['sentiment']['compound'] 
                    for comment in post.get('community_comments', [])
                ]
                
                post_data = {
                    'post_id': post.get('post_id', ''),
                    'title': post.get('title', ''),
                    'url': post.get('url', ''),
                    'author': post.get('author', ''),
                    'subreddit': post.get('subreddit', ''),
                    'created_utc': post.get('created_utc', ''),
                    'score': post.get('score', 0),
                    'num_comments': post.get('num_comments', 0),
                    'title_sentiment': post.get('title_sentiment', {}),
                    'relevance_score': post_info['relevance_score'],
                    'quality_score': post_info['quality_score'],
                    'metrics': post.get('metrics', {}),
                    'author_trajectory': author_trajectory,
                    'community_trajectory': community_trajectory,
                    'author_replies_count': post_info['author_replies_count'],
                    'community_comments_count': post_info['community_comments_count'],
                    'category': category_name
                }
                
                posts_list.append(post_data)
            return posts_list
        
        output_data['usability_posts'] = process_posts(usability_posts, 'usability')
        output_data['defect_posts'] = process_posts(defect_posts, 'defect')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        total_posts = len(output_data['usability_posts']) + len(output_data['defect_posts'])
        print(f"Saved {len(output_data['usability_posts'])} usability posts and {len(output_data['defect_posts'])} defect posts ({total_posts} total) to {output_file}")
        return output_data


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Find usability and software defect posts for sentiment trajectory analysis'
    )
    parser.add_argument(
        '--no-refresh',
        action='store_true',
        help='Skip automatic data refresh check'
    )
    parser.add_argument(
        '--max-age',
        type=int,
        default=7,
        help='Maximum age in days before data is considered outdated (default: 7)'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default='data/enhanced_automated_sentiment_results.json',
        help='Path to data file'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FINDING USABILITY AND SOFTWARE DEFECT POSTS")
    print("=" * 80)
    
    finder = UsabilityDefectPostFinder(data_file=args.data_file)
    
    auto_refresh = not args.no_refresh
    if not finder.load_data(auto_refresh=auto_refresh, max_age_days=args.max_age):
        print("Failed to load data. Exiting.")
        return
    
    usability_posts = finder.find_relevant_posts(category='usability', min_relevance=3)
    defect_posts = finder.find_relevant_posts(category='defect', min_relevance=3)
    
    if len(usability_posts) == 0:
        print("No usability posts found. Try lowering the relevance threshold.")
        return
    
    if len(defect_posts) == 0:
        print("No defect posts found. Try lowering the relevance threshold.")
        return
    
    selected_usability = finder.select_top_posts(usability_posts, category='usability', top_n=3)
    selected_defects = finder.select_top_posts(defect_posts, category='defect', top_n=3)
    
    output_file = "data/usability_defect_posts.json"
    finder.save_selected_posts(selected_usability, selected_defects, output_file)
    
    print("\n" + "=" * 80)
    print("POST SELECTION COMPLETE")
    print("=" * 80)
    print(f"Selected: {len(selected_usability)} usability posts + {len(selected_defects)} defect posts = {len(selected_usability) + len(selected_defects)} total")
    print(f"Output file: {output_file}")
    print(f"Next step: Run 'python scripts/compare_usability_defect_trajectories.py'")
    print("=" * 80)


if __name__ == "__main__":
    main()
