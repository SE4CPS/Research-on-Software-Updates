"""
Script to check if the processed data file is outdated and optionally refresh it from the API.
Can be used as a standalone utility or imported by other scripts.
"""

import os
import json
import requests
from datetime import datetime, timedelta
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from scripts.enhanced_automated_sentiment_analysis import EnhancedAutomatedSentimentAnalyzer


class DataRefreshManager:
    """Manages checking and refreshing processed data from API"""
    
    def __init__(self, data_file="data/enhanced_automated_sentiment_results.json", 
                 max_age_days=7, api_url="https://releasetrain.io/api/reddit"):
        self.data_file = data_file
        self.max_age_days = max_age_days
        self.api_url = api_url
        
    def check_file_exists(self):
        """Check if the data file exists"""
        return os.path.exists(self.data_file)
    
    def get_file_age(self):
        """Get the age of the data file in days"""
        if not self.check_file_exists():
            return None
        
        file_time = os.path.getmtime(self.data_file)
        file_date = datetime.fromtimestamp(file_time)
        age = datetime.now() - file_date
        return age.days
    
    def get_file_metadata(self):
        """Get metadata from the JSON file to check analysis date"""
        if not self.check_file_exists():
            return None
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            metadata = data.get('analysis_metadata', {})
            return metadata
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def is_data_outdated(self):
        """Check if the data file is outdated based on file age and metadata"""
        if not self.check_file_exists():
            print("Data file does not exist. Need to fetch from API.")
            return True
        
        file_age_days = self.get_file_age()
        metadata = self.get_file_metadata()
        
        if file_age_days is None:
            return True
        
        if file_age_days > self.max_age_days:
            print(f"Data file is {file_age_days} days old (threshold: {self.max_age_days} days)")
            return True
        
        if metadata:
            analysis_date_str = metadata.get('date', '')
            if analysis_date_str:
                try:
                    analysis_date = datetime.strptime(analysis_date_str, "%Y-%m-%d %H:%M:%S")
                    analysis_age = (datetime.now() - analysis_date).days
                    if analysis_age > self.max_age_days:
                        print(f"Analysis data is {analysis_age} days old (threshold: {self.max_age_days} days)")
                        return True
                except ValueError:
                    pass
        
        print(f"Data file is up to date ({file_age_days} days old)")
        return False
    
    def check_api_available(self):
        """Check if the API endpoint is available"""
        try:
            response = requests.get(self.api_url, timeout=5)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"API endpoint not available: {e}")
            return False
    
    def refresh_data(self, force=False):
        """Refresh the data file by fetching from API and processing"""
        if not force and not self.is_data_outdated():
            print("Data is up to date. No refresh needed.")
            return False
        
        if not self.check_api_available():
            print("Cannot refresh: API endpoint is not available.")
            return False
        
        print("\n" + "=" * 80)
        print("REFRESHING DATA FROM API")
        print("=" * 80)
        print(f"API Endpoint: {self.api_url}")
        print(f"Output File: {self.data_file}")
        print("\nThis will:")
        print("  1. Fetch all posts from API")
        print("  2. Run sentiment analysis on all posts")
        print("  3. Extract trajectories")
        print("  4. Calculate quality metrics")
        print("  5. Filter and save results")
        print("\nThis may take several minutes...")
        print("=" * 80 + "\n")
        
        os.makedirs(os.path.dirname(self.data_file) if os.path.dirname(self.data_file) else '.', exist_ok=True)
        
        analyzer = EnhancedAutomatedSentimentAnalyzer(api_url=self.api_url)
        
        if not analyzer.fetch_reddit_posts():
            print("Failed to fetch posts from API. Aborting refresh.")
            return False
        
        analyzed_posts = analyzer.analyze_all_posts(
            min_comments=10,
            min_author_replies=3,
            min_community_comments=5
        )
        
        if not analyzed_posts:
            print("No posts found meeting quality criteria. Aborting refresh.")
            return False
        
        top_posts = analyzer.select_top_posts_enhanced(
            analyzed_posts,
            top_n=3,
            min_quality_score=0.3
        )
        
        analyzer.save_enhanced_results(analyzed_posts, top_posts, self.data_file)
        
        print("\n" + "=" * 80)
        print("DATA REFRESH COMPLETE")
        print("=" * 80)
        print(f"Saved {len(analyzed_posts)} analyzed posts to {self.data_file}")
        print("=" * 80 + "\n")
        
        return True
    
    def ensure_data_fresh(self, auto_refresh=True):
        """Ensure data is fresh, refresh if needed"""
        if self.is_data_outdated():
            if auto_refresh:
                return self.refresh_data(force=True)
            else:
                print("\nData is outdated. Run with --refresh flag to update.")
                return False
        return True


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Check and optionally refresh processed sentiment analysis data'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check if data file is outdated (default action)'
    )
    parser.add_argument(
        '--refresh',
        action='store_true',
        help='Refresh data from API if outdated'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force refresh even if data is up to date'
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
        help='Path to data file (default: data/enhanced_automated_sentiment_results.json)'
    )
    
    args = parser.parse_args()
    
    manager = DataRefreshManager(
        data_file=args.data_file,
        max_age_days=args.max_age
    )
    
    if args.force:
        manager.refresh_data(force=True)
    elif args.refresh:
        manager.refresh_data(force=False)
    else:
        is_outdated = manager.is_data_outdated()
        if is_outdated:
            print("\nTo refresh the data, run:")
            print(f"  python3 scripts/check_and_refresh_data.py --refresh")
            sys.exit(1)
        else:
            print("\nData is up to date!")
            sys.exit(0)


if __name__ == "__main__":
    main()
