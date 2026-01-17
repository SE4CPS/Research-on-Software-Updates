"""
Script to compare sentiment trajectories of 3 usability posts and 3 software defect posts.
Creates visualization and generates analysis report.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from datetime import datetime


def load_selected_posts(data_file="data/usability_defect_posts.json"):
    """Load the selected posts data, separated by category"""
    print("Loading selected posts data...")
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        usability_posts = data.get('usability_posts', [])
        defect_posts = data.get('defect_posts', [])
        print(f"Loaded {len(usability_posts)} usability posts and {len(defect_posts)} defect posts")
        return usability_posts, defect_posts, data.get('analysis_metadata', {})
    except FileNotFoundError:
        print(f"File not found: {data_file}")
        return None, None, None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None, None, None


def calculate_trajectory_stats(trajectory):
    """Calculate statistics for a sentiment trajectory"""
    if not trajectory or len(trajectory) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'trend': 0.0,
            'volatility': 0.0
        }
    
    trajectory_array = np.array(trajectory)
    mean = float(np.mean(trajectory_array))
    std = float(np.std(trajectory_array))
    min_val = float(np.min(trajectory_array))
    max_val = float(np.max(trajectory_array))
    
    if len(trajectory) > 1:
        x = np.arange(len(trajectory))
        trend = float(np.polyfit(x, trajectory_array, 1)[0])
    else:
        trend = 0.0
    
    if len(trajectory) > 1:
        changes = np.abs(np.diff(trajectory_array))
        volatility = float(np.mean(changes))
    else:
        volatility = 0.0
    
    return {
        'mean': round(mean, 3),
        'std': round(std, 3),
        'min': round(min_val, 3),
        'max': round(max_val, 3),
        'trend': round(trend, 3),
        'volatility': round(volatility, 3)
    }


def create_comparison_visualization(usability_posts, defect_posts, output_file="visualizations/usability_defect_trajectory_comparison.png"):
    """Create visualization comparing sentiment trajectories for both categories"""
    print(f"\nCreating comparison visualization...")
    
    if len(usability_posts) != 3:
        print(f"Expected 3 usability posts, found {len(usability_posts)}")
    if len(defect_posts) != 3:
        print(f"Expected 3 defect posts, found {len(defect_posts)}")
    
    usability_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    defect_colors = ['#d62728', '#9467bd', '#8c564b']
    
    fig, axes = plt.subplots(6, 1, figsize=(16, 20))
    fig.suptitle('Sentiment Trajectory Comparison: Usability vs Software Defect Posts\n' +
                 'Author vs Community Sentiment Over Time', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    def plot_post(ax, post, color, post_num, category):
        """Helper function to plot a single post"""
        title = post.get('title', 'N/A')
        display_title = title[:75] + "..." if len(title) > 75 else title
        
        ax.set_title(f"{category} Post {post_num}: {display_title}", fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Comment Sequence', fontsize=10, fontweight='bold')
        ax.set_ylabel('Sentiment Score', fontsize=10, fontweight='bold')
        
        author_trajectory = post.get('author_trajectory', [])
        community_trajectory = post.get('community_trajectory', [])
        
        if len(author_trajectory) > 0:
            author_x = list(range(1, len(author_trajectory) + 1))
            ax.plot(author_x, author_trajectory, color=color, linestyle='-', 
                   linewidth=3, alpha=0.9, marker='o', markersize=5,
                   label=f'Author ({len(author_trajectory)} replies)')
        
        if len(community_trajectory) > 0:
            community_x = list(range(1, len(community_trajectory) + 1))
            ax.plot(community_x, community_trajectory, color=color, linestyle='--', 
                   linewidth=2, alpha=0.7, marker='s', markersize=3,
                   label=f'Community ({len(community_trajectory)} comments)')
        
        all_values = author_trajectory + community_trajectory
        if all_values:
            y_min = min(all_values) - 0.2
            y_max = max(all_values) + 0.2
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(-1, 1)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9)
        
        ax.text(0.02, 0.98, 'Positive', transform=ax.transAxes, fontsize=8, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax.text(0.02, 0.02, 'Negative', transform=ax.transAxes, fontsize=8, 
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    for idx, (post, color) in enumerate(zip(usability_posts, usability_colors)):
        plot_post(axes[idx], post, color, idx + 1, "Usability")
    
    for idx, (post, color) in enumerate(zip(defect_posts, defect_colors)):
        plot_post(axes[idx + 3], post, color, idx + 1, "Defect")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    plt.close()
    
    return output_file


def generate_analysis_report(usability_posts, defect_posts, metadata, output_file="documentation/usability_defect_analysis_report.md"):
    """Generate analysis report with findings for both categories"""
    print(f"\nGenerating analysis report...")
    
    report = []
    report.append("# Usability and Software Defect Posts - Sentiment Trajectory Analysis\n")
    report.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Source:** {metadata.get('source_file', 'N/A') if metadata else 'N/A'}\n")
    report.append("\n---\n")
    
    report.append("## Executive Summary\n")
    report.append("This report analyzes the sentiment trajectories of 6 Reddit posts: ")
    report.append("3 posts related to usability issues and 3 posts related to software defects. ")
    report.append("The analysis compares how the original post authors' sentiment evolves compared ")
    report.append("to the community's sentiment over time.\n\n")
    
    report.append("## Usability Posts\n\n")
    
    for idx, post in enumerate(usability_posts, 1):
        report.append(f"### Post {idx}: {post.get('title', 'N/A')}\n")
        report.append(f"- **URL:** {post.get('url', 'N/A')}\n")
        report.append(f"- **Subreddit:** r/{post.get('subreddit', 'N/A')}\n")
        report.append(f"- **Author:** u/{post.get('author', 'N/A')}\n")
        report.append(f"- **Post Score:** {post.get('score', 0)}\n")
        report.append(f"- **Total Comments:** {post.get('num_comments', 0)}\n")
        report.append(f"- **Relevance Score:** {post.get('relevance_score', 0)}\n")
        report.append(f"- **Quality Score:** {post.get('quality_score', 0):.3f}\n")
        report.append(f"- **Title Sentiment:** {post.get('title_sentiment', {}).get('label', 'N/A')} ")
        report.append(f"(Score: {post.get('title_sentiment', {}).get('compound', 0):.3f})\n\n")
        
        author_traj = post.get('author_trajectory', [])
        community_traj = post.get('community_trajectory', [])
        
        author_stats = calculate_trajectory_stats(author_traj)
        community_stats = calculate_trajectory_stats(community_traj)
        
        report.append("#### Author Sentiment Trajectory\n")
        report.append(f"- **Number of Replies:** {len(author_traj)}\n")
        report.append(f"- **Mean Sentiment:** {author_stats['mean']}\n")
        report.append(f"- **Sentiment Range:** {author_stats['min']} to {author_stats['max']}\n")
        report.append(f"- **Trend:** {author_stats['trend']:+.4f} per comment ")
        report.append(f"({'increasing' if author_stats['trend'] > 0 else 'decreasing' if author_stats['trend'] < 0 else 'stable'})\n")
        report.append(f"- **Volatility:** {author_stats['volatility']:.3f}\n\n")
        
        report.append("#### Community Sentiment Trajectory\n")
        report.append(f"- **Number of Comments:** {len(community_traj)}\n")
        report.append(f"- **Mean Sentiment:** {community_stats['mean']}\n")
        report.append(f"- **Sentiment Range:** {community_stats['min']} to {community_stats['max']}\n")
        report.append(f"- **Trend:** {community_stats['trend']:+.4f} per comment ")
        report.append(f"({'increasing' if community_stats['trend'] > 0 else 'decreasing' if community_stats['trend'] < 0 else 'stable'})\n")
        report.append(f"- **Volatility:** {community_stats['volatility']:.3f}\n\n")
        
        sentiment_divergence = abs(author_stats['mean'] - community_stats['mean'])
        report.append("#### Author vs Community Comparison\n")
        report.append(f"- **Sentiment Divergence:** {sentiment_divergence:.3f}\n")
        if sentiment_divergence < 0.2:
            report.append("  - *Low divergence: Author and community sentiment are aligned*\n")
        elif sentiment_divergence < 0.5:
            report.append("  - *Moderate divergence: Some difference in sentiment*\n")
        else:
            report.append("  - *High divergence: Significant difference in sentiment*\n")
        
        report.append("\n---\n\n")
    
    report.append("## Software Defect Posts\n\n")
    
    for idx, post in enumerate(defect_posts, 1):
        report.append(f"### Post {idx}: {post.get('title', 'N/A')}\n")
        report.append(f"- **URL:** {post.get('url', 'N/A')}\n")
        report.append(f"- **Subreddit:** r/{post.get('subreddit', 'N/A')}\n")
        report.append(f"- **Author:** u/{post.get('author', 'N/A')}\n")
        report.append(f"- **Post Score:** {post.get('score', 0)}\n")
        report.append(f"- **Total Comments:** {post.get('num_comments', 0)}\n")
        report.append(f"- **Relevance Score:** {post.get('relevance_score', 0)}\n")
        report.append(f"- **Quality Score:** {post.get('quality_score', 0):.3f}\n")
        report.append(f"- **Title Sentiment:** {post.get('title_sentiment', {}).get('label', 'N/A')} ")
        report.append(f"(Score: {post.get('title_sentiment', {}).get('compound', 0):.3f})\n\n")
        
        author_traj = post.get('author_trajectory', [])
        community_traj = post.get('community_trajectory', [])
        
        author_stats = calculate_trajectory_stats(author_traj)
        community_stats = calculate_trajectory_stats(community_traj)
        
        report.append("#### Author Sentiment Trajectory\n")
        report.append(f"- **Number of Replies:** {len(author_traj)}\n")
        report.append(f"- **Mean Sentiment:** {author_stats['mean']}\n")
        report.append(f"- **Sentiment Range:** {author_stats['min']} to {author_stats['max']}\n")
        report.append(f"- **Trend:** {author_stats['trend']:+.4f} per comment ")
        report.append(f"({'increasing' if author_stats['trend'] > 0 else 'decreasing' if author_stats['trend'] < 0 else 'stable'})\n")
        report.append(f"- **Volatility:** {author_stats['volatility']:.3f}\n\n")
        
        report.append("#### Community Sentiment Trajectory\n")
        report.append(f"- **Number of Comments:** {len(community_traj)}\n")
        report.append(f"- **Mean Sentiment:** {community_stats['mean']}\n")
        report.append(f"- **Sentiment Range:** {community_stats['min']} to {community_stats['max']}\n")
        report.append(f"- **Trend:** {community_stats['trend']:+.4f} per comment ")
        report.append(f"({'increasing' if community_stats['trend'] > 0 else 'decreasing' if community_stats['trend'] < 0 else 'stable'})\n")
        report.append(f"- **Volatility:** {community_stats['volatility']:.3f}\n\n")
        
        sentiment_divergence = abs(author_stats['mean'] - community_stats['mean'])
        report.append("#### Author vs Community Comparison\n")
        report.append(f"- **Sentiment Divergence:** {sentiment_divergence:.3f}\n")
        if sentiment_divergence < 0.2:
            report.append("  - *Low divergence: Author and community sentiment are aligned*\n")
        elif sentiment_divergence < 0.5:
            report.append("  - *Moderate divergence: Some difference in sentiment*\n")
        else:
            report.append("  - *High divergence: Significant difference in sentiment*\n")
        
        report.append("\n---\n\n")
    
    report.append("## Comparative Analysis\n\n")
    
    usability_author_means = [calculate_trajectory_stats(p.get('author_trajectory', []))['mean'] 
                              for p in usability_posts]
    usability_community_means = [calculate_trajectory_stats(p.get('community_trajectory', []))['mean'] 
                                 for p in usability_posts]
    
    defect_author_means = [calculate_trajectory_stats(p.get('author_trajectory', []))['mean'] 
                           for p in defect_posts]
    defect_community_means = [calculate_trajectory_stats(p.get('community_trajectory', []))['mean'] 
                             for p in defect_posts]
    
    all_author_means = usability_author_means + defect_author_means
    all_community_means = usability_community_means + defect_community_means
    
    report.append("### Usability Posts Summary\n\n")
    report.append(f"- **Average Author Sentiment:** {np.mean(usability_author_means):.3f}\n")
    report.append(f"- **Average Community Sentiment:** {np.mean(usability_community_means):.3f}\n")
    report.append(f"- **Author Sentiment Range:** {min(usability_author_means):.3f} to {max(usability_author_means):.3f}\n")
    report.append(f"- **Community Sentiment Range:** {min(usability_community_means):.3f} to {max(usability_community_means):.3f}\n\n")
    
    report.append("### Software Defect Posts Summary\n\n")
    report.append(f"- **Average Author Sentiment:** {np.mean(defect_author_means):.3f}\n")
    report.append(f"- **Average Community Sentiment:** {np.mean(defect_community_means):.3f}\n")
    report.append(f"- **Author Sentiment Range:** {min(defect_author_means):.3f} to {max(defect_author_means):.3f}\n")
    report.append(f"- **Community Sentiment Range:** {min(defect_community_means):.3f} to {max(defect_community_means):.3f}\n\n")
    
    report.append("### Overall Key Findings\n\n")
    report.append(f"1. **Overall Author Sentiment Range:** {min(all_author_means):.3f} to {max(all_author_means):.3f}\n")
    report.append(f"2. **Overall Community Sentiment Range:** {min(all_community_means):.3f} to {max(all_community_means):.3f}\n")
    report.append(f"3. **Overall Average Author Sentiment:** {np.mean(all_author_means):.3f}\n")
    report.append(f"4. **Overall Average Community Sentiment:** {np.mean(all_community_means):.3f}\n\n")
    
    report.append("### Observed Patterns\n\n")
    report.append("- **Sentiment Evolution:** How sentiment changes from initial post to later interactions\n")
    report.append("- **Author-Community Alignment:** Whether authors and community share similar sentiment\n")
    report.append("- **Volatility:** How much sentiment fluctuates throughout the discussion\n")
    report.append("- **Trend Direction:** Whether sentiment improves, worsens, or remains stable\n")
    report.append("- **Category Differences:** Comparison between usability and defect post sentiment patterns\n\n")
    
    report.append("## Visualization\n\n")
    report.append("See the comparison chart: `visualizations/usability_defect_trajectory_comparison.png`\n\n")
    
    report.append("## Conclusion\n\n")
    report.append("This analysis provides insights into how users express sentiment when discussing ")
    report.append("usability issues and software defects. The trajectory comparison reveals patterns ")
    report.append("in how initial concerns evolve through community interaction and problem-solving.\n\n")
    
    report.append("---\n")
    report.append(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    report_text = ''.join(report)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"Report saved to {output_file}")
    return output_file


def main():
    """Main execution function"""
    print("=" * 80)
    print("COMPARING USABILITY AND DEFECT POST SENTIMENT TRAJECTORIES")
    print("=" * 80)
    
    usability_posts, defect_posts, metadata = load_selected_posts()
    if usability_posts is None or defect_posts is None:
        print("Failed to load posts. Make sure to run find_usability_defect_posts.py first.")
        return
    
    if len(usability_posts) == 0:
        print("No usability posts found in data file.")
        return
    
    if len(defect_posts) == 0:
        print("No defect posts found in data file.")
        return
    
    viz_file = create_comparison_visualization(usability_posts, defect_posts)
    report_file = generate_analysis_report(usability_posts, defect_posts, metadata)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Visualization: {viz_file}")
    print(f"Report: {report_file}")
    print("\nSummary:")
    print("\n  Usability Posts:")
    for idx, post in enumerate(usability_posts, 1):
        title = post.get('title', 'N/A')
        print(f"    {idx}. {title[:55]}...")
    print("\n  Defect Posts:")
    for idx, post in enumerate(defect_posts, 1):
        title = post.get('title', 'N/A')
        print(f"    {idx}. {title[:55]}...")
    print("=" * 80)


if __name__ == "__main__":
    main()
