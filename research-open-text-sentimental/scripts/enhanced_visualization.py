import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import json
import os


def load_enhanced_results(results_file="data/enhanced_automated_sentiment_results.json"):
    """Load enhanced automated sentiment analysis results"""
    print(f"📂 Loading enhanced results from {results_file}...")
    
    if not os.path.exists(results_file):
        print(f"❌ Enhanced results file not found: {results_file}")
        print(f"💡 Please run 'python scripts/enhanced_automated_sentiment_analysis.py' first")
        return None
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"✅ Loaded enhanced analysis results")
    return results


def prepare_enhanced_visualization_data(results):
    """Prepare data for enhanced visualization"""
    top_posts = results['top_posts']
    
    visualization_data = {}
    colors = {
        'positive': ['#1f77b4', '#ff7f0e', '#2ca02c'],
        'negative': ['#d62728', '#9467bd', '#8c564b'],
        'neutral': ['#17becf', '#bcbd22', '#7f7f7f']
    }
    
    for category in ['positive', 'negative', 'neutral']:
        posts = top_posts[category]
        
        for i, post in enumerate(posts):
            subreddit = post['subreddit']
            post_key = f"{category.capitalize()} {i+1} ({subreddit})"
            
            metrics = post['metrics']
            
            visualization_data[post_key] = {
                'author_sentiment_trajectory': metrics['author_trajectory'],
                'replies_trajectory': metrics['community_trajectory'],
                'color': colors[category][i],
                'category': category.capitalize(),
                'url': post['url'],
                'title': post['title'],
                'sentiment_score': post['title_sentiment']['compound'],
                'num_comments': post['num_comments'],
                'author_replies': metrics['author_replies_count'],
                'other_comments': metrics['community_comments_count'],
                # Enhanced metrics
                'quality_score': metrics['overall_quality_score'],
                'author_reliability': metrics['author_trajectory_reliability'],
                'community_reliability': metrics['community_trajectory_reliability']
            }
    
    return visualization_data


def create_enhanced_visualization(visualization_data, output_file="visualizations/enhanced_sentiment_trajectory.png"):
    """Create enhanced sentiment trajectory visualization with quality indicators"""
    print(f"\n📊 Creating enhanced sentiment trajectory visualization...")
    
    # Create figure with 3 subplots (wider to accommodate legend)
    fig, axes = plt.subplots(3, 1, figsize=(20, 20))
    fig.suptitle('Enhanced Sentiment Trajectory Analysis: Author vs Community Comments\n' +
                 'Quality-Filtered Posts with Reliability Metrics', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    category_titles = ['Positive Posts', 'Negative Posts', 'Neutral Posts']
    categories = ['Positive', 'Negative', 'Neutral']
    
    for idx, category in enumerate(categories):
        ax = axes[idx]
        
        ax.set_xlabel('Comment Sequence', fontsize=11, fontweight='bold')
        ax.set_ylabel('Sentiment Score (-1 to +1)', fontsize=11, fontweight='bold')
        ax.set_title(category_titles[idx], fontsize=13, fontweight='bold', pad=10)
        
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
        
        legend_elements = []
        max_x_length = 0
        
        # Plot data for this category
        for post_name, data in visualization_data.items():
            if data['category'] == category:
                color = data['color']
                author_trajectory = data['author_sentiment_trajectory']
                community_trajectory = data['replies_trajectory']
                author_replies = data['author_replies']
                other_comments = data['other_comments']
                quality_score = data['quality_score']
                author_reliability = data['author_reliability']
                community_reliability = data['community_reliability']
                
                # Determine line style based on reliability
                author_alpha = 0.9 if author_reliability > 0.5 else 0.5
                community_alpha = 0.7 if community_reliability > 0.5 else 0.4
                
                # Plot author sentiment trajectory
                if len(author_trajectory) > 1:
                    author_x_points = list(range(1, len(author_trajectory) + 1))
                    ax.plot(author_x_points, author_trajectory, color=color, linestyle='-', 
                           linewidth=3, alpha=author_alpha, marker='o', markersize=6)
                    max_x_length = max(max_x_length, len(author_trajectory))
                elif len(author_trajectory) == 1:
                    ax.axhline(y=author_trajectory[0], color=color, linestyle='-', 
                              linewidth=3, alpha=author_alpha)
                
                # Plot community trajectory
                if len(community_trajectory) > 0:
                    community_x_points = list(range(1, len(community_trajectory) + 1))
                    ax.plot(community_x_points, community_trajectory, color=color, 
                           linestyle='--', linewidth=2, alpha=community_alpha)
                    max_x_length = max(max_x_length, len(community_trajectory))
                
                # Clean legend without quality indicators
                if len(author_trajectory) > 1:
                    legend_elements.append(Line2D([0], [0], color=color, lw=3, marker='o', linestyle='-',
                                                 label=f'{post_name} - Author ({author_replies})'))
                elif len(author_trajectory) == 1:
                    legend_elements.append(Line2D([0], [0], color=color, lw=3, linestyle='-',
                                                 label=f'{post_name} - Author ({author_replies})'))
                
                if len(community_trajectory) > 0:
                    legend_elements.append(Line2D([0], [0], color=color, lw=2, linestyle='--', 
                                                 label=f'{post_name} - Community ({other_comments})'))
        
        # Adjust x-axis
        ax.set_xlim(0, max(max_x_length + 2, 10))
        
        # Clean legend with just line styles
        combined_legend = legend_elements.copy()
        
        # Add line style explanations
        combined_legend.extend([
            Line2D([0], [0], linestyle='None', label='--- Line Styles ---'),
            Line2D([0], [0], color='black', lw=3, linestyle='-', label='Author Trajectory (solid)'),
            Line2D([0], [0], color='black', lw=2, linestyle='--', label='Community Trajectory (dashed)')
        ])
        
        # Add clean legend
        ax.legend(handles=combined_legend, loc='upper right', fontsize=9, 
                 bbox_to_anchor=(1.02, 1), borderaxespad=0)
        
        # Add sentiment zone labels
        ax.text(0.02, 0.98, 'Positive Zone', transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax.text(0.02, 0.02, 'Negative Zone', transform=ax.transAxes, fontsize=9, 
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        ax.text(0.02, 0.5, 'Neutral Zone', transform=ax.transAxes, fontsize=9, 
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Enhanced visualization saved to: {output_file}")
    
    plt.show()
    
    return fig


def print_enhanced_url_reference(visualization_data):
    """Print enhanced URL reference with quality metrics"""
    print("\n" + "=" * 80)
    print("🔗 ENHANCED URL REFERENCE WITH QUALITY METRICS")
    print("=" * 80)
    
    categories = ['Positive', 'Negative', 'Neutral']
    
    for category in categories:
        print(f"\n📊 {category.upper()} POSTS:")
        print("-" * 80)
        
        for post_name, data in visualization_data.items():
            if data['category'] == category:
                print(f"\n{post_name}:")
                print(f"  📝 Title: {data['title'][:100]}...")
                print(f"  🔗 URL: {data['url']}")
                print(f"  📊 Title Sentiment Score: {data['sentiment_score']:.3f}")
                print(f"  💬 Total Comments: {data['num_comments']}")
                print(f"  👤 Author Replies: {data['author_replies']}")
                print(f"  👥 Community Comments: {data['other_comments']}")
                print(f"  📈 Author Trajectory Length: {len(data['author_sentiment_trajectory'])}")
                print(f"  📊 Community Trajectory Length: {len(data['replies_trajectory'])}")
                print(f"  ⭐ Overall Quality Score: {data['quality_score']:.3f}")
                print(f"  🔹 Author Reliability: {data['author_reliability']:.3f}")
                print(f"  🔹 Community Reliability: {data['community_reliability']:.3f}")


def save_enhanced_url_reference(visualization_data, output_file="visualizations/enhanced_posts_reference.txt"):
    """Save enhanced URL reference to text file"""
    print(f"\n💾 Saving enhanced URL reference to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ENHANCED AUTOMATED SENTIMENT ANALYSIS - REDDIT POSTS REFERENCE\n")
        f.write("=" * 80 + "\n\n")
        f.write("LEGEND:\n")
        f.write("- Solid lines: Author sentiment trajectories\n")
        f.write("- Dashed lines: Community sentiment trajectories\n")
        f.write("- Numbers in parentheses: Comment counts\n\n")
        
        categories = ['Positive', 'Negative', 'Neutral']
        
        for category in categories:
            f.write(f"\n{category.upper()} POSTS:\n")
            f.write("-" * 80 + "\n")
            
            for post_name, data in visualization_data.items():
                if data['category'] == category:
                    f.write(f"\n{post_name}:\n")
                    f.write(f"  Title: {data['title']}\n")
                    f.write(f"  URL: {data['url']}\n")
                    f.write(f"  Title Sentiment Score: {data['sentiment_score']:.3f}\n")
                    f.write(f"  Total Comments: {data['num_comments']}\n")
                    f.write(f"  Author Replies: {data['author_replies']}\n")
                    f.write(f"  Community Comments: {data['other_comments']}\n")
                    f.write(f"  Author Trajectory Length: {len(data['author_sentiment_trajectory'])}\n")
                    f.write(f"  Community Trajectory Length: {len(data['replies_trajectory'])}\n")
                    f.write(f"  Overall Quality Score: {data['quality_score']:.3f}\n")
                    f.write(f"  Author Reliability: {data['author_reliability']:.3f}\n")
                    f.write(f"  Community Reliability: {data['community_reliability']:.3f}\n")
                    f.write("\n")
    
    print(f"✅ Enhanced URL reference saved!")


def main():
    """Main execution function for enhanced visualization"""
    print("=" * 80)
    print("📈 ENHANCED AUTOMATED SENTIMENT TRAJECTORY VISUALIZATION")
    print("=" * 80)
    
    # Load enhanced results
    results = load_enhanced_results()
    if not results:
        return
    
    # Prepare enhanced visualization data
    visualization_data = prepare_enhanced_visualization_data(results)
    
    if not visualization_data:
        print("❌ No enhanced visualization data available")
        return
    
    print(f"\n📊 Preparing to visualize {len(visualization_data)} quality-filtered posts...")
    
    # Create enhanced visualization
    create_enhanced_visualization(visualization_data)
    
    # Print enhanced URL reference
    print_enhanced_url_reference(visualization_data)
    
    # Save enhanced URL reference
    save_enhanced_url_reference(visualization_data)
    
    print("\n" + "=" * 80)
    print("✅ ENHANCED VISUALIZATION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  📊 Chart: visualizations/enhanced_sentiment_trajectory.png")
    print("  📝 Reference: visualizations/enhanced_posts_reference.txt")
    print("  💾 Full data: data/enhanced_automated_sentiment_results.json")
    print("\nLegend:")
    print("  - Solid lines: Author sentiment trajectories")
    print("  - Dashed lines: Community sentiment trajectories")
    print("  - Numbers in parentheses: Comment counts")
    print("=" * 80)


if __name__ == "__main__":
    main()
