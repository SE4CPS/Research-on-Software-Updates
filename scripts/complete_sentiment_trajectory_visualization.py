import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def create_complete_sentiment_visualization():
    """
    Create comprehensive sentiment trajectory visualization for all Reddit posts.
    Shows author sentiment vs replies sentiment over time for 9 selected posts across 3 categories.
    """
    
    posts_data = {
        'Positive 1 (Rust)': {
            'author_sentiment_trajectory': [0.4, -0.2, 0.3, 0.1, -0.1, 0.6, 0.2, 0.2, 0.2, 0.8, 0.1, 0.9, 0.7],
            'replies_trajectory': [-0.1, -0.05, 0.03, 0.08, 0.16, 0.15, 0.17, 0.18, 0.20, 0.22, 0.21, 0.23, 0.22, 0.25, 0.24, 0.26, 0.25, 0.28, 0.29, 0.30, 0.31, 0.30, 0.29, 0.26, 0.27, 0.28, 0.28, 0.28, 0.27, 0.28, 0.28, 0.27, 0.27, 0.28, 0.29, 0.30, 0.29, 0.28, 0.28, 0.30, 0.30, 0.31, 0.32, 0.32, 0.34, 0.34, 0.35, 0.36, 0.36, 0.36, 0.37],
            'color': '#1f77b4',  
            'category': 'Positive',
            'url': 'https://reddit.com/r/rust/comments/1mt1kf1/is_the_nom_crate_a_good_option_for_parsing_a/',
            'title': 'Is the *Nom* crate a good option for parsing a complex syntax?',
            'sentiment_score': 0.898,
            'num_comments': 63,
            'author_replies': 12,
            'other_comments': 51
        },
        'Positive 2 (Linux)': {
            'author_sentiment_trajectory': [0.7],
            'replies_trajectory': [0.8, 0.7, 0.75, 0.6, 0.85, 0.8, 0.9, 0.85, 0.3, 0.6, 0.1, 0.2, 0.3, 0.4, 0.35, 0.5, 0.4, 0.45, 0.55, 0.5, 0.6, 0.4, 0.45, 0.65, 0.9, 0.7, 0.85, 0.75, 0.7, 0.8, 0.85, 0.9, 0.8, 0.2, 0.6, 0.1, 0.0, 0.75, 0.0, 0.4, 0.3, 0.2, 0.15, 0.1, 0.8],
            'color': '#ff7f0e',  
            'category': 'Positive',
            'url': 'https://reddit.com/r/linux/comments/1mrxkfd/im_using_linux_mint_now_daily_for_the_last_4/',
            'title': 'I\'m using Linux Mint now daily for the last 4 months and I start to love the flexibility...',
            'sentiment_score': 0.869,
            'num_comments': 49,
            'author_replies': 0,
            'other_comments': 49
        },
        'Positive 3 (WordPress)': {
            'author_sentiment_trajectory': [-0.2, 0.1],
            'replies_trajectory': [0.3, 0.4, 0.5, -0.2, 0.6, 0.2, -0.3, 0.1, -0.4, 0.7, 0.1, 0.2, 0.4, 0.3, 0.1, 0.5, 0.6, 0.8, 0.2, 0.1, 0.2, 0.9, -0.2, 0.4, 0.2, 0.1, 0.3, 0.0, 0.4, 0.1, 0.3, 0.2, 0.1, 0.8],
            'color': '#2ca02c',  
            'category': 'Positive',
            'url': 'https://reddit.com/r/Wordpress/comments/1mtq0bi/wp_rocket_is_it_still_the_best_option/',
            'title': 'WP Rocket, is it still the best option?',
            'sentiment_score': 0.786,
            'num_comments': 35,
            'author_replies': 1,
            'other_comments': 34
        },
        
        'Negative 1 (Django)': {
            'author_sentiment_trajectory': [-0.8, -0.6, -0.4],
            'replies_trajectory': [-0.3, 0.4, 0.3, 0.8, 0.5, 0.9, 0.1, 0.2, 0.1, 0.0],
            'color': '#d62728',  
            'category': 'Negative',
            'url': 'https://reddit.com/r/django/comments/1mql9p0/dreaded_django_mistake/',
            'title': 'Dreaded Django mistake',
            'sentiment_score': -0.727,
            'num_comments': 13,
            'author_replies': 3,
            'other_comments': 10
        },
        'Negative 2 (IBM)': {
            'author_sentiment_trajectory': [-0.7],
            'replies_trajectory': [-0.8, 0.2, 0.1, -0.4, -0.3, -0.6, -0.2, 0.4, -0.5, -0.3, -0.9, -0.4, -0.8, -0.5, -0.7, 0.0, 0.2, 0.1, -0.4, -0.6, -0.2, 0.1, 0.2, -0.5, -0.8, -0.6],
            'color': '#9467bd',  
            'category': 'Negative',
            'url': 'https://reddit.com/r/IBM/comments/1msecmu/why_is_askibm_so_crappy/',
            'title': 'Why is AskIBM so crappy?',
            'sentiment_score': -0.682,
            'num_comments': 26,
            'author_replies': 0,
            'other_comments': 26
        },
        'Negative 3 (ComfyUI)': {
            'author_sentiment_trajectory': [-0.8, -0.2, 0.3, 0.5, -0.4, -0.5, 0.1, -0.3, -0.7, 0.2, 0.4, 0.4],
            'replies_trajectory': [0.6, 0.8, 0.7, 0.4, 0.3, 0.6, 0.5, 0.3, 0.4, 0.2, 0.5, 0.7, 0.6, 0.4, 0.8, 0.3, 0.5, 0.4, 0.6, -0.2, 0.7, 0.5, 0.8, 0.6, 0.4, 0.3, 0.5, 0.7, 0.6, 0.4, 0.8, 0.5, 0.6, 0.7, 0.5, 0.4, 0.6, 0.3, 0.8, 0.5, 0.7, 0.6, 0.4, 0.5, 0.3, 0.6, 0.4, 0.2, 0.5],
            'color': '#8c564b',  
            'category': 'Negative',
            'url': 'https://reddit.com/r/comfyui/comments/1mqqs0f/are_you_in_dependecies_hell_everytime_you_use_new/',
            'title': 'Are you in dependecies hell everytime you use new workflow you found on internet?',
            'sentiment_score': -0.681,
            'num_comments': 60,
            'author_replies': 11,
            'other_comments': 49
        },
        
        'Neutral 1 (Kdenlive)': {
            'author_sentiment_trajectory': [0.0],
            'replies_trajectory': [0.1, 0.2, -0.2, -0.4, 0.0, -0.3, -0.3, 0.1, -0.7, 0.0, -0.8, -0.5, 0.0, 0.1, 0.0],
            'color': '#17becf',  
            'category': 'Neutral',
            'url': 'https://reddit.com/r/linux/comments/1mtpflw/kdenlive_2508_released_with_over_300_commits_of/',
            'title': 'Kdenlive 25.08 released with over 300 commits of bug fixes and polishing',
            'sentiment_score': 0.026,
            'num_comments': 15,
            'author_replies': 0,
            'other_comments': 15
        },
        'Neutral 2 (Neovim)': {
            'author_sentiment_trajectory': [0.0, 0.0, 0.4, 0.5, 0.2, 0.1, 0.0],
            'replies_trajectory': [0.6, -0.1, 0.8, 0.3, 0.8, 0.2, 0.7, 0.9, 0.6, 0.2, 0.5, 0.6],
            'color': '#bcbd22',  
            'category': 'Neutral',
            'url': 'https://reddit.com/r/neovim/comments/1mrlrmm/how_to_prevent_split_windows_from_inheriting/',
            'title': 'How to prevent split windows from inheriting window options from origin window',
            'sentiment_score': 0.026,
            'num_comments': 18,
            'author_replies': 6,
            'other_comments': 12
        },
        'Neutral 3 (Rust)': {
            'author_sentiment_trajectory': [0.0, 0.6, 0.4, 0.2, 0.1, 0.0, 0.1, 0.3, 0.2, 0.3],
            'replies_trajectory': [0.8, 0.2, 0.6, -0.1, 0.1, 0.5, 0.2, 0.8, 0.9, 0.1, 0.8, 0.7, 0.9, 0.6, 0.0, 0.5, 0.6, 0.8, 0.7],
            'color': '#7f7f7f',  
            'category': 'Neutral',
            'url': 'https://reddit.com/r/rust/comments/1mq4woz/media_simple_optimization_not_so_safe/',
            'title': '[Media] Simple optimization (not so safe)',
            'sentiment_score': -0.006,
            'num_comments': 29,
            'author_replies': 9,
            'other_comments': 20
        }
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    fig.suptitle('Sentiment Trajectory Analysis: Author vs Community Comments\n' +
                 'Comment Count by Author/Review with Adjusted X-axis Length', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    category_titles = ['Positive Posts', 'Negative Posts', 'Neutral Posts']
    
    for idx, category in enumerate(['Positive', 'Negative', 'Neutral']):
        ax = axes[idx]
        
        ax.set_xlabel('Comment Sequence', fontsize=11, fontweight='bold')
        ax.set_ylabel('Sentiment Score (-3 to +3)', fontsize=11, fontweight='bold')
        ax.set_title(category_titles[idx], fontsize=13, fontweight='bold', pad=10)
        
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
        
        legend_elements = []
        
        max_x_length = 0  # Track maximum x-axis length for adjustment
        
        for post_name, data in posts_data.items():
            if data['category'] == category:
                color = data['color']
                author_sentiment_trajectory = data['author_sentiment_trajectory']
                replies_trajectory = data['replies_trajectory']
                author_replies = data['author_replies']
                other_comments = data['other_comments']
                
                # Plot author sentiment trajectory
                if len(author_sentiment_trajectory) > 1:
                    author_x_points = list(range(1, len(author_sentiment_trajectory) + 1))
                    ax.plot(author_x_points, author_sentiment_trajectory, color=color, linestyle='-', 
                           linewidth=3, alpha=0.9, marker='o', markersize=6, 
                           label=f'{post_name} - Author ({author_replies} replies)')
                    max_x_length = max(max_x_length, len(author_sentiment_trajectory))
                else:
                    # Single point for authors with no replies
                    ax.axhline(y=author_sentiment_trajectory[0], color=color, linestyle='-', linewidth=3, 
                              alpha=0.9, label=f'{post_name} - Author ({author_replies} replies)')
                
                # Plot replies trajectory
                replies_x_points = list(range(1, len(replies_trajectory) + 1))
                ax.plot(replies_x_points, replies_trajectory, color=color, linestyle='--', 
                       linewidth=2, alpha=0.7, label=f'{post_name} - Community ({other_comments} comments)')
                max_x_length = max(max_x_length, len(replies_trajectory))
                
                # Update legend elements with comment counts
                if len(author_sentiment_trajectory) > 1:
                    legend_elements.append(Line2D([0], [0], color=color, lw=3, marker='o', 
                                                 label=f'{post_name} - Author ({author_replies})'))
                else:
                    legend_elements.append(Line2D([0], [0], color=color, lw=3, 
                                                 label=f'{post_name} - Author ({author_replies})'))
                legend_elements.append(Line2D([0], [0], color=color, lw=2, linestyle='--', 
                                             label=f'{post_name} - Community ({other_comments})'))
        
        # Adjust x-axis length based on maximum trajectory length
        ax.set_xlim(0, max_x_length + 2)
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
                 bbox_to_anchor=(1.02, 1), borderaxespad=0)
        
        ax.text(0.02, 0.98, 'Positive Zone', transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax.text(0.02, 0.02, 'Negative Zone', transform=ax.transAxes, fontsize=9, 
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        ax.text(0.02, 0.5, 'Neutral Zone', transform=ax.transAxes, fontsize=9, 
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('visualizations/complete_sentiment_trajectory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Complete sentiment trajectory visualization created successfully!")
    print("📊 Saved as: visualizations/complete_sentiment_trajectory_analysis.png")
    print("📈 Shows all 9 posts across 3 categories (Positive, Negative, Neutral)")
    print("\n🔗 URL Reference for Reddit Posts:")
    print("=" * 60)
    
    # Print URL reference table
    categories = ['Positive', 'Negative', 'Neutral']
    
    for category in categories:
        print(f"\n📊 {category.upper()} POSTS:")
        print("-" * 30)
        
        for post_name, data in posts_data.items():
            if data['category'] == category:
                print(f"\n{post_name}:")
                print(f"  📝 Title: {data['title']}")
                print(f"  🔗 URL: {data['url']}")
                print(f"  📊 Sentiment Score: {data['sentiment_score']:.3f}")
                print(f"  💬 Total Comments: {data['num_comments']}")
                print(f"  👤 Author Replies: {data['author_replies']}")
                print(f"  👥 Other Comments: {data['other_comments']}")
                print(f"  📈 Author Trajectory Length: {len(data['author_sentiment_trajectory'])}")
                print(f"  📊 Reply Trajectory Length: {len(data['replies_trajectory'])}")
    
    print("\n" + "=" * 60)
    print("📋 Copy and paste URLs to view original Reddit discussions")
    
    return fig



if __name__ == "__main__":
    create_complete_sentiment_visualization()
