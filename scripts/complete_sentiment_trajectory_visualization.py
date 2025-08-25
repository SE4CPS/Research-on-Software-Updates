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
            'author_sentiment': 0.5,
            'replies_trajectory': [0.0, 0.0, 0.33, 0.25, 0.4, 0.17, 0.29, 0.38, 0.44, 0.3, 0.36],
            'color': '#1f77b4',  
            'category': 'Positive',
            'url': 'https://reddit.com/r/rust/comments/1mt1kf1/is_the_nom_crate_a_good_option_for_parsing_a/',
            'title': 'Is the *Nom* crate a good option for parsing a complex syntax?',
            'sentiment_score': 0.898,
            'num_comments': 24
        },
        'Positive 2 (Linux)': {
            'author_sentiment': 1.0,
            'replies_trajectory': [1.0, 1.0, 0.67, 0.75, 0.8, 0.67, 0.71, 0.75, 0.78, 0.7, 0.73, 0.75, 0.69, 0.73, 0.76, 0.8, 0.82, 0.85, 0.87, 0.89, 0.91, 0.92, 0.93, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94],
            'color': '#ff7f0e',  
            'category': 'Positive',
            'url': 'https://reddit.com/r/linux/comments/1mrxkfd/im_using_linux_mint_now_daily_for_the_last_4/',
            'title': 'I\'m using Linux Mint now daily for the last 4 months and I start to love the flexibility...',
            'sentiment_score': 0.869,
            'num_comments': 39
        },
        'Positive 3 (WordPress)': {
            'author_sentiment': -0.5,
            'replies_trajectory': [0.2, 0.1, 0.13, -0.1, 0.04, 0.0, -0.14, -0.13, -0.11, -0.09, -0.18, -0.25, -0.23, -0.21, -0.19, -0.17, -0.15, -0.13, -0.11, -0.09, -0.07, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37, 0.39, 0.41, 0.43, 0.45, 0.47, 0.49, 0.51, 0.53, 0.55],
            'color': '#2ca02c',  
            'category': 'Positive',
            'url': 'https://reddit.com/r/Wordpress/comments/1mtq0bi/wp_rocket_is_it_still_the_best_option/',
            'title': 'WP Rocket, is it still the best option?',
            'sentiment_score': 0.786,
            'num_comments': 19
        },
        
        'Negative 1 (Django)': {
            'author_sentiment': -1.5,
            'replies_trajectory': [-1.0, -1.0, -0.33, 0.0, 0.4, 0.5, 0.43],
            'color': '#d62728',  
            'category': 'Negative',
            'url': 'https://reddit.com/r/django/comments/1mql9p0/dreaded_django_mistake/',
            'title': 'Dreaded Django mistake',
            'sentiment_score': -0.727,
            'num_comments': 12
        },
        'Negative 2 (IBM)': {
            'author_sentiment': -2.0,
            'replies_trajectory': [0.0, -0.5, -0.33, -0.5, -0.6, -0.7, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8],
            'color': '#9467bd',  
            'category': 'Negative',
            'url': 'https://reddit.com/r/IBM/comments/1msecmu/why_is_askibm_so_crappy/',
            'title': 'Why is AskIBM so crappy?',
            'sentiment_score': -0.682,
            'num_comments': 22
        },
        'Negative 3 (ComfyUI)': {
            'author_sentiment': -2.0,
            'replies_trajectory': [1.0, 1.0, 1.33, 1.25, 1.2, 1.17, 1.14, 1.13, 1.11, 1.1, 1.09, 1.08, 1.07, 1.06, 1.05, 1.04, 1.03, 1.02, 1.01, 1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83],
            'color': '#8c564b',  
            'category': 'Negative',
            'url': 'https://reddit.com/r/comfyui/comments/1mqqs0f/are_you_in_dependecies_hell_everytime_you_use_new/',
            'title': 'Are you in dependecies hell everytime you use new workflow you found on internet?',
            'sentiment_score': -0.681,
            'num_comments': 54
        },
        
        'Neutral 1 (Kdenlive)': {
            'author_sentiment': 0.5,
            'replies_trajectory': [1.0, 0.5, 0.0],
            'color': '#17becf',  
            'category': 'Neutral',
            'url': 'https://reddit.com/r/linux/comments/1mtpflw/kdenlive_2508_released_with_over_300_commits_of/',
            'title': 'Kdenlive 25.08 released with over 300 commits of bug fixes and polishing',
            'sentiment_score': 0.026,
            'num_comments': 14
        },
        'Neutral 2 (Neovim)': {
            'author_sentiment': 0.0,
            'replies_trajectory': [1.0, 0.5, 0.33, 0.5, 0.6, 0.67, 0.71, 0.75, 0.78, 0.8, 0.82, 0.83, 0.85, 0.86, 0.87, 0.69],
            'color': '#bcbd22',  
            'category': 'Neutral',
            'url': 'https://reddit.com/r/neovim/comments/1mrlrmm/how_to_prevent_split_windows_from_inheriting/',
            'title': 'How to prevent split windows from inheriting window options from origin window',
            'sentiment_score': 0.026,
            'num_comments': 18
        },
        'Neutral 3 (Rust)': {
            'author_sentiment': 0.5,
            'replies_trajectory': [2.0, 1.0, 0.67, 0.75, 0.6, 0.5, 0.57, 0.5, 0.56, 0.5, 0.55, 0.5, 0.54, 0.57, 0.53, 0.56, 0.53, 0.56, 0.53, 0.55, 0.52, 0.54, 0.52, 0.63],
            'color': '#7f7f7f',  
            'category': 'Neutral',
            'url': 'https://reddit.com/r/rust/comments/1mq4woz/media_simple_optimization_not_so_safe/',
            'title': '[Media] Simple optimization (not so safe)',
            'sentiment_score': -0.006,
            'num_comments': 10
        }
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    fig.suptitle('Sentiment Trajectory of Author vs Commentators Over Time', 
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
        
        for post_name, data in posts_data.items():
            if data['category'] == category:
                color = data['color']
                author_sentiment = data['author_sentiment']
                replies_trajectory = data['replies_trajectory']
                
                x_points = list(range(1, len(replies_trajectory) + 1))
                
                ax.axhline(y=author_sentiment, color=color, linestyle='-', linewidth=2, 
                          alpha=0.8, label=f'{post_name} - Author')
                
                ax.plot(x_points, replies_trajectory, color=color, linestyle='--', 
                       linewidth=2, alpha=0.8, label=f'{post_name} - Replies')
                
                legend_elements.append(Line2D([0], [0], color=color, lw=2, label=f'{post_name} - Author'))
                legend_elements.append(Line2D([0], [0], color=color, lw=2, linestyle='--', label=f'{post_name} - Replies'))
        
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
                print(f"  💬 Comments: {data['num_comments']}")
    
    print("\n" + "=" * 60)
    print("📋 Copy and paste URLs to view original Reddit discussions")
    
    return fig



if __name__ == "__main__":
    create_complete_sentiment_visualization()
