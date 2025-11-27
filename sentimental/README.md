# Research on Software Updates - Sentiment Trajectory Analysis

## 📊 Project Overview
This project analyzes sentiment trajectories of Reddit posts related to software updates, examining how author sentiment compares to community replies over time.

## 📁 Project Structure

### Core Files
- `scripts/complete_sentiment_trajectory_visualization.py` - Main visualization script
- `visualizations/complete_sentiment_trajectory_analysis.png` - Sentiment trajectory chart
- `visualizations/reddit_posts_urls_reference.txt` - URL reference for all analyzed posts

### Data Files
- `data/` - Contains Reddit post data and analysis results
- `documentation/` - Project documentation and review templates

## 🚀 How to Use

### Generate Visualization
```bash
python scripts/complete_sentiment_trajectory_visualization.py
```

This will:
1. Create the sentiment trajectory visualization
2. Save it as `visualizations/complete_sentiment_trajectory_analysis.png`
3. Print URL references for all 9 analyzed Reddit posts

### Files for Professor Review
- **Chart**: `visualizations/complete_sentiment_trajectory_analysis.png`
- **URL Reference**: `visualizations/reddit_posts_urls_reference.txt`

## 📈 Analysis Details
- **9 Reddit posts** analyzed across 3 sentiment categories
- **Positive Posts**: Rust, Linux, WordPress discussions
- **Negative Posts**: Django, IBM, ComfyUI discussions  
- **Neutral Posts**: Kdenlive, Neovim, Rust discussions
- **Sentiment Range**: -1.0 (negative) to +1.0 (positive)

## 🔗 Accessing Original Posts
Copy and paste URLs from `reddit_posts_urls_reference.txt` to view the original Reddit discussions.
