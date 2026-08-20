# Research on Software Updates - Sentiment Trajectory Analysis

## 📊 Project Overview
This project analyzes sentiment trajectories of Reddit posts related to software updates, examining how author sentiment compares to community replies over time. The project includes both manual analysis (for detailed research) and automated analysis (for scale and reproducibility).

## 📁 Project Structure

### 🎯 **Current Systems (Recommended)**

#### **Enhanced Automated System (Primary)**
- `scripts/enhanced_automated_sentiment_analysis.py` - Enhanced analysis with quality filtering
- `scripts/enhanced_visualization.py` - Quality-aware visualization with comprehensive legends
- `data/enhanced_automated_sentiment_results.json` - Quality-filtered results (324 posts)
- `visualizations/enhanced_sentiment_trajectory.png` - Enhanced chart with quality indicators
- `visualizations/enhanced_posts_reference.txt` - Quality metrics reference

#### **Manual System (For Detailed Research)**
- `scripts/complete_sentiment_trajectory_visualization.py` - Manual analysis visualization
- `visualizations/complete_sentiment_trajectory_analysis.png` - Manual analysis chart
- `visualizations/reddit_posts_urls_reference.txt` - Manual analysis reference

### 📊 **Data Files**
- `data/` - Contains Reddit post data and analysis results
- `documentation/` - Project documentation and review templates

## 🚀 How to Use

### **Option 1: Enhanced Automated Analysis (Recommended)**
```bash
# Step 1: Run enhanced analysis (quality-filtered, 324 posts)
python scripts/enhanced_automated_sentiment_analysis.py

# Step 2: Generate enhanced visualization
python scripts/enhanced_visualization.py
```

**Benefits:**
- ✅ Quality-filtered posts (3+ author replies, 5+ community comments)
- ✅ Reliability metrics and quality scores
- ✅ Comprehensive legends and zone labels
- ✅ 324 posts analyzed (vs 9 manual posts)
- ✅ Reproducible and scalable

### **Option 2: Manual Analysis (For Detailed Research)**
```bash
python scripts/complete_sentiment_trajectory_visualization.py
```

**Benefits:**
- ✅ Human-verified sentiment accuracy
- ✅ Detailed manual review and selection
- ✅ Suitable for academic paper validation

## 📈 Analysis Details

### **Enhanced Automated System:**
- **324 posts** analyzed with quality filtering
- **9 posts** selected (3 per category) with high engagement
- **Quality scores**: 2.0-3.9 (all high quality)
- **Reliability scores**: 0.4-0.7 (all reliable)
- **Sentiment Range**: -1.0 (negative) to +1.0 (positive)

### **Manual System:**
- **9 Reddit posts** manually selected and analyzed
- **Positive Posts**: Rust, Linux, WordPress discussions
- **Negative Posts**: Django, IBM, ComfyUI discussions  
- **Neutral Posts**: Kdenlive, Neovim, Rust discussions

## 🔗 Accessing Original Posts
- **Enhanced System**: Copy URLs from `visualizations/enhanced_posts_reference.txt`
- **Manual System**: Copy URLs from `visualizations/reddit_posts_urls_reference.txt`

## 📚 Documentation
- `AUTOMATED_SYSTEM_README.md` - Complete automated system guide
- `ENHANCED_SYSTEM_COMPARISON.md` - Comparison between systems
- `documentation/` - Manual review templates and guides
