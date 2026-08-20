class SentimentDataLoader {
    constructor() {
        this.data = null;
        this.processedData = null;
    }

    async loadData() {
        try {
            const response = await fetch('../data/enhanced_automated_sentiment_results.json');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            this.data = await response.json();
            this.processedData = this.processData();
            return this.processedData;
            
        } catch (error) {
            console.error('Error loading data:', error);
            throw new Error('Failed to load sentiment data. Please ensure the data file exists.');
        }
    }
    processData() {
        if (!this.data || !this.data.top_posts) {
            throw new Error('No valid data to process');
        }

        const processed = {
            positive: [],
            negative: [],
            neutral: []
        };

        const colors = {
            positive: ['#1f77b4', '#ff7f0e', '#2ca02c'],
            negative: ['#d62728', '#9467bd', '#8c564b'],
            neutral: ['#17becf', '#bcbd22', '#7f7f7f']
        };

        ['positive', 'negative', 'neutral'].forEach(category => {
            const posts = this.data.top_posts[category] || [];
            
            posts.forEach((post, index) => {
                const color = colors[category][index] || '#000000';
                const subreddit = post.subreddit || 'Unknown';
                const postKey = `${category.charAt(0).toUpperCase() + category.slice(1)} ${index + 1} (${subreddit})`;
                
                const metrics = post.metrics || {};
                const authorTrajectory = metrics.author_trajectory || [];
                const communityTrajectory = metrics.community_trajectory || [];
                
                const datasets = [];
                
                if (authorTrajectory.length > 0) {
                    datasets.push({
                        label: `${postKey} - Author (${metrics.author_replies_count || 0})`,
                        data: authorTrajectory.map((value, idx) => ({
                            x: idx + 1,
                            y: value
                        })),
                        borderColor: color,
                        backgroundColor: color + '20',
                        borderWidth: 3,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        pointBackgroundColor: color,
                        pointBorderColor: color,
                        pointBorderWidth: 2,
                        fill: false,
                        tension: 0.1,
                        spanGaps: false
                    });
                }
                
                if (communityTrajectory.length > 0) {
                    datasets.push({
                        label: `${postKey} - Community (${metrics.community_comments_count || 0})`,
                        data: communityTrajectory.map((value, idx) => ({
                            x: idx + 1,
                            y: value
                        })),
                        borderColor: color,
                        backgroundColor: color + '10',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.1,
                        spanGaps: false
                    });
                }
                
                const postInfo = {
                    postKey,
                    title: post.title || 'No title',
                    url: post.url || '#',
                    subreddit: subreddit,
                    sentimentScore: post.title_sentiment?.compound || 0,
                    numComments: post.num_comments || 0,
                    authorReplies: metrics.author_replies_count || 0,
                    communityComments: metrics.community_comments_count || 0,
                    qualityScore: metrics.overall_quality_score || 0,
                    authorReliability: metrics.author_trajectory_reliability || 0,
                    communityReliability: metrics.community_trajectory_reliability || 0,
                    datasets: datasets
                };
                
                processed[category].push(postInfo);
            });
        });

        return processed;
    }

    getCategoryData(category) {
        if (!this.processedData) {
            throw new Error('Data not loaded yet');
        }
        return this.processedData[category] || [];
    }

    getPostKeys(category) {
        const data = this.getCategoryData(category);
        return data.map(post => post.postKey);
    }

    getPostInfo(category, postKey) {
        const data = this.getCategoryData(category);
        return data.find(post => post.postKey === postKey);
    }
}

window.SentimentDataLoader = SentimentDataLoader;
