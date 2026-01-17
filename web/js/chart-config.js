class SentimentChartConfig {
    constructor() {
        this.colors = {
            positive: '#28a745',
            negative: '#dc3545',
            neutral: '#6c757d',
            background: {
                positive: 'rgba(40, 167, 69, 0.1)',
                negative: 'rgba(220, 53, 69, 0.1)',
                neutral: 'rgba(108, 117, 125, 0.1)'
            }
        };
    }

    getBaseConfig() {
        return {
            type: 'line',
            data: {
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    title: {
                        display: false
                    },
                    legend: {
                        display: true,
                        position: 'right',
                        align: 'start',
                        labels: {
                            usePointStyle: true,
                            pointStyle: 'circle',
                            padding: 15,
                            font: {
                                size: 11,
                                family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
                            },
                            generateLabels: function(chart) {
                                const original = Chart.defaults.plugins.legend.labels.generateLabels;
                                const labels = original.call(this, chart);
                                
                                labels.forEach(label => {
                                    if (label.text.includes('Author')) {
                                        label.pointStyle = 'circle';
                                        label.lineWidth = 3;
                                    } else if (label.text.includes('Community')) {
                                        label.pointStyle = 'line';
                                        label.lineWidth = 2;
                                        label.borderDash = [5, 5];
                                    }
                                });
                                
                                return labels;
                            }
                        }
                    },
                    tooltip: {
                        enabled: true,
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: '#fff',
                        borderWidth: 1,
                        cornerRadius: 6,
                        displayColors: true,
                        callbacks: {
                            title: function(context) {
                                return `Comment ${context[0].parsed.x}`;
                            },
                            label: function(context) {
                                const label = context.dataset.label || '';
                                const value = context.parsed.y;
                                const sentiment = value > 0.1 ? 'Positive' : value < -0.1 ? 'Negative' : 'Neutral';
                                return `${label}: ${value.toFixed(3)} (${sentiment})`;
                            },
                            afterBody: function(context) {
                                const dataset = context[0].dataset;
                                if (dataset.postInfo) {
                                    return [
                                        `Quality Score: ${dataset.postInfo.qualityScore.toFixed(3)}`,
                                        `Reliability: ${dataset.postInfo.authorReliability.toFixed(3)}`
                                    ];
                                }
                                return [];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: 'Comment Sequence',
                            font: {
                                size: 12,
                                weight: 'bold',
                                family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
                            },
                            color: '#495057'
                        },
                        grid: {
                            display: true,
                            color: 'rgba(0, 0, 0, 0.1)',
                            lineWidth: 1
                        },
                        ticks: {
                            font: {
                                size: 10,
                                family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
                            },
                            color: '#6c757d',
                            stepSize: 10
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Sentiment Score (-1 to +1)',
                            font: {
                                size: 12,
                                weight: 'bold',
                                family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
                            },
                            color: '#495057'
                        },
                        min: -1,
                        max: 1,
                        grid: {
                            display: true,
                            color: 'rgba(0, 0, 0, 0.1)',
                            lineWidth: 1
                        },
                        ticks: {
                            font: {
                                size: 10,
                                family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
                            },
                            color: '#6c757d',
                            stepSize: 0.2,
                            callback: function(value) {
                                return value.toFixed(1);
                            }
                        }
                    }
                },
                elements: {
                    point: {
                        hoverRadius: 8,
                        hoverBorderWidth: 2
                    },
                    line: {
                        tension: 0.1
                    }
                },
                onClick: function(event, elements) {
                    if (elements.length > 0) {
                        const datasetIndex = elements[0].datasetIndex;
                        const dataset = this.data.datasets[datasetIndex];
                        if (dataset.postInfo && dataset.postInfo.url) {
                            window.open(dataset.postInfo.url, '_blank');
                        }
                    }
                }
            }
        };
    }

    /**
     * Get configuration for positive sentiment chart
     */
    getPositiveConfig() {
        const config = this.getBaseConfig();
        config.options.scales.y.grid.color = 'rgba(40, 167, 69, 0.1)';
        return config;
    }

    /**
     * Get configuration for negative sentiment chart
     */
    getNegativeConfig() {
        const config = this.getBaseConfig();
        config.options.scales.y.grid.color = 'rgba(220, 53, 69, 0.1)';
        return config;
    }

    /**
     * Get configuration for neutral sentiment chart
     */
    getNeutralConfig() {
        const config = this.getBaseConfig();
        config.options.scales.y.grid.color = 'rgba(108, 117, 125, 0.1)';
        return config;
    }

    /**
     * Create sentiment zone annotations
     */
    getSentimentZones() {
        return {
            annotations: {
                box1: {
                    type: 'box',
                    yMin: 0.25,
                    yMax: 1,
                    backgroundColor: 'rgba(144, 238, 144, 0.1)',
                    borderColor: 'rgba(144, 238, 144, 0.3)',
                    borderWidth: 1,
                    label: {
                        display: true,
                        content: 'Positive Zone',
                        position: 'start',
                        backgroundColor: 'rgba(144, 238, 144, 0.7)',
                        color: '#2d5a2d',
                        font: {
                            size: 10,
                            weight: 'bold'
                        }
                    }
                },
                box2: {
                    type: 'box',
                    yMin: -0.25,
                    yMax: 0.25,
                    backgroundColor: 'rgba(211, 211, 211, 0.1)',
                    borderColor: 'rgba(211, 211, 211, 0.3)',
                    borderWidth: 1,
                    label: {
                        display: true,
                        content: 'Neutral Zone',
                        position: 'start',
                        backgroundColor: 'rgba(211, 211, 211, 0.7)',
                        color: '#555',
                        font: {
                            size: 10,
                            weight: 'bold'
                        }
                    }
                },
                box3: {
                    type: 'box',
                    yMin: -1,
                    yMax: -0.25,
                    backgroundColor: 'rgba(255, 182, 193, 0.1)',
                    borderColor: 'rgba(255, 182, 193, 0.3)',
                    borderWidth: 1,
                    label: {
                        display: true,
                        content: 'Negative Zone',
                        position: 'start',
                        backgroundColor: 'rgba(255, 182, 193, 0.7)',
                        color: '#8b0000',
                        font: {
                            size: 10,
                            weight: 'bold'
                        }
                    }
                }
            }
        };
    }

    /**
     * Add sentiment zones to chart configuration
     */
    addSentimentZones(config) {
        if (!config.options.plugins) {
            config.options.plugins = {};
        }
        config.options.plugins.annotation = this.getSentimentZones();
        return config;
    }

    /**
     * Create dataset for a specific post
     */
    createPostDataset(postInfo, isAuthor = true) {
        const trajectory = isAuthor ? 
            postInfo.datasets.find(d => d.label.includes('Author'))?.data || [] :
            postInfo.datasets.find(d => d.label.includes('Community'))?.data || [];

        if (trajectory.length === 0) return null;

        const baseDataset = {
            label: `${postInfo.postKey} - ${isAuthor ? 'Author' : 'Community'} (${isAuthor ? postInfo.authorReplies : postInfo.communityComments})`,
            data: trajectory,
            postInfo: postInfo,
            borderColor: postInfo.datasets.find(d => d.label.includes(isAuthor ? 'Author' : 'Community'))?.borderColor || '#000000',
            backgroundColor: postInfo.datasets.find(d => d.label.includes(isAuthor ? 'Author' : 'Community'))?.backgroundColor || '#00000020',
            borderWidth: isAuthor ? 3 : 2,
            pointRadius: isAuthor ? 6 : 0,
            pointHoverRadius: isAuthor ? 8 : 4,
            fill: false,
            tension: 0.1,
            spanGaps: false
        };

        if (isAuthor) {
            baseDataset.pointBackgroundColor = baseDataset.borderColor;
            baseDataset.pointBorderColor = baseDataset.borderColor;
            baseDataset.pointBorderWidth = 2;
        } else {
            baseDataset.borderDash = [5, 5];
        }

        return baseDataset;
    }

    /**
     * Filter datasets based on selected posts
     */
    filterDatasets(datasets, selectedPosts) {
        if (!selectedPosts || selectedPosts.length === 0) {
            return datasets;
        }
        
        return datasets.filter(dataset => {
            const postKey = dataset.label.split(' - ')[0];
            return selectedPosts.includes(postKey);
        });
    }
}

// Export for use in other modules
window.SentimentChartConfig = SentimentChartConfig;
