class SentimentTrajectoryApp {
    constructor() {
        this.dataLoader = new SentimentDataLoader();
        this.chartConfig = new SentimentChartConfig();
        this.charts = {};
        this.processedData = null;
    }

    async init() {
        try {
            this.showLoading();
            this.processedData = await this.dataLoader.loadData();
            this.hideLoading();
            this.showCharts();
            await this.createCharts();
            this.setupEventListeners();
        } catch (error) {
            this.showError('Failed to load sentiment data. Please check the console for details.');
        }
    }

    showLoading() {
        document.getElementById('loading').style.display = 'block';
        document.getElementById('charts-container').style.display = 'none';
        document.getElementById('error').style.display = 'none';
    }

    hideLoading() {
        document.getElementById('loading').style.display = 'none';
    }

    showCharts() {
        document.getElementById('charts-container').style.display = 'block';
    }

    showError(message) {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('charts-container').style.display = 'none';
        document.getElementById('error').style.display = 'block';
        document.getElementById('error-message').textContent = message;
    }

    async createCharts() {
        const categories = ['positive', 'negative', 'neutral'];
        
        for (const category of categories) {
            await this.createChart(category);
        }
    }
    async createChart(category) {
        const canvasId = `${category}Chart`;
        const canvas = document.getElementById(canvasId);
        
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const categoryData = this.processedData[category];
        
        if (!categoryData || categoryData.length === 0) {
            console.log(`No data for ${category} category`);
            return;
        }
        
        console.log(`${category} data:`, categoryData);

        let config;
        switch (category) {
            case 'positive':
                config = this.chartConfig.getPositiveConfig();
                break;
            case 'negative':
                config = this.chartConfig.getNegativeConfig();
                break;
            case 'neutral':
                config = this.chartConfig.getNeutralConfig();
                break;
            default:
                config = this.chartConfig.getBaseConfig();
        }

        config = this.chartConfig.addSentimentZones(config);

        const datasets = [];
        categoryData.forEach(postInfo => {
            postInfo.datasets.forEach(dataset => {
                datasets.push(dataset);
            });
        });

        config.data.datasets = datasets;
        this.charts[category] = new Chart(ctx, config);
        this.populateFilter(category, categoryData);
    }

    populateFilter(category, categoryData) {
        const filterId = `${category}Filter`;
        const filter = document.getElementById(filterId);
        
        if (!filter) return;

        filter.innerHTML = '';

        categoryData.forEach(postInfo => {
            const option = document.createElement('option');
            option.value = postInfo.postKey;
            option.textContent = postInfo.postKey;
            option.selected = true;
            filter.appendChild(option);
        });

        filter.addEventListener('change', () => {
            this.filterChart(category);
        });
    }

    filterChart(category) {
        const chart = this.charts[category];
        if (!chart) return;

        const filterId = `${category}Filter`;
        const filter = document.getElementById(filterId);
        const selectedOptions = Array.from(filter.selectedOptions).map(option => option.value);

        chart.data.datasets.forEach(dataset => {
            const postKey = dataset.label.split(' - ')[0];
            dataset.hidden = !selectedOptions.includes(postKey);
        });

        chart.update();
    }

    setupEventListeners() {
        window.resetZoom = (chartId) => {
            const category = chartId.replace('Chart', '');
            const chart = this.charts[category];
            if (chart) {
                chart.reset();
            }
        };

        window.toggleLegend = (chartId) => {
            const category = chartId.replace('Chart', '');
            const chart = this.charts[category];
            if (chart) {
                chart.options.plugins.legend.display = !chart.options.plugins.legend.display;
                chart.update();
            }
        };
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    const app = new SentimentTrajectoryApp();
    await app.init();
    window.sentimentApp = app;
});
