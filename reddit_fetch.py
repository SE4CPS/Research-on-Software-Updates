from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import matplotlib.pyplot as plt
import json

url = "https://releasetrain.io/api/reddit"

try:
    response = requests.get(url)
    response.raise_for_status()  
    reddit_data = response.json()

    print(f"✅ Fetched {len(reddit_data)} Reddit posts.\n")
    if reddit_data:
        print("Sample Reddit post keys:", reddit_data[0].keys())
        print("Sample Reddit post:", reddit_data[0])

except requests.exceptions.RequestException as e:
    print("❌ Failed to fetch data from API.")
    print(e)

analyzer = SentimentIntensityAnalyzer()

print("\n🧠 Sentiment Analysis on Top 10 Titles (for threshold verification):\n")

sample_size = 10
sample_results = []

for i, post in enumerate(reddit_data[:sample_size]):
    title = post['title']
    sentiment_scores = analyzer.polarity_scores(title)
    compound = sentiment_scores['compound']
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    sample_results.append((title, compound, label))
    print(f"{i+1}. {title}")
    print(f"   Sentiment: {label} (Compound Score: {compound:.2f})\n")

label_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
for _, _, label in sample_results:
    label_counts[label] += 1
print("Summary of sample:")
for label, count in label_counts.items():
    print(f"  {label}: {count} out of {sample_size}")

large_sample_size = len(reddit_data)
large_label_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
for post in reddit_data[:large_sample_size]:
    title = post['title']
    sentiment_scores = analyzer.polarity_scores(title)
    compound = sentiment_scores['compound']
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    large_label_counts[label] += 1
print(f"\nSentiment distribution for {large_sample_size} Reddit post titles:")
for label, count in large_label_counts.items():
    percent = (count / large_sample_size) * 100
    print(f"  {label}: {count} ({percent:.2f}%)")

print("\nStarting update of Reddit posts with sentiment results...")
update_count = 0
for post in reddit_data:
    title = post['title']
    sentiment_scores = analyzer.polarity_scores(title)
    compound = sentiment_scores['compound']
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    update_data = {
        'sentiment_label': label,
        'sentiment_score': compound
    }
    post_id = post['_id']
    put_url = f"https://releasetrain.io/api/reddit/{post_id}"
    try:
        response = requests.put(put_url, data=json.dumps(update_data), headers={'Content-Type': 'application/json'})
        if response.status_code == 200:
            print(f"✅ Updated post {post_id} with sentiment: {label} (Score: {compound:.2f})")
            update_count += 1
        else:
            print(f"❌ Failed to update post {post_id}. Status code: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Exception updating post {post_id}: {e}")
print(f"\nTotal posts updated: {update_count}")

labels = list(large_label_counts.keys())
counts = [large_label_counts[label] for label in labels]
plt.figure(figsize=(7, 5))
plt.bar(labels, counts, color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution of Reddit Post Titles')
plt.xlabel('Sentiment')
plt.ylabel('Number of Posts')
for i, v in enumerate(counts):
    plt.text(i, v + max(counts)*0.01, str(v), ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig('sentiment_distribution_bar_chart.png')
plt.show()