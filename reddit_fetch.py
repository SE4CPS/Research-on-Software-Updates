from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import matplotlib.pyplot as plt
import json
from wordcloud import WordCloud
import re
from collections import Counter

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

# print("\nStarting update of Reddit posts with sentiment results...")
# update_count = 0
# for post in reddit_data:
#     title = post['title']
#     sentiment_scores = analyzer.polarity_scores(title)
#     compound = sentiment_scores['compound']
#     if compound >= 0.05:
#         label = "Positive"
#     elif compound <= -0.05:
#         label = "Negative"
#     else:
#         label = "Neutral"
#     # Prepare only the sentiment fields to update
#     update_data = {
#         'sentiment_label': label,
#         'sentiment_score': compound
#     }
#     post_id = post['_id']
#     put_url = f"https://releasetrain.io/api/reddit/{post_id}"
#     try:
#         response = requests.put(put_url, data=json.dumps(update_data), headers={'Content-Type': 'application/json'})
#         if response.status_code == 200:
#             print(f"✅ Updated post {post_id} with sentiment: {label} (Score: {compound:.2f})")
#             update_count += 1
#         else:
#             print(f"❌ Failed to update post {post_id}. Status code: {response.status_code}")
#             print(f"Response: {response.text}")
#     except Exception as e:
#             print(f"❌ Exception updating post {post_id}: {e}")
# print(f"\nTotal posts updated: {update_count}")

# labels = list(large_label_counts.keys())
# counts = [large_label_counts[label] for label in labels]
# plt.figure(figsize=(7, 5))
# plt.bar(labels, counts, color=['green', 'red', 'gray'])
# plt.title('Sentiment Distribution of Reddit Post Titles')
# plt.xlabel('Sentiment')
# plt.ylabel('Number of Posts')
# for i, v in enumerate(counts):
#     plt.text(i, v + max(counts)*0.01, str(v), ha='center', va='bottom', fontsize=12)
# plt.tight_layout()
# plt.savefig('sentiment_distribution_bar_chart.png')
# plt.show()


positive_titles = []
neutral_titles = []
negative_titles = []

for post in reddit_data:
    title = post['title']
    sentiment_scores = analyzer.polarity_scores(title)
    compound = sentiment_scores['compound']
    if compound >= 0.05:
        positive_titles.append(title)
    elif compound <= -0.05:
        negative_titles.append(title)
    else:
        neutral_titles.append(title)

print(f"\n📊 Sentiment breakdown:")
print(f"  Positive titles: {len(positive_titles)}")
print(f"  Neutral titles: {len(neutral_titles)}")
print(f"  Negative titles: {len(negative_titles)}")

company_product_names = [
    'chrome', 'google', 'wordpress', 'microsoft', 'apple', 'windows', 'linux', 'android', 'ios',
    'firefox', 'mozilla', 'safari', 'edge', 'opera', 'github', 'gitlab', 'bitbucket', 'amazon',
    'aws', 'azure', 'facebook', 'meta', 'instagram', 'twitter', 'x', 'reddit', 'youtube', 'zoom',
    'slack', 'teams', 'notion', 'dropbox', 'skype', 'adobe', 'photoshop', 'illustrator', 'figma',
    'jira', 'confluence', 'trello', 'atlassian', 'salesforce', 'oracle', 'sap', 'dell', 'hp',
    'lenovo', 'thinkpad', 'mac', 'macos', 'ipad', 'iphone', 'pixel', 'samsung', 'huawei', 'oneplus',
    'nvidia', 'intel', 'amd', 'arm', 'tesla', 'spacex', 'uber', 'lyft', 'airbnb', 'netflix', 'disney',
    'hbo', 'prime', 'alexa', 'siri', 'cortana', 'bixby', 'openai', 'chatgpt', 'bard', 'gemini', 'copilot'
]
company_product_names = set(company_product_names)

def remove_company_names(text):
    words = text.split()
    filtered_words = [w for w in words if w.lower() not in company_product_names]
    return ' '.join(filtered_words)

def extract_words_from_text(text):
    """Extract individual words from text, cleaning and filtering them"""
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    words = clean_text.split()
    
    stop_words = set(['the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'that', 'with', 'as', 'by', 'are', 'this', 'it', 'an', 'be', 'from', 'at', 'or', 'which', 'was', 'has', 'have', 'can', 'not', 'we', 'more', 'but', 'our', 'also', 'will', 'all', 'been', 'their', 'about', 'other', 'its', 'one', 'you', 'had', 'new', 'get', 'go', 'up', 'out', 'do', 'my', 'me', 'so', 'what', 'just', 'like', 'very', 'know', 'take', 'see', 'come', 'think', 'look', 'want', 'give', 'use', 'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call'])
    
    filtered_words = [word for word in words if word not in stop_words and word not in company_product_names and len(word) > 2]
    return filtered_words

def generate_wordcloud_for_sentiment(titles, sentiment_type, filename):
    """Generate word cloud for a specific sentiment type"""
    if not titles:
        print(f"No {sentiment_type.lower()} titles found to generate word cloud.")
        return set()
    
    combined_text = ' '.join(titles)
    filtered_text = remove_company_names(combined_text)
    
    words = extract_words_from_text(filtered_text)
    word_set = set(words)
    
    if word_set:
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                            max_words=100, colormap='viridis').generate(filtered_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{sentiment_type} Sentiment Reddit Posts')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Generated {sentiment_type} word cloud: {filename}")
        return word_set
    else:
        print(f"No words found for {sentiment_type} sentiment after filtering.")
        return set()

print("\n🎨 Generating word clouds for positive and neutral sentiments...")
positive_words = generate_wordcloud_for_sentiment(positive_titles, "Positive", "positive_sentiment_wordcloud.png")
neutral_words = generate_wordcloud_for_sentiment(neutral_titles, "Neutral", "neutral_sentiment_wordcloud.png")

print("\n🧹 Cleaning negative sentiment list by removing overlapping words...")
if negative_titles:
    all_positive_neutral_words = positive_words.union(neutral_words)
    
    negative_text = ' '.join(negative_titles)
    filtered_negative_text = remove_company_names(negative_text)
    negative_words = extract_words_from_text(filtered_negative_text)
    
    original_negative_count = len(negative_words)
    cleaned_negative_words = [word for word in negative_words if word not in all_positive_neutral_words]
    removed_count = original_negative_count - len(cleaned_negative_words)
    
    print(f"  Original negative words: {original_negative_count}")
    print(f"  Words removed (overlapping with positive/neutral): {removed_count}")
    print(f"  Cleaned negative words: {len(cleaned_negative_words)}")
    
    if cleaned_negative_words:
        cleaned_negative_text = ' '.join(cleaned_negative_words)
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                            max_words=100, colormap='Reds').generate(cleaned_negative_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Negative Sentiment Reddit Posts')
        plt.tight_layout()
        plt.savefig('negative_sentiment_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated cleaned negative word cloud: negative_sentiment_wordcloud.png")
        
        removed_words = [word for word in negative_words if word in all_positive_neutral_words]
        if removed_words:
            print(f"\n📝 Examples of removed words (found in positive/neutral): {', '.join(removed_words[:10])}")
    else:
        print("❌ No words remaining in negative list after cleaning.")
else:
    print('No negative sentiment titles found to process.')