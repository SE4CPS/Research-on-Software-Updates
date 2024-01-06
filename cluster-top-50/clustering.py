import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import re
import json

# Step 1: Read the data with specified encoding
file_path = 'data.csv'
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()


# Step 2: Pre-process and Cluster the Data
# Number of clusters
n_clusters = 50

# Create a TF-IDF Vectorizer to convert the text data to numerical data
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(lines)

# KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(X)

# Extracting cluster labels
labels = kmeans.labels_

# Grouping the data by cluster labels
clustered_items = {}
for item, label in zip(lines, labels):
    if label not in clustered_items:
        clustered_items[label] = []
    clustered_items[label].append(item.strip())

# Count the number of items in each cluster for later use
cluster_summary = {label: len(items) for label, items in clustered_items.items()}

# Step 3: Extract Top Words
def extract_top_words(texts, top_n=5):
    """ Extract the top N words from a list of texts excluding common stop words. """
    words = []
    for text in texts:
        # Remove non-alphanumeric characters for cleaner word analysis
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        words.extend(clean_text.lower().split())

    # Count the frequency of each word
    word_count = Counter(words)
    
    # Remove common stop words (a simple list for demonstration)
    stop_words = set(["the", "and", "to", "of", "a", "in", "for", "is", "on", "that", "with", "as", "by", "are", "this", "it", "an", "be", "from", "at", "or", "which", "was", "has", "have", "can", "not", "we", "more", "but", "our", "also", "will", "all", "been", "their", "about", "other", "its", "one", "you", "had", "new"])
    words_filtered = [word for word in word_count if word not in stop_words]

    # Sort the words by frequency
    most_common_words = sorted(words_filtered, key=lambda x: word_count[x], reverse=True)

    return most_common_words[:top_n]

# Extracting top words for each cluster
cluster_descriptions = {}
for label, items in clustered_items.items():
    top_words = extract_top_words(items)
    cluster_descriptions[label] = ', '.join(top_words)

cluster_descriptions_sorted = dict(sorted(cluster_descriptions.items(), key=lambda item: cluster_summary[item[0]], reverse=True))

# Combine cluster descriptions and sizes into one dictionary
combined_cluster_info = {
    str(label): {"description": description, "size": cluster_summary[label]}
    for label, description in cluster_descriptions.items()
}

# Sort the combined data by cluster size
combined_cluster_info_sorted = dict(sorted(combined_cluster_info.items(), key=lambda item: item[1]["size"], reverse=True))

# Convert the combined cluster information to a JSON format
json_data = json.dumps(combined_cluster_info_sorted, indent=4)

# Specify the path for the output file (change this to your desired file path)
output_file_path = 'cluster_descriptions_with_sizes.json'

# Writing the JSON data to a file
with open(output_file_path, 'w') as file:
    file.write(json_data)

print(f"Cluster descriptions with sizes written to {output_file_path}")