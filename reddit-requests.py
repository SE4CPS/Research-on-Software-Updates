import requests
import json

def fetch_reddit_data():
    url = "https://releasetrain.io/api/reddit"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
        # Pretty print the JSON data
        print(json.dumps(data, indent=4))
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    fetch_reddit_data()
