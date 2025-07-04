import requests

url = 'https://releasetrain.io/api/reddit'
response = requests.get(url)

print("Status Code:", response.status_code)
print("Response Content:")
print(response.text)