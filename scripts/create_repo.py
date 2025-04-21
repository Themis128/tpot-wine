import requests

# GitHub API URL
GITHUB_API_URL = "https://api.github.com/user/repos"

# Replace with your GitHub username and personal access token
GITHUB_USERNAME = "themis128"
GITHUB_TOKEN = "your-personal-access-token"

# Repository details
repo_name = "RandomForest"
repo_description = "A repository for the RandomForest project"
is_private = False  # Set to True if you want the repository to be private

# API request payload
payload = {
    "name": repo_name,
    "description": repo_description,
    "private": is_private
}

# Make the API request to create the repository
response = requests.post(
    GITHUB_API_URL,
    json=payload,
    auth=(GITHUB_USERNAME, GITHUB_TOKEN)
)

# Check the response
if response.status_code == 201:
    print(f"✅ Repository '{repo_name}' created successfully!")
    print(f"URL: {response.json()['html_url']}")
else:
    print(f"❌ Failed to create repository: {response.status_code}")
    print(response.json())
