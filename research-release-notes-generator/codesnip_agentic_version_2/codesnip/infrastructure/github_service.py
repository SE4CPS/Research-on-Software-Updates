import requests
from typing import Optional
from codesnip.infrastructure.config_service import get_github_token


class GitHubService:
    BASE = "https://api.github.com"

    def _headers(self) -> dict:
        token = get_github_token()
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def get_pr_diff(self, repo: str, pr_number: int) -> str:
        """Fetch the unified diff for a GitHub PR (owner/repo, pr_number)."""
        url = f"{self.BASE}/repos/{repo}/pulls/{pr_number}"
        headers = {**self._headers(), "Accept": "application/vnd.github.v3.diff"}
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 404:
            raise ValueError(f"PR #{pr_number} not found in {repo}. Check repo name and PR number.")
        if response.status_code == 401:
            raise ValueError("GitHub token missing or invalid. Run: codesnip config github_token <token>")
        response.raise_for_status()
        return response.text

    def get_pr_commits(self, repo: str, pr_number: int) -> str:
        """Return commit messages for a PR as a newline-separated string."""
        url = f"{self.BASE}/repos/{repo}/pulls/{pr_number}/commits"
        response = requests.get(url, headers=self._headers(), timeout=30)
        response.raise_for_status()
        commits = response.json()
        return "\n".join(
            f"- {c['commit']['message'].splitlines()[0]}" for c in commits
        )

    def get_pr_meta(self, repo: str, pr_number: int) -> dict:
        """Return basic PR metadata (title, author, base branch, head branch)."""
        url = f"{self.BASE}/repos/{repo}/pulls/{pr_number}"
        response = requests.get(url, headers=self._headers(), timeout=30)
        response.raise_for_status()
        data = response.json()
        return {
            "title": data["title"],
            "author": data["user"]["login"],
            "base": data["base"]["ref"],
            "head": data["head"]["ref"],
            "url": data["html_url"],
            "state": data["state"],
        }
