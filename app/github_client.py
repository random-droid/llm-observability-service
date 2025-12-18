"""
GitHub API client for posting PR review comments.
"""

import requests
import hmac
import hashlib
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class GitHubClient:
    """Client for interacting with GitHub API."""

    def __init__(self, token: str):
        """
        Initialize GitHub client.

        Args:
            token: GitHub personal access token or app token
        """
        self.token = token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }

    def get_pull_request(self, owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
        """Get pull request details."""
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_pull_request_files(self, owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
        """Get list of files changed in a PR."""
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/files"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_file_content(self, owner: str, repo: str, path: str, ref: str) -> Optional[str]:
        """Get file content from a specific ref (branch/commit)."""
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": ref}
        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code == 404:
            return None

        response.raise_for_status()
        data = response.json()

        # Content is base64 encoded
        import base64
        content = base64.b64decode(data["content"]).decode("utf-8")
        return content

    def post_review_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
        commit_id: str,
        path: str,
        line: int
    ) -> bool:
        """
        Post a review comment on a specific line.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            body: Comment text
            commit_id: Commit SHA
            path: File path
            line: Line number in the diff

        Returns:
            True if successful, False otherwise
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/comments"
        data = {
            "body": body,
            "commit_id": commit_id,
            "path": path,
            "line": line,
            "side": "RIGHT"  # Comment on the new version
        }

        try:
            response = requests.post(url, headers=self.headers, json=data)
            if response.status_code in (200, 201):
                logger.info(f"Posted comment to {path}:{line}")
                return True
            else:
                logger.warning(f"Failed to post line comment: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error posting line comment: {e}")
            return False

    def post_issue_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str
    ) -> bool:
        """
        Post a general comment on the PR.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            body: Comment text

        Returns:
            True if successful, False otherwise
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/issues/{pr_number}/comments"
        data = {"body": body}

        try:
            response = requests.post(url, headers=self.headers, json=data)
            if response.status_code in (200, 201):
                logger.info(f"Posted general comment to PR #{pr_number}")
                return True
            else:
                logger.warning(f"Failed to post general comment: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error posting general comment: {e}")
            return False

    def create_review(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        commit_id: str,
        body: str,
        event: str = "COMMENT",
        comments: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Create a pull request review with multiple comments.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            commit_id: Commit SHA
            body: Review summary
            event: APPROVE, REQUEST_CHANGES, or COMMENT
            comments: List of inline comments

        Returns:
            True if successful, False otherwise
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
        data = {
            "commit_id": commit_id,
            "body": body,
            "event": event,
        }

        if comments:
            data["comments"] = comments

        try:
            response = requests.post(url, headers=self.headers, json=data)
            if response.status_code in (200, 201):
                logger.info(f"Created review for PR #{pr_number}")
                return True
            else:
                logger.warning(f"Failed to create review: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error creating review: {e}")
            return False


def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """
    Verify GitHub webhook signature.

    Args:
        payload: Raw request body
        signature: X-Hub-Signature-256 header value
        secret: Webhook secret

    Returns:
        True if signature is valid
    """
    if not signature or not secret:
        return False

    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected, signature)
