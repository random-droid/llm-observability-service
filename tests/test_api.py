"""
Integration tests for the LLM Code Review API endpoints.
"""
import hashlib
import hmac
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self):
        """Health endpoint should return healthy status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_includes_version(self):
        """Health endpoint should include version."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data

    def test_health_includes_datadog_status(self):
        """Health endpoint should indicate Datadog status."""
        response = client.get("/health")
        data = response.json()
        assert "datadog_enabled" in data


class TestWebhookEndpoint:
    """Tests for the /webhook/github endpoint."""

    def _create_signature(self, payload: dict, secret: str) -> str:
        """Create HMAC-SHA256 signature for webhook payload."""
        payload_bytes = json.dumps(payload, separators=(',', ':')).encode()
        signature = hmac.new(
            secret.encode(),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"

    def test_webhook_rejects_missing_signature(self):
        """Webhook should reject requests without signature."""
        response = client.post(
            "/webhook/github",
            json={"action": "opened"},
            headers={"X-GitHub-Event": "pull_request"}
        )
        assert response.status_code == 401

    def test_webhook_rejects_invalid_signature(self):
        """Webhook should reject requests with invalid signature."""
        response = client.post(
            "/webhook/github",
            json={"action": "opened"},
            headers={
                "X-GitHub-Event": "pull_request",
                "X-Hub-Signature-256": "sha256=invalid"
            }
        )
        assert response.status_code == 401

    def test_webhook_ignores_non_pr_events(self):
        """Webhook should ignore non-pull_request events."""
        payload = {"action": "created"}

        with patch('app.config.get_settings') as mock_settings:
            mock_settings.return_value.GITHUB_WEBHOOK_SECRET = "test-secret"
            signature = self._create_signature(payload, "test-secret")

            response = client.post(
                "/webhook/github",
                json=payload,
                headers={
                    "X-GitHub-Event": "push",
                    "X-Hub-Signature-256": signature
                }
            )

        assert response.status_code == 200
        assert response.json()["status"] == "ignored"

    def test_webhook_ignores_closed_pr(self):
        """Webhook should ignore closed PR events."""
        payload = {"action": "closed"}

        with patch('app.config.get_settings') as mock_settings:
            mock_settings.return_value.GITHUB_WEBHOOK_SECRET = "test-secret"
            signature = self._create_signature(payload, "test-secret")

            response = client.post(
                "/webhook/github",
                json=payload,
                headers={
                    "X-GitHub-Event": "pull_request",
                    "X-Hub-Signature-256": signature
                }
            )

        assert response.status_code == 200
        assert response.json()["status"] == "ignored"

    @patch('app.main.github_client')
    @patch('app.main.process_pr_review')
    def test_webhook_accepts_valid_pr_opened(self, mock_process, mock_github):
        """Webhook should accept valid PR opened events."""
        payload = {
            "action": "opened",
            "pull_request": {
                "number": 123,
                "title": "Test PR",
                "head": {"sha": "abc123"}
            },
            "repository": {
                "full_name": "owner/repo"
            }
        }

        mock_github.get_pr_files.return_value = []

        with patch('app.config.get_settings') as mock_settings:
            settings = MagicMock()
            settings.GITHUB_WEBHOOK_SECRET = "test-secret"
            settings.GITHUB_TOKEN = "token"
            mock_settings.return_value = settings

            signature = self._create_signature(payload, "test-secret")

            response = client.post(
                "/webhook/github",
                json=payload,
                headers={
                    "X-GitHub-Event": "pull_request",
                    "X-Hub-Signature-256": signature
                }
            )

        assert response.status_code == 200
        assert response.json()["status"] == "processing"


class TestReviewEndpoint:
    """Tests for the /review endpoint (manual testing endpoint)."""

    @patch('app.main.GenerativeModel')
    def test_review_requires_code(self, mock_model):
        """Review endpoint should require code in request."""
        response = client.post(
            "/review",
            json={
                "repo": "owner/repo",
                "pr_number": 1,
                "file_path": "test.py"
                # Missing 'code' field
            }
        )
        assert response.status_code == 422  # Validation error

    @patch('app.main.GenerativeModel')
    @patch('app.metrics.track_review_metrics')
    def test_review_returns_analysis(self, mock_metrics, mock_model):
        """Review endpoint should return code analysis."""
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.text = "Code looks good. No issues found."
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.total_token_count = 150

        mock_instance = MagicMock()
        mock_instance.generate_content.return_value = mock_response
        mock_model.return_value = mock_instance

        response = client.post(
            "/review",
            json={
                "repo": "owner/repo",
                "pr_number": 1,
                "file_path": "test.py",
                "code": "def hello():\n    print('hello')",
                "lines_added": 2,
                "lines_deleted": 0,
                "files_changed": 1
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "review" in data
        assert "metrics" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
