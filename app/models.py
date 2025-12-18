from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class PRReviewRequest(BaseModel):
    """Manual PR review request."""
    repo: str = Field(..., description="Repository name (e.g., 'owner/repo')")
    pr_id: int = Field(..., description="Pull request number")
    diff: str = Field(..., description="Git diff content")
    files_changed: int = Field(..., ge=0, description="Number of files changed")
    lines_added: int = Field(..., ge=0, description="Lines added")
    lines_deleted: int = Field(..., ge=0, description="Lines deleted")

    class Config:
        json_schema_extra = {
            "example": {
                "repo": "myorg/myrepo",
                "pr_id": 123,
                "diff": "- old code\n+ new code",
                "files_changed": 2,
                "lines_added": 15,
                "lines_deleted": 8
            }
        }


class GitHubWebhookPayload(BaseModel):
    """GitHub webhook payload for pull_request events."""
    action: str
    number: int
    pull_request: Dict[str, Any]
    repository: Dict[str, Any]
    sender: Dict[str, Any]

    @property
    def owner(self) -> str:
        return self.repository.get("owner", {}).get("login", "")

    @property
    def repo_name(self) -> str:
        return self.repository.get("name", "")

    @property
    def full_repo(self) -> str:
        return self.repository.get("full_name", "")

    @property
    def pr_number(self) -> int:
        return self.number

    @property
    def head_sha(self) -> str:
        return self.pull_request.get("head", {}).get("sha", "")

    @property
    def head_ref(self) -> str:
        return self.pull_request.get("head", {}).get("ref", "")

    @property
    def base_ref(self) -> str:
        return self.pull_request.get("base", {}).get("ref", "")


class WebhookResponse(BaseModel):
    """Response for webhook processing."""
    status: str
    message: str
    pr_number: Optional[int] = None
    files_reviewed: Optional[int] = None
    comments_posted: Optional[int] = None


class SecurityIssue(BaseModel):
    type: str
    severity: str
    details: Optional[str] = None


class ReviewMetrics(BaseModel):
    cost_usd: float
    tokens_prompt: int
    tokens_completion: int
    tokens_total: int
    duration_ms: int
    context_utilization_pct: float
    pr_complexity: int


class ReviewResponse(BaseModel):
    review: str
    metrics: ReviewMetrics
    security_issues: List[SecurityIssue]
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    version: str
    gcp_project: str
    datadog_enabled: bool
