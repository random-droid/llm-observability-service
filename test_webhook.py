#!/usr/bin/env python3
"""
Test script to simulate a GitHub webhook for PR review.
This triggers the full RAG pipeline and metrics tracking.
"""

import requests
import json

# Webhook endpoint
WEBHOOK_URL = "http://127.0.0.1:8000/webhook/github"

# Simulated GitHub webhook payload for PR #4
# Using random-droid/ai_pr_reviewer repo
payload = {
    "action": "opened",
    "number": 4,
    "pull_request": {
        "number": 4,
        "title": "Add RAG-powered codebase context and multi-provider AI support",
        "head": {
            "sha": "8094b379c65aa9428a8577deba5818e461731940",
            "ref": "feature/rag"
        },
        "base": {
            "ref": "main"
        }
    },
    "repository": {
        "name": "ai_pr_reviewer",
        "full_name": "random-droid/ai_pr_reviewer",
        "owner": {
            "login": "random-droid"
        }
    },
    "sender": {
        "login": "random-droid"
    }
}

print("Sending simulated GitHub webhook...")
print(f"Repository: {payload['repository']['full_name']}")
print(f"PR: #{payload['number']} - {payload['pull_request']['title']}")
print()

response = requests.post(
    WEBHOOK_URL,
    json=payload,
    headers={
        "Content-Type": "application/json",
        "X-GitHub-Event": "pull_request"
    }
)

print(f"Response Status: {response.status_code}")
print(f"Response Body: {json.dumps(response.json(), indent=2)}")
print()
print("Check the server logs for RAG indexing and retrieval metrics!")
