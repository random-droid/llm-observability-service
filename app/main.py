"""
LLM Code Review Service with GitHub Webhook Integration.

This service provides:
1. AI-powered code review using Vertex AI (Gemini)
2. GitHub webhook integration for automatic PR reviews
3. Comprehensive observability with Datadog metrics
4. Security scanning for secrets, PII, and prompt injection
"""

from fastapi import FastAPI, HTTPException, Request, Header, BackgroundTasks
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import vertexai
from vertexai.generative_models import GenerativeModel
import json
import time
import logging
import os
from datetime import datetime
from typing import Optional

from app.config import get_settings
from app.models import (
    PRReviewRequest,
    ReviewResponse,
    ReviewMetrics,
    HealthResponse,
    GitHubWebhookPayload,
    WebhookResponse
)
from app.security import run_all_security_checks
from app.metrics import (
    init_datadog,
    track_review_metrics,
    track_security_event,
    track_error,
    track_rag_indexing,
    track_rag_retrieval
)
from app.rag import retriever
from app.github_client import GitHubClient, verify_webhook_signature

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

# Global GitHub client (initialized on startup)
github_client: Optional[GitHubClient] = None


# System message for AI reviewer (from ai_pr_reviewer)
SYSTEM_MESSAGE = """You are an expert code reviewer with deep knowledge of software engineering best practices, security vulnerabilities, and performance optimization.

Your role is to review code changes (git diffs) and provide actionable, accurate feedback.

CORE PRINCIPLES:
1. ACCURACY OVER QUANTITY: Only report genuine issues. False positives waste developer time.
2. CONTEXT MATTERS: Consider the codebase context provided before flagging issues.
3. BE SPECIFIC: Reference exact line numbers and explain the root cause clearly.
4. BE ACTIONABLE: Every issue should have a clear path to resolution.
5. SEVERITY APPROPRIATENESS: Reserve critical issues for actual bugs/security problems.

WHAT TO LOOK FOR:
- Critical: Security vulnerabilities, data leaks, crashes, infinite loops, race conditions
- Warning: Potential null/undefined issues, resource leaks, poor error handling, edge cases
- Suggestion: Code style, naming conventions, performance improvements, maintainability

OUTPUT FORMAT:
Print each issue in this format: "line_number : cause and effect"
If there are no issues, just say "No critical issues found".

Do not add introductory text. Start directly with the issues or "No critical issues found".
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global github_client

    # Startup
    logger.info("Starting LLM Code Review Service...")

    # Initialize Vertex AI
    if not settings.GCP_PROJECT:
        logger.error("GCP_PROJECT not set!")
    else:
        try:
            vertexai.init(
                project=settings.GCP_PROJECT,
                location=settings.GCP_LOCATION
            )
            logger.info(f"Vertex AI initialized: {settings.GCP_PROJECT}/{settings.GCP_LOCATION}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")


    # Initialize Datadog
    init_datadog()

    # Initialize GitHub client
    if settings.GITHUB_TOKEN:
        github_client = GitHubClient(settings.GITHUB_TOKEN)
        logger.info("GitHub client initialized")
    else:
        logger.warning("GITHUB_TOKEN not set - webhook functionality disabled")

    yield

    # Shutdown
    logger.info("Shutting down LLM Code Review Service...")


# Create FastAPI app
app = FastAPI(
    title="LLM Code Review Service",
    description="AI-powered code review with GitHub webhook integration and Datadog observability",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/", tags=["General"])
async def root():
    """Root endpoint - service info."""
    return {
        "message": "LLM Code Review Service",
        "docs": "/docs",
        "health": "/health",
        "webhook": "/webhook/github"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        gcp_project=settings.GCP_PROJECT,
        datadog_enabled=bool(settings.DD_API_KEY)
    )


@app.post("/webhook/github", response_model=WebhookResponse, tags=["Webhook"])
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: Optional[str] = Header(None),
    x_github_event: Optional[str] = Header(None)
):
    """
    GitHub webhook endpoint for PR events.

    Configure in GitHub:
    1. Go to repo Settings → Webhooks → Add webhook
    2. Payload URL: https://your-service-url/webhook/github
    3. Content type: application/json
    4. Secret: your webhook secret
    5. Events: Pull requests
    """
    # Get raw body for signature verification
    body = await request.body()

    # Log incoming request details for debugging
    logger.info(f"Webhook received - Event: {x_github_event}, Body length: {len(body)}")

    # Handle empty body (some ping events or health checks)
    if not body or len(body) == 0:
        logger.info("Received webhook with empty body")
        return WebhookResponse(
            status="ok",
            message="Empty body received"
        )

    # Verify webhook signature if secret is configured
    if settings.GITHUB_WEBHOOK_SECRET:
        if not verify_webhook_signature(body, x_hub_signature_256 or "", settings.GITHUB_WEBHOOK_SECRET):
            logger.warning("Invalid webhook signature")
            raise HTTPException(status_code=401, detail="Invalid signature")

    # Handle ping event (sent when webhook is first configured)
    if x_github_event == "ping":
        logger.info("Received GitHub ping event")
        return WebhookResponse(
            status="ok",
            message="Pong! Webhook configured successfully"
        )

    # Only process pull_request events
    if x_github_event != "pull_request":
        return WebhookResponse(
            status="skipped",
            message=f"Event type '{x_github_event}' not handled"
        )

    # Parse payload for PR events (use already-read body bytes)
    try:
        payload_dict = json.loads(body)
        payload = GitHubWebhookPayload(**payload_dict)
    except Exception as e:
        logger.error(f"Failed to parse webhook payload: {e}")
        logger.error(f"Body content (first 500 chars): {body[:500]}")
        raise HTTPException(status_code=400, detail="Invalid payload")

    # Only process opened, synchronize, reopened actions
    if payload.action not in ("opened", "synchronize", "reopened"):
        return WebhookResponse(
            status="skipped",
            message=f"Action '{payload.action}' not handled",
            pr_number=payload.pr_number
        )

    if not github_client:
        raise HTTPException(status_code=500, detail="GitHub client not configured")

    # Process review in background
    background_tasks.add_task(
        process_pr_review,
        payload.owner,
        payload.repo_name,
        payload.pr_number,
        payload.head_sha
    )

    return WebhookResponse(
        status="processing",
        message="PR review started",
        pr_number=payload.pr_number
    )


async def process_pr_review(owner: str, repo: str, pr_number: int, head_sha: str):
    """
    Process a PR review in the background.

    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: Pull request number
        head_sha: Head commit SHA
    """
    start_time = time.time()
    full_repo = f"{owner}/{repo}"
    files_reviewed = 0
    comments_posted = 0
    total_tokens = 0
    total_prompt_tokens = 0
    max_context_utilization = 0.0
    total_cost = 0.0

    # RAG components (optional)
    rag_retriever = None
    file_contents = {}

    try:
        logger.info(f"Starting review for {full_repo} PR #{pr_number}")

        # Get changed files
        files = github_client.get_pull_request_files(owner, repo, pr_number)
        logger.info(f"Found {len(files)} changed files")

        # Pre-fetch all file contents for RAG indexing
        for file_info in files:
            filename = file_info.get("filename", "")
            ext = os.path.splitext(filename)[1].lstrip(".")
            if ext in settings.TARGET_EXTENSIONS:
                content = github_client.get_file_content(owner, repo, filename, head_sha)
                if content:
                    file_contents[filename] = content

        # Try to build RAG index from PR files
        try:
            from app.rag import CodebaseIndexer, RAGRetriever
            indexer = CodebaseIndexer()

            indexing_start = time.time()
            codebase_index = indexer.index_files_from_github(
                files=files,
                get_content_func=lambda f: file_contents.get(f)
            )
            indexing_duration = time.time() - indexing_start

            if codebase_index.index.ntotal > 0:
                rag_retriever = RAGRetriever(codebase_index)
                logger.info(f"RAG index built with {codebase_index.index.ntotal} chunks")

                # Track RAG indexing metrics
                track_rag_indexing(
                    repo=full_repo,
                    pr_id=pr_number,
                    files_indexed=len(file_contents),
                    chunks_created=codebase_index.index.ntotal,
                    indexing_duration=indexing_duration,
                    embedding_duration=indexing_duration * 0.8  # Approximate: embeddings are ~80% of indexing time
                )
        except Exception as e:
            logger.warning(f"RAG indexing failed, proceeding without context: {e}")

        review_comments = []

        for file_info in files:
            filename = file_info.get("filename", "")
            patch = file_info.get("patch", "")

            # Skip if no patch (binary file or too large)
            if not patch:
                continue

            # Check file extension
            ext = os.path.splitext(filename)[1].lstrip(".")
            if ext not in settings.TARGET_EXTENSIONS:
                logger.info(f"Skipping {filename} - extension not in target list")
                continue

            content = file_contents.get(filename)
            if not content:
                logger.warning(f"Could not get content for {filename}")
                continue

            # Run security checks
            security_issues = run_all_security_checks(patch)
            for issue in security_issues:
                track_security_event(
                    event_type=issue.type,
                    severity=issue.severity,
                    repo=full_repo,
                    pr_id=pr_number,
                    source="webhook"
                )

            # Get RAG context if available
            rag_context = ""
            if rag_retriever:
                try:
                    retrieval_start = time.time()
                    context = rag_retriever.retrieve_context(
                        file_path=filename,
                        file_content=content,
                        diff_content=patch
                    )
                    rag_context = context.format_for_prompt(max_tokens=1500)
                    retrieval_duration = time.time() - retrieval_start

                    # Extract max similarity score from retrieved chunks
                    max_similarity = 0.0
                    if context.similar_chunks:
                        max_similarity = max(score for _, score in context.similar_chunks)

                    # Track RAG retrieval metrics
                    track_rag_retrieval(
                        repo=full_repo,
                        pr_id=pr_number,
                        file_path=filename,
                        chunks_retrieved=len(context.similar_chunks) + len(context.related_chunks),
                        retrieval_duration=retrieval_duration,
                        context_tokens=len(rag_context) // 4,  # Approximate token count
                        max_similarity=max_similarity
                    )
                except Exception as e:
                    logger.warning(f"Failed to get RAG context for {filename}: {e}")

            # Review the file with RAG context
            review_result = await review_file(
                filename=filename,
                content=content,
                patch=patch,
                rag_context=rag_context
            )

            if review_result:
                files_reviewed += 1
                total_tokens += review_result.get("tokens", 0)
                total_cost += review_result.get("cost", 0)

                # Track context utilization (max across all files)
                prompt_tokens = review_result.get("prompt_tokens", 0)
                total_prompt_tokens += prompt_tokens
                file_utilization = (prompt_tokens / settings.MAX_TOKENS) * 100
                max_context_utilization = max(max_context_utilization, file_utilization)

                # Collect comments for batch posting
                if review_result.get("comments"):
                    for comment in review_result["comments"]:
                        review_comments.append({
                            "path": filename,
                            "body": comment["text"],
                            "line": comment.get("line", 1),
                            "side": "RIGHT"
                        })

        # Post review with all comments
        if review_comments:
            success = github_client.create_review(
                owner=owner,
                repo=repo,
                pr_number=pr_number,
                commit_id=head_sha,
                body=f"AI Code Review - {len(review_comments)} issue(s) found",
                event="COMMENT",
                comments=review_comments
            )
            if success:
                comments_posted = len(review_comments)
            else:
                # Line-specific comments failed (likely invalid line numbers)
                # Fall back to posting as a general comment with all feedback
                comment_body = "## 🤖 AI Code Review\n\n"
                for comment in review_comments:
                    comment_body += f"**{comment['path']}**: {comment['body']}\n\n"
                github_client.post_issue_comment(
                    owner=owner,
                    repo=repo,
                    pr_number=pr_number,
                    body=comment_body
                )
                comments_posted = len(review_comments)
        else:
            # Post a "looks good" comment
            github_client.post_issue_comment(
                owner=owner,
                repo=repo,
                pr_number=pr_number,
                body="✅ **AI Code Review**: No critical issues found. LGTM!"
            )

        # Track metrics
        duration = time.time() - start_time
        track_review_metrics(
            repo=full_repo,
            pr_id=pr_number,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_tokens - total_prompt_tokens,
            total_tokens=total_tokens,
            cost=total_cost,
            duration=duration,
            context_utilization=max_context_utilization,
            files_changed=len(files),
            total_changes=sum(f.get("changes", 0) for f in files),
            model=settings.MODEL_NAME,
            source="webhook"
        )

        logger.info(
            f"Review completed: {full_repo} PR #{pr_number} | "
            f"Files: {files_reviewed} | Comments: {comments_posted} | "
            f"Cost: ${total_cost:.4f} | Duration: {duration:.2f}s"
        )

    except Exception as e:
        logger.error(f"Error reviewing PR {full_repo}#{pr_number}: {e}", exc_info=True)
        track_error(
            error_type=type(e).__name__,
            repo=full_repo,
            pr_id=pr_number,
            source="webhook"
        )


async def review_file(filename: str, content: str, patch: str, rag_context: str = "") -> dict:
    """
    Review a single file using Vertex AI.

    Args:
        filename: Path to the file
        content: Full file content
        patch: Git diff
        rag_context: Optional RAG context from similar code

    Returns:
        Dict with tokens, cost, and comments
    """
    try:
        # Build prompt with optional RAG context
        context_section = ""
        if rag_context:
            context_section = f"""
=== CODEBASE CONTEXT ===
Use this context to understand patterns and conventions in this codebase:
{rag_context}

"""

        prompt = f"""{SYSTEM_MESSAGE}
{context_section}
=== FILE BEING REVIEWED ===
{filename}

=== FULL FILE CONTENT ===
{content[:50000]}

=== GIT DIFF (focus your review here) ===
{patch}
"""

        # Call Vertex AI
        llm_start = time.time()
        model = GenerativeModel(settings.MODEL_NAME)
        response = model.generate_content(prompt)
        llm_duration = time.time() - llm_start

        # Extract metrics
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count
        completion_tokens = usage.candidates_token_count
        total_tokens = usage.total_token_count

        # Calculate cost
        cost = (
            (prompt_tokens / 1000 * settings.COST_PER_1K_INPUT) +
            (completion_tokens / 1000 * settings.COST_PER_1K_OUTPUT)
        )

        # Parse diff to get valid line numbers
        valid_lines = parse_diff_line_numbers(patch)
        logger.info(f"Valid diff lines for {filename}: {sorted(valid_lines)[:10]}... ({len(valid_lines)} total)")

        # Parse response into comments, filtering to valid diff lines
        comments = parse_review_response(response.text, valid_lines)

        logger.info(f"Reviewed {filename}: {len(comments)} issues, {total_tokens} tokens, ${cost:.4f}")

        return {
            "tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "cost": cost,
            "duration": llm_duration,
            "comments": comments
        }

    except Exception as e:
        logger.error(f"Error reviewing file {filename}: {e}")
        return None


def parse_diff_line_numbers(patch: str) -> set:
    """
    Extract valid line numbers from a git diff patch.

    Parses @@ -old_start,old_count +new_start,new_count @@ headers
    and tracks which lines in the new file are added/modified.

    Returns:
        Set of valid line numbers that can receive comments
    """
    import re
    valid_lines = set()

    # Match diff hunk headers: @@ -10,5 +12,7 @@
    hunk_pattern = re.compile(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@')

    current_line = 0
    in_hunk = False

    for line in patch.split('\n'):
        # Check for new hunk header
        match = hunk_pattern.match(line)
        if match:
            current_line = int(match.group(1))
            in_hunk = True
            continue

        if not in_hunk:
            continue

        if line.startswith('+') and not line.startswith('+++'):
            # Added line - valid for comments
            valid_lines.add(current_line)
            current_line += 1
        elif line.startswith('-') and not line.startswith('---'):
            # Deleted line - don't increment (not in new file)
            pass
        else:
            # Context line (unchanged) - also valid for comments
            valid_lines.add(current_line)
            current_line += 1

    return valid_lines


def parse_review_response(response: str, valid_lines: set = None) -> list:
    """
    Parse AI response into line-specific comments.

    Expected format: "line_number : description"

    Args:
        response: AI response text
        valid_lines: Optional set of valid line numbers from diff.
                     Comments on lines not in this set will be filtered out.
    """
    if not response:
        return []

    # Check for "no issues" response
    no_issues_indicators = ["no critical issues", "lgtm", "looks good"]
    if any(indicator in response.lower() for indicator in no_issues_indicators):
        return []

    comments = []
    filtered_comments = []

    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Extract line number from start
        number_str = ""
        for char in line:
            if char.isdigit():
                number_str += char
            else:
                break

        line_num = int(number_str) if number_str else None

        comment = {
            "line": line_num,
            "text": line
        }

        # Filter by valid lines if provided
        if valid_lines is not None:
            if line_num and line_num in valid_lines:
                comments.append(comment)
            else:
                # Track filtered comments for potential general comment
                filtered_comments.append(comment)
                logger.debug(f"Filtered comment on line {line_num} - not in diff")
        else:
            if line_num:
                comments.append(comment)

    return comments


# Keep the original /review endpoint for manual testing
@app.post("/review", response_model=ReviewResponse, tags=["Code Review"])
async def review_pr(request: PRReviewRequest):
    """
    Review a pull request using Vertex AI Gemini (manual endpoint).

    This endpoint:
    1. Runs security checks (secrets, PII, prompt injection)
    2. Calls Vertex AI Gemini for code review
    3. Tracks comprehensive metrics in Datadog
    4. Returns review with observability data
    """
    start_time = time.time()

    try:
        # Run security checks on PR diff
        logger.info(f"Reviewing PR #{request.pr_id} from {request.repo}")
        security_issues = run_all_security_checks(request.diff)

        # Track security events (marked as experiment since this is manual /review endpoint)
        for issue in security_issues:
            track_security_event(
                event_type=issue.type,
                severity=issue.severity,
                repo=request.repo,
                pr_id=request.pr_id,
                source="_experiment"
            )

        # Build prompt
        prompt = f"""{SYSTEM_MESSAGE}

Repository: {request.repo}
Pull Request: #{request.pr_id}
Files changed: {request.files_changed}
Lines added: {request.lines_added}
Lines deleted: {request.lines_deleted}

=== GIT DIFF ===
{request.diff}
"""

        # Call Vertex AI
        llm_start = time.time()
        model = GenerativeModel(settings.MODEL_NAME)
        response = model.generate_content(prompt)
        llm_duration = time.time() - llm_start

        # Extract metrics
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count
        completion_tokens = usage.candidates_token_count
        total_tokens = usage.total_token_count

        # Calculate cost
        cost = (
            (prompt_tokens / 1000 * settings.COST_PER_1K_INPUT) +
            (completion_tokens / 1000 * settings.COST_PER_1K_OUTPUT)
        )

        # Calculate context utilization
        context_utilization = (prompt_tokens / settings.MAX_TOKENS) * 100

        # Calculate PR complexity
        pr_complexity = request.lines_added + request.lines_deleted

        # Track metrics in Datadog (marked as experiment since this is manual /review endpoint)
        track_review_metrics(
            repo=request.repo,
            pr_id=request.pr_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            duration=llm_duration,
            context_utilization=context_utilization,
            files_changed=request.files_changed,
            total_changes=pr_complexity,
            model=settings.MODEL_NAME,
            source="_experiment"
        )

        total_duration = time.time() - start_time

        logger.info(
            f"Review completed: PR #{request.pr_id} | "
            f"Cost: ${cost:.4f} | "
            f"Tokens: {total_tokens} | "
            f"Duration: {llm_duration:.2f}s | "
            f"Security issues: {len(security_issues)}"
        )

        return ReviewResponse(
            review=response.text,
            metrics=ReviewMetrics(
                cost_usd=round(cost, 6),
                tokens_prompt=prompt_tokens,
                tokens_completion=completion_tokens,
                tokens_total=total_tokens,
                duration_ms=int(llm_duration * 1000),
                context_utilization_pct=round(context_utilization, 2),
                pr_complexity=pr_complexity
            ),
            security_issues=security_issues,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Error reviewing PR: {e}", exc_info=True)
        track_error(
            error_type=type(e).__name__,
            repo=request.repo,
            pr_id=request.pr_id,
            source="_experiment"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to review PR: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
