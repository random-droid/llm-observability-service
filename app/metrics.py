"""
Datadog metrics module with HTTP API support for Cloud Run.

Supports two modes:
1. HTTP API mode (default for Cloud Run) - sends metrics directly to Datadog
2. StatsD mode (for local dev with agent) - uses DogStatsD UDP/socket
"""

from datadog import initialize, api, statsd
from app.config import get_settings
from app import bigquery_metrics
from typing import List, Dict, Any
import logging
import os
import time

logger = logging.getLogger(__name__)

# Track which mode we're using
_use_http_api = False
_initialized = False


def init_datadog():
    """Initialize Datadog - auto-detects HTTP API vs StatsD mode"""
    global _use_http_api, _initialized

    settings = get_settings()

    if not settings.DD_API_KEY:
        logger.warning("DD_API_KEY not set - Datadog metrics disabled")
        return False

    try:
        # Check if we should use HTTP API (Cloud Run) or StatsD (local with agent)
        socket_path = os.getenv('DD_DOGSTATSD_SOCKET', '/var/run/datadog/dsd.socket')
        agent_host = os.getenv('DD_AGENT_HOST', '')
        use_http = os.getenv('DD_USE_HTTP_API', 'auto').lower()

        # Auto-detect: use HTTP API if no agent available
        # Auto-detect: use HTTP API if no agent available
        if use_http == 'auto':
            has_socket = os.path.exists(socket_path)
            has_agent_host = bool(agent_host)
            _use_http_api = not (has_socket or has_agent_host)
        else:
            _use_http_api = use_http == 'true'

        # Initialize BigQuery metrics (if enabled)
        bigquery_metrics.init_bigquery()
        
        if _use_http_api:
            # HTTP API mode - for Cloud Run (no agent needed)
            logger.info("Using Datadog HTTP API mode (no agent required)")
            initialize(
                api_key=settings.DD_API_KEY,
                app_key=settings.DD_APP_KEY,
                api_host=f"https://api.{os.getenv('DD_SITE', 'datadoghq.com')}"
            )
        elif os.path.exists(socket_path):
            # Unix socket mode
            logger.info(f"Using DogStatsD Unix socket: {socket_path}")
            initialize(
                api_key=settings.DD_API_KEY,
                app_key=settings.DD_APP_KEY,
                statsd_socket_path=socket_path,
                statsd_constant_tags=[
                    f"env:{settings.DD_ENV}",
                    f"service:{settings.DD_SERVICE}"
                ],
                statsd_disable_buffering=True
            )
        else:
            # UDP mode
            host = agent_host or 'localhost'
            port = int(os.getenv('DD_DOGSTATSD_PORT', '8125'))
            logger.info(f"Using DogStatsD UDP: {host}:{port}")
            initialize(
                api_key=settings.DD_API_KEY,
                app_key=settings.DD_APP_KEY,
                statsd_host=host,
                statsd_port=port,
                statsd_constant_tags=[
                    f"env:{settings.DD_ENV}",
                    f"service:{settings.DD_SERVICE}"
                ],
                statsd_disable_buffering=True
            )

        _initialized = True
        logger.info(f"Datadog initialized successfully (HTTP API: {_use_http_api})")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize Datadog: {e}")
        return False


def _send_metrics_http(metrics: List[Dict[str, Any]]):
    """Send metrics via HTTP API"""
    try:
        series = []
        current_time = int(time.time())

        for m in metrics:
            series.append({
                'metric': m['metric'],
                'points': [(current_time, m['value'])],
                'type': m.get('type', 'gauge'),
                'tags': m.get('tags', [])
            })

        response = api.Metric.send(series)
        if response.get('status') != 'ok':
            logger.warning(f"Datadog API response: {response}")

    except Exception as e:
        logger.error(f"Failed to send metrics via HTTP API: {e}")


def _send_metrics_statsd(metrics: List[Dict[str, Any]]):
    """Send metrics via StatsD"""
    try:
        for m in metrics:
            metric_type = m.get('type', 'gauge')
            if metric_type == 'gauge':
                statsd.gauge(m['metric'], m['value'], tags=m.get('tags', []))
            elif metric_type == 'count':
                statsd.increment(m['metric'], m['value'], tags=m.get('tags', []))
            elif metric_type == 'histogram':
                statsd.histogram(m['metric'], m['value'], tags=m.get('tags', []))

        statsd.flush()

    except Exception as e:
        logger.error(f"Failed to send metrics via StatsD: {e}")


def _send_metrics(metrics: List[Dict[str, Any]]):
    """Send metrics using the appropriate method"""
    if not _initialized:
        logger.warning("Datadog not initialized, skipping metrics")
        return

    # Send to BigQuery (fire and forget, ideally async but keeping simple for now)
    bigquery_metrics.insert_metrics(metrics)

    if _use_http_api:
        _send_metrics_http(metrics)
    else:
        _send_metrics_statsd(metrics)


def track_review_metrics(
    repo: str,
    pr_id: int,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cost: float,
    duration: float,
    context_utilization: float,
    files_changed: int,
    total_changes: int,
    model: str = "gemini-2.5-pro",
    source: str = "webhook"
):
    """Send review metrics to Datadog

    Args:
        source: Origin of the review - 'webhook' for GitHub webhook, '_experiment' for /review endpoint
    """
    settings = get_settings()

    tags = [
        f"repo:{repo}",
        f"pr:{pr_id}",
        f"model:{model}",
        f"service:{settings.DD_SERVICE}",
        f"env:{settings.DD_ENV}",
        f"source:{source}"
    ]

    metrics = [
        {'metric': 'llm.tokens.prompt', 'value': prompt_tokens, 'type': 'gauge', 'tags': tags},
        {'metric': 'llm.tokens.completion', 'value': completion_tokens, 'type': 'gauge', 'tags': tags},
        {'metric': 'llm.tokens.total', 'value': total_tokens, 'type': 'gauge', 'tags': tags},
        {'metric': 'llm.cost.per_review', 'value': cost, 'type': 'gauge', 'tags': tags},
        {'metric': 'llm.duration.seconds', 'value': duration, 'type': 'histogram', 'tags': tags},
        {'metric': 'llm.context.utilization_pct', 'value': context_utilization, 'type': 'gauge', 'tags': tags},
        {'metric': 'pr.files_changed', 'value': files_changed, 'type': 'gauge', 'tags': tags},
        {'metric': 'pr.total_changes', 'value': total_changes, 'type': 'gauge', 'tags': tags},
        {'metric': 'llm.reviews.total', 'value': 1, 'type': 'count', 'tags': tags},
    ]

    if total_changes > 0:
        metrics.append({
            'metric': 'llm.cost.per_line',
            'value': cost / total_changes,
            'type': 'gauge',
            'tags': tags
        })

    _send_metrics(metrics)
    logger.info(f"Metrics sent for PR #{pr_id} (HTTP API: {_use_http_api})")


def track_security_event(event_type: str, severity: str, repo: str, pr_id: int, source: str = "webhook"):
    """Track security events

    Args:
        source: Origin of the review - 'webhook' for GitHub webhook, '_experiment' for /review endpoint
    """
    settings = get_settings()

    tags = [
        f"event_type:{event_type}",
        f"severity:{severity}",
        f"repo:{repo}",
        f"pr:{pr_id}",
        f"service:{settings.DD_SERVICE}",
        f"env:{settings.DD_ENV}",
        f"source:{source}"
    ]

    _send_metrics([{
        'metric': 'security.events',
        'value': 1,
        'type': 'count',
        'tags': tags
    }])


def track_error(error_type: str, repo: str, pr_id: int, source: str = "webhook"):
    """Track errors

    Args:
        source: Origin of the review - 'webhook' for GitHub webhook, '_experiment' for /review endpoint
    """
    settings = get_settings()

    tags = [
        f"error_type:{error_type}",
        f"repo:{repo}",
        f"pr:{pr_id}",
        f"service:{settings.DD_SERVICE}",
        f"env:{settings.DD_ENV}",
        f"source:{source}"
    ]

    _send_metrics([{
        'metric': 'llm.errors',
        'value': 1,
        'type': 'count',
        'tags': tags
    }])


def track_rag_indexing(
    repo: str,
    pr_id: int,
    files_indexed: int,
    chunks_created: int,
    indexing_duration: float,
    embedding_duration: float
):
    """Track RAG indexing metrics"""
    settings = get_settings()

    tags = [
        f"repo:{repo}",
        f"pr:{pr_id}",
        f"service:{settings.DD_SERVICE}",
        f"env:{settings.DD_ENV}"
    ]

    metrics = [
        {'metric': 'rag.indexing.files', 'value': files_indexed, 'type': 'gauge', 'tags': tags},
        {'metric': 'rag.indexing.chunks', 'value': chunks_created, 'type': 'gauge', 'tags': tags},
        {'metric': 'rag.indexing.duration_seconds', 'value': indexing_duration, 'type': 'histogram', 'tags': tags},
        {'metric': 'rag.embedding.duration_seconds', 'value': embedding_duration, 'type': 'histogram', 'tags': tags},
        {'metric': 'rag.indexing.total', 'value': 1, 'type': 'count', 'tags': tags},
    ]

    _send_metrics(metrics)
    logger.info(f"RAG indexing metrics sent: {chunks_created} chunks from {files_indexed} files")


def track_rag_retrieval(
    repo: str,
    pr_id: int,
    file_path: str,
    chunks_retrieved: int,
    retrieval_duration: float,
    context_tokens: int,
    max_similarity: float = 0.0
):
    """Track RAG retrieval metrics"""
    settings = get_settings()

    tags = [
        f"repo:{repo}",
        f"pr:{pr_id}",
        f"service:{settings.DD_SERVICE}",
        f"env:{settings.DD_ENV}"
    ]

    metrics = [
        {'metric': 'rag.retrieval.chunks', 'value': chunks_retrieved, 'type': 'gauge', 'tags': tags},
        {'metric': 'rag.retrieval.duration_seconds', 'value': retrieval_duration, 'type': 'histogram', 'tags': tags},
        {'metric': 'rag.retrieval.context_tokens', 'value': context_tokens, 'type': 'gauge', 'tags': tags},
        {'metric': 'rag.retrieval.total', 'value': 1, 'type': 'count', 'tags': tags},
        {'metric': 'rag.retrieval.max_similarity', 'value': max_similarity, 'type': 'gauge', 'tags': tags},
    ]

    _send_metrics(metrics)
    logger.info(f"RAG retrieval for {file_path}: {chunks_retrieved} chunks, ~{context_tokens} tokens, max_sim={max_similarity:.2f}")


def track_embedding_call(
    batch_size: int,
    duration: float,
    success: bool = True
):
    """Track Vertex AI embedding API calls"""
    settings = get_settings()

    tags = [
        f"service:{settings.DD_SERVICE}",
        f"env:{settings.DD_ENV}",
        f"success:{str(success).lower()}"
    ]

    metrics = [
        {'metric': 'rag.embedding.batch_size', 'value': batch_size, 'type': 'gauge', 'tags': tags},
        {'metric': 'rag.embedding.api_duration_seconds', 'value': duration, 'type': 'histogram', 'tags': tags},
        {'metric': 'rag.embedding.api_calls', 'value': 1, 'type': 'count', 'tags': tags},
    ]

    _send_metrics(metrics)
