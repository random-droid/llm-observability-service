# LLM Code Review Service

AI-powered code review service with RAG (Retrieval-Augmented Generation) for contextual analysis. Integrates with GitHub webhooks to automatically review pull requests using Google's Gemini 2.5 Pro model.

**Why Observability Matters**: As AI transitions from standalone tools to embedded infrastructure, operational concerns become critical:
- Inference at scale rivals training in complexity and cost
- Value is shifting to deployment, evaluation, and integration
- Latency, reliability, and cost control outweigh marginal benchmark gains
- Sustainable, long-term scalable systems require visibility

This service demonstrates production-grade AI with built-in Datadog observability.

## Features

- **Automated PR Reviews**: Triggered via GitHub webhooks on pull request events
- **RAG-Powered Context**: Uses FAISS vector search to provide relevant codebase context
- **Security Scanning**: Detects secrets, PII, and potential prompt injection
- **Datadog Observability**: Comprehensive metrics for LLM costs, latency, and RAG performance
- **Multi-file Support**: Reviews Python, JavaScript, TypeScript, Go, Java, and more

## Architecture

```
                                    ┌─────────────┐
                                    │   GitHub    │
                                    │  Webhook    │
                                    └──────┬──────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                         Cloud Run                                 │
│  ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌─────────────┐   │
│  │ FastAPI │───▶│ Security │───▶│   RAG   │───▶│ Gemini LLM  │   │
│  │         │    │   Scan   │    │ Context │    │ (Vertex AI) │   │
│  └─────────┘    └──────────┘    └─────────┘    └─────────────┘   │
│       │              │               │               │            │
│       └──────────────┴───────────────┴───────────────┘            │
│                              │                                    │
└──────────────────────────────┼────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │ Datadog  │    │  GitHub  │    │  FAISS   │
        │ Metrics  │    │ PR API   │    │  Index   │
        └──────────┘    └──────────┘    └──────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- GCP Project with Vertex AI enabled
- GitHub Personal Access Token
- Datadog API keys

### Local Development

1. **Clone and setup environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Run locally**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

4. **Test the service**
   ```bash
   curl http://localhost:8000/health
   ```

## Configuration

All configuration is managed through environment variables. See `.env.example` for the full list.

### Required Variables

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT` | Google Cloud project ID |
| `GCP_LOCATION` | GCP region (default: `us-central1`) |
| `GITHUB_TOKEN` | GitHub PAT with repo access |
| `GITHUB_WEBHOOK_SECRET` | Secret for webhook signature verification |
| `DD_API_KEY` | Datadog API key |
| `DD_APP_KEY` | Datadog Application key |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `gemini-2.5-pro` | Vertex AI model to use |
| `MAX_TOKENS` | `128000` | Model context window size |
| `COST_PER_1K_INPUT` | `0.00125` | Cost per 1K input tokens (USD) |
| `COST_PER_1K_OUTPUT` | `0.005` | Cost per 1K output tokens (USD) |
| `TARGET_EXTENSIONS` | `py,js,ts,...` | File extensions to review |
| `DD_SERVICE` | `llm-code-review` | Datadog service name |
| `DD_ENV` | `production` | Datadog environment tag |
| `DD_SITE` | `datadoghq.com` | Datadog site (e.g., `us5.datadoghq.com`) |

## Deployment to GCP Cloud Run

### 1. Create secrets in Secret Manager

```bash
gcloud config set project YOUR_PROJECT_ID

echo -n "your-dd-api-key" | gcloud secrets create dd-api-key --data-file=-
echo -n "your-dd-app-key" | gcloud secrets create dd-app-key --data-file=-
echo -n "your-github-token" | gcloud secrets create github-token --data-file=-
echo -n "your-webhook-secret" | gcloud secrets create github-webhook-secret --data-file=-
```

### 2. Deploy

```bash
export GCP_PROJECT=your-project-id
export GCP_LOCATION=us-central1
./deploy.sh
```

### 3. Configure GitHub Webhook

1. Go to your repository → Settings → Webhooks → Add webhook
2. **Payload URL**: `https://YOUR-CLOUD-RUN-URL/webhook/github`
3. **Content type**: `application/json`
4. **Secret**: Same value as `github-webhook-secret`
5. **Events**: Select "Pull requests"

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/webhook/github` | POST | GitHub webhook receiver |
| `/review` | POST | Manual review endpoint (for testing) |

## Datadog Dashboard

Import the dashboard configuration from `datadog/dashboard.json` to visualize:

- **Overview KPIs**: Total reviews, avg cost, success rate, errors
- **Pipeline Performance**: LLM duration, RAG retrieval latency
- **Cost Analysis**: Per-review and cumulative costs by repository
- **Token Metrics**: Prompt vs completion tokens, context utilization
- **RAG Metrics**: Retrieval quality, chunks retrieved, indexing stats
- **Security & Errors**: Security events by type, error rates

## Project Structure

```
.
├── app/
│   ├── main.py           # FastAPI application & endpoints
│   ├── config.py         # Configuration management
│   ├── models.py         # Pydantic models
│   ├── metrics.py        # Datadog metrics (HTTP API)
│   ├── security.py       # Security scanning
│   ├── github_client.py  # GitHub API integration
│   └── rag/
│       ├── embeddings.py # Vertex AI embeddings
│       ├── code_parser.py# Code chunking
│       ├── indexer.py    # FAISS index building
│       └── retriever.py  # Context retrieval
├── datadog/
│   └── dashboard.json    # Datadog dashboard config
├── deploy.sh             # Cloud Run deployment script
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
└── .env.example          # Environment template
```


## License

MIT
