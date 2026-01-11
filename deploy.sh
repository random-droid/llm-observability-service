#!/bin/bash
# Deploy LLM Code Review Service to Cloud Run

set -e

# Configuration - update these values
# Load .env file if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

PROJECT_ID="${GCP_PROJECT:-steel-earth-470201-g1}"
REGION="${GCP_LOCATION:-us-central1}"
SERVICE_NAME="llm-code-review"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "=== LLM Code Review Service - Cloud Run Deployment ==="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo ""

# Check if gcloud is authenticated
if ! gcloud auth print-identity-token &> /dev/null; then
    echo "Error: Not authenticated with gcloud. Run 'gcloud auth login'"
    exit 1
fi

# Set project
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    aiplatform.googleapis.com \
    secretmanager.googleapis.com

# Build and push image
echo ""
echo "Building and pushing Docker image..."
gcloud builds submit --tag ${IMAGE_NAME} .

# Deploy to Cloud Run
echo ""
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --timeout 300 \
    --concurrency 10 \
    --min-instances 0 \
    --max-instances 5 \
    --set-env-vars "GCP_PROJECT=${PROJECT_ID}" \
    --set-env-vars "GCP_LOCATION=${REGION}" \
    --set-env-vars "DD_ENV=production" \
    --set-env-vars "DD_SERVICE=${SERVICE_NAME}" \
    --set-env-vars "DD_SITE=us5.datadoghq.com" \
    --set-secrets "DD_API_KEY=dd-api-key:latest" \
    --set-secrets "DD_APP_KEY=dd-app-key:latest" \
    --set-secrets "GITHUB_TOKEN=github-token:latest" \
    --set-secrets "GITHUB_WEBHOOK_SECRET=github-webhook-secret:latest" \
    --set-env-vars "ENABLE_BQ_METRICS=true"

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')

echo ""
echo "=== Deployment Complete ==="
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Next steps:"
echo "1. Create secrets in Secret Manager:"
echo "   gcloud secrets create dd-api-key --data-file=- <<< 'your-datadog-api-key'"
echo "   gcloud secrets create dd-app-key --data-file=- <<< 'your-datadog-app-key'"
echo "   gcloud secrets create github-token --data-file=- <<< 'your-github-token'"
echo "   gcloud secrets create github-webhook-secret --data-file=- <<< 'your-webhook-secret'"
echo ""
echo "2. Configure GitHub webhook:"
echo "   - URL: ${SERVICE_URL}/webhook/github"
echo "   - Content type: application/json"
echo "   - Secret: your-webhook-secret"
echo "   - Events: Pull requests"
echo ""
echo "3. Test the service:"
echo "   curl ${SERVICE_URL}/health"
