import os
from functools import lru_cache
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Settings:
    # GCP
    GCP_PROJECT: str = os.getenv("GCP_PROJECT", "")
    GCP_LOCATION: str = os.getenv("GCP_LOCATION", "us-central1")

    # GitHub
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    GITHUB_WEBHOOK_SECRET: str = os.getenv("GITHUB_WEBHOOK_SECRET", "")

    # Datadog
    DD_API_KEY: str = os.getenv("DD_API_KEY", "")
    DD_APP_KEY: str = os.getenv("DD_APP_KEY", "")
    DD_SERVICE: str = os.getenv("DD_SERVICE", "llm-code-review")
    DD_ENV: str = os.getenv("DD_ENV", "production")

    # Model
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-2.5-pro")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "128000"))

    # Costs (per 1K tokens) - Gemini 2.5 Pro pricing
    COST_PER_1K_INPUT: float = float(os.getenv("COST_PER_1K_INPUT", "0.00125"))
    COST_PER_1K_OUTPUT: float = float(os.getenv("COST_PER_1K_OUTPUT", "0.005"))

    # File extensions to review
    TARGET_EXTENSIONS: list = os.getenv(
        "TARGET_EXTENSIONS", "py,js,ts,tsx,jsx,java,go,rs,rb,php,c,cpp,h"
    ).split(",")

    # BigQuery
    BQ_DATASET: str = os.getenv("BQ_DATASET", "llm_observability")
    BQ_TABLE: str = os.getenv("BQ_TABLE", "metrics")
    ENABLE_BQ_METRICS: bool = os.getenv("ENABLE_BQ_METRICS", "false").lower() == "true"

@lru_cache()
def get_settings() -> Settings:
    return Settings()