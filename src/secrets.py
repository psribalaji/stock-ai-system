"""
secrets.py — API key management.
Production: fetches from AWS Secrets Manager.
Local dev:  falls back to .env file.
Never hardcode keys. Never commit .env to git.
"""
from __future__ import annotations
import os
import json
from functools import lru_cache
from pathlib import Path
from loguru import logger


# ── Local dev fallback (.env file) ───────────────────────────────

def _load_dotenv() -> None:
    """Load .env file for local development. Silently skips if missing."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"'))
    logger.debug("Loaded .env for local development")


_load_dotenv()


# ── AWS Secrets Manager ──────────────────────────────────────────

def _fetch_from_aws(secret_name: str) -> str:
    """Fetch a secret from AWS Secrets Manager."""
    try:
        import boto3
        client = boto3.client("secretsmanager", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        response = client.get_secret_value(SecretId=secret_name)
        value = response["SecretString"]
        # Handle JSON secrets (key-value pairs stored as JSON string)
        try:
            parsed = json.loads(value)
            return list(parsed.values())[0] if len(parsed) == 1 else value
        except (json.JSONDecodeError, IndexError):
            return value
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch secret '{secret_name}' from AWS: {e}\n"
            f"Tip: For local dev, add to your .env file instead."
        ) from e


@lru_cache(maxsize=32)
def get_secret(env_var: str, aws_secret_name: str | None = None) -> str:
    """
    Get a secret value. Priority order:
    1. Environment variable (set by .env locally or by system in prod)
    2. AWS Secrets Manager (if aws_secret_name provided and env var missing)

    Usage:
        api_key = get_secret("ALPACA_API_KEY", "stock-ai/alpaca-api-key")
    """
    # Try env var first (works both locally via .env and in prod via systemd env)
    value = os.environ.get(env_var)
    if value:
        logger.debug(f"Secret '{env_var}' loaded from environment")
        return value

    # Fall back to AWS Secrets Manager
    if aws_secret_name:
        logger.info(f"Secret '{env_var}' not in env — fetching from AWS Secrets Manager")
        return _fetch_from_aws(aws_secret_name)

    raise ValueError(
        f"Secret '{env_var}' not found in environment variables.\n"
        f"Add it to your .env file for local dev:\n"
        f"  {env_var}=your_value_here"
    )


# ── Convenience accessors ─────────────────────────────────────────

class Secrets:
    """Typed accessors for all API keys used in the system."""

    @staticmethod
    def alpaca_api_key() -> str:
        return get_secret("ALPACA_API_KEY", "stock-ai/alpaca-api-key")

    @staticmethod
    def alpaca_secret_key() -> str:
        return get_secret("ALPACA_SECRET_KEY", "stock-ai/alpaca-secret-key")

    @staticmethod
    def polygon_api_key() -> str:
        return get_secret("POLYGON_API_KEY", "stock-ai/polygon-api-key")

    @staticmethod
    def finnhub_api_key() -> str:
        return get_secret("FINNHUB_API_KEY", "stock-ai/finnhub-api-key")

    @staticmethod
    def anthropic_api_key() -> str:
        return get_secret("ANTHROPIC_API_KEY", "stock-ai/anthropic-api-key")

    @staticmethod
    def aws_role_arn() -> str:
        """Optional — only needed when assuming a role locally."""
        return os.environ.get("AWS_ROLE_ARN", "")
