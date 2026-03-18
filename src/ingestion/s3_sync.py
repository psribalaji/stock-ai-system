"""
s3_sync.py — S3 sync for Parquet data files.

Mirrors the local data/ directory to S3 for:
  - Disaster recovery (rebuild local store from S3)
  - Multi-machine access (e.g. Lightsail prod + local dev)
  - Audit trail persistence

Controlled by config.yaml:
  data.sync_to_s3: false   # Set true in Phase 3
  data.s3_bucket:  "stock-ai-system-data"
  data.s3_prefix:  "market-data"

When sync_to_s3=false, all upload/sync calls are no-ops.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_config


class S3Sync:
    """
    Syncs local Parquet files to/from S3.

    All methods are safe to call when sync_to_s3=false — they log and return
    immediately without touching S3.
    """

    def __init__(self, bucket: Optional[str] = None, prefix: Optional[str] = None):
        self.config = get_config()
        self.bucket = bucket or self.config.data.s3_bucket
        self.prefix = prefix or self.config.data.s3_prefix
        self._client = None

    # ── Public API ────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        """Returns True if sync_to_s3=true in config."""
        return self.config.data.sync_to_s3

    def validate_connection(self) -> bool:
        """
        Verify S3 bucket is reachable and accessible.

        Returns:
            True if bucket exists and credentials are valid.
        """
        try:
            client = self._get_client()
            client.head_bucket(Bucket=self.bucket)
            logger.info(f"S3 bucket '{self.bucket}' is reachable")
            return True
        except Exception as e:
            logger.error(f"S3 bucket '{self.bucket}' not reachable: {e}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def upload_file(self, local_path: Path, s3_key: Optional[str] = None) -> str:
        """
        Upload a single file to S3.

        Args:
            local_path: Path to local file.
            s3_key:     S3 object key. Defaults to prefix/filename.

        Returns:
            S3 URI of the uploaded file (s3://bucket/key), or "" if disabled.
        """
        if not self.is_enabled():
            logger.debug(f"S3 sync disabled — skipping upload of {local_path}")
            return ""

        local_path = Path(local_path)
        if not local_path.exists():
            logger.warning(f"File not found, skipping upload: {local_path}")
            return ""

        key = s3_key or f"{self.prefix}/{local_path.name}"
        client = self._get_client()
        client.upload_file(str(local_path), self.bucket, key)
        uri = f"s3://{self.bucket}/{key}"
        logger.info(f"Uploaded {local_path.name} → {uri}")
        return uri

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def download_file(self, s3_key: str, local_path: Path) -> bool:
        """
        Download a file from S3 to local path.

        Args:
            s3_key:     S3 object key.
            local_path: Destination path.

        Returns:
            True on success, False on failure.
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            client = self._get_client()
            client.download_file(self.bucket, s3_key, str(local_path))
            logger.info(f"Downloaded s3://{self.bucket}/{s3_key} → {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            return False

    def sync_directory(self, local_dir: Path, s3_prefix: Optional[str] = None) -> List[str]:
        """
        Sync all Parquet files in a local directory to S3.

        Walks local_dir recursively and uploads any .parquet file whose
        S3 counterpart is missing or older.

        Args:
            local_dir:  Local directory to sync (e.g. Path("./data")).
            s3_prefix:  S3 prefix to sync into. Defaults to config prefix.

        Returns:
            List of S3 URIs that were uploaded.
        """
        if not self.is_enabled():
            logger.debug("S3 sync disabled — skipping sync_directory")
            return []

        local_dir = Path(local_dir)
        if not local_dir.exists():
            logger.warning(f"Directory not found: {local_dir}")
            return []

        prefix = s3_prefix or self.prefix
        uploaded: List[str] = []

        parquet_files = list(local_dir.rglob("*.parquet"))
        if not parquet_files:
            logger.info(f"No Parquet files found in {local_dir}")
            return []

        logger.info(f"Syncing {len(parquet_files)} Parquet files to s3://{self.bucket}/{prefix}/")

        for file_path in parquet_files:
            # Preserve relative directory structure under prefix
            relative = file_path.relative_to(local_dir)
            s3_key = f"{prefix}/{relative.as_posix()}"

            if self._should_upload(file_path, s3_key):
                uri = self.upload_file(file_path, s3_key)
                if uri:
                    uploaded.append(uri)

        logger.info(f"Sync complete — {len(uploaded)} files uploaded")
        return uploaded

    def restore_from_s3(self, local_dir: Path, s3_prefix: Optional[str] = None) -> List[Path]:
        """
        Download all Parquet files from S3 to rebuild local data store.
        Used for disaster recovery or first-time setup on a new machine.

        Args:
            local_dir:  Local directory to restore into.
            s3_prefix:  S3 prefix to restore from. Defaults to config prefix.

        Returns:
            List of local paths that were restored.
        """
        local_dir = Path(local_dir)
        prefix = s3_prefix or self.prefix
        restored: List[Path] = []

        try:
            keys = self.list_remote(prefix)
        except Exception as e:
            logger.error(f"Could not list S3 objects: {e}")
            return []

        if not keys:
            logger.info(f"No files found at s3://{self.bucket}/{prefix}/")
            return []

        logger.info(f"Restoring {len(keys)} files from S3...")

        for key in keys:
            # Reconstruct local path by stripping the prefix
            relative = key[len(prefix):].lstrip("/")
            local_path = local_dir / relative

            if self.download_file(key, local_path):
                restored.append(local_path)

        logger.info(f"Restore complete — {len(restored)} files restored to {local_dir}")
        return restored

    def list_remote(self, prefix: Optional[str] = None) -> List[str]:
        """
        List all S3 object keys under a prefix.

        Args:
            prefix: S3 prefix to list. Defaults to config prefix.

        Returns:
            List of S3 object keys.
        """
        prefix = prefix or self.prefix
        client = self._get_client()
        paginator = client.get_paginator("list_objects_v2")

        keys: List[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])

        return keys

    # ── Helpers ───────────────────────────────────────────────────

    def _get_client(self):
        """Lazy-init boto3 S3 client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client("s3", region_name="us-east-1")
            except ImportError:
                raise ImportError(
                    "boto3 not installed. Run: pip install boto3"
                )
        return self._client

    def _should_upload(self, local_path: Path, s3_key: str) -> bool:
        """
        Check if a file needs to be uploaded (missing on S3 or local is newer).
        Returns True if upload is needed.
        """
        try:
            client = self._get_client()
            response = client.head_object(Bucket=self.bucket, Key=s3_key)
            s3_modified = response["LastModified"].timestamp()
            local_modified = local_path.stat().st_mtime
            return local_modified > s3_modified
        except Exception:
            # Object doesn't exist on S3 — upload it
            return True
