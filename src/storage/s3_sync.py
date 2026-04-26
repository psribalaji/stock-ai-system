"""
storage/s3_sync.py — Incremental Parquet → S3 sync for production hosting.

Only runs when config.data.sync_to_s3 = true.
Uses boto3 with ETag-based change detection — only uploads new or modified files.
Preserves the local directory structure under the configured S3 prefix.

Usage:
    sync = S3Sync()
    result = sync.sync()           # sync ./data/ → s3://bucket/prefix/
    result = sync.sync_file(path)  # sync a single file

AWS credentials: standard boto3 chain (env vars → ~/.aws → IAM role on Lightsail).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from loguru import logger

from src.config import get_config


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class SyncResult:
    """Summary of a completed sync operation."""
    started_at: datetime
    finished_at: datetime
    files_uploaded: int
    files_skipped: int
    files_failed: int
    bytes_uploaded: int
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.files_failed == 0

    def summary(self) -> str:
        elapsed = (self.finished_at - self.started_at).total_seconds()
        return (
            f"S3 sync complete in {elapsed:.1f}s — "
            f"{self.files_uploaded} uploaded, "
            f"{self.files_skipped} skipped, "
            f"{self.files_failed} failed "
            f"({self.bytes_uploaded / 1024:.1f} KB)"
        )


# ── Sync ──────────────────────────────────────────────────────────────────────

class S3Sync:
    """
    Incrementally syncs local Parquet data files to S3.

    Only uploads files that are new or have changed (MD5 check against S3 ETag).
    All sync operations are no-ops when config.data.sync_to_s3 is False.

    Args:
        bucket:     S3 bucket name (defaults to config.data.s3_bucket)
        prefix:     S3 key prefix (defaults to config.data.s3_prefix)
        local_path: Local data directory (defaults to config.data.storage_path)
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
        local_path: Optional[str] = None,
    ) -> None:
        self.config     = get_config()
        self.bucket     = bucket     or self.config.data.s3_bucket
        self.prefix     = prefix     or self.config.data.s3_prefix
        self.local_path = Path(local_path or self.config.data.storage_path)
        self._s3        = None   # lazy-init boto3 client

        # Load .env explicitly so AWS_REGION is available
        import os
        from src.secrets import _load_dotenv
        _load_dotenv()
        self._region = os.environ.get("AWS_REGION", "us-east-1")

        logger.info(
            f"[S3Sync] Initialised — "
            f"local: {self.local_path} → s3://{self.bucket}/{self.prefix}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def validate_connection(self) -> bool:
        """Return True if the S3 bucket is reachable and accessible."""
        try:
            self._get_s3().head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False

    def sync(self) -> SyncResult:
        """
        Sync all Parquet files under local_path to S3.
        Skips entirely if config.data.sync_to_s3 is False.

        Returns:
            SyncResult with counts and any errors.
        """
        started = datetime.now(timezone.utc)

        if not self.config.data.sync_to_s3:
            logger.debug("[S3Sync] sync_to_s3=false — skipping")
            return SyncResult(
                started_at=started,
                finished_at=datetime.now(timezone.utc),
                files_uploaded=0,
                files_skipped=0,
                files_failed=0,
                bytes_uploaded=0,
            )

        parquet_files = sorted(self.local_path.rglob("*.parquet"))
        if not parquet_files:
            logger.info("[S3Sync] No Parquet files found — nothing to sync")
            return SyncResult(
                started_at=started,
                finished_at=datetime.now(timezone.utc),
                files_uploaded=0,
                files_skipped=0,
                files_failed=0,
                bytes_uploaded=0,
            )

        logger.info(f"[S3Sync] Syncing {len(parquet_files)} file(s) to S3")
        uploaded = skipped = failed = 0
        total_bytes = 0
        errors: List[str] = []

        for fpath in parquet_files:
            try:
                result = self.sync_file(fpath)
                if result == "uploaded":
                    uploaded   += 1
                    total_bytes += fpath.stat().st_size
                elif result == "skipped":
                    skipped += 1
                else:
                    failed += 1
            except Exception as exc:
                failed += 1
                msg = f"{fpath.name}: {exc}"
                errors.append(msg)
                logger.error(f"[S3Sync] Upload failed — {msg}")

        result = SyncResult(
            started_at=started,
            finished_at=datetime.now(timezone.utc),
            files_uploaded=uploaded,
            files_skipped=skipped,
            files_failed=failed,
            bytes_uploaded=total_bytes,
            errors=errors,
        )
        logger.info(f"[S3Sync] {result.summary()}")
        return result

    def sync_file(self, local_file: Path) -> str:
        """
        Sync a single file to S3.

        Returns:
            "uploaded"  — file was new or changed, uploaded successfully
            "skipped"   — file matches S3 ETag, no upload needed
            "failed"    — upload raised an exception (caller should catch)
        """
        s3_key = self._s3_key(local_file)

        # Check if already up-to-date
        if self._matches_s3(local_file, s3_key):
            logger.debug(f"[S3Sync] Skipped (unchanged): {s3_key}")
            return "skipped"

        s3 = self._get_s3()
        s3.upload_file(
            Filename=str(local_file),
            Bucket=self.bucket,
            Key=s3_key,
            ExtraArgs={"ContentType": "application/octet-stream"},
        )
        size_kb = local_file.stat().st_size / 1024
        logger.info(f"[S3Sync] Uploaded {s3_key} ({size_kb:.1f} KB)")
        return "uploaded"

    def download_all(self, dest_path: Optional[Path] = None) -> int:
        """
        Download all files from S3 prefix to dest_path (or local_path).
        Used to restore data on a fresh Lightsail instance.

        Returns:
            Number of files downloaded.
        """
        dest = dest_path or self.local_path
        s3   = self._get_s3()

        paginator = s3.get_paginator("list_objects_v2")
        pages     = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix + "/")

        count = 0
        for page in pages:
            for obj in page.get("Contents", []):
                key       = obj["Key"]
                rel_path  = key[len(self.prefix) + 1:]   # strip prefix/
                local_out = dest / rel_path
                local_out.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(self.bucket, key, str(local_out))
                logger.debug(f"[S3Sync] Downloaded {key} → {local_out}")
                count += 1

        logger.info(f"[S3Sync] Downloaded {count} file(s) from S3")
        return count

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _s3_key(self, local_file: Path) -> str:
        """Build S3 key preserving relative directory structure."""
        try:
            rel = local_file.relative_to(self.local_path)
        except ValueError:
            rel = Path(local_file.name)
        # Always use forward slashes in S3 keys
        return f"{self.prefix}/{rel.as_posix()}"

    def _matches_s3(self, local_file: Path, s3_key: str) -> bool:
        """
        Return True if the local file's MD5 matches the S3 object ETag.
        S3 ETag for non-multipart uploads is the MD5 hex digest.
        """
        try:
            s3  = self._get_s3()
            obj = s3.head_object(Bucket=self.bucket, Key=s3_key)
            s3_etag   = obj["ETag"].strip('"')
            local_md5 = self._md5(local_file)
            return s3_etag == local_md5
        except Exception:
            # Object doesn't exist or head failed → treat as new
            return False

    @staticmethod
    def _md5(path: Path) -> str:
        """Compute MD5 hex digest of a file."""
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _get_s3(self):
        """Lazy-init boto3 S3 client."""
        if self._s3 is None:
            try:
                import boto3
                self._s3 = boto3.client("s3", region_name=self._region)
            except ImportError:
                raise ImportError(
                    "boto3 not installed. Run: pip install boto3"
                )
        return self._s3
