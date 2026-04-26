"""
src/notifications/notifier.py — Notification service with activity log + Telegram.

Usage:
    notifier = Notifier()
    notifier.notify("NVDA BUY placed @ $139.50", level="trade", ticker="NVDA")
    notifier.notify("MRVL discovered (3.2x spike)", level="discovery")
    notifier.notify("Kill switch activated!", level="critical")

Activity log is stored in Parquet for the dashboard feed.
Telegram messages are sent via bot API (optional, needs config).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.config import get_config


# Notification levels with emoji
_ICONS = {
    "trade":     "🟢",
    "sell":      "🔴",
    "stop":      "🛑",
    "tp":        "🎯",
    "discovery": "🔔",
    "approval":  "⏳",
    "warning":   "⚠️",
    "critical":  "☠️",
    "info":      "📊",
}


class Notifier:
    """
    Sends notifications to activity log (Parquet) and optionally Telegram.
    """

    def __init__(self) -> None:
        self.config = get_config()
        self._log_path = Path(self.config.data.storage_path) / "notifications"
        self._log_path.mkdir(parents=True, exist_ok=True)
        self._telegram_enabled = bool(
            getattr(self.config, "notifications", None)
            and getattr(self.config.notifications, "telegram_enabled", False)
        )

    def notify(
        self,
        message: str,
        level: str = "info",
        ticker: str = "",
        data: Optional[dict] = None,
    ) -> None:
        """
        Send a notification to all enabled channels.

        Args:
            message: Human-readable notification text
            level:   One of: trade, sell, stop, tp, discovery, approval, warning, critical, info
            ticker:  Optional ticker symbol
            data:    Optional extra data dict
        """
        icon = _ICONS.get(level, "📌")
        timestamp = datetime.now(timezone.utc)

        # Always log to activity feed
        self._log_to_parquet(timestamp, level, icon, message, ticker, data)

        # Telegram (if enabled)
        if self._telegram_enabled:
            self._send_telegram(f"{icon} {message}")

        logger.info(f"[Notifier] {icon} {message}")

    def get_feed(self, limit: int = 50) -> pd.DataFrame:
        """Load recent activity feed entries."""
        files = sorted(self._log_path.glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        # Load last 2 files (current + previous month)
        dfs = [pd.read_parquet(f) for f in files[-2:]]
        df = pd.concat(dfs).sort_values("timestamp", ascending=False)
        return df.head(limit).reset_index(drop=True)

    # ── Activity log ──────────────────────────────────────────────

    def _log_to_parquet(
        self, timestamp, level, icon, message, ticker, data
    ) -> None:
        month = timestamp.strftime("%Y-%m")
        path = self._log_path / f"{month}.parquet"

        row = pd.DataFrame([{
            "timestamp": timestamp,
            "level": level,
            "icon": icon,
            "message": message,
            "ticker": ticker,
            "data": json.dumps(data) if data else "",
        }])

        if path.exists():
            existing = pd.read_parquet(path)
            row = pd.concat([existing, row])

        row.to_parquet(path, engine="pyarrow", compression="snappy", index=False)

        # S3 sync
        try:
            from src.ingestion.storage import ParquetStore
            store = ParquetStore(self.config.data.storage_path)
            store._s3_upload(path)
        except Exception:
            pass

    # ── Telegram ──────────────────────────────────────────────────

    def _send_telegram(self, text: str) -> None:
        try:
            import os
            import httpx
            token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
            if not token or not chat_id:
                logger.debug("[Telegram] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in env")
                return
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            resp = httpx.post(url, json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
            }, timeout=10)
            if resp.status_code != 200:
                logger.warning(f"[Telegram] Send failed: {resp.text}")
        except Exception as e:
            logger.warning(f"[Telegram] Send failed: {e}")


# ── Convenience functions ─────────────────────────────────────────

_notifier: Optional[Notifier] = None


def get_notifier() -> Notifier:
    global _notifier
    if _notifier is None:
        _notifier = Notifier()
    return _notifier


def notify(message: str, level: str = "info", ticker: str = "", data: Optional[dict] = None) -> None:
    """Shortcut: notify("message", level="trade", ticker="NVDA")"""
    get_notifier().notify(message, level, ticker, data)
