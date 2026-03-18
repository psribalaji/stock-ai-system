"""
monitoring/cloudwatch_alerts.py — AWS CloudWatch custom metrics and alarms.

Publishes per-strategy and portfolio-level metrics to CloudWatch so that
on-call alerts fire before losses become severe.

Metrics published (namespace: "StockAI/Trading"):
  - WinRate          — per strategy, fraction
  - DailyPnLPct      — portfolio daily P&L, fraction
  - MaxDrawdown      — current peak-to-trough, fraction
  - OpenPositions    — count of open positions
  - SignalsToday     — count of approved signals today
  - SharpeRatio      — per strategy, rolling

Alarms created (one-time setup via setup_alarms()):
  - DailyLoss        — daily P&L < -2%   → SNS alert
  - DrawdownCritical — drawdown  > 15%   → SNS alert
  - WinRateLow       — win rate  < 0.40  → SNS alert

Falls back silently if boto3 is unavailable or AWS credentials are not configured.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from loguru import logger

from src.config import get_config


# ── Metric model ──────────────────────────────────────────────────────────────

NAMESPACE = "StockAI/Trading"


class CloudWatchAlerts:
    """
    Publishes custom CloudWatch metrics and manages alarms.

    All methods are safe to call without AWS credentials — they will log a
    warning and return gracefully instead of raising exceptions.

    Args:
        namespace:   CloudWatch metric namespace (default: "StockAI/Trading")
        region:      AWS region (default: "us-east-1")
        sns_arn:     SNS topic ARN for alarm notifications (optional)
        cloudwatch:  Pre-built boto3 CloudWatch client (for testing)
    """

    def __init__(
        self,
        namespace: str = NAMESPACE,
        region: str = "us-east-1",
        sns_arn: Optional[str] = None,
        cloudwatch=None,
    ) -> None:
        self.config    = get_config()
        self.namespace = namespace
        self.region    = region
        self.sns_arn   = sns_arn
        self._cw       = cloudwatch   # lazy init via _get_cw()

        logger.info(
            f"[CloudWatch] Initialised — namespace: {namespace}, region: {region}"
        )

    # ── Metrics ───────────────────────────────────────────────────────────────

    def publish_portfolio_metrics(
        self,
        daily_pnl_pct: float,
        max_drawdown: float,
        open_positions: int,
        signals_today: int,
    ) -> bool:
        """
        Publish portfolio-level metrics to CloudWatch.

        Args:
            daily_pnl_pct:  Today's P&L as a fraction (e.g. -0.015 = -1.5%)
            max_drawdown:   Current max drawdown as a positive fraction
            open_positions: Number of currently open positions
            signals_today:  Number of approved signals generated today

        Returns:
            True if published successfully, False on failure.
        """
        metrics = [
            self._metric("DailyPnLPct",   daily_pnl_pct,  "None"),
            self._metric("MaxDrawdown",    max_drawdown,   "None"),
            self._metric("OpenPositions",  open_positions, "Count"),
            self._metric("SignalsToday",   signals_today,  "Count"),
        ]
        return self._put_metrics(metrics)

    def publish_strategy_metrics(
        self,
        strategy: str,
        win_rate: float,
        sharpe: float,
    ) -> bool:
        """
        Publish per-strategy performance metrics.

        Args:
            strategy: Strategy name (e.g. "momentum")
            win_rate: Rolling win rate fraction
            sharpe:   Rolling Sharpe ratio

        Returns:
            True if published successfully, False on failure.
        """
        dims = [{"Name": "Strategy", "Value": strategy}]
        metrics = [
            self._metric("WinRate",     win_rate, "None", dims),
            self._metric("SharpeRatio", sharpe,   "None", dims),
        ]
        return self._put_metrics(metrics)

    def publish_all(
        self,
        daily_pnl_pct: float,
        max_drawdown: float,
        open_positions: int,
        signals_today: int,
        strategy_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> bool:
        """
        Convenience: publish all metrics in a single call.

        Args:
            daily_pnl_pct:     Portfolio daily P&L
            max_drawdown:      Current drawdown
            open_positions:    Open position count
            signals_today:     Approved signals today
            strategy_metrics:  Dict of {strategy: {"win_rate": float, "sharpe": float}}

        Returns:
            True if all publishes succeeded, False if any failed.
        """
        ok = self.publish_portfolio_metrics(
            daily_pnl_pct, max_drawdown, open_positions, signals_today
        )
        if strategy_metrics:
            for strategy, vals in strategy_metrics.items():
                ok = self.publish_strategy_metrics(
                    strategy,
                    win_rate=vals.get("win_rate", 0.0),
                    sharpe=vals.get("sharpe", 0.0),
                ) and ok
        return ok

    # ── Alarm setup ───────────────────────────────────────────────────────────

    def setup_alarms(self) -> List[str]:
        """
        Create (or update) standard CloudWatch alarms.
        Safe to call repeatedly — CloudWatch is idempotent on alarm names.

        Returns:
            List of alarm names that were created/updated.
        """
        cfg    = self.config.risk
        alarms = [
            {
                "AlarmName":          "StockAI-DailyLoss",
                "AlarmDescription":   "Daily P&L below circuit breaker threshold",
                "MetricName":         "DailyPnLPct",
                "Namespace":          self.namespace,
                "Statistic":          "Minimum",
                "Period":             3600,       # 1 hour
                "EvaluationPeriods":  1,
                "Threshold":          -cfg.daily_loss_limit,
                "ComparisonOperator": "LessThanThreshold",
                "TreatMissingData":   "notBreaching",
            },
            {
                "AlarmName":          "StockAI-DrawdownCritical",
                "AlarmDescription":   "Drawdown exceeds kill switch threshold",
                "MetricName":         "MaxDrawdown",
                "Namespace":          self.namespace,
                "Statistic":          "Maximum",
                "Period":             3600,
                "EvaluationPeriods":  1,
                "Threshold":          cfg.max_drawdown_pct,
                "ComparisonOperator": "GreaterThanThreshold",
                "TreatMissingData":   "notBreaching",
            },
            {
                "AlarmName":          "StockAI-WinRateLow",
                "AlarmDescription":   "Portfolio win rate below minimum threshold",
                "MetricName":         "WinRate",
                "Namespace":          self.namespace,
                "Statistic":          "Minimum",
                "Period":             86400,      # 1 day
                "EvaluationPeriods":  1,
                "Threshold":          cfg.min_confidence - 0.20,
                "ComparisonOperator": "LessThanThreshold",
                "TreatMissingData":   "notBreaching",
            },
        ]

        # Attach SNS action if provided
        if self.sns_arn:
            for alarm in alarms:
                alarm["AlarmActions"]            = [self.sns_arn]
                alarm["OKActions"]               = [self.sns_arn]
                alarm["InsufficientDataActions"] = []

        created = []
        try:
            cw = self._get_cw()
            for alarm in alarms:
                cw.put_metric_alarm(**alarm)
                created.append(alarm["AlarmName"])
                logger.info(f"[CloudWatch] Alarm configured: {alarm['AlarmName']}")
        except Exception as exc:
            logger.warning(f"[CloudWatch] setup_alarms failed: {exc}")

        return created

    def get_alarm_states(self) -> Dict[str, str]:
        """
        Return current state of all StockAI alarms.

        Returns:
            Dict mapping alarm name → state ("OK" | "ALARM" | "INSUFFICIENT_DATA")
        """
        try:
            cw       = self._get_cw()
            response = cw.describe_alarms(AlarmNamePrefix="StockAI-")
            return {
                a["AlarmName"]: a["StateValue"]
                for a in response.get("MetricAlarms", [])
            }
        except Exception as exc:
            logger.warning(f"[CloudWatch] get_alarm_states failed: {exc}")
            return {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _put_metrics(self, metric_data: list) -> bool:
        """Send metric data to CloudWatch. Returns True on success."""
        try:
            cw = self._get_cw()
            cw.put_metric_data(Namespace=self.namespace, MetricData=metric_data)
            logger.debug(
                f"[CloudWatch] Published {len(metric_data)} metric(s) "
                f"to {self.namespace}"
            )
            return True
        except Exception as exc:
            logger.warning(f"[CloudWatch] put_metric_data failed: {exc}")
            return False

    @staticmethod
    def _metric(
        name: str,
        value: float,
        unit: str,
        dimensions: Optional[list] = None,
    ) -> dict:
        """Build a CloudWatch MetricDatum dict."""
        datum: dict = {
            "MetricName": name,
            "Value":      float(value),
            "Unit":       unit,
            "Timestamp":  datetime.now(timezone.utc),
        }
        if dimensions:
            datum["Dimensions"] = dimensions
        return datum

    def _get_cw(self):
        """Lazy-init boto3 CloudWatch client."""
        if self._cw is None:
            try:
                import boto3
                self._cw = boto3.client("cloudwatch", region_name=self.region)
            except ImportError:
                raise ImportError(
                    "boto3 not installed. Run: pip install boto3"
                )
        return self._cw
