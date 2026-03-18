"""
cloudwatch_reporter.py — Push trading metrics to AWS CloudWatch.

Metrics are published under the namespace "StockAI" and can be
used to set alarms in the AWS Console (e.g. page on drawdown > 10%).

Dimensions used:
  Environment: paper | live

Metrics published:
  PortfolioValue      — total portfolio value in USD
  CashBalance         — uninvested cash in USD
  BuyingPower         — available buying power in USD
  DailyPnL            — day's profit/loss in USD (positive = profit)
  DrawdownPct         — current drawdown as a percentage (0–100)
  OpenPositions       — number of open positions (count)
  SignalsGenerated    — number of signals generated in last cycle (count)
  SignalsApproved     — number of approved (tradeable) signals (count)
  CircuitBreakerTripped — 1 if tripped, 0 if not (count)
  DriftWarnings       — count of active WARNING-level drift alerts
  DriftCriticals      — count of active CRITICAL-level drift alerts
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from src.config import get_config


class CloudWatchReporter:
    """
    Publishes StockAI metrics to AWS CloudWatch.

    Metrics are batched in groups of 20 (CloudWatch API limit per call).
    All methods silently log and continue on failure — never crash the
    trading system due to a metrics reporting error.
    """

    NAMESPACE = "StockAI"
    MAX_METRICS_PER_CALL = 20

    def __init__(self, region: str = "us-east-1", client=None):
        """
        Args:
            region: AWS region for CloudWatch.
            client: Pre-built boto3 CloudWatch client (injected for testing).
        """
        self.config = get_config()
        self.region = region
        self._client = client
        self._environment = "live" if self.config.is_live else "paper"

    # ── Public API ────────────────────────────────────────────────

    def put_metric(
        self,
        name: str,
        value: float,
        unit: str = "None",
        dimensions: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Publish a single metric to CloudWatch.

        Args:
            name:       Metric name (e.g. "PortfolioValue").
            value:      Numeric value.
            unit:       CloudWatch unit string (None|Count|Percent|Seconds|Bytes|...).
            dimensions: Extra dimensions dict. Environment is always added.

        Returns:
            True on success, False on failure.
        """
        dims = {"Environment": self._environment}
        if dimensions:
            dims.update(dimensions)

        metric_data = self._build_metric_datum(name, value, unit, dims)
        return self._put_metric_data([metric_data])

    def push_portfolio_metrics(self, account: Dict[str, Any]) -> bool:
        """
        Push account/portfolio metrics from AlpacaClient.get_account().

        Args:
            account: Dict from AlpacaClient.get_account() with keys:
                     portfolio_value, cash, buying_power, equity.

        Returns:
            True if all metrics were published successfully.
        """
        dims = {"Environment": self._environment}
        metrics = []

        field_map = {
            "portfolio_value": ("PortfolioValue", "None"),
            "cash":            ("CashBalance", "None"),
            "buying_power":    ("BuyingPower", "None"),
            "equity":          ("Equity", "None"),
        }

        for field, (metric_name, unit) in field_map.items():
            value = account.get(field)
            if value is not None:
                metrics.append(self._build_metric_datum(metric_name, float(value), unit, dims))

        if not metrics:
            logger.warning("push_portfolio_metrics: no usable fields in account dict")
            return False

        return self._put_metric_data(metrics)

    def push_signal_metrics(self, signals_generated: int, signals_approved: int) -> bool:
        """
        Push signal pipeline counts.

        Args:
            signals_generated: Total signals from signal detector this cycle.
            signals_approved:  Signals that passed confidence + risk filters.

        Returns:
            True on success.
        """
        dims = {"Environment": self._environment}
        metrics = [
            self._build_metric_datum("SignalsGenerated", float(signals_generated), "Count", dims),
            self._build_metric_datum("SignalsApproved",  float(signals_approved),  "Count", dims),
        ]
        return self._put_metric_data(metrics)

    def push_risk_metrics(
        self,
        drawdown_pct: float,
        open_positions: int,
        circuit_breaker_tripped: bool,
    ) -> bool:
        """
        Push risk/drawdown metrics.

        Args:
            drawdown_pct:            Current drawdown as a fraction (0.0–1.0).
                                     Stored in CloudWatch as 0–100 Percent.
            open_positions:          Number of open positions.
            circuit_breaker_tripped: True if daily loss circuit breaker is active.

        Returns:
            True on success.
        """
        dims = {"Environment": self._environment}
        metrics = [
            self._build_metric_datum(
                "DrawdownPct",
                round(drawdown_pct * 100, 4),
                "Percent",
                dims,
            ),
            self._build_metric_datum("OpenPositions", float(open_positions), "Count", dims),
            self._build_metric_datum(
                "CircuitBreakerTripped",
                1.0 if circuit_breaker_tripped else 0.0,
                "Count",
                dims,
            ),
        ]
        return self._put_metric_data(metrics)

    def push_drift_alerts(self, alerts: List[Any]) -> bool:
        """
        Push counts of active drift alerts by severity.

        Args:
            alerts: List of DriftAlert objects from ModelMonitor.

        Returns:
            True on success.
        """
        dims = {"Environment": self._environment}
        warnings  = sum(1 for a in alerts if getattr(a, "severity", "") == "WARNING")
        criticals = sum(1 for a in alerts if getattr(a, "severity", "") == "CRITICAL")

        metrics = [
            self._build_metric_datum("DriftWarnings",  float(warnings),  "Count", dims),
            self._build_metric_datum("DriftCriticals", float(criticals), "Count", dims),
        ]
        return self._put_metric_data(metrics)

    def push_daily_pnl(self, pnl_usd: float) -> bool:
        """
        Push today's realised P&L.

        Args:
            pnl_usd: Dollar P&L for the day (positive = profit).

        Returns:
            True on success.
        """
        dims = {"Environment": self._environment}
        metric = self._build_metric_datum("DailyPnL", pnl_usd, "None", dims)
        return self._put_metric_data([metric])

    # ── Helpers ───────────────────────────────────────────────────

    def _build_metric_datum(
        self,
        name: str,
        value: float,
        unit: str,
        dimensions: Dict[str, str],
    ) -> dict:
        """Build a CloudWatch MetricDatum dict."""
        return {
            "MetricName": name,
            "Value": value,
            "Unit": unit,
            "Timestamp": datetime.now(timezone.utc),
            "Dimensions": [
                {"Name": k, "Value": v} for k, v in dimensions.items()
            ],
        }

    def _put_metric_data(self, metrics: List[dict]) -> bool:
        """
        Send metrics to CloudWatch in batches of MAX_METRICS_PER_CALL.
        Never raises — logs errors and returns False on failure.
        """
        if not metrics:
            return True

        try:
            client = self._get_client()
            # Batch into groups of 20 (API limit)
            for i in range(0, len(metrics), self.MAX_METRICS_PER_CALL):
                batch = metrics[i : i + self.MAX_METRICS_PER_CALL]
                client.put_metric_data(
                    Namespace=self.NAMESPACE,
                    MetricData=batch,
                )
            logger.debug(f"CloudWatch: pushed {len(metrics)} metrics to {self.NAMESPACE}")
            return True
        except Exception as e:
            logger.error(f"CloudWatch put_metric_data failed: {e}")
            return False

    def _get_client(self):
        """Lazy-init boto3 CloudWatch client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client("cloudwatch", region_name=self.region)
            except ImportError:
                raise ImportError("boto3 not installed. Run: pip install boto3")
        return self._client
