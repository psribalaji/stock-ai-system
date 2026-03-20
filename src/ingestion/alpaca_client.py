"""
alpaca_client.py — Alpaca Markets REST + WebSocket wrapper.
Used for: real-time price streaming, portfolio state, order execution.
Paper trading and live trading use the IDENTICAL API — only the base URL differs.
"""
from __future__ import annotations
from datetime import date, datetime, timedelta
from typing import Optional, AsyncGenerator, Callable
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.secrets import Secrets
from src.config import get_config


class AlpacaClient:
    """
    Wraps alpaca-py for both data and trading operations.
    Automatically switches between paper and live endpoints
    based on config.yaml trading.mode setting.
    """

    PAPER_BASE_URL = "https://paper-api.alpaca.markets"
    LIVE_BASE_URL  = "https://api.alpaca.markets"

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        self.config = get_config()
        self.api_key    = api_key    or Secrets.alpaca_api_key()
        self.secret_key = secret_key or Secrets.alpaca_secret_key()
        self.is_paper   = self.config.is_paper
        self.base_url   = self.PAPER_BASE_URL if self.is_paper else self.LIVE_BASE_URL

        self._trading_client = None
        self._data_client    = None
        self._stream_client  = None

        self._init_clients()

        mode = "PAPER" if self.is_paper else "LIVE"
        logger.info(f"Alpaca client initialized — mode: {mode}")

    def _init_clients(self) -> None:
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.live import StockDataStream

            self._trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.is_paper,
            )
            self._data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
            )
            self._stream_client = StockDataStream(
                api_key=self.api_key,
                secret_key=self.secret_key,
            )
        except ImportError:
            raise ImportError("alpaca-py not installed. Run: pip install alpaca-py")

    # ── PORTFOLIO & ACCOUNT ──────────────────────────────────────

    def get_account(self) -> dict:
        """Get account details: cash, portfolio value, buying power."""
        account = self._trading_client.get_account()
        return {
            "portfolio_value": float(account.portfolio_value),
            "cash":            float(account.cash),
            "buying_power":    float(account.buying_power),
            "equity":          float(account.equity),
            "daytrade_count":  account.daytrade_count,
            "trading_blocked": account.trading_blocked,
        }

    def get_positions(self) -> pd.DataFrame:
        """Get all open positions as a DataFrame."""
        positions = self._trading_client.get_all_positions()
        if not positions:
            return pd.DataFrame()

        records = []
        for p in positions:
            records.append({
                "ticker":          p.symbol,
                "qty":             float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price":   float(p.current_price),
                "market_value":    float(p.market_value),
                "unrealized_pl":   float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "side":            p.side,
            })
        return pd.DataFrame(records)

    # ── HISTORICAL DATA (recent bars only — use Polygon for backtesting) ──

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def get_recent_bars(
        self,
        ticker: str,
        days: int = 60,
    ) -> pd.DataFrame:
        """
        Fetch recent daily bars from Alpaca.
        Use this for live signal computation, NOT for backtesting.
        For backtesting, always use PolygonClient.
        """
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        start = datetime.now() - timedelta(days=days)
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start,
            feed=self.config.data.alpaca_feed,
        )
        bars = self._data_client.get_stock_bars(request)
        if not bars or ticker not in bars.data:
            return pd.DataFrame()

        records = [
            {
                "timestamp": bar.timestamp,
                "open":      bar.open,
                "high":      bar.high,
                "low":       bar.low,
                "close":     bar.close,
                "volume":    bar.volume,
                "vwap":      bar.vwap,
                "ticker":    ticker,
                "source":    "alpaca",
            }
            for bar in bars.data[ticker]
        ]
        return pd.DataFrame(records)

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Get the latest trade price for a ticker."""
        try:
            from alpaca.data.requests import StockLatestTradeRequest
            req = StockLatestTradeRequest(symbol_or_symbols=ticker)
            trade = self._data_client.get_stock_latest_trade(req)
            return float(trade[ticker].price) if ticker in trade else None
        except Exception as e:
            logger.warning(f"Could not fetch latest price for {ticker}: {e}")
            return None

    # ── ORDER EXECUTION ──────────────────────────────────────────

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
    def _submit_market_order_only(self, ticker: str, qty: float, side: str):
        """Submit just the market order with retry."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        order_data = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        return self._trading_client.submit_order(order_data)

    def submit_market_order(
        self,
        ticker: str,
        qty: float,
        side: str,
        stop_loss_price: Optional[float] = None,
    ) -> dict:
        """
        Submit a market order with optional stop loss.

        Args:
            ticker:          Stock symbol
            qty:             Number of shares (fractional supported)
            side:            "buy" or "sell"
            stop_loss_price: If provided, attaches a stop loss order after the buy fills

        Returns:
            Order details dict
        """
        # Market order is retried independently — stop loss failure won't re-trigger it
        order = self._submit_market_order_only(ticker, qty, side)
        logger.info(
            f"Order submitted: {side.upper()} {qty} {ticker} "
            f"| ID: {order.id} | Mode: {'PAPER' if self.is_paper else 'LIVE'}"
        )

        result = {
            "order_id":  str(order.id),
            "ticker":    ticker,
            "side":      side,
            "qty":       qty,
            "status":    str(order.status),
            "submitted_at": str(order.submitted_at),
            "paper":     self.is_paper,
        }

        # Attach stop loss separately — failure here won't retry the market order
        if stop_loss_price and side.lower() == "buy":
            try:
                self._attach_stop_loss(ticker, qty, stop_loss_price)
                result["stop_loss_price"] = stop_loss_price
            except Exception as exc:
                logger.warning(
                    f"Stop loss attachment failed for {ticker} "
                    f"(order {order.id} still submitted): {exc}"
                )

        return result

    def _attach_stop_loss(self, ticker: str, qty: float, stop_price: float) -> None:
        """Submit a stop loss order after a buy."""
        from alpaca.trading.requests import StopOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        stop_order = StopOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,  # Good till cancelled
            stop_price=round(stop_price, 2),
        )
        self._trading_client.submit_order(stop_order)
        logger.info(f"Stop loss attached: {ticker} @ ${stop_price:.2f}")

    def cancel_all_orders(self) -> None:
        """Cancel all open orders. Used by kill switch."""
        self._trading_client.cancel_orders()
        logger.warning("All open orders cancelled")

    def close_all_positions(self) -> None:
        """
        Emergency close all positions.
        ONLY called by kill switch — use with extreme caution.
        """
        logger.critical("KILL SWITCH ACTIVATED — closing all positions")
        self._trading_client.close_all_positions(cancel_orders=True)

    # ── REAL-TIME STREAMING ──────────────────────────────────────

    async def stream_bars(
        self,
        tickers: list[str],
        on_bar: Callable,
    ) -> None:
        """
        Stream real-time minute bars via WebSocket.

        Args:
            tickers: List of symbols to stream
            on_bar:  Async callback called with each new bar
        """
        logger.info(f"Starting real-time stream for: {tickers}")

        @self._stream_client.handler
        async def handle_bar(bar):
            await on_bar({
                "ticker":    bar.symbol,
                "timestamp": bar.timestamp,
                "open":      bar.open,
                "high":      bar.high,
                "low":       bar.low,
                "close":     bar.close,
                "volume":    bar.volume,
                "vwap":      bar.vwap,
            })

        self._stream_client.subscribe_bars(handle_bar, *tickers)
        await self._stream_client._run_forever()

    # ── VALIDATION ───────────────────────────────────────────────

    def validate_connection(self) -> bool:
        """Check Alpaca API connection is working."""
        try:
            account = self.get_account()
            mode = "PAPER" if self.is_paper else "LIVE"
            logger.info(
                f"Alpaca connection OK ({mode}) — "
                f"Portfolio: ${account['portfolio_value']:,.2f}"
            )
            return True
        except Exception as e:
            logger.error(f"Alpaca connection failed: {e}")
            return False
