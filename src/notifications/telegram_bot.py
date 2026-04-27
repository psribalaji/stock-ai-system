"""
src/notifications/telegram_bot.py — Interactive Telegram bot with commands + LLM Q&A.

Commands:
  /status     — portfolio overview
  /positions  — open positions with P&L
  /stops      — active trailing stops and take-profits
  /signals    — today's signals summary
  /why TICKER — explain why a trade was made/blocked
  /approve TICKER — approve a discovery candidate
  /deny TICKER    — ignore a discovery candidate
  /pause      — activate circuit breaker
  /resume     — deactivate circuit breaker
  /help       — list commands

Free-form text → sent to Claude with system context for Q&A.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import date, timedelta
from typing import Optional

from loguru import logger


class TelegramBot:
    """
    Interactive Telegram bot that runs alongside the trading scheduler.

    Usage:
        bot = TelegramBot(executor=executor, risk_manager=rm, store=store)
        bot.start_background()  # non-blocking polling thread
    """

    def __init__(self, executor=None, risk_manager=None, store=None):
        self._token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        self._executor = executor
        self._risk_manager = risk_manager
        self._store = store
        self._running = False
        self._offset = 0  # Telegram update offset

        if not self._token or not self._chat_id:
            logger.warning("[TelegramBot] Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")

    @property
    def is_configured(self) -> bool:
        return bool(self._token and self._chat_id)

    def start_background(self) -> None:
        if not self.is_configured:
            logger.info("[TelegramBot] Not configured — skipping")
            return
        self._running = True
        thread = threading.Thread(target=self._poll_loop, daemon=True)
        thread.start()
        logger.info("[TelegramBot] Started polling")

    def stop(self) -> None:
        self._running = False

    # ── Polling loop ──────────────────────────────────────────────

    def _poll_loop(self) -> None:
        import httpx
        url = f"https://api.telegram.org/bot{self._token}/getUpdates"
        while self._running:
            try:
                resp = httpx.get(url, params={"offset": self._offset, "timeout": 10}, timeout=15)
                data = resp.json()
                for update in data.get("result", []):
                    self._offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    text = msg.get("text", "").strip()
                    chat_id = str(msg.get("chat", {}).get("id", ""))
                    if text and chat_id == self._chat_id:
                        self._handle(text, chat_id)
            except Exception as e:
                logger.debug(f"[TelegramBot] Poll error: {e}")
            time.sleep(2)

    # ── Message handler ───────────────────────────────────────────

    def _handle(self, text: str, chat_id: str) -> None:
        parts = text.split()
        cmd = parts[0].lower()
        arg = parts[1].upper() if len(parts) > 1 else ""

        handlers = {
            "/start": lambda: self._send("🤖 StockAI Bot ready. Type /help for commands."),
            "/help": self._cmd_help,
            "/status": self._cmd_status,
            "/positions": self._cmd_positions,
            "/stops": self._cmd_stops,
            "/signals": self._cmd_signals,
            "/why": lambda: self._cmd_why(arg),
            "/approve": lambda: self._cmd_approve(arg),
            "/deny": lambda: self._cmd_deny(arg),
            "/pause": self._cmd_pause,
            "/resume": self._cmd_resume,
        }

        if cmd in handlers:
            try:
                handlers[cmd]()
            except Exception as e:
                self._send(f"❌ Error: {e}")
        elif text.startswith("/"):
            self._send(f"Unknown command: {cmd}\nType /help for available commands.")
        else:
            # Free-form question → LLM
            self._cmd_ask(text)

    # ── Commands ──────────────────────────────────────────────────

    def _cmd_help(self) -> None:
        self._send(
            "📋 <b>Commands:</b>\n"
            "/status — portfolio overview\n"
            "/positions — open positions with P&L\n"
            "/stops — trailing stops & take-profits\n"
            "/signals — today's signals\n"
            "/why TICKER — explain a trade decision\n"
            "/approve TICKER — approve discovery candidate\n"
            "/deny TICKER — ignore discovery candidate\n"
            "/pause — pause trading (circuit breaker)\n"
            "/resume — resume trading\n\n"
            "Or just type a question and I'll answer using AI."
        )

    def _cmd_status(self) -> None:
        try:
            from src.ingestion.alpaca_client import AlpacaClient
            acct = AlpacaClient().get_account()
            total = acct.get("portfolio_value", 0)
            cash = acct.get("cash", 0)
            pnl = total - 100_000
            paused = self._risk_manager.is_paused if self._risk_manager else False
            killed = self._risk_manager.is_killed if self._risk_manager else False
            status = "☠️ KILL SWITCH" if killed else "⏸️ PAUSED" if paused else "✅ Active"
            self._send(
                f"📊 <b>Portfolio Status</b>\n"
                f"Value: ${total:,.2f}\n"
                f"Cash: ${cash:,.2f}\n"
                f"P&L: ${pnl:+,.2f}\n"
                f"System: {status}"
            )
        except Exception as e:
            self._send(f"Could not fetch status: {e}")

    def _cmd_positions(self) -> None:
        try:
            from src.ingestion.alpaca_client import AlpacaClient
            positions = AlpacaClient().get_positions()
            if positions.empty:
                self._send("No open positions.")
                return
            lines = ["📈 <b>Open Positions</b>"]
            for _, p in positions.iterrows():
                ticker = p.get("ticker", "?")
                pnl = p.get("unrealized_pl", 0)
                icon = "🟢" if pnl >= 0 else "🔴"
                lines.append(f"{icon} {ticker}: ${pnl:+,.2f}")
            self._send("\n".join(lines))
        except Exception as e:
            self._send(f"Could not fetch positions: {e}")

    def _cmd_stops(self) -> None:
        if not self._executor:
            self._send("Executor not available.")
            return
        trailing = getattr(self._executor, "_trailing_stops", {})
        take_profits = getattr(self._executor, "_take_profits", {})
        if not trailing and not take_profits:
            self._send("No active trailing stops or take-profits.")
            return
        lines = ["🛑 <b>Active Stops & TPs</b>"]
        all_tickers = set(list(trailing.keys()) + list(take_profits.keys()))
        for t in sorted(all_tickers):
            ts = trailing.get(t, {})
            tp = take_profits.get(t)
            stop = f"${ts['stop_price']:,.2f}" if ts else "—"
            high = f"${ts['high_water']:,.2f}" if ts else "—"
            target = f"${tp:,.2f}" if tp else "—"
            lines.append(f"<b>{t}</b>: stop={stop} high={high} TP={target}")
        self._send("\n".join(lines))

    def _cmd_signals(self) -> None:
        if not self._store:
            self._send("Store not available.")
            return
        try:
            end = date.today()
            start = end - timedelta(days=1)
            df = self._store.load_signals(start=start, end=end)
            if df.empty:
                self._send("No signals today.")
                return
            total = len(df)
            approved = len(df[df["approved"] == True]) if "approved" in df.columns else total
            blocked = total - approved
            lines = [f"📡 <b>Signals Today</b>: {total} total, ✅ {approved} approved, 🚫 {blocked} blocked"]
            if "ticker" in df.columns and "direction" in df.columns:
                approved_df = df[df["approved"] == True] if "approved" in df.columns else df
                for _, r in approved_df.head(5).iterrows():
                    lines.append(f"  {r['ticker']} {r['direction']} (conf={r.get('confidence', 0):.2f})")
            self._send("\n".join(lines))
        except Exception as e:
            self._send(f"Could not load signals: {e}")

    def _cmd_why(self, ticker: str) -> None:
        if not ticker:
            self._send("Usage: /why TICKER\nExample: /why NVDA")
            return
        if not self._store:
            self._send("Store not available.")
            return
        try:
            end = date.today()
            start = end - timedelta(days=7)
            df = self._store.load_signals(start=start, end=end)
            if df.empty or "ticker" not in df.columns:
                self._send(f"No recent signals for {ticker}.")
                return
            subset = df[df["ticker"] == ticker].tail(1)
            if subset.empty:
                self._send(f"No recent signals for {ticker}.")
                return
            r = subset.iloc[0]
            approved = "✅ APPROVED" if r.get("approved", False) else f"🚫 BLOCKED: {r.get('block_reason', '?')}"
            self._send(
                f"🔍 <b>Why {ticker}?</b>\n"
                f"Direction: {r.get('direction', '?')}\n"
                f"Strategy: {r.get('strategy', '?')}\n"
                f"Confidence: {r.get('confidence', 0):.2%}\n"
                f"Reason: {r.get('signal_reason', r.get('pattern', '?'))}\n"
                f"Result: {approved}"
            )
        except Exception as e:
            self._send(f"Could not explain {ticker}: {e}")

    def _cmd_approve(self, ticker: str) -> None:
        if not ticker:
            self._send("Usage: /approve TICKER")
            return
        try:
            from src.discovery.universe_manager import UniverseManager
            mgr = UniverseManager()
            if mgr.approve(ticker):
                self._send(f"✅ {ticker} approved for trading universe.")
            else:
                self._send(f"❌ {ticker} not found in candidates.")
        except Exception as e:
            self._send(f"Failed to approve {ticker}: {e}")

    def _cmd_deny(self, ticker: str) -> None:
        if not ticker:
            self._send("Usage: /deny TICKER")
            return
        try:
            from src.discovery.universe_manager import UniverseManager
            mgr = UniverseManager()
            if mgr.ignore(ticker):
                self._send(f"🚫 {ticker} ignored.")
            else:
                self._send(f"❌ {ticker} not found in candidates.")
        except Exception as e:
            self._send(f"Failed to deny {ticker}: {e}")

    def _cmd_pause(self) -> None:
        if self._risk_manager:
            self._risk_manager._paused = True
            self._send("⏸️ Trading paused (circuit breaker activated manually).")
        else:
            self._send("Risk manager not available.")

    def _cmd_resume(self) -> None:
        if self._risk_manager:
            self._risk_manager.reset_daily()
            self._send("▶️ Trading resumed (circuit breaker cleared).")
        else:
            self._send("Risk manager not available.")

    # ── LLM Q&A ──────────────────────────────────────────────────

    def _cmd_ask(self, question: str) -> None:
        """Send free-form question to Claude with system context."""
        try:
            # Build context from current state
            context_parts = [f"Question: {question}"]

            if self._store:
                try:
                    end = date.today()
                    start = end - timedelta(days=1)
                    signals = self._store.load_signals(start=start, end=end)
                    if not signals.empty:
                        context_parts.append(f"Today's signals: {len(signals)} total, "
                                             f"{len(signals[signals.get('approved', True) == True]) if 'approved' in signals.columns else len(signals)} approved")
                except Exception:
                    pass

            if self._risk_manager:
                status = self._risk_manager.get_status()
                context_parts.append(f"Risk status: paused={status['paused']}, killed={status['killed']}")

            context = "\n".join(context_parts)

            from src.secrets import Secrets
            import anthropic
            client = anthropic.Anthropic(api_key=Secrets.anthropic_api_key())
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                system=(
                    "You are a helpful assistant for a stock trading system called StockAI. "
                    "Answer concisely in 2-4 sentences. Use the provided context about the system's "
                    "current state. If you don't know, say so. Keep responses under 300 chars for Telegram."
                ),
                messages=[{"role": "user", "content": context}],
            )
            answer = resp.content[0].text.strip()
            self._send(f"🤖 {answer}")

        except Exception as e:
            self._send(f"🤖 Sorry, couldn't process that: {e}")

    # ── Send helper ───────────────────────────────────────────────

    def _send(self, text: str) -> None:
        try:
            import httpx
            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            # Telegram max message length is 4096
            for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
                httpx.post(url, json={
                    "chat_id": self._chat_id,
                    "text": chunk,
                    "parse_mode": "HTML",
                }, timeout=10)
        except Exception as e:
            logger.warning(f"[TelegramBot] Send failed: {e}")
