"""
llm/llm_analysis_service.py — LLM-based signal enrichment via Claude API.

SCOPE: The LLM ONLY generates reasoning text and news summaries for the
dashboard. It does NOT score confidence, make trade decisions, or compute
any indicators. All numerical scoring is done by ConfidenceScorer.
"""
from __future__ import annotations

import json
from typing import Optional

from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config import get_config
from src.secrets import Secrets
from src.signals.confidence_scorer import ScoredSignal


class LLMAnalysisService:
    """
    Enriches scored signals with human-readable reasoning from Claude.

    The LLM receives:
      - Signal direction + strategy name
      - Key features (RSI, MACD, ADX, regime)
      - Recent news headlines + sentiment summary

    The LLM returns:
      - A 2-3 sentence reasoning paragraph for the dashboard
      - A one-line summary suitable for a notification

    The LLM does NOT return confidence scores, position sizes, or decisions.
    """

    MODEL = "claude-sonnet-4-6"
    MAX_TOKENS = 512

    def __init__(self) -> None:
        self.config = get_config()
        self._client = None  # lazy init to avoid key errors in tests

    def _get_client(self):
        """Lazy-init Anthropic client (allows import without ANTHROPIC_API_KEY)."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=Secrets.anthropic_api_key()
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to initialize Anthropic client: {exc}\n"
                    "Ensure ANTHROPIC_API_KEY is set in .env"
                ) from exc
        return self._client

    def enrich(
        self,
        scored_signal: ScoredSignal,
        news_summary: Optional[str] = None,
    ) -> dict:
        """
        Generate LLM reasoning text for a scored signal.

        Args:
            scored_signal: ScoredSignal from ConfidenceScorer
            news_summary:  Optional news text from NewsService

        Returns:
            Dict with keys:
              - reasoning:   2-3 sentence analysis paragraph
              - summary:     One-line notification string
              - llm_used:    True (so callers know this was enriched)
        """
        prompt = self._build_prompt(scored_signal, news_summary)

        try:
            return self._call_llm(prompt, scored_signal)
        except Exception as exc:
            logger.warning(
                f"[LLMAnalysisService] LLM enrichment failed for "
                f"{scored_signal.ticker}: {exc}. Using fallback reasoning."
            )
            return self._fallback_reasoning(scored_signal)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=False,
    )
    def _call_llm(self, prompt: str, scored_signal: ScoredSignal) -> dict:
        """Call Claude API with retry logic."""
        client = self._get_client()
        response = client.messages.create(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
            system=self._system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        # Parse structured response
        try:
            parsed = json.loads(text)
            return {
                "reasoning": parsed.get("reasoning", text),
                "summary":   parsed.get("summary", scored_signal.reason),
                "llm_used":  True,
            }
        except json.JSONDecodeError:
            # Fallback: treat the whole text as reasoning
            return {
                "reasoning": text,
                "summary":   scored_signal.reason,
                "llm_used":  True,
            }

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a financial analyst assistant for a trading system. "
            "Your ONLY job is to explain trade signals in plain English for a dashboard. "
            "You do NOT make trade decisions, assign confidence scores, or recommend position sizes. "
            "Be concise, factual, and mention specific indicator values. "
            'Always respond with valid JSON: {"reasoning": "...", "summary": "..."}'
        )

    @staticmethod
    def _build_prompt(signal: ScoredSignal, news_summary: Optional[str]) -> str:
        """Build the user prompt from signal data."""
        feats = signal.features_snapshot
        rsi   = feats.get("rsi_14")
        macd  = feats.get("macd_hist")
        adx   = feats.get("adx")
        regime = "bull" if feats.get("bull_regime") else "bear/neutral"

        rsi_str  = f"{rsi:.1f}"  if rsi  is not None else "N/A"
        macd_str = f"{macd:.4f}" if macd is not None else "N/A"
        adx_str  = f"{adx:.1f}"  if adx  is not None else "N/A"

        news_section = f"\nRecent news:\n{news_summary}" if news_summary else ""

        return (
            f"Signal for {signal.ticker}:\n"
            f"  Direction:  {signal.direction}\n"
            f"  Strategy:   {signal.strategy}\n"
            f"  Pattern:    {signal.pattern}\n"
            f"  Confidence: {signal.confidence:.2f} (statistical, not from you)\n"
            f"  Trigger:    {signal.reason}\n"
            f"\nKey indicators:\n"
            f"  RSI-14: {rsi_str}\n"
            f"  MACD Histogram: {macd_str}\n"
            f"  ADX: {adx_str}\n"
            f"  Market regime: {regime}"
            f"{news_section}\n\n"
            "Explain in 2-3 sentences WHY this signal triggered and what the indicators suggest. "
            "Then write a one-line summary suitable for a mobile notification. "
            'Respond ONLY with JSON: {"reasoning": "...", "summary": "..."}'
        )

    @staticmethod
    def _fallback_reasoning(signal: ScoredSignal) -> dict:
        """Return safe fallback when LLM is unavailable."""
        return {
            "reasoning": (
                f"{signal.ticker} shows a {signal.direction} signal via the "
                f"{signal.strategy} strategy ({signal.pattern}). "
                f"Trigger: {signal.reason}. "
                f"Statistical confidence: {signal.confidence:.2f}."
            ),
            "summary": (
                f"{signal.ticker} {signal.direction} - {signal.strategy} "
                f"(conf={signal.confidence:.2f})"
            ),
            "llm_used": False,
        }

    def enrich_news(self, ticker: str, headlines: list) -> str:
        """
        Summarize recent news headlines for a ticker.

        Args:
            ticker:    Ticker symbol
            headlines: List of recent news headline strings

        Returns:
            A 1-2 sentence news summary string.
        """
        if not headlines:
            return ""

        headlines_text = "\n".join(f"- {h}" for h in headlines[:10])
        prompt = (
            f"Summarize the following {ticker} news headlines in 1-2 sentences "
            f"focusing on what matters for short-term price action:\n\n"
            f"{headlines_text}\n\n"
            'Respond ONLY with JSON: {"summary": "..."}'
        )

        try:
            result = self._call_llm_raw(prompt)
            parsed = json.loads(result)
            return parsed.get("summary", result)
        except Exception as exc:
            logger.warning(f"[LLMAnalysisService] News summary failed for {ticker}: {exc}")
            return f"Recent news: {headlines[0]}" if headlines else ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=False,
    )
    def _call_llm_raw(self, prompt: str) -> str:
        """Raw LLM call returning text."""
        client = self._get_client()
        response = client.messages.create(
            model=self.MODEL,
            max_tokens=256,
            system="You are a concise financial news analyst. Always respond with valid JSON.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
