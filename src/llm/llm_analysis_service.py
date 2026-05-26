"""
llm/llm_analysis_service.py — LLM-based signal enrichment via Claude API.

SCOPE: The LLM ONLY generates reasoning text and news summaries for the
dashboard. It does NOT score confidence, make trade decisions, or compute
any indicators. All numerical scoring is done by ConfidenceScorer.

Improvements (inspired by Anthropic financial-services repo):
  - Structured output schema: thesis + bullets + risks + source citations
  - Data freshness guardrail: today's date injected, "use only provided data"
  - Source citation: Finnhub article IDs tagged in output
  - Earnings context: beat/miss history + upcoming earnings date
"""
from __future__ import annotations

import json
from datetime import date
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
      - Recent news headlines with source IDs
      - Earnings context (beat/miss history, upcoming date)
      - Today's date (prevents use of stale training knowledge)

    The LLM returns structured JSON:
      - thesis:            One-line trade thesis
      - reasoning:         2-3 sentence analysis paragraph
      - supporting_points: 3 bullet points grounded in provided data
      - risks:             2 key risks to the trade
      - sentiment:         bullish | bearish | neutral
      - news_sources:      Finnhub article IDs cited
      - summary:           One-line mobile notification string

    The LLM does NOT return confidence scores, position sizes, or decisions.
    """

    MODEL = "claude-sonnet-4-6"
    MAX_TOKENS = 600

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

    # ── PRIMARY ENRICHMENT ────────────────────────────────────────

    def enrich(
        self,
        scored_signal: ScoredSignal,
        news_summary: Optional[str] = None,
        news_articles: Optional[list[dict]] = None,
        earnings_context: Optional[dict] = None,
    ) -> dict:
        """
        Generate LLM reasoning text for a scored signal.

        Args:
            scored_signal:    ScoredSignal from ConfidenceScorer
            news_summary:     Legacy plain-text news summary (still accepted)
            news_articles:    Structured articles [{id, headline, source, datetime}]
                              Preferred over news_summary — enables source citation
            earnings_context: Dict from NewsService.get_earnings_context()

        Returns:
            Dict with keys: thesis, reasoning, supporting_points, risks,
                            sentiment, news_sources, summary, llm_used
        """
        prompt = self._build_prompt(scored_signal, news_summary, news_articles, earnings_context)

        try:
            return self._call_llm(prompt, scored_signal)
        except Exception as exc:
            logger.warning(
                f"[LLMAnalysisService] LLM enrichment failed for "
                f"{scored_signal.ticker}: {exc}. Using fallback reasoning."
            )
            return self._fallback_reasoning(scored_signal)

    # ── EARNINGS ANALYSIS ─────────────────────────────────────────

    def earnings_analysis(
        self,
        ticker: str,
        earnings_context: dict,
        news_articles: Optional[list[dict]] = None,
    ) -> dict:
        """
        Dedicated earnings analysis — called when a ticker has upcoming or
        very recent earnings (within 7 days either side).

        Analyzes: beat/miss history, EPS trend, guidance language,
        upcoming catalyst risk, and short-term price implications.

        Args:
            ticker:           Ticker symbol
            earnings_context: Dict from NewsService.get_earnings_context()
            news_articles:    Recent news articles with earnings coverage

        Returns:
            Dict with keys: beat_miss_trend, upcoming_risk, thesis,
                            key_metrics, risks, summary, llm_used
        """
        if not earnings_context or not earnings_context.get("eps_history"):
            return {"llm_used": False, "reason": "No earnings data available"}

        prompt = self._build_earnings_prompt(ticker, earnings_context, news_articles)

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.MODEL,
                max_tokens=500,
                system=self._earnings_system_prompt(),
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            try:
                parsed = json.loads(text)
                parsed["llm_used"] = True
                return parsed
            except json.JSONDecodeError:
                return {"thesis": text, "llm_used": True}
        except Exception as exc:
            logger.warning(f"[LLMAnalysisService] Earnings analysis failed for {ticker}: {exc}")
            return {"llm_used": False, "reason": str(exc)}

    # ── LLM CALLS ─────────────────────────────────────────────────

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

        try:
            parsed = json.loads(text)
            return {
                "thesis":            parsed.get("thesis", ""),
                "reasoning":         parsed.get("reasoning", text),
                "supporting_points": parsed.get("supporting_points", []),
                "risks":             parsed.get("risks", []),
                "sentiment":         parsed.get("sentiment", "neutral"),
                "news_sources":      parsed.get("news_sources", []),
                "summary":           parsed.get("summary", scored_signal.reason),
                "llm_used":          True,
            }
        except json.JSONDecodeError:
            return {
                "thesis":            "",
                "reasoning":         text,
                "supporting_points": [],
                "risks":             [],
                "sentiment":         "neutral",
                "news_sources":      [],
                "summary":           scored_signal.reason,
                "llm_used":          True,
            }

    # ── PROMPTS ───────────────────────────────────────────────────

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a financial analyst assistant for a trading system. "
            "Your ONLY job is to explain trade signals in plain English for a dashboard. "
            "You do NOT make trade decisions, assign confidence scores, or recommend position sizes. "
            f"Today's date is {date.today().isoformat()}. "
            "CRITICAL: Use ONLY the data provided in the prompt. "
            "Do not use your training knowledge about specific stock prices, "
            "earnings, or news — it may be outdated. Cite only what is given. "
            "Always respond with valid JSON matching the requested schema exactly."
        )

    @staticmethod
    def _earnings_system_prompt() -> str:
        return (
            "You are a financial analyst specializing in earnings analysis. "
            f"Today's date is {date.today().isoformat()}. "
            "Analyze ONLY the earnings data provided. Do not use training knowledge "
            "about specific EPS figures — it may be outdated. "
            "Be concise and factual. Respond with valid JSON matching the requested schema."
        )

    @staticmethod
    def _build_prompt(
        signal: ScoredSignal,
        news_summary: Optional[str],
        news_articles: Optional[list[dict]],
        earnings_context: Optional[dict],
    ) -> str:
        feats  = signal.features_snapshot
        rsi    = feats.get("rsi_14")
        macd   = feats.get("macd_hist")
        adx    = feats.get("adx")
        regime = "bull" if feats.get("bull_regime") else "bear/neutral"

        rsi_str  = f"{rsi:.1f}"  if rsi  is not None else "N/A"
        macd_str = f"{macd:.4f}" if macd is not None else "N/A"
        adx_str  = f"{adx:.1f}"  if adx  is not None else "N/A"

        # News section — prefer structured articles (enables citation)
        news_section = ""
        if news_articles:
            lines = []
            for a in news_articles[:8]:
                art_id  = a.get("id", "unknown")
                headline = a.get("headline", "")
                source   = a.get("source", "")
                lines.append(f"  [{art_id}] {headline} ({source})")
            news_section = "\nRecent news (cite by ID):\n" + "\n".join(lines)
        elif news_summary:
            news_section = f"\nRecent news:\n{news_summary}"

        # Earnings section
        earnings_section = ""
        if earnings_context:
            upcoming = earnings_context.get("upcoming_date")
            days_until = earnings_context.get("days_until_earnings")
            beat_miss = earnings_context.get("beat_miss")
            surprise_pct = earnings_context.get("last_eps_surprise_pct")
            consecutive = earnings_context.get("consecutive_beats", 0)

            parts = []
            if upcoming and days_until is not None and days_until <= 14:
                parts.append(f"Earnings in {days_until} days ({upcoming}) — elevated uncertainty")
            if beat_miss:
                surprise_str = f"{surprise_pct:+.1f}%" if surprise_pct is not None else ""
                parts.append(f"Last quarter: {beat_miss} {surprise_str}")
            if consecutive > 0:
                parts.append(f"{consecutive}/4 recent quarters beat estimates")
            if parts:
                earnings_section = "\nEarnings context:\n" + "\n".join(f"  {p}" for p in parts)

        schema = (
            '{\n'
            '  "thesis": "one-line trade thesis",\n'
            '  "reasoning": "2-3 sentence explanation of WHY this signal triggered",\n'
            '  "supporting_points": ["point 1", "point 2", "point 3"],\n'
            '  "risks": ["risk 1", "risk 2"],\n'
            '  "sentiment": "bullish|bearish|neutral",\n'
            '  "news_sources": ["article_id_1", "article_id_2"],\n'
            '  "summary": "one-line mobile notification"\n'
            '}'
        )

        return (
            f"Signal for {signal.ticker} — {date.today().isoformat()}:\n"
            f"  Direction:  {signal.direction}\n"
            f"  Strategy:   {signal.strategy}\n"
            f"  Pattern:    {signal.pattern}\n"
            f"  Confidence: {signal.confidence:.2f} (statistical score, not from you)\n"
            f"  Trigger:    {signal.reason}\n"
            f"\nKey indicators:\n"
            f"  RSI-14: {rsi_str}\n"
            f"  MACD Histogram: {macd_str}\n"
            f"  ADX: {adx_str}\n"
            f"  Market regime: {regime}"
            f"{earnings_section}"
            f"{news_section}\n\n"
            "Using ONLY the data above, explain why this signal triggered and what the "
            "indicators suggest. Cite news by article ID where relevant.\n"
            f"Respond ONLY with this JSON schema:\n{schema}"
        )

    @staticmethod
    def _build_earnings_prompt(
        ticker: str,
        earnings_context: dict,
        news_articles: Optional[list[dict]],
    ) -> str:
        history = earnings_context.get("eps_history", [])
        upcoming = earnings_context.get("upcoming_date")
        days_until = earnings_context.get("days_until_earnings")

        history_lines = []
        for e in history:
            actual   = e.get("actual")
            estimate = e.get("estimate")
            sp       = e.get("surprise_pct")
            period   = e.get("period", "?")
            actual_str   = f"${actual:.2f}"   if actual   is not None else "N/A"
            estimate_str = f"${estimate:.2f}" if estimate is not None else "N/A"
            sp_str       = f"{sp:+.1f}%"      if sp       is not None else "N/A"
            history_lines.append(f"  {period}: actual={actual_str} est={estimate_str} surprise={sp_str}")

        news_section = ""
        if news_articles:
            lines = [
                f"  [{a.get('id','?')}] {a.get('headline','')} ({a.get('source','')})"
                for a in news_articles[:6]
            ]
            news_section = "\nRecent news:\n" + "\n".join(lines)

        upcoming_str = (
            f"{upcoming} ({days_until} days away)"
            if upcoming and days_until is not None
            else "Not within 30 days"
        )

        schema = (
            '{\n'
            '  "beat_miss_trend": "improving|deteriorating|stable",\n'
            '  "upcoming_risk": "high|medium|low",\n'
            '  "thesis": "one-line earnings thesis",\n'
            '  "key_metrics": ["metric observation 1", "metric observation 2"],\n'
            '  "risks": ["risk 1", "risk 2"],\n'
            '  "summary": "one-line notification string"\n'
            '}'
        )

        return (
            f"Earnings analysis for {ticker} — {date.today().isoformat()}\n\n"
            f"Next earnings: {upcoming_str}\n\n"
            f"Recent EPS history (provided data only — do not use training knowledge):\n"
            + "\n".join(history_lines)
            + f"{news_section}\n\n"
            "Analyze the beat/miss trend, assess risk heading into next earnings, "
            "and note any signals from recent news.\n"
            f"Respond ONLY with this JSON schema:\n{schema}"
        )

    # ── NEWS SUMMARY ──────────────────────────────────────────────

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
            f"Today is {date.today().isoformat()}. "
            f"Summarize the following {ticker} news headlines in 1-2 sentences "
            f"focusing on what matters for short-term price action. "
            f"Use ONLY these headlines — do not add outside knowledge:\n\n"
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

    def score_sentiment(self, ticker: str, headlines: list[str]) -> float:
        """
        Score sentiment of headlines via LLM. Returns -1.0 to 1.0.
        Falls back to 0.0 (neutral) on any failure.
        """
        if not headlines:
            return 0.0

        headlines_text = "\n".join(f"- {h}" for h in headlines[:10])
        prompt = (
            f"Today is {date.today().isoformat()}. "
            f"Score the overall sentiment of these {ticker} headlines "
            f"for short-term stock price impact. Use ONLY these headlines.\n\n"
            f"{headlines_text}\n\n"
            'Respond ONLY with JSON: {"score": <float between -1.0 and 1.0>}\n'
            "-1.0 = very bearish, 0.0 = neutral, 1.0 = very bullish"
        )

        try:
            result = self._call_llm_raw(prompt)
            parsed = json.loads(result)
            score = float(parsed.get("score", 0.0))
            return max(-1.0, min(1.0, score))
        except Exception as exc:
            logger.warning(f"[LLMAnalysisService] Sentiment scoring failed for {ticker}: {exc}")
            return 0.0

    # ── HELPERS ───────────────────────────────────────────────────

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
            system=(
                f"You are a concise financial news analyst. "
                f"Today is {date.today().isoformat()}. "
                "Always respond with valid JSON."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    @staticmethod
    def _fallback_reasoning(signal: ScoredSignal) -> dict:
        """Return safe fallback when LLM is unavailable."""
        return {
            "thesis":            f"{signal.ticker} {signal.direction} via {signal.strategy}",
            "reasoning":         (
                f"{signal.ticker} shows a {signal.direction} signal via the "
                f"{signal.strategy} strategy ({signal.pattern}). "
                f"Trigger: {signal.reason}. "
                f"Statistical confidence: {signal.confidence:.2f}."
            ),
            "supporting_points": [signal.reason],
            "risks":             ["LLM enrichment unavailable — using statistical signal only"],
            "sentiment":         "bullish" if signal.direction == "BUY" else "bearish",
            "news_sources":      [],
            "summary":           (
                f"{signal.ticker} {signal.direction} - {signal.strategy} "
                f"(conf={signal.confidence:.2f})"
            ),
            "llm_used":          False,
        }
