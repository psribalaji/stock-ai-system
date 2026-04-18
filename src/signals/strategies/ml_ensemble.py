"""
signals/strategies/ml_ensemble.py — ML-based ensemble signal strategy.

Uses a LightGBM model trained on historical trade outcomes to predict
win probability from the full feature set. Returns HOLD when the model
is not yet trained (insufficient data).

Enable via config: ml_ensemble.enabled = true (default: false)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_config
from src.features.feature_engine import FeatureEngine
from src.signals.strategies.momentum import RawSignal


class MLEnsembleStrategy:
    """
    ML ensemble strategy: predicts win probability from all features.

    Returns HOLD when:
      - Model is not trained yet
      - ml_ensemble.enabled is False
      - Prediction is between sell_threshold and buy_threshold
    """

    NAME = "ml_ensemble"

    def __init__(self, store=None) -> None:
        self.config = get_config()
        self.store = store
        self._model = None
        self._feature_cols = FeatureEngine.get_feature_columns()
        self._load_model()

    def _load_model(self) -> None:
        """Load trained model from disk if it exists."""
        model_path = Path(self.config.ml_ensemble.model_path)
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    self._model = pickle.load(f)
                logger.info(f"[{self.NAME}] Model loaded from {model_path}")
            except Exception as exc:
                logger.warning(f"[{self.NAME}] Failed to load model: {exc}")

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def evaluate(self, features: dict, ticker: str = "") -> RawSignal:
        """Predict using the trained model, or return HOLD if not ready."""
        if not self.is_ready:
            return self._hold(ticker, "Model not trained yet")

        cfg = self.config.ml_ensemble
        try:
            X = self._features_to_array(features)
            if X is None:
                return self._hold(ticker, "Insufficient features")

            prob = float(self._model.predict_proba(X)[:, 1][0])

            if prob > cfg.buy_threshold:
                return RawSignal(
                    direction="BUY", strategy=self.NAME,
                    strength=prob, pattern="ml_ensemble_buy",
                    reason=f"ML ensemble win probability {prob:.2f}",
                    features_snapshot=features,
                )
            elif prob < cfg.sell_threshold:
                return RawSignal(
                    direction="SELL", strategy=self.NAME,
                    strength=1.0 - prob, pattern="ml_ensemble_sell",
                    reason=f"ML ensemble loss probability {1.0 - prob:.2f}",
                    features_snapshot=features,
                )
            else:
                return self._hold(ticker, f"ML probability {prob:.2f} in neutral zone")

        except Exception as exc:
            logger.warning(f"[{self.NAME}] Prediction failed for {ticker}: {exc}")
            return self._hold(ticker, f"Prediction error: {exc}")

    def train(self, signals_df: pd.DataFrame) -> bool:
        """
        Train the model from historical signal outcomes.

        Args:
            signals_df: DataFrame with feature columns + 'outcome' column ('WIN'/'LOSS')

        Returns:
            True if training succeeded, False otherwise.
        """
        cfg = self.config.ml_ensemble
        if len(signals_df) < cfg.min_training_trades:
            logger.info(
                f"[{self.NAME}] Not enough data to train: "
                f"{len(signals_df)}/{cfg.min_training_trades}"
            )
            return False

        try:
            from lightgbm import LGBMClassifier

            available_cols = [c for c in self._feature_cols if c in signals_df.columns]
            if len(available_cols) < 10:
                logger.warning(f"[{self.NAME}] Too few feature columns: {len(available_cols)}")
                return False

            X = signals_df[available_cols].fillna(0).values
            y = (signals_df["outcome"] == "WIN").astype(int).values

            model = LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
            )
            model.fit(X, y)
            self._model = model

            # Save to disk
            model_path = Path(cfg.model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            accuracy = float((model.predict(X) == y).mean())
            logger.info(
                f"[{self.NAME}] Model trained on {len(signals_df)} trades, "
                f"training accuracy={accuracy:.2%}"
            )
            return True

        except ImportError:
            logger.warning(f"[{self.NAME}] lightgbm not installed — skipping training")
            return False
        except Exception as exc:
            logger.error(f"[{self.NAME}] Training failed: {exc}")
            return False

    def _features_to_array(self, features: dict) -> Optional[np.ndarray]:
        """Convert feature dict to numpy array matching training columns."""
        vals = []
        for col in self._feature_cols:
            v = features.get(col)
            vals.append(float(v) if v is not None else 0.0)
        return np.array([vals])

    def _hold(self, ticker: str, reason: str) -> RawSignal:
        logger.debug(f"[{self.NAME}] HOLD for {ticker}: {reason}")
        return RawSignal(
            direction="HOLD", strategy=self.NAME, strength=0.0,
            reason=reason, pattern="hold", features_snapshot={},
        )
