"""
dashboard/pages/ml_ensemble_status.py — ML Ensemble Status page.
Shows training data progress, model status, and feature importance.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def render_ml_ensemble_page() -> None:
    st.header("ML Ensemble Status")
    from src.config import get_config
    from src.ingestion.storage import ParquetStore

    cfg = get_config()
    store = ParquetStore(cfg.data.storage_path)
    ml_cfg = cfg.ml_ensemble

    # ── Status
    st.subheader("Configuration")
    c1, c2, c3 = st.columns(3)
    if ml_cfg.enabled:
        c1.success("✅ ENABLED")
    else:
        c1.warning("⏸️ DISABLED")
    c2.metric("Min Training Trades", ml_cfg.min_training_trades)
    c3.metric("Buy / Sell Thresholds", f"{ml_cfg.buy_threshold} / {ml_cfg.sell_threshold}")

    # ── Training data progress
    st.subheader("Training Data Progress")
    try:
        signals = store.load_signals()
        if not signals.empty and "outcome" in signals.columns:
            labeled = signals[signals["outcome"].notna()]
            total = len(labeled)
            target = ml_cfg.min_training_trades
            pct = min(total / target, 1.0)

            st.progress(pct, text=f"{total} / {target} labeled trades ({pct:.0%})")

            if total >= target:
                st.success(f"🎉 Enough data to train! You have {total} labeled trades.")
            else:
                remaining = target - total
                st.info(f"Need {remaining} more labeled trades before training.")

            # Outcome breakdown
            if total > 0:
                wins = (labeled["outcome"] == "WIN").sum()
                losses = (labeled["outcome"] == "LOSS").sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Labeled", total)
                c2.metric("Wins", f"{wins} ({wins/total:.0%})" if total > 0 else "0")
                c3.metric("Losses", f"{losses} ({losses/total:.0%})" if total > 0 else "0")
        else:
            st.progress(0.0, text=f"0 / {ml_cfg.min_training_trades} labeled trades (0%)")
            st.info("No labeled trade data yet. Trades need WIN/LOSS outcomes to train the model.")
    except Exception:
        st.progress(0.0, text=f"0 / {ml_cfg.min_training_trades} labeled trades")
        st.info("No signal data available yet.")

    # ── Model status
    st.subheader("Model Status")
    model_path = Path(ml_cfg.model_path)
    if model_path.exists():
        st.success(f"✅ Trained model found at `{ml_cfg.model_path}`")
        stat = model_path.stat()
        st.caption(f"Last modified: {pd.Timestamp(stat.st_mtime, unit='s')}")

        # Feature importance (if model is loadable)
        try:
            import pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            if hasattr(model, "feature_importances_"):
                from src.features.feature_engine import FeatureEngine
                cols = FeatureEngine.get_feature_columns()
                importances = model.feature_importances_
                if len(cols) == len(importances):
                    fi_df = pd.DataFrame({"Feature": cols, "Importance": importances})
                    fi_df = fi_df.sort_values("Importance", ascending=False).head(15)
                    st.subheader("Top 15 Feature Importances")
                    st.bar_chart(fi_df.set_index("Feature"))
        except Exception as e:
            st.caption(f"Could not load model for feature importance: {e}")
    else:
        st.info(f"No trained model yet. Will be saved to `{ml_cfg.model_path}` after training.")

    # ── How to enable
    if not ml_cfg.enabled:
        st.subheader("How to Enable")
        st.markdown(f"""
        1. Accumulate **{ml_cfg.min_training_trades}** labeled trades (check progress bar above)
        2. Set `ml_ensemble.enabled: true` in `config.yaml`
        3. Restart the system — model will auto-train on first run
        """)
