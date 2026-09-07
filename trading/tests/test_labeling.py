"""Triple-barrier labels: entry on the next open, barriers in ATR units."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.features import add_features
from ml.labeling import label_summary, triple_barrier_labels


def _frame(closes: list[float], atr: float = 1.0) -> pd.DataFrame:
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(close), freq="1h", tz="UTC"),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "atr": atr,
        }
    )


def test_take_profit_before_stop_loss_is_a_positive_label() -> None:
    # Entry at open[1] = 100; +2 ATR = 102 is reached at index 3.
    df = _frame([100, 100, 101, 102.5, 99])
    labels = triple_barrier_labels(df, horizon=3, tp_atr=2.0, sl_atr=1.0)
    assert labels.loc[0, "label"] == 1.0
    assert labels.loc[0, "label_hit"] == "tp"
    assert labels.loc[0, "label_bars"] == 3


def test_stop_loss_first_is_a_negative_label() -> None:
    df = _frame([100, 100, 98.5, 103, 103])
    labels = triple_barrier_labels(df, horizon=3, tp_atr=2.0, sl_atr=1.0)
    assert labels.loc[0, "label"] == 0.0
    assert labels.loc[0, "label_hit"] == "sl"


def test_timeout_counts_as_a_loss() -> None:
    df = _frame([100, 100, 100.2, 100.1, 100.3])
    labels = triple_barrier_labels(df, horizon=3, tp_atr=2.0, sl_atr=1.0)
    assert labels.loc[0, "label"] == 0.0
    assert labels.loc[0, "label_hit"] == "timeout"


def test_ambiguous_candle_resolves_pessimistically() -> None:
    """Both barriers inside one candle: OHLC cannot say which came first."""
    df = _frame([100, 100, 100, 100])
    df.loc[2, ["high", "low"]] = [103.0, 98.0]
    labels = triple_barrier_labels(df, horizon=2, tp_atr=2.0, sl_atr=1.0)
    assert labels.loc[0, "label"] == 0.0
    assert labels.loc[0, "label_hit"] == "sl"


def test_entry_price_is_the_next_open_not_the_signal_close() -> None:
    df = _frame([100, 110, 112, 112, 112])
    # Entry at 110, so +2 ATR = 112 is a win; measuring from 100 it would be
    # a win far too early, and from the close of the signal candle it would be
    # unexecutable.
    labels = triple_barrier_labels(df, horizon=3, tp_atr=2.0, sl_atr=1.0)
    assert labels.loc[0, "label"] == 1.0
    assert np.isclose(labels.loc[0, "label_ret"], 112 / 110 - 1.0)


def test_unobservable_tail_is_unlabelled() -> None:
    df = _frame(list(np.linspace(100, 110, 20)))
    labels = triple_barrier_labels(df, horizon=5, tp_atr=2.0, sl_atr=1.0)
    assert labels["label"].iloc[-5:].isna().all()
    assert labels["label"].iloc[:-5].notna().all()


def test_summary_reports_the_class_balance(ohlcv: pd.DataFrame) -> None:
    labels = triple_barrier_labels(add_features(ohlcv), horizon=24)
    summary = label_summary(labels)
    assert summary["samples"] > 0
    assert 0.0 < summary["positive_rate"] < 1.0
    assert summary["tp"] + summary["sl"] + summary["timeout"] == summary["samples"]
