"""The features must not be able to see the future.

If any of these fail, every backtest built on top of the model is fiction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features import STARTUP_CANDLES, add_features, add_market_features, feature_columns


def test_features_only_depend_on_the_past(ohlcv: pd.DataFrame) -> None:
    """Truncating the tail must not change any earlier value.

    This is the property that catches ``shift(-n)``, centered rolling windows,
    ``bfill()`` and any whole-sample normalisation.
    """
    full = add_features(ohlcv)
    cut = 2000
    truncated = add_features(ohlcv.iloc[:cut].copy())

    cols = feature_columns(full)
    assert cols, "no features produced"
    pd.testing.assert_frame_equal(
        full.loc[: cut - 1, cols],
        truncated.loc[: cut - 1, cols],
        check_exact=False,
        rtol=1e-12,
    )


def test_market_features_only_depend_on_the_past(
    ohlcv: pd.DataFrame, market_ohlcv: pd.DataFrame
) -> None:
    full = add_market_features(add_features(ohlcv), market_ohlcv)
    cut = 2000
    truncated = add_market_features(
        add_features(ohlcv.iloc[:cut].copy()), market_ohlcv.iloc[:cut].copy()
    )
    cols = [col for col in feature_columns(full) if "btc" in col]
    assert cols, "no market features produced"
    pd.testing.assert_frame_equal(
        full.loc[: cut - 1, cols],
        truncated.loc[: cut - 1, cols],
        check_exact=False,
        rtol=1e-12,
    )


def test_features_are_defined_after_warmup(ohlcv: pd.DataFrame) -> None:
    """After the warm-up window every feature must be a real number."""
    out = add_features(ohlcv)
    tail = out.iloc[STARTUP_CANDLES:]
    bad = {
        col: int(tail[col].isna().sum() + np.isinf(tail[col]).sum())
        for col in feature_columns(out)
        if tail[col].isna().any() or np.isinf(tail[col]).any()
    }
    assert not bad, f"non-finite features after warm-up: {bad}"


def test_startup_window_covers_the_longest_indicator(ohlcv: pd.DataFrame) -> None:
    """A shorter warm-up than STARTUP_CANDLES must still be leaving NaNs."""
    out = add_features(ohlcv)
    cols = feature_columns(out)
    assert out.loc[: STARTUP_CANDLES // 4, cols].isna().any().any()


@pytest.mark.parametrize("column", ["f_rsi_14", "f_adx", "f_bb_pos"])
def test_bounded_features_stay_in_range(ohlcv: pd.DataFrame, column: str) -> None:
    values = add_features(ohlcv)[column].dropna()
    assert not values.empty
    if column in ("f_rsi_14", "f_adx"):
        assert values.between(0.0, 1.0).all()
