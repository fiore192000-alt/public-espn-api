"""Test fixtures for the trading ML package."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

USER_DATA = Path(__file__).resolve().parents[1] / "user_data"
if str(USER_DATA) not in sys.path:
    sys.path.insert(0, str(USER_DATA))


def make_ohlcv(rows: int = 3000, seed: int = 7, start: str = "2020-01-01") -> pd.DataFrame:
    """A synthetic but well-formed hourly OHLCV series (geometric random walk)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.01, rows)
    # A slow regime cycle keeps the series from being pure noise, so a model
    # trained on it has something to find.
    steps += 0.004 * np.sin(np.arange(rows) / 250.0)
    close = 100.0 * np.exp(np.cumsum(steps))

    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0.0, 0.006, rows)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(mean=8.0, sigma=0.5, size=rows)

    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=rows, freq="1h", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def market_ohlcv() -> pd.DataFrame:
    return make_ohlcv(seed=11)
