"""Feature engineering shared by the training pipeline and the live strategy.

Every feature here is *causal*: the value at row ``i`` only depends on rows
``<= i``.  That property is what keeps backtests honest, and it is enforced by
``trading/tests/test_features.py`` (a truncated dataframe must produce the same
values as the full one for every row they share).

The module deliberately depends on nothing but pandas/numpy so that the exact
same code path runs inside ``train.py`` and inside the Freqtrade strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Longest lookback used below (EMA 200 needs a good deal more than 200 candles
# to converge).  Freqtrade's ``startup_candle_count`` must be >= this value.
STARTUP_CANDLES = 400


# --------------------------------------------------------------------------- #
# Indicator primitives (pure pandas, no TA-Lib dependency)
# --------------------------------------------------------------------------- #
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing == EWM with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 means a pure uptrend window -> RSI 100
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return ``(adx, plus_di, minus_di)`` using Wilder's smoothing."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    alpha = 1 / period
    tr_s = true_range(high, low, close).ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_s = pd.Series(plus_dm, index=high.index).ewm(
        alpha=alpha, adjust=False, min_periods=period
    ).mean()
    minus_s = pd.Series(minus_dm, index=high.index).ewm(
        alpha=alpha, adjust=False, min_periods=period
    ).mean()

    tr_safe = tr_s.replace(0.0, np.nan)
    plus_di = 100.0 * plus_s / tr_safe
    minus_di = 100.0 * minus_s / tr_safe
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean(), plus_di, minus_di


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Least-squares slope of ``series`` over ``window`` bars, per bar."""
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denom = float((x_centered**2).sum())

    def _slope(values: np.ndarray) -> float:
        return float(np.dot(values - values.mean(), x_centered) / denom)

    return series.rolling(window, min_periods=window).apply(_slope, raw=True)


# --------------------------------------------------------------------------- #
# Feature block
# --------------------------------------------------------------------------- #
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the model features to an OHLCV dataframe (copy, never in place).

    Expects lowercase ``open/high/low/close/volume`` columns and, optionally, a
    ``date`` column for the session-of-day features.
    """
    out = df.copy()
    close, high, low, volume = out["close"], out["high"], out["low"], out["volume"]

    # --- returns / momentum ------------------------------------------------ #
    log_close = np.log(close.replace(0.0, np.nan))
    for n in (1, 3, 5, 10, 20, 50):
        out[f"f_ret_{n}"] = log_close.diff(n)

    # --- trend structure --------------------------------------------------- #
    ema_20, ema_50, ema_200 = ema(close, 20), ema(close, 50), ema(close, 200)
    out["f_close_ema20"] = close / ema_20 - 1.0
    out["f_ema20_ema50"] = ema_20 / ema_50 - 1.0
    out["f_ema50_ema200"] = ema_50 / ema_200 - 1.0
    out["f_slope_20"] = rolling_slope(log_close, 20) * 100.0
    out["f_slope_50"] = rolling_slope(log_close, 50) * 100.0

    # --- oscillators ------------------------------------------------------- #
    out["f_rsi_14"] = rsi(close, 14) / 100.0
    out["f_rsi_7"] = rsi(close, 7) / 100.0

    macd_line = ema(close, 12) - ema(close, 26)
    macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    # Normalised by price so the feature is comparable across pairs and regimes.
    out["f_macd"] = macd_line / close
    out["f_macd_hist"] = (macd_line - macd_signal) / close

    adx_v, plus_di, minus_di = adx(high, low, close, 14)
    out["f_adx"] = adx_v / 100.0
    out["f_di_spread"] = (plus_di - minus_di) / 100.0

    # --- volatility -------------------------------------------------------- #
    atr_14 = atr(high, low, close, 14)
    out["atr"] = atr_14  # kept unprefixed: the strategy sizes stops with it
    out["f_atr_pct"] = atr_14 / close
    out["f_atr_ratio"] = atr_14 / atr_14.rolling(100, min_periods=100).mean()
    ret_1 = log_close.diff(1)
    out["f_vol_20"] = ret_1.rolling(20, min_periods=20).std()
    out["f_vol_50"] = ret_1.rolling(50, min_periods=50).std()
    out["f_vol_ratio"] = out["f_vol_20"] / out["f_vol_50"]

    # --- bollinger position ------------------------------------------------ #
    sma_20 = close.rolling(20, min_periods=20).mean()
    std_20 = close.rolling(20, min_periods=20).std()
    out["f_bb_pos"] = (close - sma_20) / (2.0 * std_20).replace(0.0, np.nan)
    out["f_bb_width"] = (4.0 * std_20) / sma_20

    # --- candle shape ------------------------------------------------------ #
    candle_range = (high - low).replace(0.0, np.nan)
    out["f_body"] = (close - out["open"]) / candle_range
    out["f_upper_wick"] = (high - np.maximum(close, out["open"])) / candle_range
    out["f_lower_wick"] = (np.minimum(close, out["open"]) - low) / candle_range
    out["f_range_pct"] = candle_range / close

    # --- volume ------------------------------------------------------------ #
    vol_mean = volume.rolling(20, min_periods=20).mean()
    vol_std = volume.rolling(20, min_periods=20).std()
    out["f_vol_z"] = (volume - vol_mean) / vol_std.replace(0.0, np.nan)
    out["f_vol_ratio_20"] = volume / vol_mean.replace(0.0, np.nan)
    # Signed volume pressure: where did the volume push price?
    out["f_vol_pressure"] = (
        (ret_1 * volume).rolling(20, min_periods=20).sum()
        / volume.rolling(20, min_periods=20).sum().replace(0.0, np.nan)
    )

    # --- distance from recent extremes ------------------------------------- #
    for n in (20, 50):
        out[f"f_dist_high_{n}"] = close / high.rolling(n, min_periods=n).max() - 1.0
        out[f"f_dist_low_{n}"] = close / low.rolling(n, min_periods=n).min() - 1.0

    # --- session of day (cyclical) ----------------------------------------- #
    if "date" in out.columns:
        hours = pd.to_datetime(out["date"], utc=True).dt.hour
        dow = pd.to_datetime(out["date"], utc=True).dt.dayofweek
        out["f_hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        out["f_hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        out["f_dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        out["f_dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    return out


def add_market_features(df: pd.DataFrame, market: pd.DataFrame, prefix: str = "btc") -> pd.DataFrame:
    """Merge features describing the wider market (usually BTC) into ``df``.

    ``market`` must carry a ``date`` column on the *same timeframe* as ``df``.
    Only closed-candle information is used, and the merge is a left join on the
    candle timestamp, so no future bar can leak in.
    """
    out = df.copy()
    # De-duplicated so the left join can never multiply rows of ``df``.
    ref = market[["date", "close"]].drop_duplicates(subset="date").copy()
    log_ref = np.log(ref["close"].replace(0.0, np.nan))
    ref[f"f_{prefix}_ret_1"] = log_ref.diff(1)
    ref[f"f_{prefix}_ret_20"] = log_ref.diff(20)
    ref[f"f_{prefix}_vol_20"] = log_ref.diff(1).rolling(20, min_periods=20).std()
    ref_ema50 = ema(ref["close"], 50)
    ref[f"f_{prefix}_trend"] = ref["close"] / ref_ema50 - 1.0
    ref = ref.drop(columns=["close"])

    merged = out.merge(ref, on="date", how="left")
    merged.index = out.index

    # Rolling correlation of the pair against the market on 1-bar returns.
    pair_ret = np.log(merged["close"].replace(0.0, np.nan)).diff(1)
    market_ret = merged[f"f_{prefix}_ret_1"]
    merged[f"f_{prefix}_corr_50"] = pair_ret.rolling(50, min_periods=50).corr(market_ret)
    # Relative strength: is the pair outperforming the market?
    merged[f"f_{prefix}_rs_20"] = (
        np.log(merged["close"].replace(0.0, np.nan)).diff(20) - merged[f"f_{prefix}_ret_20"]
    )
    return merged


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All model input columns present in ``df`` (everything prefixed ``f_``)."""
    return sorted(col for col in df.columns if col.startswith("f_"))
