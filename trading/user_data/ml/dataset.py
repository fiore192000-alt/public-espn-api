"""Loading Freqtrade OHLCV files and turning them into a supervised dataset."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import add_features, add_market_features, feature_columns
from .labeling import LABEL_COL, LABEL_META_COLS, triple_barrier_labels

OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
DATA_SUFFIXES = (".feather", ".parquet", ".json", ".json.gz")


def pair_to_filename(pair: str) -> str:
    """``BTC/USDT:USDT`` -> ``BTC_USDT_USDT`` (Freqtrade's on-disk naming)."""
    return pair.replace("/", "_").replace(":", "_")


def find_data_file(data_dir: Path, pair: str, timeframe: str) -> Path:
    """Locate the OHLCV file Freqtrade wrote for ``pair``/``timeframe``."""
    stem = f"{pair_to_filename(pair)}-{timeframe}"
    for suffix in DATA_SUFFIXES:
        candidate = data_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No OHLCV data for {pair} {timeframe} in {data_dir}. "
        f"Download it first:  freqtrade download-data --pairs {pair} --timeframes {timeframe}"
    )


def load_ohlcv(data_dir: Path, pair: str, timeframe: str) -> pd.DataFrame:
    """Read one Freqtrade OHLCV file into a tz-aware, de-duplicated dataframe."""
    path = find_data_file(Path(data_dir), pair, timeframe)

    if path.suffix == ".feather":
        df = pd.read_feather(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        opener = gzip.open if path.name.endswith(".json.gz") else open
        with opener(path, "rt") as handle:
            df = pd.DataFrame(json.load(handle), columns=OHLCV_COLUMNS)
        df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)

    df = df[OHLCV_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_dataset(
    df: pd.DataFrame,
    market: pd.DataFrame | None = None,
    horizon: int = 24,
    tp_atr: float = 2.0,
    sl_atr: float = 1.0,
) -> pd.DataFrame:
    """OHLCV -> features + triple-barrier labels, with unusable rows dropped."""
    out = add_features(df)
    if market is not None:
        out = add_market_features(out, market)
    labels = triple_barrier_labels(out, horizon=horizon, tp_atr=tp_atr, sl_atr=sl_atr)
    out = pd.concat([out, labels], axis=1)

    cols = feature_columns(out)
    out = out.replace([np.inf, -np.inf], np.nan)
    # A row is usable only when every feature and the label are known: the
    # warm-up head (indicators still converging) and the last ``horizon`` rows
    # (label not yet observable) both fall away here.
    return out.dropna(subset=cols + [LABEL_COL]).reset_index(drop=True)


def load_pair_dataset(
    data_dir: Path,
    pair: str,
    timeframe: str,
    market_pair: str | None = "BTC/USDT",
    horizon: int = 24,
    tp_atr: float = 2.0,
    sl_atr: float = 1.0,
) -> pd.DataFrame:
    """Full pipeline for a single pair, tagged with a ``pair`` column."""
    ohlcv = load_ohlcv(data_dir, pair, timeframe)
    market = None
    if market_pair:
        market = load_ohlcv(data_dir, market_pair, timeframe)
    dataset = build_dataset(
        ohlcv, market=market, horizon=horizon, tp_atr=tp_atr, sl_atr=sl_atr
    )
    dataset.insert(0, "pair", pair)
    return dataset


@dataclass(frozen=True)
class Split:
    """One time-ordered train/validation/test split of a dataset."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> str:
        def _line(name: str, part: pd.DataFrame) -> str:
            if part.empty:
                return f"  {name:<5} empty"
            return (
                f"  {name:<5} {len(part):>8,} rows  "
                f"{part['date'].min():%Y-%m-%d} -> {part['date'].max():%Y-%m-%d}  "
                f"positives {part[LABEL_COL].mean():.1%}"
            )

        return "\n".join(
            [_line("train", self.train), _line("valid", self.valid), _line("test", self.test)]
        )


def time_split(
    dataset: pd.DataFrame,
    train_end: str,
    test_start: str,
    valid_fraction: float = 0.15,
    purge_bars: int = 24,
) -> Split:
    """Split strictly by time, purging the rows whose labels straddle a cut.

    A triple-barrier label at row ``i`` peeks at candles up to ``i + horizon``.
    Rows within ``purge_bars`` of a boundary therefore describe an outcome that
    lives in the next block, so they are dropped rather than leaked across it.
    """
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be in (0, 1)")

    data = dataset.sort_values(["date", "pair"] if "pair" in dataset else "date")
    data = data.reset_index(drop=True)
    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    test_start_ts = pd.Timestamp(test_start, tz="UTC")
    if test_start_ts < train_end_ts:
        raise ValueError("test_start must not be before train_end")

    head = data[data["date"] < train_end_ts]
    test = data[data["date"] >= test_start_ts].reset_index(drop=True)

    # Validation is the tail of the in-sample block: it stays *before* the test
    # period, so the threshold picked on it never sees out-of-sample data.
    cut = int(len(head) * (1.0 - valid_fraction))
    train, valid = head.iloc[:cut], head.iloc[cut:]

    train = _purge_tail(train, purge_bars)
    valid = _purge_tail(valid, purge_bars)
    return Split(
        train=train.reset_index(drop=True),
        valid=valid.reset_index(drop=True),
        test=test,
    )


def _purge_tail(part: pd.DataFrame, purge_bars: int) -> pd.DataFrame:
    """Drop the last ``purge_bars`` rows *per pair* to stop label overlap."""
    if purge_bars <= 0 or part.empty:
        return part
    if "pair" not in part.columns:
        return part.iloc[: max(len(part) - purge_bars, 0)]
    keep = part.groupby("pair", sort=False).cumcount(ascending=False) >= purge_bars
    return part[keep]


def xy(dataset: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    return dataset[cols], dataset[LABEL_COL].astype(int)


__all__ = [
    "LABEL_COL",
    "LABEL_META_COLS",
    "Split",
    "build_dataset",
    "find_data_file",
    "load_ohlcv",
    "load_pair_dataset",
    "pair_to_filename",
    "time_split",
    "xy",
]
