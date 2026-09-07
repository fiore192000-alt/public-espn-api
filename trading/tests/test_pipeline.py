"""End-to-end: data on disk -> dataset -> split -> model bundle -> prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from ml.dataset import build_dataset, load_ohlcv, load_pair_dataset, time_split
from ml.evaluate import select_threshold, signal_economics
from ml.features import feature_columns
from ml.model_io import load_bundle


@pytest.fixture
def data_dir(tmp_path):
    """A Freqtrade-shaped data directory with two pairs."""
    directory = tmp_path / "data" / "binance"
    directory.mkdir(parents=True)
    for pair, seed in (("BTC/USDT", 3), ("ETH/USDT", 5)):
        name = pair.replace("/", "_")
        make_ohlcv(rows=8000, seed=seed).to_feather(directory / f"{name}-1h.feather")
    return directory


def test_load_ohlcv_reads_freqtrade_layout(data_dir) -> None:
    df = load_ohlcv(data_dir, "BTC/USDT", "1h")
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert df["date"].is_monotonic_increasing
    assert df["date"].dt.tz is not None


def test_missing_data_names_the_download_command(data_dir) -> None:
    with pytest.raises(FileNotFoundError, match="download-data"):
        load_ohlcv(data_dir, "DOGE/USDT", "1h")


def test_dataset_drops_warmup_and_unlabelled_tail(data_dir) -> None:
    raw = load_ohlcv(data_dir, "BTC/USDT", "1h")
    dataset = build_dataset(raw, horizon=24)
    assert not dataset.empty
    assert len(dataset) < len(raw)
    assert dataset[feature_columns(dataset)].notna().all().all()
    assert dataset["label"].isin([0.0, 1.0]).all()
    # The last labelled candle must leave room for its own barrier window.
    assert dataset["date"].max() <= raw["date"].max() - pd.Timedelta(hours=24)


def test_split_is_ordered_and_purged(data_dir) -> None:
    frames = [
        load_pair_dataset(data_dir, pair, "1h", market_pair="BTC/USDT", horizon=24)
        for pair in ("BTC/USDT", "ETH/USDT")
    ]
    dataset = pd.concat(frames, ignore_index=True)
    split = time_split(dataset, train_end="2020-08-01", test_start="2020-08-01", purge_bars=24)

    assert not split.train.empty and not split.valid.empty and not split.test.empty
    assert split.train["date"].max() < split.valid["date"].min()
    assert split.valid["date"].max() < split.test["date"].min()
    # Purging must leave a gap of at least the label horizon at each boundary.
    gap = split.valid["date"].min() - split.train["date"].max()
    assert gap >= pd.Timedelta(hours=24)


def test_training_produces_a_loadable_bundle(data_dir, tmp_path) -> None:
    pytest.importorskip("xgboost")
    from ml.train import main

    model_dir = tmp_path / "model"
    exit_code = main(
        [
            "--data-dir", str(data_dir),
            "--model-dir", str(model_dir),
            "--pairs", "BTC/USDT", "ETH/USDT",
            "--timeframe", "1h",
            "--train-end", "2020-08-01",
            "--test-start", "2020-08-01",
            "--n-estimators", "60",
            "--early-stopping", "20",
            "--min-trades", "20",
        ]
    )
    assert exit_code == 0

    bundle = load_bundle(model_dir)
    assert bundle.features
    assert 0.0 < bundle.threshold < 1.0
    assert bundle.meta["horizon"] == 24
    assert bundle.meta["test_rows"] > 0

    # The bundle must score a raw OHLCV frame the same way training did —
    # including the market-context features it was trained with.
    raw = load_ohlcv(data_dir, "BTC/USDT", "1h")
    scored = build_dataset(raw, market=load_ohlcv(data_dir, "BTC/USDT", "1h"), horizon=24)
    probabilities = bundle.predict_proba(scored)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_predict_proba_abstains_during_warmup(data_dir, tmp_path) -> None:
    pytest.importorskip("xgboost")
    from ml.features import add_features
    from ml.train import main

    model_dir = tmp_path / "model"
    assert main(
        [
            "--data-dir", str(data_dir), "--model-dir", str(model_dir),
            "--pairs", "BTC/USDT", "--market-pair", "none", "--timeframe", "1h",
            "--train-end", "2020-08-01", "--test-start", "2020-08-01",
            "--n-estimators", "40", "--early-stopping", "10", "--min-trades", "20",
        ]
    ) == 0

    bundle = load_bundle(model_dir)
    featured = add_features(load_ohlcv(data_dir, "BTC/USDT", "1h"))
    probabilities = bundle.predict_proba(featured)
    # Warm-up candles have undefined indicators: the model must return NaN
    # there rather than a confident-looking guess.
    assert np.isnan(probabilities[:50]).all()
    assert np.isfinite(probabilities[-50:]).all()


def test_threshold_selection_prefers_profitable_signals() -> None:
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(0.3, 0.9, 5000)
    # Returns that genuinely improve with the score.
    label_ret = (probabilities - 0.6) * 0.1 + rng.normal(0.0, 0.01, 5000)

    threshold, table = select_threshold(probabilities, label_ret, min_trades=100)
    assert threshold > 0.5
    assert (table["trades"] > 0).any()
    assert signal_economics(probabilities, label_ret, threshold)["expectancy"] > 0
