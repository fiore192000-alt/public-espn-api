"""Contract tests for ``MLProbStrategy``.

Freqtrade is a heavy dependency (and on Windows it usually lives inside
Docker), so these tests drive the strategy against a minimal stand-in for the
few Freqtrade objects it touches.  That proves *our* logic — column contracts,
ATR-anchored exits, graceful behaviour with no model — and deliberately proves
nothing about Freqtrade itself; ``freqtrade backtesting`` is that check.
"""

from __future__ import annotations

import sys
import types
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv

STRATEGY_DIR = Path(__file__).resolve().parents[1] / "user_data" / "strategies"


class FakeTrade:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _install_freqtrade_stub() -> list[str]:
    """Register the slice of the Freqtrade API the strategy imports."""

    class IStrategy:
        def __init__(self, config):
            self.config = config

    class Parameter:
        def __init__(self, *args, default=None, **kwargs):
            self.value = default

    def stoploss_from_open(open_relative_stop, current_profit, is_short=False, leverage=1.0):
        # Same conversion Freqtrade performs: a stop expressed against the open
        # price, restated relative to the current rate.
        return max((open_relative_stop + 1) / (current_profit + 1) - 1, -1.0)

    modules = {
        "freqtrade": types.ModuleType("freqtrade"),
        "freqtrade.persistence": types.ModuleType("freqtrade.persistence"),
        "freqtrade.strategy": types.ModuleType("freqtrade.strategy"),
        "freqtrade.exchange": types.ModuleType("freqtrade.exchange"),
    }
    modules["freqtrade.persistence"].Trade = FakeTrade
    modules["freqtrade.strategy"].IStrategy = IStrategy
    modules["freqtrade.strategy"].DecimalParameter = Parameter
    modules["freqtrade.strategy"].IntParameter = Parameter
    modules["freqtrade.strategy"].stoploss_from_open = stoploss_from_open
    modules["freqtrade.exchange"].timeframe_to_seconds = lambda timeframe: 3600

    added = []
    for name, module in modules.items():
        if name not in sys.modules:
            sys.modules[name] = module
            added.append(name)
    return added


class FakeDataProvider:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames
        self.analyzed: dict[str, pd.DataFrame] = {}

    def get_pair_dataframe(self, pair: str, timeframe: str) -> pd.DataFrame:
        return self.frames.get(pair, pd.DataFrame())

    def get_analyzed_dataframe(self, pair: str, timeframe: str):
        return self.analyzed.get(pair, pd.DataFrame()), {}


@pytest.fixture(scope="module")
def strategy_module():
    added = _install_freqtrade_stub()
    if str(STRATEGY_DIR) not in sys.path:
        sys.path.insert(0, str(STRATEGY_DIR))
    import MLProbStrategy as module

    yield module

    for name in added:
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A model bundle plus the OHLCV frames it was trained on."""
    pytest.importorskip("xgboost")
    from ml.dataset import load_ohlcv
    from ml.train import main

    tmp = tmp_path_factory.mktemp("strategy")
    data_dir = tmp / "data" / "binance"
    data_dir.mkdir(parents=True)
    for pair, seed in (("BTC/USDT", 3), ("ETH/USDT", 5)):
        make_ohlcv(rows=6000, seed=seed).to_feather(
            data_dir / f"{pair.replace('/', '_')}-1h.feather"
        )

    model_dir = tmp / "entry"
    assert main(
        [
            "--data-dir", str(data_dir), "--model-dir", str(model_dir),
            "--pairs", "BTC/USDT", "ETH/USDT", "--train-end", "2020-06-01",
            "--test-start", "2020-06-01", "--n-estimators", "50",
            "--early-stopping", "15", "--min-trades", "20",
        ]
    ) == 0
    frames = {pair: load_ohlcv(data_dir, pair, "1h") for pair in ("BTC/USDT", "ETH/USDT")}
    return model_dir, frames


@pytest.fixture
def strategy(strategy_module, trained, monkeypatch):
    model_dir, frames = trained
    monkeypatch.setattr(strategy_module, "MODEL_DIR", model_dir)
    instance = strategy_module.MLProbStrategy({"runmode": "backtest"})
    instance.dp = FakeDataProvider(frames)
    instance.bot_start()
    return instance


@pytest.fixture
def analyzed(strategy, trained):
    _, frames = trained
    metadata = {"pair": "ETH/USDT"}
    df = strategy.populate_indicators(frames["ETH/USDT"].copy(), metadata)
    df = strategy.populate_entry_trend(df, metadata)
    df = strategy.populate_exit_trend(df, metadata)
    strategy.dp.analyzed["ETH/USDT"] = df
    return df


def test_bot_start_adopts_the_trained_settings(strategy) -> None:
    assert 0.0 < strategy.entry_threshold.value < 1.0
    assert strategy.tp_atr > strategy.sl_atr > 0
    assert strategy.informative_pairs() == [("BTC/USDT", "1h")]


def test_signals_are_produced_and_warmup_abstains(analyzed) -> None:
    assert analyzed["ml_prob"].iloc[:100].isna().all()
    assert analyzed["ml_prob"].notna().any()
    assert analyzed["enter_long"].fillna(0).sum() > 0


def test_entries_respect_threshold_and_minimum_edge(strategy, analyzed) -> None:
    entries = analyzed[analyzed["enter_long"].fillna(0) == 1]
    assert (entries["ml_prob"] >= strategy.entry_threshold.value).all()
    assert (entries["tp_distance_pct"] >= strategy.min_edge_pct.value).all()
    assert (entries["enter_tag"] == "ml_prob").all()


def _first_trade(strategy, analyzed, offset: int = 10) -> FakeTrade:
    index = analyzed.index[analyzed["enter_long"].fillna(0) == 1][offset]
    fill = analyzed.loc[index + 1]  # Freqtrade fills at the next candle's open
    return FakeTrade(
        id=1,
        pair="ETH/USDT",
        open_rate=float(fill["open"]),
        open_date_utc=fill["date"].to_pydatetime(),
        leverage=1.0,
    )


def test_entry_atr_reads_the_signal_candle(strategy, analyzed) -> None:
    index = analyzed.index[analyzed["enter_long"].fillna(0) == 1][10]
    trade = _first_trade(strategy, analyzed)
    assert strategy.entry_atr("ETH/USDT", trade) == pytest.approx(analyzed.loc[index, "atr"])


def test_stoploss_sits_one_atr_below_the_entry(strategy, analyzed) -> None:
    trade = _first_trade(strategy, analyzed)
    atr_value = strategy.entry_atr("ETH/USDT", trade)

    ratio = strategy.custom_stoploss(
        "ETH/USDT", trade, trade.open_date_utc, trade.open_rate, 0.0
    )
    assert trade.open_rate * (1 + ratio) == pytest.approx(
        trade.open_rate - strategy.sl_atr * atr_value
    )


def test_exits_fire_at_the_target_and_at_the_horizon(strategy, analyzed) -> None:
    trade = _first_trade(strategy, analyzed)
    atr_value = strategy.entry_atr("ETH/USDT", trade)
    now = trade.open_date_utc

    assert strategy.custom_exit("ETH/USDT", trade, now, trade.open_rate, 0.0) is None
    target = trade.open_rate + strategy.tp_atr * atr_value
    assert strategy.custom_exit("ETH/USDT", trade, now, target, 0.05) == "atr_take_profit"

    expired = now + timedelta(hours=strategy.max_hold_candles.value)
    assert (
        strategy.custom_exit("ETH/USDT", trade, expired, trade.open_rate, 0.0)
        == "horizon_timeout"
    )


def test_without_a_model_the_strategy_stays_flat(strategy_module, trained, monkeypatch) -> None:
    """A missing bundle must disable trading, not crash the bot."""
    _, frames = trained
    monkeypatch.setattr(strategy_module, "MODEL_DIR", Path("/nonexistent/model"))
    instance = strategy_module.MLProbStrategy({"runmode": "backtest"})
    instance.dp = FakeDataProvider(frames)
    instance.bot_start()

    metadata = {"pair": "ETH/USDT"}
    df = instance.populate_indicators(frames["ETH/USDT"].copy(), metadata)
    df = instance.populate_entry_trend(df, metadata)
    assert instance.bundle is None
    assert df["ml_prob"].isna().all()
    assert "enter_long" not in df.columns or df["enter_long"].fillna(0).sum() == 0
