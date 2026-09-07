"""Freqtrade strategy driven by the XGBoost entry model.

The model answers one question per candle — *will a +tp_atr move arrive before
a -sl_atr move?* — and this strategy trades that answer with the same barriers
it was trained on, so backtest results and training metrics describe the same
thing.

Nothing here trains or refits: the bundle in ``user_data/models/entry`` is
frozen at ``python -m ml.train`` time, on data strictly older than the period
you backtest.  If the bundle is missing the strategy simply never enters.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IStrategy, IntParameter, stoploss_from_open

# user_data/ is the package root for the shared ML code; the strategies folder
# is what Freqtrade puts on sys.path, so the parent has to be added by hand.
USER_DATA_DIR = Path(__file__).resolve().parents[1]
if str(USER_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(USER_DATA_DIR))

from ml.features import STARTUP_CANDLES, add_features, add_market_features  # noqa: E402
from ml.model_io import load_bundle  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_DIR = USER_DATA_DIR / "models" / "entry"


class MLProbStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Exits are ATR-based and live in custom_exit / custom_stoploss, so the
    # fixed ladders are pushed out of the way rather than deleted (Freqtrade
    # requires both attributes to exist).
    minimal_roi = {"0": 10.0}
    stoploss = -0.10
    use_custom_stoploss = True
    trailing_stop = False

    startup_candle_count: int = STARTUP_CANDLES

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # --- tunables (hyperoptable, but sane out of the box) ------------------- #
    # Default is overwritten in bot_start() by the threshold the trainer picked
    # on validation data, unless you hyperopt it.
    entry_threshold = DecimalParameter(0.50, 0.85, default=0.60, decimals=3, space="buy")
    # The take-profit distance has to be worth the round trip: below this the
    # fees and the spread eat the whole edge.
    min_edge_pct = DecimalParameter(0.002, 0.020, default=0.006, decimals=4, space="buy")
    # Hard time limit, mirroring the label's third barrier.
    max_hold_candles = IntParameter(6, 96, default=24, space="sell")

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.bundle = None
        self.tp_atr = 2.0
        self.sl_atr = 1.0
        self.market_pair: Optional[str] = "BTC/USDT"
        # trade_id -> ATR at the signal candle, so exits stay anchored to the
        # volatility that justified the entry.
        self._entry_atr: dict[int, float] = {}

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def bot_start(self, **kwargs) -> None:
        try:
            self.bundle = load_bundle(MODEL_DIR)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            logger.error(
                "No usable model bundle in %s (%s). The strategy will not enter any "
                "trade until you run:  python -m ml.train", MODEL_DIR, exc
            )
            return

        meta = self.bundle.meta
        self.tp_atr = float(meta.get("tp_atr", self.tp_atr))
        self.sl_atr = float(meta.get("sl_atr", self.sl_atr))
        market_pair = meta.get("market_pair", self.market_pair)
        self.market_pair = None if str(market_pair).lower() == "none" else market_pair

        horizon = int(meta.get("horizon", self.max_hold_candles.value))
        # Trust the values the trainer chose — except under hyperopt, which owns
        # the parameters it is searching over.
        if not self._is_hyperopt():
            self.entry_threshold.value = float(self.bundle.threshold)
            self.max_hold_candles.value = horizon

        trained_timeframe = meta.get("timeframe")
        if trained_timeframe and trained_timeframe != self.timeframe:
            logger.warning(
                "Model was trained on %s but the strategy runs on %s — retrain or "
                "change the timeframe, the features are not transferable.",
                trained_timeframe, self.timeframe,
            )
        logger.info(
            "Loaded model: threshold=%.3f tp=%.1f*ATR sl=%.1f*ATR horizon=%d features=%d",
            self.entry_threshold.value, self.tp_atr, self.sl_atr, horizon,
            len(self.bundle.features),
        )

    def _is_hyperopt(self) -> bool:
        runmode = self.config.get("runmode")
        return str(getattr(runmode, "value", runmode)).lower() == "hyperopt"

    def informative_pairs(self):
        if not self.market_pair:
            return []
        return [(self.market_pair, self.timeframe)]

    # ------------------------------------------------------------------ #
    # signal generation
    # ------------------------------------------------------------------ #
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_features(dataframe)

        if self.market_pair:
            market = self.dp.get_pair_dataframe(self.market_pair, self.timeframe)
            if market is not None and not market.empty:
                # Same timeframe, joined on the candle timestamp: the reference
                # candle closes exactly when ours does, and entries fill on the
                # next open, so this reads no future information.
                dataframe = add_market_features(dataframe, market)
            else:
                logger.warning("No data for market pair %s", self.market_pair)

        # Expected take-profit distance as a fraction of price — the edge the
        # trade has to clear before costs.  Set before any early return so the
        # entry logic always has the column, model or no model.
        dataframe["tp_distance_pct"] = self.tp_atr * dataframe["atr"] / dataframe["close"]
        dataframe["ml_prob"] = float("nan")

        if self.bundle is None:
            return dataframe

        missing = [col for col in self.bundle.features if col not in dataframe.columns]
        if missing:
            logger.error("Feature mismatch for %s: %s missing", metadata["pair"], missing)
            return dataframe

        dataframe["ml_prob"] = self.bundle.predict_proba(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["ml_prob"] >= self.entry_threshold.value)
            & (dataframe["tp_distance_pct"] >= self.min_edge_pct.value)
            & (dataframe["atr"] > 0)
            & (dataframe["volume"] > 0),
            ["enter_long", "enter_tag"],
        ] = (1, "ml_prob")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exits are handled by custom_exit / custom_stoploss.
        return dataframe

    # ------------------------------------------------------------------ #
    # exits
    # ------------------------------------------------------------------ #
    def entry_atr(self, pair: str, trade: Trade) -> Optional[float]:
        """ATR of the candle that produced the entry signal."""
        cached = self._entry_atr.get(trade.id)
        if cached is not None:
            return cached

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        # The signal candle is the last one that closed strictly before the fill.
        candles = dataframe.loc[dataframe["date"] < trade.open_date_utc]
        if candles.empty:
            return None
        value = candles.iloc[-1].get("atr")
        if value is None or not float(value) > 0:
            return None
        value = float(value)
        if trade.id is not None:
            self._entry_atr[trade.id] = value
        return value

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool = False,
        **kwargs,
    ) -> Optional[float]:
        atr_value = self.entry_atr(pair, trade)
        if atr_value is None or trade.open_rate <= 0:
            return None  # keep whatever stoploss is already in force

        stop_from_open = -(self.sl_atr * atr_value) / trade.open_rate
        return stoploss_from_open(
            stop_from_open, current_profit, is_short=False, leverage=trade.leverage or 1.0
        )

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        atr_value = self.entry_atr(pair, trade)
        if atr_value is not None and current_rate >= trade.open_rate + self.tp_atr * atr_value:
            return "atr_take_profit"

        # Third barrier: if neither target was reached in time, the reason for
        # the trade has expired — close it and free the slot.
        held = (current_time - trade.open_date_utc).total_seconds()
        limit = self.max_hold_candles.value * self.timeframe_to_seconds()
        if limit > 0 and held >= limit:
            return "horizon_timeout"
        return None

    def timeframe_to_seconds(self) -> int:
        from freqtrade.exchange import timeframe_to_seconds

        return timeframe_to_seconds(self.timeframe)

    # ------------------------------------------------------------------ #
    # plotting
    # ------------------------------------------------------------------ #
    plot_config = {
        "main_plot": {},
        "subplots": {
            "model": {"ml_prob": {"color": "blue"}},
            "volatility": {"tp_distance_pct": {"color": "orange"}},
        },
    }
