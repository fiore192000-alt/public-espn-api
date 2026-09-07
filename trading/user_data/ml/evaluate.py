"""Calibration, threshold selection and the metrics that decide go / no-go.

Accuracy is close to useless here: the model can be right 55% of the time and
still lose money, or right 40% of the time and print.  Everything below is
framed in terms of the money the signal would have made.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Binance spot taker fee, applied on entry and on exit.
DEFAULT_FEE = 0.001


def fit_isotonic(y_true: np.ndarray, probabilities: np.ndarray, grid: int = 201) -> dict:
    """Fit an isotonic calibration curve, serialised as a lookup table.

    Gradient boosting scores are ranks, not probabilities — a raw 0.70 does not
    mean "wins 70% of the time".  Since the strategy gates on an absolute
    number, the score has to be mapped onto observed frequencies first.
    """
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(probabilities, y_true)
    x = np.linspace(0.0, 1.0, grid)
    return {"type": "isotonic", "x": x.tolist(), "y": model.predict(x).tolist()}


def ranking_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return {"samples": int(len(y_true)), "base_rate": float(np.mean(y_true))}
    return {
        "samples": int(len(y_true)),
        "base_rate": float(np.mean(y_true)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "avg_precision": float(average_precision_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, np.clip(probabilities, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }


def signal_economics(
    probabilities: np.ndarray,
    label_ret: np.ndarray,
    threshold: float,
    fee: float = DEFAULT_FEE,
) -> dict[str, float]:
    """What the rule "enter whenever p >= threshold" would have paid.

    ``label_ret`` is the realised return of the barrier trade, so this is a
    signal-quality estimate, not a backtest: it ignores position sizing, slots,
    slippage and overlapping trades.  Freqtrade's backtest is the real check.
    """
    selected = np.asarray(probabilities) >= threshold
    count = int(selected.sum())
    if count == 0:
        return {"threshold": float(threshold), "trades": 0}

    # Round-trip cost: taker in, taker out.
    returns = np.asarray(label_ret, dtype=float)[selected] - 2.0 * fee
    wins, losses = returns[returns > 0], returns[returns <= 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    return {
        "threshold": float(threshold),
        "trades": count,
        "win_rate": float((returns > 0).mean()),
        "expectancy": float(returns.mean()),
        "total_return": float(returns.sum()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        # Per-trade Sharpe-like ratio; not annualised, only useful for ranking.
        "return_over_risk": float(returns.mean() / returns.std()) if returns.std() > 0 else 0.0,
    }


def threshold_table(
    probabilities: np.ndarray,
    label_ret: np.ndarray,
    fee: float = DEFAULT_FEE,
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    if grid is None:
        grid = np.round(np.arange(0.40, 0.91, 0.025), 4)
    rows = [signal_economics(probabilities, label_ret, t, fee) for t in grid]
    return pd.DataFrame(rows).fillna(0.0)


def select_threshold(
    probabilities: np.ndarray,
    label_ret: np.ndarray,
    fee: float = DEFAULT_FEE,
    min_trades: int = 200,
    grid: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """Pick the entry threshold on *validation* data, never on the test set.

    Chosen by best expectancy among thresholds that still fire often enough to
    be measurable — a threshold with twelve trades is noise wearing a suit.
    """
    table = threshold_table(probabilities, label_ret, fee=fee, grid=grid)
    liquid = table[table["trades"] >= min_trades]
    if liquid.empty:
        # Nothing clears the bar; fall back to the busiest threshold available
        # so the caller still gets a usable (if weak) model.
        liquid = table[table["trades"] > 0]
    if liquid.empty:
        return 0.5, table
    best = liquid.loc[liquid["expectancy"].idxmax()]
    return float(best["threshold"]), table


def format_metrics(name: str, metrics: dict[str, float]) -> str:
    body = "  ".join(
        f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in metrics.items()
    )
    return f"{name:<12} {body}"
