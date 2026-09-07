"""Train the entry model.

    python -m ml.train --pairs BTC/USDT ETH/USDT SOL/USDT --timeframe 1h

Design rules that this script enforces, because they are the difference
between a backtest and a fantasy:

* the test period is never touched — not for early stopping, not for the
  threshold, not for calibration;
* rows whose label window crosses a split boundary are purged;
* the feature code is the same module the live strategy imports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import LABEL_COL, load_pair_dataset, time_split, xy
from .evaluate import (
    DEFAULT_FEE,
    fit_isotonic,
    format_metrics,
    ranking_metrics,
    select_threshold,
    signal_economics,
    threshold_table,
)
from .features import feature_columns
from .model_io import save_bundle

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "binance"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "entry"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the XGBoost entry model")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--pairs", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument(
        "--market-pair",
        default="BTC/USDT",
        help="Pair used for the market-context features ('none' to disable)",
    )
    parser.add_argument("--train-end", default="2024-01-01", help="Exclusive, UTC")
    parser.add_argument("--test-start", default="2024-01-01", help="Inclusive, UTC")
    parser.add_argument("--valid-fraction", type=float, default=0.15)
    parser.add_argument("--horizon", type=int, default=24, help="Barrier time limit, in candles")
    parser.add_argument("--tp-atr", type=float, default=2.0)
    parser.add_argument("--sl-atr", type=float, default=1.0)
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE, help="Per-side fee")
    parser.add_argument("--min-trades", type=int, default=200)
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def load_all(args: argparse.Namespace) -> pd.DataFrame:
    market_pair = None if args.market_pair.lower() == "none" else args.market_pair
    frames = []
    for pair in args.pairs:
        frame = load_pair_dataset(
            args.data_dir,
            pair,
            args.timeframe,
            market_pair=market_pair,
            horizon=args.horizon,
            tp_atr=args.tp_atr,
            sl_atr=args.sl_atr,
        )
        print(f"  {pair:<12} {len(frame):>8,} usable rows  "
              f"{frame['date'].min():%Y-%m-%d} -> {frame['date'].max():%Y-%m-%d}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_model(args: argparse.Namespace, positive_rate: float):
    import xgboost as xgb

    # Barrier labels are usually unbalanced; re-weight so the model does not
    # collapse onto the majority class.
    scale_pos_weight = (1.0 - positive_rate) / positive_rate if positive_rate > 0 else 1.0
    return xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.5,
        gamma=0.1,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=args.early_stopping,
        tree_method="hist",
        random_state=args.seed,
        n_jobs=-1,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("Loading data")
    dataset = load_all(args)

    print("\nSplitting (purge = horizon, so no label crosses a boundary)")
    split = time_split(
        dataset,
        train_end=args.train_end,
        test_start=args.test_start,
        valid_fraction=args.valid_fraction,
        purge_bars=args.horizon,
    )
    print(split.describe())
    if split.train.empty or split.valid.empty or split.test.empty:
        print("\nOne of the splits is empty — check --train-end / --test-start "
              "against the date ranges printed above.")
        return 1

    features = feature_columns(dataset)
    x_train, y_train = xy(split.train, features)
    x_valid, y_valid = xy(split.valid, features)
    x_test, y_test = xy(split.test, features)

    print(f"\nTraining on {len(features)} features")
    model = build_model(args, float(y_train.mean()))
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    booster = model.get_booster()
    best_iteration = getattr(model, "best_iteration", None)
    print(f"  best iteration: {best_iteration}")

    # Calibrate and pick the threshold on validation only.
    raw_valid = model.predict_proba(x_valid)[:, 1]
    calibration = fit_isotonic(y_valid.to_numpy(), raw_valid)
    cal_valid = np.interp(raw_valid, calibration["x"], calibration["y"])
    threshold, table = select_threshold(
        cal_valid,
        split.valid["label_ret"].to_numpy(),
        fee=args.fee,
        min_trades=args.min_trades,
    )

    raw_test = model.predict_proba(x_test)[:, 1]
    cal_test = np.interp(raw_test, calibration["x"], calibration["y"])

    print("\nRanking quality")
    print(" ", format_metrics("valid", ranking_metrics(y_valid.to_numpy(), cal_valid)))
    print(" ", format_metrics("test", ranking_metrics(y_test.to_numpy(), cal_test)))

    print(f"\nThreshold picked on validation: {threshold:.3f}")
    print("\nSignal economics (barrier trades, fees included, no slippage)")
    print(" ", format_metrics("valid", signal_economics(
        cal_valid, split.valid["label_ret"].to_numpy(), threshold, args.fee)))
    print(" ", format_metrics("test", signal_economics(
        cal_test, split.test["label_ret"].to_numpy(), threshold, args.fee)))

    print("\nTest-set threshold sweep (diagnostic only — do not tune on this)")
    test_table = threshold_table(cal_test, split.test["label_ret"].to_numpy(), fee=args.fee)
    print(test_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    importance = sorted(
        booster.get_score(importance_type="gain").items(), key=lambda kv: -kv[1]
    )[:15]
    print("\nTop features by gain")
    for name, gain in importance:
        print(f"  {name:<24} {gain:10.2f}")

    meta = {
        "pairs": args.pairs,
        "timeframe": args.timeframe,
        "market_pair": args.market_pair,
        "horizon": args.horizon,
        "tp_atr": args.tp_atr,
        "sl_atr": args.sl_atr,
        "fee": args.fee,
        "train_end": args.train_end,
        "test_start": args.test_start,
        "best_iteration": best_iteration,
        "train_rows": int(len(split.train)),
        "valid_rows": int(len(split.valid)),
        "test_rows": int(len(split.test)),
        "train_range": [str(split.train["date"].min()), str(split.train["date"].max())],
        "test_range": [str(split.test["date"].min()), str(split.test["date"].max())],
        "metrics_valid": ranking_metrics(y_valid.to_numpy(), cal_valid),
        "metrics_test": ranking_metrics(y_test.to_numpy(), cal_test),
        "economics_valid": signal_economics(
            cal_valid, split.valid["label_ret"].to_numpy(), threshold, args.fee),
        "economics_test": signal_economics(
            cal_test, split.test["label_ret"].to_numpy(), threshold, args.fee),
        "params": {
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "seed": args.seed,
        },
    }
    out = save_bundle(args.model_dir, booster, features, threshold, calibration, meta)
    (out / "threshold_valid.csv").write_text(table.to_csv(index=False))
    print(f"\nSaved model bundle to {out}")
    print(json.dumps({"threshold": threshold, "test": meta["economics_test"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
