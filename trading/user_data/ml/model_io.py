"""Persisting and loading a trained model bundle.

A bundle is a plain directory:

    model.json   XGBoost booster (native format, portable across versions)
    meta.json    feature order, training window, calibration curve, threshold

Nothing is pickled on purpose: the strategy has to load this inside whatever
Freqtrade image the user is running, and pickles break across library versions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODEL_FILE = "model.json"
META_FILE = "meta.json"


@dataclass
class ModelBundle:
    """A booster plus everything needed to reproduce its inputs and outputs."""

    booster: Any
    features: list[str]
    threshold: float = 0.5
    calibration: dict[str, list[float]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated P(take-profit before stop-loss) for every row of ``df``.

        Rows with any missing feature return ``NaN`` — during warm-up the
        indicators have not converged, and a model asked about a row it could
        never have been trained on should abstain, not guess.
        """
        missing = [col for col in self.features if col not in df.columns]
        if missing:
            raise KeyError(f"Missing features for prediction: {missing}")

        matrix = df[self.features].to_numpy(dtype=np.float32, copy=True)
        matrix[~np.isfinite(matrix)] = np.nan
        usable = ~np.isnan(matrix).any(axis=1)

        out = np.full(len(df), np.nan)
        if usable.any():
            raw = np.asarray(self.booster.inplace_predict(matrix[usable])).astype(float)
            out[usable] = self.apply_calibration(raw)
        return out

    def apply_calibration(self, probabilities: np.ndarray) -> np.ndarray:
        """Map raw scores through the isotonic curve fitted on validation."""
        if not self.calibration:
            return probabilities
        return np.interp(
            probabilities,
            np.asarray(self.calibration["x"], dtype=float),
            np.asarray(self.calibration["y"], dtype=float),
        )


def save_bundle(
    directory: Path,
    booster: Any,
    features: list[str],
    threshold: float,
    calibration: dict[str, list[float]] | None,
    meta: dict[str, Any],
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(directory / MODEL_FILE))
    payload = {
        "features": list(features),
        "threshold": float(threshold),
        "calibration": calibration,
        **meta,
    }
    (directory / META_FILE).write_text(json.dumps(payload, indent=2, default=str))
    return directory


def load_bundle(directory: Path) -> ModelBundle:
    import xgboost as xgb  # imported lazily: only inference needs it

    directory = Path(directory)
    meta_path = directory / META_FILE
    model_path = directory / MODEL_FILE
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"No model bundle in {directory} (expected {MODEL_FILE} and {META_FILE}). "
            "Train one first:  python -m ml.train"
        )

    meta = json.loads(meta_path.read_text())
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    return ModelBundle(
        booster=booster,
        features=list(meta["features"]),
        threshold=float(meta.get("threshold", 0.5)),
        calibration=meta.get("calibration"),
        meta=meta,
    )
