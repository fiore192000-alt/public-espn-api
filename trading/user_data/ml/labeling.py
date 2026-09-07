"""Target construction.

The model answers one question: *if I enter at the next candle's open, does
price reach my take-profit before my stop-loss?*  That is the triple-barrier
label (upper barrier, lower barrier, time limit) — it matches what the strategy
actually does with the prediction, which a plain "will the close be higher in N
bars" target does not.

Barriers are sized in ATR multiples, so the label means the same thing in a
quiet market and in a violent one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_COL = "label"
LABEL_META_COLS = ("label_hit", "label_bars", "label_ret")


def triple_barrier_labels(
    df: pd.DataFrame,
    horizon: int = 24,
    tp_atr: float = 2.0,
    sl_atr: float = 1.0,
    atr_col: str = "atr",
) -> pd.DataFrame:
    """Label every row of ``df`` with the triple-barrier outcome.

    Entry is assumed at ``open`` of the *next* candle — never at the close of
    the signal candle, which would not be executable — and barriers are checked
    against the high/low of candles ``i+1 .. i+horizon``.

    Returns a frame indexed like ``df`` with:

    ``label``      1 = take-profit hit first, 0 = stop-loss first or timeout.
    ``label_hit``  ``"tp"`` / ``"sl"`` / ``"timeout"``.
    ``label_bars`` bars until the barrier was touched (``horizon`` on timeout).
    ``label_ret``  realised return of the trade the label describes.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    n = len(df)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    atr_v = df[atr_col].to_numpy(dtype=float)

    label = np.full(n, np.nan)
    hit = np.full(n, "", dtype=object)
    bars = np.full(n, np.nan)
    ret = np.full(n, np.nan)

    # Row i needs candles i+1 .. i+horizon to exist, so the last horizon
    # rows stay unlabelled — their outcome has not happened yet.
    for i in range(max(n - horizon, 0)):
        entry = open_[i + 1]
        band = atr_v[i]
        if not np.isfinite(entry) or not np.isfinite(band) or band <= 0.0:
            continue

        tp = entry + tp_atr * band
        sl = entry - sl_atr * band
        window = slice(i + 1, i + 1 + horizon)
        hi, lo = high[window], low[window]

        up_hits = np.flatnonzero(hi >= tp)
        dn_hits = np.flatnonzero(lo <= sl)
        first_up = up_hits[0] if up_hits.size else np.inf
        first_dn = dn_hits[0] if dn_hits.size else np.inf

        if first_up == np.inf and first_dn == np.inf:
            label[i], hit[i], bars[i] = 0.0, "timeout", horizon
            ret[i] = close[i + horizon] / entry - 1.0
        elif first_up < first_dn:
            label[i], hit[i], bars[i] = 1.0, "tp", first_up + 1
            ret[i] = tp / entry - 1.0
        elif first_dn < first_up:
            label[i], hit[i], bars[i] = 0.0, "sl", first_dn + 1
            ret[i] = sl / entry - 1.0
        else:
            # Both barriers inside the same candle: we cannot know the order
            # from OHLC alone, so assume the pessimistic outcome.
            label[i], hit[i], bars[i] = 0.0, "sl", first_dn + 1
            ret[i] = sl / entry - 1.0

    return pd.DataFrame(
        {
            LABEL_COL: label,
            "label_hit": pd.Series(hit, index=df.index).replace("", np.nan),
            "label_bars": bars,
            "label_ret": ret,
        },
        index=df.index,
    )


def label_summary(labels: pd.DataFrame) -> dict[str, float]:
    """Sanity numbers to print before training on a fresh label set."""
    valid = labels[LABEL_COL].notna()
    total = int(valid.sum())
    if total == 0:
        return {"samples": 0}
    counts = labels.loc[valid, "label_hit"].value_counts()
    return {
        "samples": total,
        "positive_rate": float(labels.loc[valid, LABEL_COL].mean()),
        "tp": int(counts.get("tp", 0)),
        "sl": int(counts.get("sl", 0)),
        "timeout": int(counts.get("timeout", 0)),
        "mean_bars_to_barrier": float(labels.loc[valid, "label_bars"].mean()),
        "mean_label_ret": float(labels.loc[valid, "label_ret"].mean()),
    }
