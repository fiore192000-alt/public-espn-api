"""Removing the bookmaker's margin from a set of prices.

A quoted book always implies more than 100% probability; the excess is the
margin. Recovering what the book actually believes means deciding *how* that
excess is distributed across the selections, and the choice is not cosmetic — it
moves the implied probability of a longshot far more than that of a favourite,
which is exactly where a model's disagreement with the price tends to live.

Three methods, cheapest assumption first:

``proportional``
    Scale every implied probability by the same factor. Assumes the margin is
    spread evenly, which is known to be wrong: books load more margin onto
    longshots, so this method overstates their true probability.

``power``
    Raise each implied probability to a common power. Bends the correction so
    longshots lose more than favourites.

``shin``
    Shin's model, which derives the margin from an assumed share of
    insider-informed money. It is the most principled of the three and usually
    the best-calibrated on football 1X2 books.

Which one is actually best is an empirical question about a given book and league,
so `measure_devig_methods` scores them against real results rather than assuming.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PROPORTIONAL = "proportional"
POWER = "power"
SHIN = "shin"
METHODS = (PROPORTIONAL, POWER, SHIN)

_TOLERANCE = 1e-10
_MAX_ITERATIONS = 100
_MIN_PROBABILITY = 1e-9


class DevigError(ValueError):
    """Raised when a set of prices cannot be devigged."""


def implied(prices: dict[str, float]) -> dict[str, float]:
    """Raw implied probabilities, margin still included."""
    usable = {
        selection: 1.0 / price for selection, price in prices.items() if price and price > 1.0
    }
    if len(usable) < 2:
        raise DevigError("At least two priced selections are needed to remove a margin.")
    return usable


def overround(prices: dict[str, float]) -> float:
    """Total implied probability. 1.0 is a margin-free book."""
    return sum(implied(prices).values())


def margin(prices: dict[str, float]) -> float:
    """The bookmaker's edge as a share of the book."""
    total = overround(prices)
    return (total - 1.0) / total if total > 0 else 0.0


def proportional(prices: dict[str, float]) -> dict[str, float]:
    """Scale all implied probabilities down by the same factor."""
    raw = implied(prices)
    total = sum(raw.values())
    return {selection: value / total for selection, value in raw.items()}


def power(prices: dict[str, float]) -> dict[str, float]:
    """Raise implied probabilities to the common power that makes them sum to one."""
    raw = implied(prices)
    total = sum(raw.values())
    if abs(total - 1.0) < _TOLERANCE:
        return dict(raw)

    # The sum decreases as the exponent grows, so bisect on the exponent.
    low, high = 1e-6, 100.0
    exponent = 1.0
    for _ in range(_MAX_ITERATIONS):
        exponent = 0.5 * (low + high)
        current = sum(value**exponent for value in raw.values())
        if abs(current - 1.0) < _TOLERANCE:
            break
        if current > 1.0:
            low = exponent
        else:
            high = exponent

    adjusted = {selection: value**exponent for selection, value in raw.items()}
    normaliser = sum(adjusted.values())
    return {selection: value / normaliser for selection, value in adjusted.items()}


def _shin_probabilities(raw: dict[str, float], total: float, z: float) -> dict[str, float]:
    if z >= 1.0:
        raise DevigError("Degenerate insider share.")
    result = {}
    for selection, value in raw.items():
        discriminant = z * z + 4.0 * (1.0 - z) * value * value / total
        result[selection] = (math.sqrt(max(discriminant, 0.0)) - z) / (2.0 * (1.0 - z))
    return result


def shin(prices: dict[str, float]) -> dict[str, float]:
    """Shin's method: solve for the insider share that makes the book sum to one."""
    raw = implied(prices)
    total = sum(raw.values())
    if total <= 1.0:
        return proportional(prices)

    low, high = 0.0, 0.99
    z = 0.0
    for _ in range(_MAX_ITERATIONS):
        z = 0.5 * (low + high)
        current = sum(_shin_probabilities(raw, total, z).values())
        if abs(current - 1.0) < _TOLERANCE:
            break
        # A larger insider share shrinks the recovered probabilities.
        if current > 1.0:
            low = z
        else:
            high = z

    probabilities = _shin_probabilities(raw, total, z)
    normaliser = sum(probabilities.values())
    if normaliser <= 0:
        raise DevigError("Shin devig produced no usable probabilities.")
    return {
        selection: max(value / normaliser, _MIN_PROBABILITY)
        for selection, value in probabilities.items()
    }


def remove_margin(prices: dict[str, float], method: str = SHIN) -> dict[str, float]:
    """Devig a complete market with the named method."""
    if method == PROPORTIONAL:
        return proportional(prices)
    if method == POWER:
        return power(prices)
    if method == SHIN:
        return shin(prices)
    raise DevigError(f"Unknown devig method {method!r}; expected one of {', '.join(METHODS)}.")


@dataclass
class MethodScore:
    """How well one devig method's probabilities matched what happened."""

    method: str
    matches: int
    log_loss: float
    brier: float

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "matches": self.matches,
            "log_loss": round(self.log_loss, 4),
            "brier": round(self.brier, 4),
        }


def measure_devig_methods(
    books: list[tuple[dict[str, float], str]],
    methods: tuple[str, ...] = METHODS,
) -> list[MethodScore]:
    """Score each method on real books and outcomes, best first.

    ``books`` pairs a complete set of prices with the selection that actually won.
    Scoring on results is the only way to choose between methods: each makes a
    different assumption about how the margin sits on the book, and only the data
    knows which assumption fits.
    """
    scores: list[MethodScore] = []

    for method in methods:
        log_loss = brier = 0.0
        counted = 0
        for prices, actual in books:
            try:
                fair = remove_margin(prices, method)
            except DevigError:
                continue
            if actual not in fair:
                continue
            counted += 1
            log_loss -= math.log(max(fair[actual], _MIN_PROBABILITY))
            for selection, probability in fair.items():
                brier += (probability - (1.0 if selection == actual else 0.0)) ** 2

        if counted:
            scores.append(
                MethodScore(
                    method=method,
                    matches=counted,
                    log_loss=log_loss / counted,
                    brier=brier / counted,
                )
            )

    scores.sort(key=lambda score: score.log_loss)
    return scores
