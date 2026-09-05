"""Does a model know anything the market price does not?

Predicting football well and being useful for betting are different things. A
model can be accurate and still worthless if everything it knows is already in
the price. The test that separates the two is forecast combination: pool the
market's probabilities with the candidate's, fit the pooling weights, and ask
whether the combination beats the market **alone** on matches neither was fitted
on.

The pool is logarithmic — the standard form for combining probability forecasts::

    p(outcome) proportional to  prod_m  p_m(outcome) ** w_m

with the weights fitted by maximum likelihood. Read them directly: a candidate
whose weight lands near zero is being told by the data that it adds nothing.

Weights only, no per-outcome intercepts. An intercept would let the pool fix a
global bias in the market and show an "improvement" that has nothing to do with
the candidate's information, which is exactly the illusion this module exists to
avoid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

OUTCOMES = ("home", "draw", "away")
_PROBABILITY_FLOOR = 1e-9
_MAX_ITERATIONS = 500
_INITIAL_STEP = 0.5
_TOLERANCE = 1e-9

# Below this improvement in held-out log-loss, a combination is not meaningfully
# better than the market on its own.
MEANINGFUL_IMPROVEMENT = 0.001


@dataclass
class Sample:
    """One match with each source's probabilities and what actually happened."""

    probabilities: dict[str, dict[str, float]]
    actual: str

    def log_probabilities(self, source: str) -> dict[str, float]:
        source_probabilities = self.probabilities[source]
        return {
            outcome: math.log(max(source_probabilities[outcome], _PROBABILITY_FLOOR))
            for outcome in OUTCOMES
        }


@dataclass
class PoolFit:
    """Fitted pooling weights and how they scored."""

    sources: list[str]
    weights: dict[str, float]
    train_matches: int
    holdout_matches: int
    train_log_loss: float
    holdout_log_loss: float
    converged: bool

    def to_dict(self) -> dict:
        return {
            "sources": self.sources,
            "weights": {name: round(weight, 4) for name, weight in self.weights.items()},
            "train_matches": self.train_matches,
            "holdout_matches": self.holdout_matches,
            "train_log_loss": round(self.train_log_loss, 4),
            "holdout_log_loss": round(self.holdout_log_loss, 4),
            "converged": self.converged,
        }


@dataclass
class IncrementalReport:
    """Whether each candidate adds information the market lacks."""

    market_log_loss: float | None = None
    holdout_matches: int = 0
    combinations: list[PoolFit] = field(default_factory=list)
    verdicts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "market_only_log_loss": (
                round(self.market_log_loss, 4) if self.market_log_loss is not None else None
            ),
            "holdout_matches": self.holdout_matches,
            "combinations": [fit.to_dict() for fit in self.combinations],
            "verdicts": self.verdicts,
        }


def pooled_probabilities(
    sample: Sample,
    weights: dict[str, float],
) -> dict[str, float]:
    """Combine the sources for one match under the given weights."""
    scores = {}
    for outcome in OUTCOMES:
        total = 0.0
        for source, weight in weights.items():
            total += weight * math.log(
                max(sample.probabilities[source][outcome], _PROBABILITY_FLOOR)
            )
        scores[outcome] = total

    largest = max(scores.values())
    exponentials = {outcome: math.exp(score - largest) for outcome, score in scores.items()}
    normaliser = sum(exponentials.values())
    return {outcome: value / normaliser for outcome, value in exponentials.items()}


def log_loss(samples: list[Sample], weights: dict[str, float]) -> float:
    if not samples:
        return float("nan")
    total = 0.0
    for sample in samples:
        pooled = pooled_probabilities(sample, weights)
        total -= math.log(max(pooled[sample.actual], _PROBABILITY_FLOOR))
    return total / len(samples)


def _gradient(samples: list[Sample], weights: dict[str, float]) -> dict[str, float]:
    """Derivative of the log-likelihood with respect to each weight."""
    gradient = dict.fromkeys(weights, 0.0)
    for sample in samples:
        pooled = pooled_probabilities(sample, weights)
        for source in weights:
            logs = sample.log_probabilities(source)
            expected = sum(pooled[outcome] * logs[outcome] for outcome in OUTCOMES)
            gradient[source] += logs[sample.actual] - expected
    count = len(samples)
    return {source: value / count for source, value in gradient.items()}


def fit_pool(
    train: list[Sample],
    holdout: list[Sample],
    sources: list[str],
    *,
    max_iterations: int = _MAX_ITERATIONS,
) -> PoolFit:
    """Fit pooling weights on ``train`` and score them on ``holdout``."""
    weights = dict.fromkeys(sources, 1.0 / len(sources))
    current = -log_loss(train, weights)
    step = _INITIAL_STEP
    converged = False

    for _ in range(max_iterations):
        gradient = _gradient(train, weights)
        candidate = {source: weights[source] + step * gradient[source] for source in sources}
        candidate_score = -log_loss(train, candidate)

        if candidate_score > current:
            improvement = candidate_score - current
            weights, current = candidate, candidate_score
            step *= 1.3
            if improvement < _TOLERANCE:
                converged = True
                break
        else:
            step *= 0.5
            if step < 1e-12:
                converged = True
                break

    return PoolFit(
        sources=list(sources),
        weights=weights,
        train_matches=len(train),
        holdout_matches=len(holdout),
        train_log_loss=-current,
        holdout_log_loss=log_loss(holdout, weights),
        converged=converged,
    )


def split(samples: list[Sample], train_fraction: float = 0.5) -> tuple[list[Sample], list[Sample]]:
    """Chronological split. Samples must already be in date order."""
    cut = max(1, int(len(samples) * train_fraction))
    return samples[:cut], samples[cut:]


def assess(
    samples: list[Sample],
    market: str,
    candidates: list[str],
    *,
    train_fraction: float = 0.5,
) -> IncrementalReport:
    """Test each candidate, and all of them together, for information beyond the price.

    Every combination is fitted on the earlier half and scored on the later half,
    so nothing is judged on data that set its own weights.
    """
    report = IncrementalReport()
    if len(samples) < 4:
        return report

    train, holdout = split(samples, train_fraction)
    if not train or not holdout:
        return report

    report.holdout_matches = len(holdout)
    market_alone = fit_pool(train, holdout, [market])
    report.market_log_loss = market_alone.holdout_log_loss
    report.combinations.append(market_alone)

    groups = [[candidate] for candidate in candidates]
    if len(candidates) > 1:
        groups.append(list(candidates))

    for group in groups:
        combination = fit_pool(train, holdout, [market, *group])
        report.combinations.append(combination)

        improvement = report.market_log_loss - combination.holdout_log_loss
        report.verdicts.append(
            {
                "candidates": group,
                "holdout_log_loss": round(combination.holdout_log_loss, 4),
                "improvement_over_market": round(improvement, 4),
                "weights": {name: round(weight, 4) for name, weight in combination.weights.items()},
                "adds_information": improvement > MEANINGFUL_IMPROVEMENT,
            }
        )

    return report
