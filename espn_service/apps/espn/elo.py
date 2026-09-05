"""Elo ratings and an Elo-based 1X2 model.

A deliberately independent second opinion. Dixon-Coles models goals; Elo models
only who is stronger, updated match by match from results. They fail differently,
which is the point of having both — a model that agrees with Dixon-Coles by
construction would add nothing to it.

Two pieces:

1. **Ratings.** Standard Elo with a home-advantage term folded into the expected
   score, and an optional margin-of-victory multiplier so a 4-0 moves ratings
   further than a 1-0. Ratings are updated in date order and never look ahead.

2. **Outcome model.** Elo alone gives an expected score, not three probabilities;
   football needs a draw. The rating difference is mapped to 1X2 through an
   ordered logit with two thresholds, fitted by maximum likelihood on past
   matches. That is what makes the draw a per-fixture estimate — tight matches
   get a higher draw probability than mismatches — rather than a league constant.

Nothing here is fitted on a match it later predicts; the walk-forward in
`backtest.py` keeps ratings and thresholds strictly behind the match being scored.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from apps.espn.dixon_coles import MatchObservation

DEFAULT_K = 20.0
# Home advantage in rating points. Roughly the historical edge in top European
# leagues; it is a starting value, not a fitted one.
DEFAULT_HOME_ADVANTAGE = 65.0
DEFAULT_INITIAL_RATING = 1500.0
# Elo's usual 400-point scale: a 400-point gap implies a 10:1 expected score.
RATING_SCALE = 400.0

# Enough matches for the ordered-logit thresholds to mean anything.
MINIMUM_FIT_SAMPLES = 50

_MAX_ITERATIONS = 400
_LEARNING_RATE = 0.05
_TOLERANCE = 1e-8
_MIN_THRESHOLD_GAP = 1e-3
_PROBABILITY_FLOOR = 1e-9

OUTCOME_HOME = "home"
OUTCOME_DRAW = "draw"
OUTCOME_AWAY = "away"


class NotEnoughData(Exception):
    """Raised when there is too little history to fit the outcome mapping."""


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


@dataclass
class EloConfig:
    """Knobs for the rating update."""

    k: float = DEFAULT_K
    home_advantage: float = DEFAULT_HOME_ADVANTAGE
    initial_rating: float = DEFAULT_INITIAL_RATING
    margin_of_victory: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EloRatings:
    """Current ratings, advanced match by match."""

    config: EloConfig = field(default_factory=EloConfig)
    ratings: dict[int, float] = field(default_factory=dict)
    matches: int = 0
    # (scaled rating difference, outcome) for every match seen, which is what the
    # outcome model is fitted on.
    samples: list[tuple[float, str]] = field(default_factory=list)

    def rating(self, team_id: int) -> float:
        return self.ratings.get(team_id, self.config.initial_rating)

    def difference(self, home_id: int, away_id: int) -> float:
        """Home rating advantage, including the home bonus, on the Elo scale."""
        return self.rating(home_id) + self.config.home_advantage - self.rating(away_id)

    def scaled_difference(self, home_id: int, away_id: int) -> float:
        return self.difference(home_id, away_id) / RATING_SCALE

    def expected_score(self, home_id: int, away_id: int) -> float:
        """Expected points share for the home side, counting a draw as a half."""
        return 1.0 / (1.0 + 10.0 ** (-self.difference(home_id, away_id) / RATING_SCALE))

    def update(self, match: MatchObservation) -> None:
        """Advance both ratings by one played match."""
        expected = self.expected_score(match.home_id, match.away_id)
        margin = match.home_goals - match.away_goals
        actual = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)

        multiplier = _margin_multiplier(margin) if self.config.margin_of_victory else 1.0
        change = self.config.k * multiplier * (actual - expected)

        self.samples.append(
            (
                self.scaled_difference(match.home_id, match.away_id),
                OUTCOME_HOME if margin > 0 else (OUTCOME_DRAW if margin == 0 else OUTCOME_AWAY),
            )
        )
        self.ratings[match.home_id] = self.rating(match.home_id) + change
        self.ratings[match.away_id] = self.rating(match.away_id) - change
        self.matches += 1

    def to_dict(self) -> dict:
        return {
            "matches": self.matches,
            "teams": len(self.ratings),
            "config": self.config.to_dict(),
        }


def _margin_multiplier(margin: int) -> float:
    """Dampened weight for the winning margin.

    Logarithmic so that a fourth goal counts for less than the second; a draw and
    a one-goal win both weigh 1.0.
    """
    return math.log1p(abs(margin)) / math.log(2.0) if abs(margin) > 1 else 1.0


def build_ratings(
    observations: list[MatchObservation],
    config: EloConfig | None = None,
) -> EloRatings:
    """Walk matches in date order, updating ratings as it goes."""
    ratings = EloRatings(config=config or EloConfig())
    for match in sorted(observations, key=lambda item: item.date):
        ratings.update(match)
    return ratings


@dataclass
class OutcomeModel:
    """Ordered logit mapping a scaled rating difference to 1X2 probabilities.

    ``beta`` is how strongly the rating gap moves the result; the two thresholds
    carve out the draw band between an away win and a home win. A wider band means
    the league draws more.
    """

    beta: float
    lower: float
    upper: float
    samples: int
    log_likelihood: float
    converged: bool

    def probabilities(self, scaled_difference: float) -> dict[str, float]:
        latent = self.beta * scaled_difference
        away = _sigmoid(self.lower - latent)
        not_home = _sigmoid(self.upper - latent)
        home = 1.0 - not_home
        draw = max(not_home - away, _PROBABILITY_FLOOR)

        total = away + draw + home
        return {
            OUTCOME_HOME: home / total,
            OUTCOME_DRAW: draw / total,
            OUTCOME_AWAY: away / total,
        }

    def to_dict(self) -> dict:
        return {
            "beta": round(self.beta, 4),
            "draw_band": [round(self.lower, 4), round(self.upper, 4)],
            "samples": self.samples,
            "log_likelihood": round(self.log_likelihood, 3),
            "converged": self.converged,
        }


def _log_likelihood(
    samples: list[tuple[float, str]], beta: float, lower: float, upper: float
) -> float:
    total = 0.0
    for difference, outcome in samples:
        latent = beta * difference
        away = _sigmoid(lower - latent)
        not_home = _sigmoid(upper - latent)
        if outcome == OUTCOME_AWAY:
            probability = away
        elif outcome == OUTCOME_HOME:
            probability = 1.0 - not_home
        else:
            probability = not_home - away
        total += math.log(max(probability, _PROBABILITY_FLOOR))
    return total


def _gradient(
    samples: list[tuple[float, str]],
    beta: float,
    lower: float,
    upper: float,
) -> tuple[float, float, float]:
    gradient_beta = gradient_lower = gradient_upper = 0.0

    for difference, outcome in samples:
        latent = beta * difference
        away = _sigmoid(lower - latent)
        not_home = _sigmoid(upper - latent)

        if outcome == OUTCOME_AWAY:
            gradient_lower += 1.0 - away
            gradient_beta += -difference * (1.0 - away)
        elif outcome == OUTCOME_HOME:
            gradient_upper += -not_home
            gradient_beta += difference * not_home
        else:
            band = max(not_home - away, _PROBABILITY_FLOOR)
            away_slope = away * (1.0 - away)
            home_slope = not_home * (1.0 - not_home)
            gradient_lower += -away_slope / band
            gradient_upper += home_slope / band
            gradient_beta += difference * (away_slope - home_slope) / band

    count = len(samples)
    return gradient_beta / count, gradient_lower / count, gradient_upper / count


def fit_outcome_model(
    samples: list[tuple[float, str]],
    *,
    max_iterations: int = _MAX_ITERATIONS,
    learning_rate: float = _LEARNING_RATE,
    initial: OutcomeModel | None = None,
) -> OutcomeModel:
    """Fit beta and the two draw thresholds by maximum likelihood.

    Gradient ascent with an adaptive step: the step grows while the likelihood
    improves and is halved when it does not, which reaches the optimum in a
    fraction of the iterations a fixed rate needs.

    Pass ``initial`` to warm-start from a previous fit. In a walk-forward the
    parameters barely move between refits, so this turns each refit into a handful
    of iterations instead of thousands.
    """
    if len(samples) < MINIMUM_FIT_SAMPLES:
        raise NotEnoughData(
            f"{len(samples)} matches is too few to fit the Elo outcome mapping; "
            f"at least {MINIMUM_FIT_SAMPLES} are needed."
        )

    if initial is not None:
        beta, lower, upper = initial.beta, initial.lower, initial.upper
    else:
        beta, lower, upper = 1.0, -0.5, 0.5

    current = _log_likelihood(samples, beta, lower, upper)
    step = learning_rate
    converged = False
    # A collapsed step can mean the optimum, or just that the adaptive step walked
    # itself into a corner. Reset it once before believing the first case.
    restarts_left = 1

    for _ in range(max_iterations):
        gradient_beta, gradient_lower, gradient_upper = _gradient(samples, beta, lower, upper)

        candidate_beta = beta + step * gradient_beta
        candidate_lower = lower + step * gradient_lower
        candidate_upper = upper + step * gradient_upper
        if candidate_upper <= candidate_lower:
            candidate_upper = candidate_lower + _MIN_THRESHOLD_GAP

        candidate = _log_likelihood(samples, candidate_beta, candidate_lower, candidate_upper)

        if candidate > current:
            improvement = candidate - current
            beta, lower, upper, current = (
                candidate_beta,
                candidate_lower,
                candidate_upper,
                candidate,
            )
            step *= 1.3
            if improvement < _TOLERANCE:
                converged = True
                break
        else:
            step *= 0.5
            if step < 1e-12:
                if restarts_left:
                    restarts_left -= 1
                    step = learning_rate
                    continue
                converged = True
                break

    return OutcomeModel(
        beta=beta,
        lower=lower,
        upper=upper,
        samples=len(samples),
        log_likelihood=current,
        converged=converged,
    )


@dataclass
class EloModel:
    """Ratings plus the mapping that turns them into probabilities."""

    ratings: EloRatings
    outcome: OutcomeModel

    def probabilities(self, home_id: int, away_id: int) -> dict[str, float]:
        return self.outcome.probabilities(self.ratings.scaled_difference(home_id, away_id))

    def to_dict(self) -> dict:
        return {"ratings": self.ratings.to_dict(), "outcome": self.outcome.to_dict()}


def fit(observations: list[MatchObservation], config: EloConfig | None = None) -> EloModel:
    """Build ratings from history and fit the outcome mapping to the same history."""
    ratings = build_ratings(observations, config)
    return EloModel(ratings=ratings, outcome=fit_outcome_model(ratings.samples))
