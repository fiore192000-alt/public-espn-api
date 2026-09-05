"""Dixon-Coles scoreline model for football.

Implements the model from Dixon & Coles (1997), "Modelling Association Football
Scores and Inefficiencies in the Football Betting Market". Goals are Poisson with
team attack and defence strengths and a home advantage term, plus a correction
that inflates the probability of low-scoring results, where the independent
Poisson assumption is known to fail::

    lambda = attack[home] * defence[away] * home_advantage   (home goals)
    mu     = attack[away] * defence[home]                    (away goals)

    P(x, y) = tau(x, y) * Poisson(x; lambda) * Poisson(y; mu)

Recent matches carry more weight through an exponential decay on match age.

Fitting is deliberately dependency-free (no numpy or scipy). Attack, defence and
home advantage come from multiplicative coordinate ascent on the weighted Poisson
likelihood, which has a closed form per parameter and converges reliably. The
correlation term rho is then fitted by a one-dimensional search conditional on
those values — a standard, stable approximation to the joint fit.

Measured behaviour of that approximation, over simulated leagues with known
parameters: attack and defence ratings correlate above 0.98 with the truth on a
full season, and rho tracks its true value with a slight pull towards zero
(-0.15 recovered as -0.14, +0.10 as +0.08). On data with no correlation at all
it centres near -0.01 rather than exactly 0, with a per-fit standard deviation
of about 0.06 at 500 matches. That bias is small enough to leave scoreline
probabilities essentially unchanged, but it is a property of the two-stage fit
rather than of the data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

DEFAULT_HALF_LIFE_DAYS = 120.0
DEFAULT_MAX_GOALS = 10
DEFAULT_MAX_ITERATIONS = 200
DEFAULT_TOLERANCE = 1e-9

# Below this many matches the ratings are too unstable to bet on.
RELIABLE_MATCH_COUNT = 60

RHO_SEARCH_BOUND = 0.30
RHO_GRID_STEPS = 60
RHO_REFINEMENTS = 4

_MIN_RATE = 1e-6


class NotEnoughData(Exception):
    """Raised when there are too few matches to fit anything at all."""


@dataclass(frozen=True)
class MatchObservation:
    """One completed match used to fit the model."""

    home_id: int
    away_id: int
    home_goals: int
    away_goals: int
    date: datetime


@dataclass
class TeamRating:
    """Fitted strengths for one team.

    ``attack`` above 1.0 means the team scores more than the league average side;
    ``defence`` above 1.0 means it *concedes* more, so lower is better.
    """

    team_id: int
    attack: float
    defence: float


@dataclass
class ModelFit:
    """Result of fitting the model to a set of matches."""

    ratings: dict[int, TeamRating]
    home_advantage: float
    rho: float
    matches: int
    effective_matches: float
    half_life_days: float
    log_likelihood: float
    iterations: int
    converged: bool
    reference_date: datetime

    @property
    def is_reliable(self) -> bool:
        return self.converged and self.effective_matches >= RELIABLE_MATCH_COUNT

    def to_dict(self) -> dict:
        return {
            "teams": len(self.ratings),
            "matches": self.matches,
            "effective_matches": round(self.effective_matches, 2),
            "half_life_days": self.half_life_days,
            "home_advantage": round(self.home_advantage, 4),
            "rho": round(self.rho, 4),
            "log_likelihood": round(self.log_likelihood, 3),
            "iterations": self.iterations,
            "converged": self.converged,
            "reliable": self.is_reliable,
        }


@dataclass
class ScoreGrid:
    """Joint distribution over scorelines for one fixture."""

    home_id: int
    away_id: int
    expected_home_goals: float
    expected_away_goals: float
    matrix: list[list[float]] = field(default_factory=list)

    @property
    def max_goals(self) -> int:
        return len(self.matrix) - 1


def poisson_pmf(k: int, rate: float) -> float:
    rate = max(rate, _MIN_RATE)
    return math.exp(-rate + k * math.log(rate) - math.lgamma(k + 1))


def tau(home_goals: int, away_goals: int, home_rate: float, away_rate: float, rho: float) -> float:
    """Dixon-Coles low-score correction; 1.0 for any scoreline above 1-1."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - home_rate * away_rate * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + home_rate * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + away_rate * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def _decay_weights(
    observations: list[MatchObservation],
    reference_date: datetime,
    half_life_days: float,
) -> list[float]:
    if half_life_days <= 0:
        return [1.0] * len(observations)
    decay = math.log(2.0) / half_life_days
    weights = []
    for match in observations:
        age_days = max((reference_date - match.date).total_seconds() / 86400.0, 0.0)
        weights.append(math.exp(-decay * age_days))
    return weights


def _rates(
    match: MatchObservation,
    attack: dict[int, float],
    defence: dict[int, float],
    home_advantage: float,
) -> tuple[float, float]:
    home_rate = attack[match.home_id] * defence[match.away_id] * home_advantage
    away_rate = attack[match.away_id] * defence[match.home_id]
    return max(home_rate, _MIN_RATE), max(away_rate, _MIN_RATE)


def _log_likelihood(
    observations: list[MatchObservation],
    weights: list[float],
    attack: dict[int, float],
    defence: dict[int, float],
    home_advantage: float,
    rho: float,
) -> float:
    total = 0.0
    for match, weight in zip(observations, weights, strict=True):
        home_rate, away_rate = _rates(match, attack, defence, home_advantage)
        correction = tau(match.home_goals, match.away_goals, home_rate, away_rate, rho)
        if correction <= 0:
            return -math.inf
        total += weight * (
            math.log(poisson_pmf(match.home_goals, home_rate))
            + math.log(poisson_pmf(match.away_goals, away_rate))
            + math.log(correction)
        )
    return total


def _rho_bounds(rates: list[tuple[float, float]]) -> tuple[float, float]:
    """Keep every tau strictly positive for the fitted rates."""
    lower, upper = -RHO_SEARCH_BOUND, RHO_SEARCH_BOUND
    for home_rate, away_rate in rates:
        lower = max(lower, -1.0 / home_rate, -1.0 / away_rate)
        upper = min(upper, 1.0 / (home_rate * away_rate), 1.0)
    margin = 1e-6
    return lower + margin, upper - margin


def _fit_rho(
    observations: list[MatchObservation],
    weights: list[float],
    rates: list[tuple[float, float]],
) -> float:
    """Maximise the weighted tau term; only 0-0, 0-1, 1-0 and 1-1 contribute."""
    relevant = [
        (match, weight, rate_pair)
        for match, weight, rate_pair in zip(observations, weights, rates, strict=True)
        if match.home_goals <= 1 and match.away_goals <= 1
    ]
    if not relevant:
        return 0.0

    lower, upper = _rho_bounds(rates)
    if lower >= upper:
        return 0.0

    def objective(candidate: float) -> float:
        total = 0.0
        for match, weight, (home_rate, away_rate) in relevant:
            correction = tau(match.home_goals, match.away_goals, home_rate, away_rate, candidate)
            if correction <= 0:
                return -math.inf
            total += weight * math.log(correction)
        return total

    best = 0.0
    for _ in range(RHO_REFINEMENTS):
        step = (upper - lower) / RHO_GRID_STEPS
        best, best_score = lower, -math.inf
        for index in range(RHO_GRID_STEPS + 1):
            candidate = lower + index * step
            score = objective(candidate)
            if score > best_score:
                best, best_score = candidate, score
        lower, upper = max(lower, best - step), min(upper, best + step)
    return best


def fit(
    observations: list[MatchObservation],
    *,
    reference_date: datetime | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ModelFit:
    """Fit attack, defence, home advantage and rho to the given matches."""
    if not observations:
        raise NotEnoughData("No matches supplied.")

    team_ids = sorted(
        {match.home_id for match in observations} | {match.away_id for match in observations}
    )
    if len(team_ids) < 2:
        raise NotEnoughData("At least two distinct teams are required.")

    reference = reference_date or max(match.date for match in observations)
    weights = _decay_weights(observations, reference, half_life_days)

    attack = dict.fromkeys(team_ids, 1.0)
    defence = dict.fromkeys(team_ids, 1.0)
    home_advantage = 1.0

    # Weighted goals scored and conceded per team, which the updates below divide
    # by the corresponding weighted expected rates.
    scored: dict[int, float] = dict.fromkeys(team_ids, 0.0)
    conceded: dict[int, float] = dict.fromkeys(team_ids, 0.0)
    for match, weight in zip(observations, weights, strict=True):
        scored[match.home_id] += weight * match.home_goals
        scored[match.away_id] += weight * match.away_goals
        conceded[match.away_id] += weight * match.home_goals
        conceded[match.home_id] += weight * match.away_goals

    total_home_goals = sum(w * m.home_goals for m, w in zip(observations, weights, strict=True))

    # Index each team's fixtures once; the sweeps below would otherwise rescan
    # every match for every team, which dominates the cost on a full season.
    home_fixtures: dict[int, list[tuple[int, float]]] = {team: [] for team in team_ids}
    away_fixtures: dict[int, list[tuple[int, float]]] = {team: [] for team in team_ids}
    for match, weight in zip(observations, weights, strict=True):
        home_fixtures[match.home_id].append((match.away_id, weight))
        away_fixtures[match.away_id].append((match.home_id, weight))

    converged = False
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        largest_change = 0.0

        for team in team_ids:
            denominator = sum(
                weight * defence[opponent] * home_advantage
                for opponent, weight in home_fixtures[team]
            ) + sum(weight * defence[opponent] for opponent, weight in away_fixtures[team])
            if denominator > 0:
                updated = max(scored[team] / denominator, _MIN_RATE)
                largest_change = max(largest_change, abs(updated - attack[team]))
                attack[team] = updated

        # Attack and defence trade off against each other by a constant factor,
        # so pin the attack scale to a geometric mean of one.
        log_mean = sum(math.log(attack[team]) for team in team_ids) / len(team_ids)
        scale = math.exp(log_mean)
        for team in team_ids:
            attack[team] /= scale
            defence[team] *= scale

        for team in team_ids:
            denominator = sum(
                weight * attack[opponent] * home_advantage
                for opponent, weight in away_fixtures[team]
            ) + sum(weight * attack[opponent] for opponent, weight in home_fixtures[team])
            if denominator > 0:
                updated = max(conceded[team] / denominator, _MIN_RATE)
                largest_change = max(largest_change, abs(updated - defence[team]))
                defence[team] = updated

        expected_home = sum(
            weight * attack[match.home_id] * defence[match.away_id]
            for match, weight in zip(observations, weights, strict=True)
        )
        if expected_home > 0:
            updated = max(total_home_goals / expected_home, _MIN_RATE)
            largest_change = max(largest_change, abs(updated - home_advantage))
            home_advantage = updated

        if largest_change < tolerance:
            converged = True
            break

    rates = [_rates(match, attack, defence, home_advantage) for match in observations]
    rho = _fit_rho(observations, weights, rates)

    return ModelFit(
        ratings={
            team: TeamRating(team_id=team, attack=attack[team], defence=defence[team])
            for team in team_ids
        },
        home_advantage=home_advantage,
        rho=rho,
        matches=len(observations),
        effective_matches=sum(weights),
        half_life_days=half_life_days,
        log_likelihood=_log_likelihood(observations, weights, attack, defence, home_advantage, rho),
        iterations=iterations,
        converged=converged,
        reference_date=reference,
    )


def score_grid(
    model: ModelFit,
    home_id: int,
    away_id: int,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> ScoreGrid:
    """Build the normalised scoreline distribution for one fixture."""
    for team in (home_id, away_id):
        if team not in model.ratings:
            raise NotEnoughData(f"Team {team} was not part of the fitted match set.")

    home_rate = max(
        model.ratings[home_id].attack * model.ratings[away_id].defence * model.home_advantage,
        _MIN_RATE,
    )
    away_rate = max(model.ratings[away_id].attack * model.ratings[home_id].defence, _MIN_RATE)

    matrix = [[0.0] * (max_goals + 1) for _ in range(max_goals + 1)]
    total = 0.0
    for home_goals in range(max_goals + 1):
        home_pmf = poisson_pmf(home_goals, home_rate)
        for away_goals in range(max_goals + 1):
            probability = (
                home_pmf
                * poisson_pmf(away_goals, away_rate)
                * max(tau(home_goals, away_goals, home_rate, away_rate, model.rho), 0.0)
            )
            matrix[home_goals][away_goals] = probability
            total += probability

    if total <= 0:
        raise NotEnoughData("Degenerate scoreline distribution.")
    for row in matrix:
        for index in range(len(row)):
            row[index] /= total

    return ScoreGrid(
        home_id=home_id,
        away_id=away_id,
        expected_home_goals=home_rate,
        expected_away_goals=away_rate,
        matrix=matrix,
    )
