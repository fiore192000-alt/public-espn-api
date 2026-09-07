"""A forecaster built on chances created rather than on goals scored.

Every model in this repository so far learns from the scoreline. A scoreline is
a very small sample of a football match: roughly two or three Bernoulli draws
from a process that generated twenty-odd chances. Expected goals sums those
chances instead, and is correspondingly less noisy — a side that loses 0-1 having
created 2.6 xG to 0.3 is telling you something the result actively hides.

This is the layer most often claimed to be where an amateur's edge lives, and it
is the last one in this project to be measured. The claim it has to answer is not
"is xG a better description of a match" — it plainly is — but the one every other
candidate here has failed: **does it know anything the price does not?**

The ratings are deliberately plain. Each side carries an attack and a defence
multiplier relative to the league, estimated as an exponentially time-weighted
mean of the xG it created and conceded. A fixture's two rates are then

    home = league_mean · attack[home] · defence[away] · home_advantage
    away = league_mean · attack[away] · defence[home]

and the scoreline distribution comes from the same Poisson grid with the same
Dixon-Coles low-score correction the rest of the project uses, so a comparison
against Dixon-Coles measures the *input* rather than two different estimators.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from apps.espn.dixon_coles import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_MAX_GOALS,
    NotEnoughData,
    ScoreGrid,
    poisson_pmf,
    tau,
)

# Fewer matches than this and a league's ratings are noise dressed as strength.
MINIMUM_MATCHES = 40
# Shrinkage toward the league mean, in matches. A side with this much weighted
# history sits halfway between its own record and the league average — which is
# what stops two fixtures crowning a promoted club, the failure that made
# Dixon-Coles return 89.8% for Frosinone.
PRIOR_MATCHES = 5.0
_MIN_RATE = 1e-6


@dataclass(frozen=True)
class XgObservation:
    """One match's chances, for both sides."""

    home_id: int
    away_id: int
    home_xg: float
    away_xg: float
    date: datetime


@dataclass
class XgRating:
    """How much of the league's average a side creates and concedes."""

    team_id: int
    attack: float = 1.0
    defence: float = 1.0
    weight: float = 0.0

    def to_dict(self) -> dict:
        return {
            "team_id": self.team_id,
            "attack": round(self.attack, 4),
            "defence": round(self.defence, 4),
            "effective_matches": round(self.weight, 2),
        }


@dataclass
class XgModel:
    """Fitted ratings plus the league constants they are relative to."""

    ratings: dict[int, XgRating] = field(default_factory=dict)
    league_mean: float = 1.3
    home_advantage: float = 1.0
    rho: float = 0.0
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS
    matches: int = 0

    def rates(self, home_id: int, away_id: int) -> tuple[float, float]:
        """Expected goals for each side of a fixture."""
        home = self.ratings.get(home_id)
        away = self.ratings.get(away_id)
        if home is None or away is None:
            raise NotEnoughData(f"No xG rating for team {home_id if home is None else away_id}.")
        return (
            max(self.league_mean * home.attack * away.defence * self.home_advantage, _MIN_RATE),
            max(self.league_mean * away.attack * home.defence, _MIN_RATE),
        )

    def score_grid(
        self, home_id: int, away_id: int, max_goals: int = DEFAULT_MAX_GOALS
    ) -> ScoreGrid:
        home_rate, away_rate = self.rates(home_id, away_id)
        matrix = [
            [
                poisson_pmf(home_goals, home_rate)
                * poisson_pmf(away_goals, away_rate)
                * tau(home_goals, away_goals, home_rate, away_rate, self.rho)
                for away_goals in range(max_goals + 1)
            ]
            for home_goals in range(max_goals + 1)
        ]
        total = sum(sum(row) for row in matrix)
        return ScoreGrid(
            home_id=home_id,
            away_id=away_id,
            expected_home_goals=home_rate,
            expected_away_goals=away_rate,
            matrix=[[value / total for value in row] for row in matrix],
        )

    def probabilities(self, home_id: int, away_id: int) -> dict[str, float]:
        grid = self.score_grid(home_id, away_id)
        size = len(grid.matrix)
        home = sum(grid.matrix[i][j] for i in range(size) for j in range(size) if i > j)
        draw = sum(grid.matrix[i][i] for i in range(size))
        away = sum(grid.matrix[i][j] for i in range(size) for j in range(size) if i < j)
        total = home + draw + away
        return {"home": home / total, "draw": draw / total, "away": away / total}

    def to_dict(self) -> dict:
        return {
            "matches": self.matches,
            "league_mean_xg": round(self.league_mean, 4),
            "home_advantage": round(self.home_advantage, 4),
            "rho": round(self.rho, 4),
            "half_life_days": self.half_life_days,
            "teams": len(self.ratings),
        }


def _decay(observations: list[XgObservation], reference: datetime, half_life_days: float):
    if half_life_days <= 0:
        return [1.0] * len(observations)
    rate = math.log(2.0) / half_life_days
    return [
        math.exp(-rate * max((reference - match.date).total_seconds() / 86400.0, 0.0))
        for match in observations
    ]


def fit(
    observations: list[XgObservation],
    *,
    reference_date: datetime,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    rho: float = 0.0,
) -> XgModel:
    """Estimate each side's attack and defence from the chances it has created.

    Ratings are shrunk toward the league average in proportion to how little
    weighted history a side has, so a promoted club with two fixtures sits near
    1.0 rather than wherever those two fixtures happened to land.
    """
    if len(observations) < MINIMUM_MATCHES:
        raise NotEnoughData(
            f"{len(observations)} matches with xG is too few; {MINIMUM_MATCHES} are needed."
        )

    weights = _decay(observations, reference_date, half_life_days)
    total_weight = sum(weights)
    home_mean = (
        sum(w * m.home_xg for w, m in zip(weights, observations, strict=True)) / total_weight
    )
    away_mean = (
        sum(w * m.away_xg for w, m in zip(weights, observations, strict=True)) / total_weight
    )
    league_mean = max((home_mean + away_mean) / 2.0, _MIN_RATE)
    home_advantage = home_mean / max(away_mean, _MIN_RATE)

    created: dict[int, float] = defaultdict(float)
    conceded: dict[int, float] = defaultdict(float)
    seen: dict[int, float] = defaultdict(float)
    for weight, match in zip(weights, observations, strict=True):
        # A side's own home advantage is divided out before its attack is
        # measured, so a team that happens to have played more home games is not
        # rated stronger for it.
        created[match.home_id] += weight * match.home_xg / home_advantage
        conceded[match.home_id] += weight * match.away_xg
        created[match.away_id] += weight * match.away_xg
        conceded[match.away_id] += weight * match.home_xg / home_advantage
        seen[match.home_id] += weight
        seen[match.away_id] += weight

    ratings = {}
    for team_id, weight in seen.items():
        shrink = weight / (weight + PRIOR_MATCHES)
        attack = (created[team_id] / weight) / league_mean if weight else 1.0
        defence = (conceded[team_id] / weight) / league_mean if weight else 1.0
        ratings[team_id] = XgRating(
            team_id=team_id,
            attack=1.0 + shrink * (attack - 1.0),
            defence=1.0 + shrink * (defence - 1.0),
            weight=weight,
        )

    return XgModel(
        ratings=ratings,
        league_mean=league_mean,
        home_advantage=math.sqrt(max(home_advantage, _MIN_RATE)),
        rho=rho,
        half_life_days=half_life_days,
        matches=len(observations),
    )


def collect_observations(league, before: datetime | None = None) -> list[XgObservation]:
    """Every stored match in a league that carries expected goals."""
    from apps.espn.analysis import _sided_competitors
    from apps.espn.models import ExpectedGoals

    query = ExpectedGoals.objects.filter(event__league=league).select_related("event")
    if before is not None:
        query = query.filter(event__date__lt=before)

    built = []
    for entry in query.prefetch_related("event__competitors__team"):
        sides = _sided_competitors(entry.event)
        if sides is None:
            continue
        home, away = sides
        built.append(
            XgObservation(
                home_id=home.team_id,
                away_id=away.team_id,
                home_xg=entry.home,
                away_xg=entry.away,
                date=entry.event.date,
            )
        )
    return sorted(built, key=lambda match: match.date)
