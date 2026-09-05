"""Bridge between stored events and the Dixon-Coles model.

`analysis.py` describes what has happened; this module predicts what will happen.
Both read only from the local database and both respect the same rule: when
forecasting an event, only matches that finished strictly before it are used.
"""

from __future__ import annotations

from datetime import datetime

from apps.espn import markets
from apps.espn.analysis import _completed_events, _sided_competitors
from apps.espn.dixon_coles import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_MAX_GOALS,
    MatchObservation,
    ModelFit,
    NotEnoughData,
    fit,
    score_grid,
)
from apps.espn.models import Event, League

# A league needs at least this many matches before a fit is attempted at all.
MINIMUM_MATCHES = 20


def collect_observations(
    league: League,
    before: datetime | None = None,
    max_matches: int | None = None,
) -> list[MatchObservation]:
    """Completed two-sided matches for a league, most recent first."""
    observations: list[MatchObservation] = []
    for event in _completed_events(league, before):
        if max_matches is not None and len(observations) >= max_matches:
            break
        sides = _sided_competitors(event)
        if sides is None:
            continue
        home, away = sides
        home_goals, away_goals = home.score_int, away.score_int
        if home_goals is None or away_goals is None:
            continue
        observations.append(
            MatchObservation(
                home_id=home.team_id,
                away_id=away.team_id,
                home_goals=home_goals,
                away_goals=away_goals,
                date=event.date,
            )
        )
    return observations


def fit_league_model(
    league: League,
    before: datetime | None = None,
    *,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    max_matches: int | None = None,
) -> ModelFit:
    """Fit the scoreline model to a league's history."""
    observations = collect_observations(league, before=before, max_matches=max_matches)
    if len(observations) < MINIMUM_MATCHES:
        raise NotEnoughData(
            f"{league.slug} has {len(observations)} completed matches on record; "
            f"at least {MINIMUM_MATCHES} are needed to fit the model."
        )
    return fit(observations, reference_date=before, half_life_days=half_life_days)


def forecast_event(
    event: Event,
    *,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    max_goals: int = DEFAULT_MAX_GOALS,
    lines: tuple[float, ...] = markets.DEFAULT_TOTAL_LINES,
    correct_score_limit: int = markets.DEFAULT_CORRECT_SCORE_COUNT,
    model: ModelFit | None = None,
) -> dict:
    """Model-based market probabilities for one fixture.

    Pass ``model`` to reuse a fit across several fixtures (the backtest does this);
    otherwise one is fitted from the league history preceding the event.
    """
    sides = _sided_competitors(event)
    if sides is None:
        raise NotEnoughData(
            f"Event {event.espn_id} does not have exactly two competitors to model."
        )
    home, away = sides

    model = model or fit_league_model(
        event.league, before=event.date, half_life_days=half_life_days
    )
    grid = score_grid(model, home.team_id, away.team_id, max_goals=max_goals)

    return {
        "model": model.to_dict(),
        "teams": {
            "home": _rating_of(model, home.team_id, home.team.abbreviation),
            "away": _rating_of(model, away.team_id, away.team.abbreviation),
        },
        "markets": markets.summarise(grid, lines=lines, correct_score_limit=correct_score_limit),
    }


def _rating_of(model: ModelFit, team_id: int, abbreviation: str) -> dict:
    rating = model.ratings[team_id]
    return {
        "abbreviation": abbreviation,
        "attack": round(rating.attack, 4),
        "defence": round(rating.defence, 4),
    }
