"""Betting markets derived from a Dixon-Coles scoreline distribution.

Every market here is a different way of summing the same joint distribution, so
they are guaranteed to be mutually consistent — the 1X2 prices, the totals and
the correct scores all come from one grid rather than from separate models.
"""

from __future__ import annotations

from apps.espn.dixon_coles import ScoreGrid

DEFAULT_TOTAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)
DEFAULT_CORRECT_SCORE_COUNT = 8

# Markets are keyed by these names throughout the service (odds, value, backtest).
MARKET_MATCH_ODDS = "1x2"
MARKET_DOUBLE_CHANCE = "double_chance"
MARKET_TOTALS = "totals"
MARKET_BTTS = "btts"

SELECTION_HOME = "home"
SELECTION_DRAW = "draw"
SELECTION_AWAY = "away"


def fair_odds(probability: float) -> float | None:
    """Decimal odds that would make a bet at this probability break even."""
    return round(1.0 / probability, 3) if probability > 0 else None


def match_odds(grid: ScoreGrid) -> dict[str, float]:
    """1X2 probabilities."""
    home = draw = away = 0.0
    for home_goals, row in enumerate(grid.matrix):
        for away_goals, probability in enumerate(row):
            if home_goals > away_goals:
                home += probability
            elif home_goals == away_goals:
                draw += probability
            else:
                away += probability
    return {SELECTION_HOME: home, SELECTION_DRAW: draw, SELECTION_AWAY: away}


def double_chance(outcomes: dict[str, float]) -> dict[str, float]:
    """1X, 12 and X2, from the 1X2 probabilities."""
    return {
        "home_or_draw": outcomes[SELECTION_HOME] + outcomes[SELECTION_DRAW],
        "home_or_away": outcomes[SELECTION_HOME] + outcomes[SELECTION_AWAY],
        "draw_or_away": outcomes[SELECTION_DRAW] + outcomes[SELECTION_AWAY],
    }


def totals(
    grid: ScoreGrid, lines: tuple[float, ...] = DEFAULT_TOTAL_LINES
) -> dict[str, dict[str, float]]:
    """Over/under probabilities for each line. Half-lines only, so no pushes."""
    result: dict[str, dict[str, float]] = {}
    for line in lines:
        over = 0.0
        for home_goals, row in enumerate(grid.matrix):
            for away_goals, probability in enumerate(row):
                if home_goals + away_goals > line:
                    over += probability
        result[f"{line}"] = {"over": over, "under": 1.0 - over}
    return result


def both_teams_to_score(grid: ScoreGrid) -> dict[str, float]:
    yes = sum(
        probability
        for home_goals, row in enumerate(grid.matrix)
        for away_goals, probability in enumerate(row)
        if home_goals > 0 and away_goals > 0
    )
    return {"yes": yes, "no": 1.0 - yes}


def correct_scores(
    grid: ScoreGrid,
    limit: int = DEFAULT_CORRECT_SCORE_COUNT,
) -> list[dict[str, float | str]]:
    """The most likely exact scorelines, most probable first."""
    scores = [
        {"score": f"{home_goals}-{away_goals}", "probability": probability}
        for home_goals, row in enumerate(grid.matrix)
        for away_goals, probability in enumerate(row)
    ]
    scores.sort(key=lambda entry: entry["probability"], reverse=True)
    return [
        {
            "score": entry["score"],
            "probability": round(entry["probability"], 5),
            "fair_odds": fair_odds(entry["probability"]),
        }
        for entry in scores[:limit]
    ]


def expected_totals(grid: ScoreGrid) -> dict[str, float]:
    return {
        "home_goals": round(grid.expected_home_goals, 3),
        "away_goals": round(grid.expected_away_goals, 3),
        "total_goals": round(grid.expected_home_goals + grid.expected_away_goals, 3),
        "supremacy": round(grid.expected_home_goals - grid.expected_away_goals, 3),
    }


def _with_fair_odds(probabilities: dict[str, float]) -> dict[str, dict[str, float | None]]:
    return {
        selection: {"probability": round(probability, 5), "fair_odds": fair_odds(probability)}
        for selection, probability in probabilities.items()
    }


def summarise(
    grid: ScoreGrid,
    *,
    lines: tuple[float, ...] = DEFAULT_TOTAL_LINES,
    correct_score_limit: int = DEFAULT_CORRECT_SCORE_COUNT,
) -> dict:
    """All supported markets for one fixture, with fair odds alongside each price."""
    outcomes = match_odds(grid)
    return {
        "expected": expected_totals(grid),
        MARKET_MATCH_ODDS: _with_fair_odds(outcomes),
        MARKET_DOUBLE_CHANCE: _with_fair_odds(double_chance(outcomes)),
        MARKET_TOTALS: {
            line: _with_fair_odds(sides) for line, sides in totals(grid, lines).items()
        },
        MARKET_BTTS: _with_fair_odds(both_teams_to_score(grid)),
        "correct_score": correct_scores(grid, correct_score_limit),
    }
