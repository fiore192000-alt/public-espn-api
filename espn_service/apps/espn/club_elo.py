"""Ratings computed over every match a club plays, not only this league's.

Dixon-Coles and the Elo in ``elo.py`` both learn a club's strength from matches
inside the league being modelled. Mid-season that is fine. At the start of one it
is a handicap, and for a promoted side it is severe: two fixtures of evidence
against fourteen for everyone else, with a time decay that has already discarded
whatever the club did in the division below.

The failure is not subtle. Asked for Frosinone against Venezia on the third
weekend of 2026/27, Dixon-Coles gave the promoted side **89.8%** and 2.60 expected
goals to 0.15 — having crowned it the best attack in Italy on one 3-0 win, with an
effective sample of 2.0 matches against roughly 14 for an established club.

ClubElo is maintained externally over every competitive fixture a club plays, so
the two years Frosinone spent in Serie B are in the number. The loader already
stores the pre-match pair on each event; this module only maps the difference to
1X2, reusing the ordered logit from ``elo.py`` so that the two rating sources are
compared through identical arithmetic rather than through two estimators.

What it is worth, measured rather than assumed:

**As a forecaster it is the best model here.** On 7,730 out-of-sample Serie A
matches its log loss is 0.9771, against Elo's 0.9843 and Dixon-Coles's 0.9984 —
paired *t* of +6.42 and +4.77, both established.

**As a source of information the market lacks it is worth nothing.** It loses to
the price (0.9610, *t* = −7.50); pooled with the market it adds less than its own
shuffled forecasts; and on 4,424 matches with a closing line its anticipation
slope is **−0.0166** (*t* = −3.14), which is where Dixon-Coles and Elo already
were. Being a better forecaster did not produce one usable scrap of information
the price did not already contain.

That gap between the two paragraphs is the whole point of the module.
"""

from __future__ import annotations

from dataclasses import dataclass

from apps.espn import elo

# Where ingest_football_data leaves the pre-match pair.
HOME_RATING_KEY = "home_elo"
AWAY_RATING_KEY = "away_elo"

# ClubElo runs on roughly the same scale as chess Elo. Dividing by this keeps the
# fitted beta near the magnitude the optimiser in elo.py expects; any constant
# would do, since beta absorbs it.
RATING_SCALE = 100.0


def ratings_of(event) -> tuple[float, float] | None:
    """The pre-match pair stored on an event, or None when it carries none.

    A rating is read only when both sides have one: a difference computed against
    a missing opponent would be a strength claim about nobody.
    """
    raw = getattr(event, "raw_data", None) or {}
    home, away = raw.get(HOME_RATING_KEY), raw.get(AWAY_RATING_KEY)
    try:
        return float(home), float(away)
    except (TypeError, ValueError):
        return None


def difference(home_rating: float, away_rating: float) -> float:
    return (home_rating - away_rating) / RATING_SCALE


@dataclass(frozen=True)
class ClubEloModel:
    """The fitted mapping from a rating gap to 1X2."""

    outcome: elo.OutcomeModel

    def probabilities(self, home_rating: float, away_rating: float) -> dict[str, float]:
        return self.outcome.probabilities(difference(home_rating, away_rating))

    @property
    def samples(self) -> int:
        return self.outcome.samples

    def to_dict(self) -> dict:
        return {"scale": RATING_SCALE, **self.outcome.to_dict()}


def fit(
    samples: list[tuple[float, str]],
    *,
    initial: ClubEloModel | None = None,
) -> ClubEloModel:
    """Fit the ordered logit on (rating difference, outcome) pairs.

    Raises ``elo.NotEnoughData`` below the sample floor, so a caller can treat a
    thin league exactly as it treats a thin Elo fit.
    """
    return ClubEloModel(
        outcome=elo.fit_outcome_model(
            samples, initial=initial.outcome if initial is not None else None
        )
    )


def sample_of(event, actual: str) -> tuple[float, str] | None:
    """One training pair, or None when the event carries no usable ratings."""
    pair = ratings_of(event)
    return (difference(*pair), actual) if pair else None
