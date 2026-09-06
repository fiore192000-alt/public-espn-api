"""Does a forecast know something the market has not finished pricing?

Every other verdict in this project is settled against results, which is the
noisiest possible target: a match yields one bit, so telling a real edge from
luck takes tens of thousands of them. The closing line is a far better target. It
is continuous, it is the market's own final opinion, and on this data selections
that shorten before kick-off go on to win more often than the price you took
implied — so a forecast that systematically beats the close is producing real
edge, and a few thousand matches are enough to say so.

The central quantity is **anticipation**: of the distance between the model and
the opening price, how much does the market itself go on to travel?

    closing − opening  =  b · (model − opening)  +  error

``b`` near zero says the model's disagreements are noise the market never
ratifies. ``b`` near one says the market ends up where the model already was, and
the model got there first. Fitted through the origin on purpose: a model that
agrees with the opening price predicts no movement, and an intercept would let a
general drift in the market masquerade as the model's insight.

Standard errors are clustered by match. The three outcomes of one fixture move
together — money on the home side has to come out of the other two — so treating
the legs as independent would overstate the evidence by roughly the square root
of three.

Nothing here may be used to select a bet. The closing price exists only after the
moment a bet would have had to be struck; it is a measuring instrument, never a
signal.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

# Below this many matches an anticipation slope is not worth quoting.
MINIMUM_MATCHES = 200
CONFIDENCE_Z = 1.96


@dataclass(frozen=True)
class Observation:
    """One selection: what the model said, and what the market did afterwards."""

    match: str
    selection: str
    model: float
    opening: float
    closing: float
    opening_price: float
    closing_price: float
    won: bool

    @property
    def disagreement(self) -> float:
        """How far the model sits from the opening price, in probability."""
        return self.model - self.opening

    @property
    def movement(self) -> float:
        """How far the market itself moved, in probability."""
        return self.closing - self.opening

    @property
    def closing_line_value(self) -> float:
        """Profit per unit staked if the close were a fair settlement price.

        A positive figure means the price taken was longer than the one the
        market settled on — the bet beat the close.
        """
        return self.opening_price / self.closing_price - 1.0 if self.closing_price else 0.0


@dataclass
class Anticipation:
    """How much of the model's disagreement the market went on to ratify."""

    slope: float
    stderr: float
    observations: int
    matches: int

    @property
    def t_stat(self) -> float:
        return self.slope / self.stderr if self.stderr > 0 else 0.0

    def interval(self, z: float = CONFIDENCE_Z) -> tuple[float, float]:
        half = z * self.stderr
        return self.slope - half, self.slope + half

    @property
    def anticipates_the_market(self) -> bool:
        """Is the slope positive by more than the sample can explain away?"""
        return self.matches >= MINIMUM_MATCHES and self.interval()[0] > 0

    def to_dict(self) -> dict:
        low, high = self.interval()
        return {
            "slope": round(self.slope, 4),
            "stderr": round(self.stderr, 4),
            "t": round(self.t_stat, 2),
            "interval": [round(low, 4), round(high, 4)],
            "observations": self.observations,
            "matches": self.matches,
            "anticipates_the_market": self.anticipates_the_market,
        }


def fit_anticipation(observations: list[Observation]) -> Anticipation:
    """Regress the market's movement on the model's disagreement, through the origin.

    The slope is ``Σxy / Σx²``. Its variance is the cluster-robust estimator with
    matches as clusters: square the *sum* of ``x·residual`` within each match
    before adding across matches, which is what keeps three dependent legs from
    counting as three independent pieces of evidence.
    """
    xx = sum(o.disagreement**2 for o in observations)
    if xx <= 0:
        return Anticipation(slope=0.0, stderr=0.0, observations=len(observations), matches=0)

    slope = sum(o.disagreement * o.movement for o in observations) / xx

    per_match: dict[str, float] = defaultdict(float)
    for o in observations:
        per_match[o.match] += o.disagreement * (o.movement - slope * o.disagreement)

    meat = sum(score**2 for score in per_match.values())
    return Anticipation(
        slope=slope,
        stderr=math.sqrt(meat) / xx if meat > 0 else 0.0,
        observations=len(observations),
        matches=len(per_match),
    )


@dataclass
class BeatingTheClose:
    """Realised closing-line value of the selections a model actually liked."""

    picks: int
    matches: int
    mean: float
    stderr: float
    hit_rate: float

    @property
    def t_stat(self) -> float:
        return self.mean / self.stderr if self.stderr > 0 else 0.0

    @property
    def beats_the_close(self) -> bool:
        return self.matches >= MINIMUM_MATCHES and self.mean - CONFIDENCE_Z * self.stderr > 0

    def to_dict(self) -> dict:
        return {
            "picks": self.picks,
            "matches": self.matches,
            "clv": round(self.mean, 4),
            "stderr": round(self.stderr, 4),
            "t": round(self.t_stat, 2),
            "hit_rate": round(self.hit_rate, 4),
            "beats_the_close": self.beats_the_close,
        }


def beat_the_close(observations: list[Observation], *, edge: float = 0.0) -> BeatingTheClose:
    """Take every selection the model rates above the opening price, and see.

    Picks are pooled per match before being averaged, for the same reason the
    slope's errors are clustered: several picks in one fixture are one event.
    """
    per_match: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        if observation.disagreement > edge:
            per_match[observation.match].append(observation)

    values = [
        sum(pick.closing_line_value for pick in picks) / len(picks) for picks in per_match.values()
    ]
    picks = sum(len(p) for p in per_match.values())
    wins = sum(1 for p in per_match.values() for pick in p if pick.won)

    if not values:
        return BeatingTheClose(picks=0, matches=0, mean=0.0, stderr=0.0, hit_rate=0.0)

    count = len(values)
    mean = sum(values) / count
    if count > 1:
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        stderr = math.sqrt(variance / count)
    else:
        stderr = 0.0

    return BeatingTheClose(
        picks=picks,
        matches=count,
        mean=mean,
        stderr=stderr,
        hit_rate=wins / picks if picks else 0.0,
    )


def log_loss(observations: list[Observation], source: str) -> float:
    """Score one probability series on the matches these observations cover.

    ``source`` is an attribute name — "model", "opening" or "closing" — so the
    three are scored on exactly the same fixtures and the comparison is paired.
    """
    winners = [o for o in observations if o.won]
    if not winners:
        return 0.0
    return -sum(math.log(max(getattr(o, source), 1e-12)) for o in winners) / len(winners)


@dataclass
class Report:
    """One model, measured against the line rather than against the result."""

    name: str
    anticipation: Anticipation
    closing_line: BeatingTheClose
    losses: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "model": self.name,
            "anticipation": self.anticipation.to_dict(),
            "beating_the_close": self.closing_line.to_dict(),
            "log_loss": {name: round(value, 4) for name, value in self.losses.items()},
        }


def assess(name: str, observations: list[Observation], *, edge: float = 0.0) -> Report:
    return Report(
        name=name,
        anticipation=fit_anticipation(observations),
        closing_line=beat_the_close(observations, edge=edge),
        losses={
            source: log_loss(observations, source) for source in ("model", "opening", "closing")
        },
    )


def observations_from(records, source: str) -> list[Observation]:
    """Build observations from backtest forecasts that carry both lines.

    ``source`` names the attribute holding the model's probabilities, so
    Dixon-Coles and Elo go through identical arithmetic. Records missing either
    line are dropped rather than filled in: a missing closing price is not a
    closing price of zero.
    """
    built: list[Observation] = []
    for record in records:
        model = getattr(record, source, None)
        opening, closing = record.market_probabilities, record.closing_probabilities
        if not (model and opening and closing and record.opening_prices and record.closing_prices):
            continue
        for selection, probability in model.items():
            if selection not in opening or selection not in closing:
                continue
            opening_price = record.opening_prices.get(selection)
            closing_price = record.closing_prices.get(selection)
            if not opening_price or not closing_price:
                continue
            built.append(
                Observation(
                    match=record.event_espn_id,
                    selection=selection,
                    model=probability,
                    opening=opening[selection],
                    closing=closing[selection],
                    opening_price=opening_price,
                    closing_price=closing_price,
                    won=selection == record.actual,
                )
            )
    return built
