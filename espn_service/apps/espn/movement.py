"""Does a feature predict the market's own move from opening to closing price?

This is the question with the statistical power. Settling a bet on the result
gives one noisy bit per match, and demonstrating a true 2% yield takes tens of
thousands of them. The market's movement between open and close is continuous,
it is the market's own correction of its first guess, and the same signal shows
up in a couple of hundred matches instead. `clv.py` asks it of one thing — how
far a model sits from the opening price. This module asks it of anything.

**The trap this module exists to refuse.** The target is ``closing - opening``.
A feature built as ``something - opening`` therefore shares a term with its own
target, with opposite signs, and regresses beautifully on it while meaning
nothing. That is not hypothetical: it is exactly how the pre-registered H-0001
produced `t = -15.65` and had to be thrown away. Two defences are wired in and
neither is optional:

``anchor_permuted``
    Lends each match **another match's feature** while keeping its **own**
    opening and closing, so whatever the shared anchor alone can produce
    survives the permutation and is measured rather than assumed.

``anchor_share``
    Says what fraction of the raw slope the anchor already accounts for. A
    feature whose share is near 1 is not predicting the market; it is the
    opening price wearing a different name.

`Fit.predicts_the_market` refuses to certify anything without both.

Errors are clustered by match throughout: three selections of one fixture move
together and are not three independent observations.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# Below this the cluster-robust standard error is not worth reporting.
MINIMUM_MATCHES = 200
CONFIDENCE_Z = 1.96
NULL_SEED = 20260907
# A feature whose null already explains this much of its slope is reported as
# anchor-driven whatever its t-statistic says.
ANCHOR_SUSPICION = 0.5


class MovementError(ValueError):
    """Raised when a candidate cannot be measured as given."""


@dataclass(frozen=True)
class Movement:
    """One selection: a candidate's own value, and what the market did afterwards.

    The candidate is stored as ``raw`` — the number the source produced — rather
    than as the finished feature, and ``anchored`` says whether the feature is
    that number's distance from the opening price. Keeping the two apart is the
    whole reason the null works: permuting a *finished* feature carries each
    match's own opening away with it and quietly destroys the structure the null
    exists to preserve. Permuting ``raw`` leaves own-opening in place, which is
    what makes the anchor's contribution measurable instead of invisible.
    """

    match: str
    selection: str
    raw: float
    opening: float
    closing: float
    anchored: bool = False
    opening_price: float = 0.0
    closing_price: float = 0.0

    @property
    def feature(self) -> float:
        """The regressor: a distance from the opening price, or the value itself."""
        return self.raw - self.opening if self.anchored else self.raw

    @property
    def travelled(self) -> float:
        """How far the market moved, in probability. The target."""
        return self.closing - self.opening


@dataclass(frozen=True)
class Fit:
    """A slope through the origin, with the anchor's own contribution beside it."""

    name: str
    slope: float
    stderr: float
    observations: int
    matches: int
    feature_sd: float
    mean_opening: float
    null_slope: float | None = None

    @property
    def t_stat(self) -> float:
        return self.slope / self.stderr if self.stderr > 0 else 0.0

    def interval(self, z: float = CONFIDENCE_Z) -> tuple[float, float]:
        half = z * self.stderr
        return self.slope - half, self.slope + half

    @property
    def net_slope(self) -> float | None:
        """The slope with the anchor's own contribution taken out."""
        return None if self.null_slope is None else self.slope - self.null_slope

    @property
    def anchor_share(self) -> float | None:
        """What fraction of the slope the shared opening price already explains.

        Near 1 means the feature is the anchor in disguise. Undefined for a
        slope of zero, which is reported as None rather than as a large number.
        """
        if self.null_slope is None or abs(self.slope) < 1e-12:
            return None
        return self.null_slope / self.slope

    @property
    def anchor_driven(self) -> bool:
        share = self.anchor_share
        return share is not None and share > ANCHOR_SUSPICION

    @property
    def movement_per_sd(self) -> float | None:
        """Probability points the market moves per standard deviation of the feature."""
        net = self.net_slope
        return None if net is None else net * self.feature_sd

    def value_per_sd(self, margin: float) -> float | None:
        """The predicted move as a share of price, against the margin it must clear.

        A move of ``d`` in probability, at an opening probability ``q``, is worth
        roughly ``d / q`` of the price. Dividing by the bookmaker's margin gives
        the number that decides whether a real effect is also a useful one — the
        ratio H-0001 reported as 0.042 before being set aside.
        """
        moved = self.movement_per_sd
        if moved is None or self.mean_opening <= 0 or margin <= 0:
            return None
        return (moved / self.mean_opening) / margin

    @property
    def predicts_the_market(self) -> bool:
        """The only positive verdict, and it needs every guard to pass."""
        if self.null_slope is None:
            return False
        low, _ = self.interval()
        return (
            self.matches >= MINIMUM_MATCHES
            and low > 0
            and self.slope > self.null_slope
            and not self.anchor_driven
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "slope": round(self.slope, 6),
            "stderr": round(self.stderr, 6),
            "t": round(self.t_stat, 2),
            "interval": [round(v, 6) for v in self.interval()],
            "observations": self.observations,
            "matches": self.matches,
            "null_slope": None if self.null_slope is None else round(self.null_slope, 6),
            "net_slope": None if self.net_slope is None else round(self.net_slope, 6),
            "anchor_share": None if self.anchor_share is None else round(self.anchor_share, 3),
            "anchor_driven": self.anchor_driven,
            "movement_per_sd": (
                None if self.movement_per_sd is None else round(self.movement_per_sd, 6)
            ),
            "predicts_the_market": self.predicts_the_market,
        }


def _spread(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def fit(movements: Sequence[Movement], name: str = "") -> Fit:
    """Regress the market's movement on the feature, through the origin.

    The slope is ``Σxy / Σx²``; its variance is the cluster-robust estimator with
    matches as clusters — square the *sum* of ``x·residual`` within a match
    before adding across matches, which is what stops three dependent legs from
    counting as three independent pieces of evidence.
    """
    if not movements:
        raise MovementError("A fit needs at least one observation.")

    xx = sum(m.feature**2 for m in movements)
    features = [m.feature for m in movements]
    openings = [m.opening for m in movements]
    if xx <= 0:
        return Fit(
            name=name,
            slope=0.0,
            stderr=0.0,
            observations=len(movements),
            matches=len({m.match for m in movements}),
            feature_sd=0.0,
            mean_opening=statistics.fmean(openings),
        )

    slope = sum(m.feature * m.travelled for m in movements) / xx

    per_match: dict[str, float] = defaultdict(float)
    for m in movements:
        per_match[m.match] += m.feature * (m.travelled - slope * m.feature)
    meat = sum(score**2 for score in per_match.values())

    return Fit(
        name=name,
        slope=slope,
        stderr=math.sqrt(meat) / xx if meat > 0 else 0.0,
        observations=len(movements),
        matches=len(per_match),
        feature_sd=_spread(features),
        mean_opening=statistics.fmean(openings),
    )


def anchor_permuted(
    movements: Sequence[Movement],
    seed: int = NULL_SEED,
) -> list[Movement]:
    """Lend each match another match's feature, keeping its own open and close.

    Shuffling whole observations would destroy the shared opening price along
    with everything else, and the shared opening price is precisely what has to
    survive: it is what a contaminated feature is really regressing on. Donors
    are matched by selection so a home feature never lands on a draw.
    """
    rng = random.Random(seed)
    by_selection: dict[str, list[Movement]] = defaultdict(list)
    for m in movements:
        by_selection[m.selection].append(m)

    permuted: list[Movement] = []
    for selection, group in by_selection.items():
        if len(group) < 2:
            continue
        donors = list(group)
        rng.shuffle(donors)
        # Rotate so no observation can donate to itself.
        donors = donors[1:] + donors[:1]
        for own, donor in zip(group, donors, strict=True):
            if own.match == donor.match:
                continue
            permuted.append(
                Movement(
                    match=own.match,
                    selection=selection,
                    # The donor's own value against THIS match's opening. Lending
                    # the finished feature instead would take own-opening with it.
                    raw=donor.raw,
                    opening=own.opening,
                    closing=own.closing,
                    anchored=own.anchored,
                    opening_price=own.opening_price,
                    closing_price=own.closing_price,
                )
            )
    return permuted


def fit_against_null(
    movements: Sequence[Movement],
    name: str = "",
    seed: int = NULL_SEED,
) -> Fit:
    """Fit the slope and the anchor's own contribution together.

    The only route to ``predicts_the_market``: a Fit without ``null_slope``
    refuses to certify, by construction rather than by convention.
    """
    real = fit(movements, name)
    floor = fit(anchor_permuted(movements, seed), name)
    return Fit(
        name=real.name,
        slope=real.slope,
        stderr=real.stderr,
        observations=real.observations,
        matches=real.matches,
        feature_sd=real.feature_sd,
        mean_opening=real.mean_opening,
        null_slope=floor.slope,
    )


# --- building candidates --------------------------------------------------

FeatureFn = Callable[[object, str], float | None]


def movements_from(
    records: Sequence[object],
    feature: FeatureFn,
    *,
    anchored: bool = False,
    selections: Sequence[str] = ("home", "draw", "away"),
) -> list[Movement]:
    """Build a candidate from backtest records and a value function.

    ``feature`` returns the candidate's own number; ``anchored`` says whether the
    regressor is that number's distance from the opening price. A record
    contributes only where it carries both a devigged opening and a devigged
    closing for the selection — a partially priced fixture is dropped rather
    than half-used.
    """
    out: list[Movement] = []
    for record in records:
        opening = getattr(record, "market_probabilities", None)
        closing = getattr(record, "closing_probabilities", None)
        if not opening or not closing:
            continue
        prices = getattr(record, "opening_prices", None) or {}
        closing_prices = getattr(record, "closing_prices", None) or {}
        for selection in selections:
            if selection not in opening or selection not in closing:
                continue
            value = feature(record, selection)
            if value is None:
                continue
            out.append(
                Movement(
                    match=str(getattr(record, "event_espn_id", id(record))),
                    selection=selection,
                    raw=float(value),
                    opening=opening[selection],
                    closing=closing[selection],
                    anchored=anchored,
                    opening_price=prices.get(selection, 0.0),
                    closing_price=closing_prices.get(selection, 0.0),
                )
            )
    return out


def model_probability(source: str) -> FeatureFn:
    """A model's own probability. Use with ``anchored=True`` for `clv.py`'s question.

    Anchored, the regressor becomes the model's distance from the opening price,
    which shares a term with the target by construction. That is not a reason to
    avoid it — it is the benchmark every new feature is compared against — but it
    is the reason the anchor share is reported beside every slope.
    """

    def feature(record: object, selection: str) -> float | None:
        model = getattr(record, source, None)
        if not model or selection not in model:
            return None
        return model[selection]

    return feature


def price_level() -> FeatureFn:
    """The opening probability itself: a pure mean-reversion control.

    Any feature that fails to beat this one is telling you nothing the price did
    not already say about itself.
    """

    def feature(record: object, selection: str) -> float | None:
        opening = getattr(record, "market_probabilities", None)
        return None if not opening else opening.get(selection)

    return feature


def gaussian_noise(seed: int = NULL_SEED) -> FeatureFn:
    """A feature that cannot possibly work. If it scores, the bench is broken."""
    rng = random.Random(seed)

    def feature(record: object, selection: str) -> float | None:
        return rng.gauss(0.0, 1.0)

    return feature
