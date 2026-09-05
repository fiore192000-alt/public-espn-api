"""Search the market's own errors for something you could actually bet on.

Looking for a bias in bookmaker prices is easy. Finding one that is real is not,
because a search over enough rules will always produce a winner, and a rule that
won a search is not evidence of anything. This module runs the search *and* the
gates that make its answer trustworthy, so the negative result it usually
produces is worth as much as a positive one would be.

The gates, in the order they bite:

1. **Discovery and validation are separated.** Rules are ranked on one league and
   one period. They are then re-measured on later matches of that league and on
   leagues the search never touched. Only the second number is evidence.
2. **The search burden is stated.** Every rule scored during discovery is counted
   and reported, together with how many would clear ``|t| >= 2`` by chance alone.
   A discovery t-statistic is not a finding; it is a lottery ticket.
3. **Effects are measured in money, not in calibration.** A gap between what the
   market implies and what happens is only interesting if it is bigger than the
   margin, so every rule is settled as real flat-staked bets at real prices.
4. **Consistency is required.** A rule that pools positive because one validation
   set carries it is not a rule. Every validation set is reported separately.
5. **Price selection is separated from the bias.** The same rule is settled twice:
   at a single bookmaker and at the best price available. Line shopping recovers
   margin on any selection whatsoever, so an "edge" that exists only at the best
   price is not a bias in the market's opinion — it is a discount on the fee.

Bets are pooled **per match, not per leg**. A rule that qualifies two outcomes of
the same match has one dependent result, not two independent ones.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from apps.espn import devig
from apps.espn.market_structure import Returns
from apps.espn.markets import MARKET_MATCH_ODDS

# Discovery will not rank a rule on fewer settled matches than this.
MINIMUM_BETS = 200
# How many of the best discovery rules are carried into validation.
CANDIDATES_CARRIED = 3
# Two standard errors. Conventional, and deliberately not adjusted downwards.
SIGNIFICANT_T = 2.0
CONFIDENCE_Z = 1.96
# A rule may lose on this many validation sets and still be called consistent.
CONSISTENCY_TOLERANCE = 1
# Share of pure-noise hypotheses expected to clear SIGNIFICANT_T.
FALSE_POSITIVE_RATE = 0.05

# Price bands wide enough to hold real samples, narrow enough that a
# favourite-longshot effect would show up as a trend across them.
PRICE_BUCKETS: tuple[tuple[float, float], ...] = (
    (1.0, 1.5),
    (1.5, 2.0),
    (2.0, 3.0),
    (3.0, 4.0),
    (4.0, 6.0),
    (6.0, 10.0),
    (10.0, math.inf),
)

BUCKET_RULE = "bucket"
SELECTION_RULE = "selection"

ESTABLISHED = "established"
NOT_ESTABLISHED = "not_established"
REJECTED = "rejected"


def _bucket_label(bucket: tuple[float, float]) -> str:
    low, high = bucket
    return f"{low:.2f}+" if math.isinf(high) else f"{low:.2f}-{high:.2f}"


@dataclass
class Series:
    """Mean and standard error of a stream of observations.

    Yields are accumulated by :class:`~apps.espn.market_structure.Returns`, which
    also tracks hit rates. This carries the same statistics for quantities that
    are not bets — calibration residuals, where "winning" means nothing.
    """

    count: int = 0
    total: float = 0.0
    _squares: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self._squares += value * value

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def stderr(self) -> float:
        if self.count < 2:
            return 0.0
        variance = (self._squares - self.count * self.mean**2) / (self.count - 1)
        return math.sqrt(max(variance, 0.0) / self.count)

    @property
    def t_stat(self) -> float:
        return self.mean / self.stderr if self.stderr > 0 else 0.0


@dataclass(frozen=True)
class PricedMatch:
    """One settled match with a book, optionally a best-price line, and the result."""

    date: object
    book: dict[str, float]
    best: dict[str, float] | None
    actual: str
    fair: dict[str, float]

    def prices(self, use_best: bool) -> dict[str, float]:
        """The price level a rule is settled at, falling back when best is absent."""
        return self.best if (use_best and self.best) else self.book


@dataclass(frozen=True)
class Rule:
    """A mechanical instruction for which selections to back.

    Deliberately crude. A rule complicated enough to need fitting would need its
    parameters held out too, and the point here is to test the market, not a model.
    """

    kind: str
    key: object
    label: str

    def legs(self, match: PricedMatch, use_best: bool) -> list[tuple[str, float]]:
        prices = match.prices(use_best)
        if self.kind == SELECTION_RULE:
            price = prices.get(self.key)
            return [(self.key, price)] if price else []

        low, high = self.key
        return [(selection, price) for selection, price in prices.items() if low <= price < high]


def price_rules() -> list[Rule]:
    """One rule per price band: back anything the market prices in this range."""
    return [
        Rule(kind=BUCKET_RULE, key=bucket, label=f"price {_bucket_label(bucket)}")
        for bucket in PRICE_BUCKETS
    ]


def selection_rules(selections: tuple[str, ...]) -> list[Rule]:
    """One rule per outcome: back this side in every match."""
    return [
        Rule(kind=SELECTION_RULE, key=selection, label=f"selection {selection}")
        for selection in selections
    ]


def settle(matches: list[PricedMatch], rule: Rule, use_best: bool) -> Returns:
    """Flat-stake a rule across matches and settle it against real results.

    One observation per match. A match where the rule fires on two selections
    stakes a unit on each and returns whatever the winner paid, which is a single
    dependent outcome however many legs it covers.
    """
    returns = Returns()
    for match in matches:
        legs = rule.legs(match, use_best)
        if not legs:
            continue
        returned = next((price for selection, price in legs if selection == match.actual), 0.0)
        returns.add_portfolio(returned=returned, staked=len(legs))
    return returns


@dataclass
class CalibrationBucket:
    """What the market implied in a price band, against what happened."""

    label: str
    legs: int
    matches: int
    implied: float
    observed: float
    residual: Series

    @property
    def gap(self) -> float:
        return self.residual.mean

    def to_dict(self) -> dict:
        return {
            "band": self.label,
            "legs": self.legs,
            "matches": self.matches,
            "market_implies": round(self.implied, 4),
            "actually_happens": round(self.observed, 4),
            "gap": round(self.gap, 4),
            "z": round(self.residual.t_stat, 2),
        }


def calibrate(matches: list[PricedMatch]) -> list[CalibrationBucket]:
    """Compare the devigged market probability to reality, band by band.

    The residual is averaged **within** a match before being pooled across
    matches, for the same reason bets are: two legs of one fixture cannot both
    be surprises independently.
    """
    buckets: dict[tuple[float, float], list] = {
        bucket: [0.0, 0, 0, Series()] for bucket in PRICE_BUCKETS
    }

    for match in matches:
        per_match: dict[tuple[float, float], list[float]] = {}
        for selection, probability in match.fair.items():
            price = match.book.get(selection)
            if not price:
                continue
            for bucket in PRICE_BUCKETS:
                low, high = bucket
                if low <= price < high:
                    won = 1.0 if selection == match.actual else 0.0
                    entry = buckets[bucket]
                    entry[0] += probability
                    entry[1] += int(won)
                    entry[2] += 1
                    per_match.setdefault(bucket, []).append(won - probability)
                    break
        for bucket, residuals in per_match.items():
            buckets[bucket][3].add(statistics.fmean(residuals))

    return [
        CalibrationBucket(
            label=_bucket_label(bucket),
            legs=legs,
            matches=residual.count,
            implied=implied / legs,
            observed=hits / legs,
            residual=residual,
        )
        for bucket, (implied, hits, legs, residual) in buckets.items()
        if legs
    ]


@dataclass
class RuleScore:
    """One rule settled at both price levels over one body of matches."""

    rule: Rule
    at_book: Returns
    at_best: Returns

    @property
    def price_selection_value(self) -> float:
        """Yield the better price added — margin recovered, not information."""
        return self.at_best.mean - self.at_book.mean

    def to_dict(self) -> dict:
        return {
            "rule": self.rule.label,
            "matches": self.at_best.bets,
            "at_book": self.at_book.to_dict(),
            "at_best": self.at_best.to_dict(),
            "t_at_best": round(self.at_best.t_stat, 2),
            "price_selection_value": round(self.price_selection_value, 4),
        }


def score(matches: list[PricedMatch], rules: list[Rule]) -> list[RuleScore]:
    return [
        RuleScore(
            rule=rule,
            at_book=settle(matches, rule, use_best=False),
            at_best=settle(matches, rule, use_best=True),
        )
        for rule in rules
    ]


@dataclass
class Discovery:
    """Everything the search saw, including how much of it it looked at."""

    league: str
    matches: int
    scores: list[RuleScore]
    calibration: list[CalibrationBucket]
    minimum_bets: int

    @property
    def hypotheses(self) -> int:
        """Rules that were actually eligible to be picked.

        Both price levels count: choosing the better of two settlements of the
        same rule is itself a search.
        """
        return sum(
            (1 if score.at_book.bets >= self.minimum_bets else 0)
            + (1 if score.at_best.bets >= self.minimum_bets else 0)
            for score in self.scores
        )

    @property
    def expected_false_positives(self) -> float:
        return self.hypotheses * FALSE_POSITIVE_RATE

    def ranked(self) -> list[RuleScore]:
        eligible = [s for s in self.scores if s.at_best.bets >= self.minimum_bets]
        return sorted(eligible, key=lambda s: -s.at_best.t_stat)

    def to_dict(self) -> dict:
        return {
            "league": self.league,
            "matches": self.matches,
            "minimum_bets": self.minimum_bets,
            "hypotheses_tested": self.hypotheses,
            "expected_false_positives": round(self.expected_false_positives, 1),
            "calibration": [bucket.to_dict() for bucket in self.calibration],
            "rules": [score.to_dict() for score in self.ranked()],
        }


@dataclass
class ValidationSet:
    """One body of matches the search never influenced."""

    name: str
    score: RuleScore

    def to_dict(self) -> dict:
        return {
            "set": self.name,
            "matches": self.score.at_best.bets,
            "yield": round(self.score.at_best.mean, 4),
            "stderr": round(self.score.at_best.stderr, 4),
            "t": round(self.score.at_best.t_stat, 2),
        }


@dataclass
class Verdict:
    """A carried rule, re-measured out of sample, and what that permits."""

    rule: Rule
    discovery_t: float
    sets: list[ValidationSet]
    pooled: RuleScore
    outcome: str = NOT_ESTABLISHED
    reasons: list[str] = field(default_factory=list)

    @property
    def positive_sets(self) -> int:
        return sum(1 for entry in self.sets if entry.score.at_best.mean > 0)

    @property
    def consistent(self) -> bool:
        return len(self.sets) - self.positive_sets <= CONSISTENCY_TOLERANCE

    @property
    def only_price_selection(self) -> bool:
        """Does the rule pay at the best price purely because the price is better?"""
        return self.pooled.at_best.mean > 0 >= self.pooled.at_book.mean

    def to_dict(self) -> dict:
        low, high = self.pooled.at_best.interval(CONFIDENCE_Z)
        return {
            "rule": self.rule.label,
            "discovery_t": round(self.discovery_t, 2),
            "sets": [entry.to_dict() for entry in self.sets],
            "positive_sets": self.positive_sets,
            "total_sets": len(self.sets),
            "pooled": self.pooled.to_dict(),
            "interval": [round(low, 4), round(high, 4)],
            "only_price_selection": self.only_price_selection,
            "outcome": self.outcome,
            "reasons": self.reasons,
        }


def judge(verdict: Verdict) -> Verdict:
    """Apply the promotion gates to one carried rule, recording why."""
    pooled = verdict.pooled.at_best
    low, high = pooled.interval(CONFIDENCE_Z)

    if pooled.bets < MINIMUM_BETS:
        verdict.outcome = NOT_ESTABLISHED
        verdict.reasons.append(
            f"only {pooled.bets} out-of-sample matches; {MINIMUM_BETS} is the floor for a claim"
        )
        return verdict

    if high < 0:
        verdict.outcome = REJECTED
        verdict.reasons.append(
            f"reliably losing out of sample: yield {pooled.mean:+.4f}, "
            f"95% interval entirely below zero"
        )
        return verdict

    if low <= 0:
        verdict.outcome = NOT_ESTABLISHED
        verdict.reasons.append(
            f"95% interval [{low:+.4f}, {high:+.4f}] contains zero — the sample cannot "
            "tell this apart from no edge at all"
        )
    if not verdict.consistent:
        verdict.reasons.append(
            f"positive on only {verdict.positive_sets} of {len(verdict.sets)} validation sets; "
            "a real bias does not need one set to carry it"
        )
    if verdict.only_price_selection:
        verdict.reasons.append(
            f"loses at a single book ({verdict.pooled.at_book.mean:+.4f}) and only pays at the "
            f"best price ({pooled.mean:+.4f}) — that gap is recovered margin, not a mispricing"
        )

    if not verdict.reasons:
        verdict.outcome = ESTABLISHED
    elif verdict.outcome != REJECTED:
        verdict.outcome = NOT_ESTABLISHED
    return verdict


@dataclass
class Report:
    """The whole search, and the one line it is allowed to conclude."""

    discovery: Discovery
    verdicts: list[Verdict]
    provider: str
    best_provider: str
    devig_method: str

    @property
    def established(self) -> list[Verdict]:
        return [verdict for verdict in self.verdicts if verdict.outcome == ESTABLISHED]

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "best_provider": self.best_provider,
            "devig_method": self.devig_method,
            "discovery": self.discovery.to_dict(),
            "verdicts": [verdict.to_dict() for verdict in self.verdicts],
            "established": [verdict.rule.label for verdict in self.established],
        }


def investigate(
    discovery_matches: list[PricedMatch],
    validation: list[tuple[str, list[PricedMatch]]],
    *,
    selections: tuple[str, ...],
    discovery_league: str = "",
    provider: str = "",
    best_provider: str = "",
    devig_method: str = devig.SHIN,
    minimum_bets: int = MINIMUM_BETS,
    carried: int = CANDIDATES_CARRIED,
) -> Report:
    """Search the discovery matches, then hold the winners to the validation sets.

    ``validation`` is a list of named bodies of matches that had no chance to
    influence which rules were carried — later seasons of the discovery league,
    and whole leagues the search never read.
    """
    rules = price_rules() + list(selection_rules(selections))
    discovery = Discovery(
        league=discovery_league,
        matches=len(discovery_matches),
        scores=score(discovery_matches, rules),
        calibration=calibrate(discovery_matches),
        minimum_bets=minimum_bets,
    )

    verdicts = []
    pooled_matches = [match for _, matches in validation for match in matches]
    for candidate in discovery.ranked()[:carried]:
        verdicts.append(
            judge(
                Verdict(
                    rule=candidate.rule,
                    discovery_t=candidate.at_best.t_stat,
                    sets=[
                        ValidationSet(
                            name=name,
                            score=score(matches, [candidate.rule])[0],
                        )
                        for name, matches in validation
                    ],
                    pooled=score(pooled_matches, [candidate.rule])[0],
                )
            )
        )

    return Report(
        discovery=discovery,
        verdicts=verdicts,
        provider=provider,
        best_provider=best_provider,
        devig_method=devig_method,
    )


def collect_matches(
    league,
    *,
    provider: str,
    best_provider: str,
    devig_method: str = devig.SHIN,
    selections: tuple[str, ...],
) -> list[PricedMatch]:
    """Load every settled match in a league that carries a complete book.

    A match without a full set of quoted outcomes is skipped rather than
    partially devigged: an incomplete book has an overround that cannot be
    separated from an edge.
    """
    # Imported here because this module is otherwise pure arithmetic, and the
    # analysis module pulls in a great deal that a unit test should not need.
    from apps.espn.analysis import _sided_competitors
    from apps.espn.backtest import outcome_of
    from apps.espn.models import Event

    wanted = set(selections)
    matches: list[PricedMatch] = []

    for event in (
        Event.objects.filter(league=league, status=Event.STATUS_FINAL)
        .prefetch_related("competitors__team", "odds")
        .order_by("date")
    ):
        sides = _sided_competitors(event)
        if sides is None:
            continue
        home, away = sides
        if home.score_int is None or away.score_int is None:
            continue

        quotes: dict[str, dict[str, float]] = {}
        for odds in event.odds.all():
            if odds.market == MARKET_MATCH_ODDS and not odds.line:
                quotes.setdefault(odds.provider_espn_id, {})[odds.selection] = odds.decimal_odds

        book = quotes.get(provider)
        if not book or set(book) != wanted:
            continue

        try:
            fair = devig.remove_margin(book, devig_method)
        except devig.DevigError:
            continue

        best = quotes.get(best_provider)
        matches.append(
            PricedMatch(
                date=event.date,
                book=book,
                best=best if best and set(best) == wanted else None,
                actual=outcome_of(home.score_int, away.score_int),
                fair=fair,
            )
        )

    return matches
