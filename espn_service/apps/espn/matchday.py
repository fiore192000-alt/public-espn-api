"""Settling one day's card under staking rules that were fixed in advance.

The point of this module is not to find bets. It is to answer a question that
sounds simple and is easy to answer dishonestly: *given the rules we already
measured, what would they have staked on a particular day, and what would that
have done to a bankroll?*

Three things keep the answer honest.

**The rules are named and fixed.** Each one is a function registered in
`RULES`, chosen before the results are known, with the thresholds already in
`apps.espn.value` — a 5% edge, quarter Kelly, a 5% cap. A rule invented after
seeing the card is a rule that wins on the card.

**The market's own price is the truth used for the null.** Every fixture's
prices are devigged (Shin by default) into the book's fair probabilities. Those
are what `distribution` uses to enumerate every way the day could have gone, so
a day's profit or loss can be placed against the spread of days that could
plausibly have happened instead of being read as a verdict.

**A settled day is a sample of one.** `Ledger.roi` is the realised return, and
`Outcome.percentile` says where it sits in the distribution above. On a
four-match card the standard deviation of a day's return is on the order of the
return itself, which is exactly why nothing here reports a rule as good or bad.
The measurements that can carry that weight live in `market_bias` and `clv`,
over thousands of matches.

Everything is pure Python; the enumeration is exponential in the number of
fixtures a rule touches, which is fine for a day's card and deliberately not
offered for a season.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from apps.espn import value

HOME = "home"
DRAW = "draw"
AWAY = "away"
SELECTIONS = (HOME, DRAW, AWAY)

# Thresholds are imported rather than restated so a change to the staking policy
# reaches this simulation too.
EDGE_THRESHOLD = value.DEFAULT_EDGE_THRESHOLD
KELLY_FRACTION = value.DEFAULT_KELLY_FRACTION
MAX_STAKE_FRACTION = value.DEFAULT_MAX_STAKE_FRACTION
DEVIG_METHOD = value.DEFAULT_DEVIG_METHOD
# Rules that stake a fixed amount regardless of edge use this share of bankroll.
FLAT_STAKE_FRACTION = 0.02
# Two legs is the smallest multiple a punter actually builds.
ACCUMULATOR_LEGS = 2

MARKET = "market"
MODEL = "model"


class CardError(ValueError):
    """Raised when a fixture cannot be used as priced."""


def _normalise(probabilities: dict[str, float]) -> dict[str, float]:
    total = sum(probabilities.values())
    if total <= 0:
        raise CardError("Model probabilities must sum to something positive.")
    return {selection: weight / total for selection, weight in probabilities.items()}


@dataclass(frozen=True)
class Fixture:
    """One match: what the book charges, what the model believes, what happened."""

    home: str
    away: str
    prices: dict[str, float]
    model: dict[str, float]
    result: str | None = None
    kickoff: str = ""
    book: str = ""

    def __post_init__(self) -> None:
        missing = [selection for selection in SELECTIONS if selection not in self.prices]
        if missing:
            raise CardError(f"{self.name}: no price for {', '.join(missing)}.")
        if self.result is not None and self.result not in SELECTIONS:
            raise CardError(f"{self.name}: {self.result!r} is not a 1X2 result.")
        object.__setattr__(self, "model", _normalise(self.model))

    @property
    def name(self) -> str:
        return f"{self.home}-{self.away}"

    @property
    def settled(self) -> bool:
        return self.result is not None

    @property
    def fair(self) -> dict[str, float]:
        """What the book believes, with its margin removed."""
        removed = value.remove_margin(self.prices, DEVIG_METHOD)
        if not removed:
            raise CardError(f"{self.name}: prices cannot be devigged.")
        return removed

    @property
    def overround(self) -> float:
        return sum(1.0 / price for price in self.prices.values() if price > 1.0)

    @property
    def margin(self) -> float:
        total = self.overround
        return (total - 1.0) / total if total > 0 else 0.0

    def price(self, selection: str) -> float:
        return self.prices[selection]

    def edge(self, selection: str) -> float:
        """How far the model's probability sits above the devigged price."""
        return self.model[selection] - self.fair[selection]

    def expected_value(self, selection: str) -> float:
        """Profit per unit staked, if the model is right."""
        return value.expected_value(self.model[selection], self.prices[selection])

    def probabilities(self, source: str) -> dict[str, float]:
        if source == MODEL:
            return dict(self.model)
        if source == MARKET:
            return self.fair
        raise CardError(f"{source!r} is not a probability source.")


@dataclass(frozen=True)
class Leg:
    """One selection inside a bet."""

    fixture: Fixture
    selection: str

    @property
    def price(self) -> float:
        return self.fixture.price(self.selection)

    @property
    def model_probability(self) -> float:
        return self.fixture.model[self.selection]

    @property
    def fair_probability(self) -> float:
        return self.fixture.fair[self.selection]

    @property
    def edge(self) -> float:
        return self.fixture.edge(self.selection)

    @property
    def expected_value(self) -> float:
        return self.fixture.expected_value(self.selection)

    @property
    def won(self) -> bool | None:
        if not self.fixture.settled:
            return None
        return self.fixture.result == self.selection


@dataclass(frozen=True)
class Bet:
    """A stake on one selection, or on several that must all land."""

    rule: str
    legs: tuple[Leg, ...]
    stake: float

    def __post_init__(self) -> None:
        if not self.legs:
            raise CardError("A bet needs at least one leg.")
        names = [leg.fixture.name for leg in self.legs]
        if len(set(names)) != len(names):
            raise CardError("A multiple cannot take two legs from the same match.")

    @property
    def price(self) -> float:
        return math.prod(leg.price for leg in self.legs)

    @property
    def model_probability(self) -> float:
        return math.prod(leg.model_probability for leg in self.legs)

    @property
    def fair_probability(self) -> float:
        return math.prod(leg.fair_probability for leg in self.legs)

    @property
    def expected_value(self) -> float:
        """Per unit staked, under the model."""
        return self.model_probability * self.price - 1.0

    @property
    def market_expected_value(self) -> float:
        """Per unit staked, under the book's own devigged probabilities.

        This is the margin the punter pays, and it is negative by construction
        for any bet a bookmaker is willing to take.
        """
        return self.fair_probability * self.price - 1.0

    @property
    def label(self) -> str:
        return " + ".join(f"{leg.fixture.name} {leg.selection}" for leg in self.legs)

    @property
    def settled(self) -> bool:
        return all(leg.fixture.settled for leg in self.legs)

    @property
    def won(self) -> bool | None:
        if not self.settled:
            return None
        return all(leg.won for leg in self.legs)

    @property
    def returned(self) -> float:
        """Stake plus profit; zero on a loser, zero while unsettled."""
        return self.stake * self.price if self.won else 0.0

    @property
    def profit(self) -> float:
        if not self.settled:
            return 0.0
        return self.returned - self.stake

    def profit_if(self, results: dict[str, str]) -> float:
        """Profit under a hypothetical set of results, keyed by fixture name."""
        landed = all(results[leg.fixture.name] == leg.selection for leg in self.legs)
        return self.stake * (self.price - 1.0) if landed else -self.stake


def _stake(bankroll: float, fraction: float) -> float:
    return round(bankroll * min(fraction, MAX_STAKE_FRACTION), 2)


def _kelly_stake(bankroll: float, probability: float, price: float) -> float:
    fraction = value.kelly_fraction(probability, price) * KELLY_FRACTION
    return _stake(bankroll, fraction)


def _ranked(fixtures: Iterable[Fixture], key: Callable[[Fixture, str], float]) -> list[Leg]:
    """Every fixture/selection pair on the card, best first by ``key``."""
    legs = [Leg(fixture, selection) for fixture in fixtures for selection in SELECTIONS]
    return sorted(legs, key=lambda leg: key(leg.fixture, leg.selection), reverse=True)


# --- the rules ------------------------------------------------------------
#
# Each takes the card and a bankroll and returns the bets it would have placed.
# None of them look at ``Fixture.result``.


def value_rule(fixtures: Sequence[Fixture], bankroll: float) -> list[Bet]:
    """The staking policy the service already implements: edge, then Kelly."""
    bets = []
    for leg in _ranked(fixtures, lambda fixture, selection: fixture.edge(selection)):
        if leg.edge < EDGE_THRESHOLD:
            continue
        stake = _kelly_stake(bankroll, leg.model_probability, leg.price)
        if stake > 0:
            bets.append(Bet("value", (leg,), stake))
    return bets


def best_edge_rule(fixtures: Sequence[Fixture], bankroll: float) -> list[Bet]:
    """One flat bet on the day's largest disagreement with the price.

    Measured over 3,004 Serie A matches this returned -7.2% (t = -3.31): the
    model's own edge signal points the wrong way, which is why the rule is here
    as a documented loser rather than as a suggestion.
    """
    ranked = _ranked(fixtures, lambda fixture, selection: fixture.edge(selection))
    if not ranked:
        return []
    return [Bet("best-edge", (ranked[0],), _stake(bankroll, FLAT_STAKE_FRACTION))]


def shortest_price_rule(fixtures: Sequence[Fixture], bankroll: float) -> list[Bet]:
    """One flat bet on the day's shortest price, ignoring the model entirely.

    The favourite-longshot bias is the one effect that survived the sweep, at
    +1.82% (t = 1.09) — a sign, not a finding.
    """
    ranked = _ranked(fixtures, lambda fixture, selection: -fixture.price(selection))
    if not ranked:
        return []
    return [Bet("shortest-price", (ranked[0],), _stake(bankroll, FLAT_STAKE_FRACTION))]


def market_favourites_rule(fixtures: Sequence[Fixture], bankroll: float) -> list[Bet]:
    """A flat stake on every market favourite: the control the others answer to."""
    bets = []
    for fixture in fixtures:
        selection = min(SELECTIONS, key=fixture.price)
        bets.append(
            Bet(
                "market-favourites",
                (Leg(fixture, selection),),
                _stake(bankroll, FLAT_STAKE_FRACTION),
            )
        )
    return bets


def accumulator_rule(fixtures: Sequence[Fixture], bankroll: float) -> list[Bet]:
    """The day's two largest edges combined into one multiple.

    A multiple multiplies the bookmaker's margin as well as the price, so its
    expected value is worse than either leg alone. It is included because it is
    what gets bet in practice, not because it is defensible.
    """
    ranked = _ranked(fixtures, lambda fixture, selection: fixture.edge(selection))
    legs: list[Leg] = []
    used: set[str] = set()
    for leg in ranked:
        if leg.fixture.name in used:
            continue
        legs.append(leg)
        used.add(leg.fixture.name)
        if len(legs) == ACCUMULATOR_LEGS:
            break
    if len(legs) < ACCUMULATOR_LEGS:
        return []
    return [Bet("accumulator", tuple(legs), _stake(bankroll, FLAT_STAKE_FRACTION))]


RULES: dict[str, Callable[[Sequence[Fixture], float], list[Bet]]] = {
    "value": value_rule,
    "best-edge": best_edge_rule,
    "shortest-price": shortest_price_rule,
    "market-favourites": market_favourites_rule,
    "accumulator": accumulator_rule,
}

DESCRIPTIONS: dict[str, str] = {
    "value": f"edge >= {EDGE_THRESHOLD:.0%}, {KELLY_FRACTION:.2f} Kelly, {MAX_STAKE_FRACTION:.0%} cap",
    "best-edge": "flat stake on the day's biggest model/market disagreement",
    "shortest-price": "flat stake on the day's shortest price",
    "market-favourites": "flat stake on every market favourite (control)",
    "accumulator": f"the {ACCUMULATOR_LEGS} biggest edges as one multiple",
}


# --- settlement -----------------------------------------------------------


@dataclass
class Ledger:
    """What one rule staked on one card, and what came back."""

    rule: str
    bankroll: float
    bets: list[Bet] = field(default_factory=list)

    @property
    def staked(self) -> float:
        return sum(bet.stake for bet in self.bets)

    @property
    def returned(self) -> float:
        return sum(bet.returned for bet in self.bets)

    @property
    def settled_stake(self) -> float:
        return sum(bet.stake for bet in self.bets if bet.settled)

    @property
    def profit(self) -> float:
        return sum(bet.profit for bet in self.bets)

    @property
    def closing(self) -> float:
        return self.bankroll + self.profit

    @property
    def roi(self) -> float:
        """Profit per unit staked on bets that have actually settled."""
        staked = self.settled_stake
        return self.profit / staked if staked > 0 else 0.0

    @property
    def growth(self) -> float:
        return self.profit / self.bankroll if self.bankroll > 0 else 0.0

    @property
    def exposure(self) -> float:
        return self.staked / self.bankroll if self.bankroll > 0 else 0.0

    @property
    def model_expected_profit(self) -> float:
        return sum(bet.stake * bet.expected_value for bet in self.bets)

    @property
    def market_expected_profit(self) -> float:
        """What the book expects to keep — the margin, in euros."""
        return sum(bet.stake * bet.market_expected_value for bet in self.bets)


def settle(fixtures: Sequence[Fixture], rule: str, bankroll: float) -> Ledger:
    """Run one named rule over the card."""
    if rule not in RULES:
        raise CardError(f"{rule!r} is not a known rule.")
    return Ledger(rule=rule, bankroll=bankroll, bets=RULES[rule](fixtures, bankroll))


def settle_all(fixtures: Sequence[Fixture], bankroll: float) -> list[Ledger]:
    return [settle(fixtures, rule, bankroll) for rule in RULES]


# --- how much of this is luck --------------------------------------------


@dataclass(frozen=True)
class Outcome:
    """The exact distribution of a day's profit, given a probability source."""

    source: str
    mean: float
    deviation: float
    realised: float | None
    percentile: float | None
    best: float
    worst: float
    profitable: float

    @property
    def z(self) -> float | None:
        """How many day-sized standard deviations the realised day sits from the mean."""
        if self.realised is None or self.deviation <= 0:
            return None
        return (self.realised - self.mean) / self.deviation


def distribution(bets: Sequence[Bet], source: str = MARKET) -> list[tuple[float, float]]:
    """Every way the day could have gone: (profit, probability), exhaustively.

    Enumerates the outcome of each fixture a bet touches, so the cost is 3^n in
    the number of such fixtures. A day's card is a handful; a season is not, and
    this is not the function for one.
    """
    fixtures: list[Fixture] = []
    for bet in bets:
        for leg in bet.legs:
            if all(leg.fixture.name != known.name for known in fixtures):
                fixtures.append(leg.fixture)
    if not fixtures:
        return []
    weights = [fixture.probabilities(source) for fixture in fixtures]
    outcomes = []
    for combination in itertools.product(SELECTIONS, repeat=len(fixtures)):
        results = {
            fixture.name: result for fixture, result in zip(fixtures, combination, strict=True)
        }
        probability = math.prod(
            weight[result] for weight, result in zip(weights, combination, strict=True)
        )
        profit = sum(bet.profit_if(results) for bet in bets)
        outcomes.append((profit, probability))
    return outcomes


def summarise(bets: Sequence[Bet], source: str = MARKET) -> Outcome | None:
    """Mean, spread, and where the realised day actually landed."""
    outcomes = distribution(bets, source)
    if not outcomes:
        return None
    total = sum(probability for _, probability in outcomes)
    if total <= 0:
        return None
    mean = sum(profit * probability for profit, probability in outcomes) / total
    variance = sum(probability * (profit - mean) ** 2 for profit, probability in outcomes) / total
    settled = all(bet.settled for bet in bets)
    realised = sum(bet.profit for bet in bets) if settled else None
    percentile = None
    if realised is not None:
        below = sum(probability for profit, probability in outcomes if profit < realised - 1e-9)
        equal = sum(
            probability for profit, probability in outcomes if abs(profit - realised) <= 1e-9
        )
        percentile = (below + equal / 2.0) / total
    profitable = sum(probability for profit, probability in outcomes if profit > 0) / total
    return Outcome(
        source=source,
        mean=mean,
        deviation=math.sqrt(variance),
        realised=realised,
        percentile=percentile,
        best=max(profit for profit, _ in outcomes),
        worst=min(profit for profit, _ in outcomes),
        profitable=profitable,
    )


# --- loading a card -------------------------------------------------------


def fixture_from_dict(entry: dict) -> Fixture:
    """Build a fixture from the JSON shape used by the card files."""
    try:
        prices = {key: float(price) for key, price in entry["prices"].items()}
        model = {key: float(weight) for key, weight in entry["model"].items()}
    except (KeyError, TypeError, ValueError) as error:
        raise CardError(f"Unusable fixture entry: {error}") from error
    return Fixture(
        home=str(entry.get("home", "")),
        away=str(entry.get("away", "")),
        prices=prices,
        model=model,
        result=entry.get("result"),
        kickoff=str(entry.get("kickoff", "")),
        book=str(entry.get("book", "")),
    )


def card_from_dict(document: dict) -> list[Fixture]:
    fixtures = document.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise CardError("A card needs a non-empty 'fixtures' list.")
    return [fixture_from_dict(entry) for entry in fixtures]
