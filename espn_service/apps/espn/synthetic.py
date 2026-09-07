"""A market whose truth is known, so a search can be shown what nothing looks like.

Every number this project has produced came out of a search: try some rules,
keep the best. The question that decides whether any of them mean anything is
not "how good is the best rule" but **"how good would the best rule have looked
if there had been nothing to find?"** No amount of real data answers that,
because real data does not come with its truth attached. Simulated data does.

`simulate` builds matches from a known set of true outcome probabilities and a
bookmaker who prices them. Two knobs decide whether the market can be beaten:

``bias``
    A multiplicative tilt on what the bookmaker believes, per selection. Zero
    everywhere means the book believes the truth exactly.

``longshot``
    Loads the margin onto longer prices instead of spreading it evenly — the one
    effect the literature reports consistently. Zero means an even spread.

With both at zero the market is **perfectly efficient in the strongest sense**:
every bet, on every selection, in every match, has the same expected return of
exactly ``-margin / (1 + margin)``. No rule can beat another, because there is
nothing to find. `search_under_the_null` then runs the project's own rule sweep
over many such markets and reports how good the winner looked anyway. That
number is the bar a real result has to clear, and it is not zero.

Turn ``longshot`` up instead and the market becomes beatable in a way we chose,
which is the positive control: a search that cannot find a bias we planted is
not a search worth trusting with one we did not.

Pure Python — the Dirichlet draws are built from `random.gammavariate`.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from apps.espn import devig, market_bias

HOME = "home"
DRAW = "draw"
AWAY = "away"
SELECTIONS = (HOME, DRAW, AWAY)

# Marginal outcome frequencies of European league football, near enough.
BASE_RATES: dict[str, float] = {HOME: 0.45, DRAW: 0.26, AWAY: 0.29}
# Dirichlet concentration. Higher makes every match look like the average one;
# lower makes mismatches more extreme. 10 puts the home probability roughly in
# the 0.15-0.75 range real fixtures occupy.
CONCENTRATION = 10.0
# Total implied probability is 1 + margin.
DEFAULT_MARGIN = 0.05
DEFAULT_MATCHES = 3000
DEFAULT_SEED = 20260907
DEFAULT_TRIALS = 200
# The draw is bounded well away from zero and one in real football; the
# Dirichlet occasionally disagrees, so probabilities are clipped.
MINIMUM_PROBABILITY = 0.01


class SpecError(ValueError):
    """Raised when a market cannot be built as specified."""


@dataclass(frozen=True)
class MarketSpec:
    """Everything that decides whether the simulated market can be beaten."""

    matches: int = DEFAULT_MATCHES
    margin: float = DEFAULT_MARGIN
    longshot: float = 0.0
    bias: Mapping[str, float] = field(default_factory=dict)
    best_price_edge: float = 0.0
    concentration: float = CONCENTRATION
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if self.matches <= 0:
            raise SpecError("A market needs at least one match.")
        if self.margin < 0:
            raise SpecError("A negative margin is not a bookmaker.")
        if self.concentration <= 0:
            raise SpecError("Concentration must be positive.")
        if self.best_price_edge < 0:
            raise SpecError("The best price cannot be worse than the book.")
        unknown = set(self.bias) - set(SELECTIONS)
        if unknown:
            raise SpecError(f"Unknown selection(s) in bias: {', '.join(sorted(unknown))}.")

    @property
    def efficient(self) -> bool:
        """True when the book believes the truth and spreads its margin evenly.

        In that case nothing in this market is beatable, and every result a
        search reports on it is search noise by construction.
        """
        return self.longshot == 0.0 and not any(self.bias.values())

    @property
    def house_edge(self) -> float:
        """The return per unit staked on *any* bet, when the market is efficient."""
        return -self.margin / (1.0 + self.margin)


@dataclass(frozen=True)
class SyntheticMatch:
    """One simulated match, with the truth kept alongside the quoted prices."""

    truth: dict[str, float]
    belief: dict[str, float]
    priced: market_bias.PricedMatch

    def expected_value(self, selection: str) -> float:
        """Profit per unit staked on a selection, under the truth."""
        return self.truth[selection] * self.priced.book[selection] - 1.0


def _dirichlet(rng: random.Random, alphas: Sequence[float]) -> list[float]:
    """A Dirichlet draw built from independent gamma variates."""
    draws = [rng.gammavariate(alpha, 1.0) for alpha in alphas]
    total = sum(draws)
    if total <= 0:
        return [1.0 / len(alphas)] * len(alphas)
    return [draw / total for draw in draws]


def _normalise(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def _clip(probabilities: dict[str, float]) -> dict[str, float]:
    floored = {key: max(value, MINIMUM_PROBABILITY) for key, value in probabilities.items()}
    return _normalise(floored)


def _implied(belief: dict[str, float], margin: float, longshot: float) -> dict[str, float]:
    """Spread ``margin`` across the book, optionally loading it onto longshots.

    ``longshot`` is the exponent bent out of a proportional loading: at zero,
    implied probability is proportional to belief and the margin is even; above
    zero, small probabilities are inflated more than large ones, which is what
    shortens a longshot's price relative to its fair value.
    """
    weights = {
        selection: probability ** (1.0 - longshot) for selection, probability in belief.items()
    }
    scale = (1.0 + margin) / sum(weights.values())
    return {selection: weight * scale for selection, weight in weights.items()}


def simulate(spec: MarketSpec) -> list[SyntheticMatch]:
    """Build a whole market: truths, beliefs, quoted prices and results."""
    rng = random.Random(spec.seed)
    alphas = [spec.concentration * BASE_RATES[selection] for selection in SELECTIONS]
    tilt = {selection: 1.0 + spec.bias.get(selection, 0.0) for selection in SELECTIONS}

    sample: list[SyntheticMatch] = []
    for index in range(spec.matches):
        truth = _clip(dict(zip(SELECTIONS, _dirichlet(rng, alphas), strict=True)))
        belief = _clip(
            _normalise({selection: truth[selection] * tilt[selection] for selection in SELECTIONS})
        )
        implied = _implied(belief, spec.margin, spec.longshot)
        book = {selection: 1.0 / value for selection, value in implied.items()}
        best = (
            {selection: price * (1.0 + spec.best_price_edge) for selection, price in book.items()}
            if spec.best_price_edge > 0
            else None
        )

        roll = rng.random()
        cumulative = 0.0
        actual = SELECTIONS[-1]
        for selection in SELECTIONS:
            cumulative += truth[selection]
            if roll < cumulative:
                actual = selection
                break

        sample.append(
            SyntheticMatch(
                truth=truth,
                belief=belief,
                priced=market_bias.PricedMatch(
                    date=index,
                    book=book,
                    best=best,
                    actual=actual,
                    # The analyst never sees `belief`; they recover it from the
                    # prices exactly as they would on real data.
                    fair=devig.remove_margin(book, devig.SHIN),
                ),
            )
        )
    return sample


def priced(sample: Sequence[SyntheticMatch]) -> list[market_bias.PricedMatch]:
    """The part of a simulated market a search is allowed to see."""
    return [match.priced for match in sample]


# --- what a search finds when there is nothing to find --------------------


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(int(fraction * len(ordered)), len(ordered) - 1)
    return ordered[position]


@dataclass(frozen=True)
class SearchOutcome:
    """The distribution of the best rule's score across repeated null markets."""

    trials: int
    matches: int
    rules: int
    yields: list[float]
    t_stats: list[float]
    labels: list[str]

    @property
    def mean_yield(self) -> float:
        return statistics.fmean(self.yields) if self.yields else 0.0

    @property
    def mean_t(self) -> float:
        return statistics.fmean(self.t_stats) if self.t_stats else 0.0

    def yield_at(self, fraction: float) -> float:
        return _percentile(self.yields, fraction)

    def t_at(self, fraction: float) -> float:
        return _percentile(self.t_stats, fraction)

    @property
    def clears_conventional_t(self) -> float:
        """Share of null searches whose winner cleared the usual |t| >= 2 bar."""
        if not self.t_stats:
            return 0.0
        return sum(1 for t in self.t_stats if t >= market_bias.SIGNIFICANT_T) / len(self.t_stats)

    def beats(self, observed_yield: float) -> float:
        """The share of null searches whose winner reached an observed yield."""
        if not self.yields:
            return 1.0
        return sum(1 for value in self.yields if value >= observed_yield) / len(self.yields)

    def beats_t(self, observed_t: float) -> float:
        """The share of null searches whose winner reached an observed t-statistic.

        This is the honest p-value for a discovery reported as "t = x", because
        the search reports the largest t out of many, not one t.
        """
        if not self.t_stats:
            return 1.0
        return sum(1 for value in self.t_stats if value >= observed_t) / len(self.t_stats)

    @property
    def favourite_share(self) -> float:
        """How often the winning rule was a selection rule rather than a price band."""
        if not self.labels:
            return 0.0
        return sum(1 for label in self.labels if label.startswith("selection")) / len(self.labels)


def sweep(
    sample: Sequence[SyntheticMatch],
    *,
    minimum_bets: int = market_bias.MINIMUM_BETS,
) -> market_bias.RuleScore | None:
    """Run the project's own rule sweep and return its winner, or None.

    Deliberately calls `market_bias.score` rather than reimplementing it: the
    point is to measure *this* search procedure, not an idealised one.
    """
    rules = market_bias.price_rules() + list(market_bias.selection_rules(SELECTIONS))
    scores = market_bias.score(priced(sample), rules)
    eligible = [entry for entry in scores if entry.at_best.bets >= minimum_bets]
    if not eligible:
        return None
    # Ranked by t-statistic, because that is what `Discovery.ranked` does. Picking
    # by yield instead would measure a search this project does not run — and
    # would flatter thin, high-variance price bands enormously.
    return max(eligible, key=lambda entry: entry.at_best.t_stat)


def search_under_the_null(
    spec: MarketSpec,
    *,
    trials: int = DEFAULT_TRIALS,
    minimum_bets: int = market_bias.MINIMUM_BETS,
) -> SearchOutcome:
    """Repeat the whole search on fresh markets and collect the winners.

    Each trial gets its own seed, so the trials are independent markets rather
    than re-reads of one.
    """
    if trials <= 0:
        raise SpecError("A null needs at least one trial.")
    yields: list[float] = []
    t_stats: list[float] = []
    labels: list[str] = []
    rules = len(market_bias.price_rules()) + len(SELECTIONS)

    for trial in range(trials):
        sample = simulate(replace(spec, seed=spec.seed + trial))
        winner = sweep(sample, minimum_bets=minimum_bets)
        if winner is None:
            continue
        yields.append(winner.at_best.mean)
        t_stats.append(winner.at_best.t_stat)
        labels.append(winner.rule.label)

    return SearchOutcome(
        trials=len(yields),
        matches=spec.matches,
        rules=rules,
        yields=yields,
        t_stats=t_stats,
        labels=labels,
    )


def realised_house_edge(sample: Sequence[SyntheticMatch]) -> float:
    """Average expected value across every selection actually quoted.

    On an efficient market this converges to `MarketSpec.house_edge`; it is the
    check that the generator built the market it was asked for.
    """
    values = [match.expected_value(selection) for match in sample for selection in SELECTIONS]
    return statistics.fmean(values) if values else 0.0


def edge_spread(sample: Sequence[SyntheticMatch]) -> float:
    """How much the expected value varies across selections.

    Zero means every bet is equally bad, which is what "no edge anywhere" means
    operationally. Anything above zero is an edge someone could look for.
    """
    values = [match.expected_value(selection) for match in sample for selection in SELECTIONS]
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def power_against(
    spec: MarketSpec,
    null: SearchOutcome,
    *,
    trials: int = 40,
    level: float = 0.95,
) -> float:
    """How often a search on a *beatable* market beats the null's threshold.

    This is the positive control: a gate that never fires on a bias we planted
    is not protecting anything, it is just refusing.
    """
    threshold = null.t_at(level)
    hits = 0
    counted = 0
    for trial in range(trials):
        sample = simulate(replace(spec, seed=spec.seed + 10_000 + trial))
        winner = sweep(sample)
        if winner is None:
            continue
        counted += 1
        if winner.at_best.t_stat > threshold:
            hits += 1
    return hits / counted if counted else 0.0
