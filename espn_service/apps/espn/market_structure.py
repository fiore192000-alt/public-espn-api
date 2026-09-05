"""What the market costs, and how much of that cost price selection recovers.

The bookmaker's margin is the headwind every bettor runs into before any question
of skill arises. This module measures it the only way that cannot be argued with:
by settling real bets at real prices against real results.

The distinction it exists to enforce: taking a better price **recovers margin**, it
does not **create an edge**. A bettor with no predictive skill who always takes the
best available price still expects to lose — just far less. Line shopping is what
makes a genuine edge survivable, not a substitute for having one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Below this many settled bets a yield is not worth quoting.
MINIMUM_BETS = 30


@dataclass
class Returns:
    """Realised return of flat-staking one selection at one price level."""

    bets: int = 0
    total: float = 0.0
    wins: int = 0
    _squares: float = 0.0

    def add(self, decimal_odds: float, won: bool) -> None:
        self._add_aggregate(profit=decimal_odds - 1.0 if won else -1.0, won=won)

    def _add_aggregate(self, profit: float, won: bool) -> None:
        """Record one observation of profit per unit staked."""
        self.bets += 1
        self.total += profit
        self._squares += profit * profit
        self.wins += 1 if won else 0

    @property
    def mean(self) -> float:
        return self.total / self.bets if self.bets else 0.0

    @property
    def stderr(self) -> float:
        """Standard error of the mean return per unit staked."""
        if self.bets < 2:
            return 0.0
        variance = (self._squares - self.bets * self.mean**2) / (self.bets - 1)
        return math.sqrt(max(variance, 0.0) / self.bets)

    def to_dict(self) -> dict:
        return {
            "bets": self.bets,
            "hit_rate": round(self.wins / self.bets, 4) if self.bets else None,
            "yield": round(self.mean, 4),
            "stderr": round(self.stderr, 4),
        }


@dataclass
class SelectionComparison:
    """One selection, flat-staked at a single book and at the best price."""

    selection: str
    at_book: Returns
    at_best: Returns

    @property
    def recovered(self) -> float:
        """How much yield the better price added, per unit staked."""
        return self.at_best.mean - self.at_book.mean

    @property
    def profitable_at_best(self) -> bool:
        """Does taking the best price alone turn this into a winning bet?

        Two standard errors, so a yield that merely looks positive does not count.
        """
        return self.at_best.bets >= MINIMUM_BETS and (
            self.at_best.mean - 2.0 * self.at_best.stderr > 0
        )

    def to_dict(self) -> dict:
        return {
            "selection": self.selection,
            "at_book": self.at_book.to_dict(),
            "at_best": self.at_best.to_dict(),
            "recovered": round(self.recovered, 4),
            "profitable_at_best": self.profitable_at_best,
        }


@dataclass
class PriceComparison:
    """Per-selection results plus a correctly pooled figure."""

    selections: list[SelectionComparison]
    # One entry per match, not per leg — see below.
    pooled_at_book: Returns
    pooled_at_best: Returns

    @property
    def recovered(self) -> float:
        return self.pooled_at_best.mean - self.pooled_at_book.mean

    @property
    def profitable_at_best(self) -> bool:
        return self.pooled_at_best.bets >= MINIMUM_BETS and (
            self.pooled_at_best.mean - 2.0 * self.pooled_at_best.stderr > 0
        )

    def to_dict(self) -> dict:
        return {
            "selections": [comparison.to_dict() for comparison in self.selections],
            "pooled": {
                "matches": self.pooled_at_book.bets,
                "at_book": self.pooled_at_book.to_dict(),
                "at_best": self.pooled_at_best.to_dict(),
                "recovered": round(self.recovered, 4),
                # The question line shopping alone cannot answer yes to, on any
                # market that is even roughly efficient.
                "profitable_at_best": self.profitable_at_best,
            },
        }


def compare_prices(
    books: list[tuple[dict[str, float], dict[str, float], str]],
    selections: tuple[str, ...],
) -> PriceComparison:
    """Flat-stake every selection at both price levels and settle against results.

    ``books`` pairs one bookmaker's prices with the best available prices and the
    selection that actually won. No probability model is involved: these are
    realised returns, so nothing here depends on a model being right.

    The pooled figure is accumulated **per match, not per leg**. Backing all three
    outcomes of one match is a single dependent event — exactly one leg wins, so
    the match returns ``winning_odds - 3`` on three units staked no matter what.
    Treating the legs as three independent bets would understate the spread of the
    pooled yield by roughly the square root of three.
    """
    comparisons = {
        selection: SelectionComparison(selection, Returns(), Returns()) for selection in selections
    }
    pooled_at_book = Returns()
    pooled_at_best = Returns()

    for book, best, actual in books:
        usable = [selection for selection in selections if selection in book and selection in best]
        if not usable:
            continue

        for selection in usable:
            won = selection == actual
            comparisons[selection].at_book.add(book[selection], won)
            comparisons[selection].at_best.add(best[selection], won)

        # Collapse the match into one observation at each price level.
        for prices, target in ((book, pooled_at_book), (best, pooled_at_best)):
            staked = len(usable)
            returned = prices[actual] if actual in usable else 0.0
            target._add_aggregate(profit=(returned - staked) / staked, won=actual in usable)

    return PriceComparison(
        selections=[comparison for comparison in comparisons.values() if comparison.at_book.bets],
        pooled_at_book=pooled_at_book,
        pooled_at_best=pooled_at_best,
    )


def summarise(comparison: PriceComparison) -> dict:
    """One statement about what price selection is worth."""
    return comparison.to_dict()
