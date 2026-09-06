"""Compare model probabilities against bookmaker prices and size stakes.

A price only carries information once the bookmaker's margin is removed, so
selections are devigged within their own market before any comparison. The
resulting "fair" probability is the market's own opinion; the model's edge is
how far it disagrees.

Two things are worth being clear about. First, devigging a single bookmaker's
prices recovers roughly that bookmaker's view, and beating it consistently is
exactly the hard part — a positive edge here is a hypothesis, not a profit.
Second, Kelly assumes the model's probability is correct; since it is not, the
fractional multiplier below is a hedge against that error, and it defaults low
on purpose.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from apps.espn import devig
from apps.espn.markets import MARKET_TOTALS

# Model probability must exceed the devigged market probability by this much.
DEFAULT_EDGE_THRESHOLD = 0.05
# Kelly is scaled down because the model's probabilities carry estimation error.
DEFAULT_KELLY_FRACTION = 0.25
# No single bet takes more than this share of the bankroll, whatever Kelly says.
DEFAULT_MAX_STAKE_FRACTION = 0.05
# Measured on 3,799 Serie A books: proportional devig is worse than both Shin and
# power by a distinguishable margin (paired t = -3.6 and -3.2), while Shin and
# power cannot be told apart (t = -1.9). Shin wins the tie on grounds the data
# cannot settle — it models *why* the margin sits where it does, rather than
# fitting an exponent to make the book sum to one.
DEFAULT_DEVIG_METHOD = devig.SHIN


@dataclass
class PricedSelection:
    """One bookmaker price, as stored by the Odds model."""

    market: str
    selection: str
    line: str
    decimal_odds: float
    provider_espn_id: str = ""
    provider_name: str = ""


@dataclass
class ValueBet:
    """A selection the model rates higher than the devigged market price."""

    market: str
    selection: str
    line: str
    decimal_odds: float
    provider_name: str
    model_probability: float
    fair_probability: float
    edge: float
    expected_value: float
    kelly_fraction: float
    stake_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


def overround(prices: list[float]) -> float:
    """Total implied probability of a market; 1.0 would be a margin-free book."""
    return sum(1.0 / price for price in prices if price > 0)


def remove_margin(
    prices: dict[str, float],
    method: str = DEFAULT_DEVIG_METHOD,
) -> dict[str, float]:
    """Recover what the book believes, with the bookmaker's margin taken out.

    Returns an empty mapping rather than raising when the prices cannot be
    devigged, since callers here treat an unusable market as one to skip.
    """
    try:
        return devig.remove_margin(prices, method)
    except devig.DevigError:
        return {}


def expected_value(probability: float, decimal_odds: float) -> float:
    """Profit per unit staked, given the model's probability."""
    return probability * decimal_odds - 1.0


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    """Full-Kelly share of bankroll; zero when the bet has no edge."""
    profit = decimal_odds - 1.0
    if profit <= 0:
        return 0.0
    fraction = (probability * decimal_odds - 1.0) / profit
    return max(fraction, 0.0)


def model_probability(
    model_markets: dict, market: str, selection: str, line: str = ""
) -> float | None:
    """Look up the model's probability for a market/selection/line."""
    if market == MARKET_TOTALS:
        entry = (model_markets.get(MARKET_TOTALS) or {}).get(line) or {}
    else:
        entry = model_markets.get(market) or {}
    priced = entry.get(selection)
    return priced.get("probability") if isinstance(priced, dict) else None


def _group_key(price: PricedSelection) -> tuple[str, str, str]:
    return price.provider_espn_id, price.market, price.line


def find_value_bets(
    model_markets: dict,
    prices: list[PricedSelection],
    *,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    kelly_multiplier: float = DEFAULT_KELLY_FRACTION,
    max_stake_fraction: float = DEFAULT_MAX_STAKE_FRACTION,
    devig_method: str = DEFAULT_DEVIG_METHOD,
) -> list[ValueBet]:
    """Find selections where the model's probability beats the devigged price.

    Prices are grouped per provider and market so each market is devigged against
    its own complement. A market with only one side quoted cannot be devigged and
    is skipped, since its overround would be indistinguishable from an edge.
    """
    groups: dict[tuple[str, str, str], list[PricedSelection]] = {}
    for price in prices:
        groups.setdefault(_group_key(price), []).append(price)

    bets: list[ValueBet] = []
    for group in groups.values():
        if len(group) < 2:
            continue

        fair = remove_margin({price.selection: price.decimal_odds for price in group}, devig_method)
        if not fair:
            continue

        for price in group:
            modelled = model_probability(model_markets, price.market, price.selection, price.line)
            fair_probability = fair.get(price.selection)
            if modelled is None or fair_probability is None:
                continue

            edge = modelled - fair_probability
            value = expected_value(modelled, price.decimal_odds)
            if edge < edge_threshold or value <= 0:
                continue

            full_kelly = kelly_fraction(modelled, price.decimal_odds)
            bets.append(
                ValueBet(
                    market=price.market,
                    selection=price.selection,
                    line=price.line,
                    decimal_odds=price.decimal_odds,
                    provider_name=price.provider_name,
                    model_probability=round(modelled, 5),
                    fair_probability=round(fair_probability, 5),
                    edge=round(edge, 5),
                    expected_value=round(value, 5),
                    kelly_fraction=round(full_kelly, 5),
                    stake_fraction=round(min(full_kelly * kelly_multiplier, max_stake_fraction), 5),
                )
            )

    bets.sort(key=lambda bet: bet.expected_value, reverse=True)
    return bets
