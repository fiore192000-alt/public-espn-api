"""Polymarket as a price source for the movement bench.

An exchange is not a bookmaker, and the difference removes a whole layer of
machinery. A bookmaker quotes three prices that imply more than 100% and the
excess has to be modelled away — that is what `devig.py` exists for, and what
Shin, power and proportional argue about. On Polymarket a share pays exactly
$1 when it resolves true, so **the price is the probability**. There is nothing
to remove.

What replaces the margin is the **spread**: the gap between the best bid and the
best ask, half of which you cross to get filled. It plays the same role in the
arithmetic — a signal has to be worth more than the cost of acting on it — but it
is observable per market and per moment rather than assumed.

The rest maps onto `movement.py` unchanged. The market's own correction between
an early price and the price before resolution is the same target the closing
line provides in football, with two advantages that matter: the trajectory is
continuous rather than two snapshots, and it exists for every resolved market
rather than for 28% of them.

> **The HTTP layer in this module has not been exercised against the live API.**
> Polymarket's hosts are blocked by this environment's egress policy, so the
> endpoints and payload shapes here were read from Polymarket's own
> `py-clob-client` source rather than from a response. The pure functions below
> are tested; `fetch_json` is not, and the first thing to do where egress works
> is run `check_polymarket` before trusting anything downstream of it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass

from apps.espn import movement

# Hosts and paths, taken from Polymarket/py-clob-client.
CLOB_HOST = "https://clob.polymarket.com"
GAMMA_HOST = "https://gamma-api.polymarket.com"
BOOK_PATH = "/book"
MIDPOINT_PATH = "/midpoint"
MARKETS_PATH = "/markets"
PRICES_HISTORY_PATH = "/prices-history"

# A share settles at 1.0, so a price outside this range is not a probability.
MIN_PRICE = 0.0
MAX_PRICE = 1.0
DEFAULT_TIMEOUT = 20.0


class PolymarketError(RuntimeError):
    """Raised when a response cannot be used as a price."""


@dataclass(frozen=True)
class Level:
    """One side of the book at one price."""

    price: float
    size: float


@dataclass(frozen=True)
class Quote:
    """A market's state at one moment, as the bench needs it."""

    token: str
    timestamp: int
    bids: tuple[Level, ...] = ()
    asks: tuple[Level, ...] = ()
    last_trade: float | None = None

    @property
    def best_bid(self) -> float | None:
        return max((level.price for level in self.bids), default=None)

    @property
    def best_ask(self) -> float | None:
        return min((level.price for level in self.asks), default=None)

    @property
    def mid(self) -> float | None:
        """The midpoint, which is this venue's fair probability.

        Falls back to whichever side exists when the book is one-sided, and to
        the last trade when it is empty — both are worse estimates, and a caller
        that cares should check `spread` rather than trusting `mid` blindly.
        """
        bid, ask = self.best_bid, self.best_ask
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        if bid is not None:
            return bid
        if ask is not None:
            return ask
        return self.last_trade

    @property
    def spread(self) -> float | None:
        """Bid-ask gap: the exchange's analogue of a bookmaker's margin."""
        bid, ask = self.best_bid, self.best_ask
        return None if bid is None or ask is None else ask - bid

    @property
    def cost_to_cross(self) -> float | None:
        """Half the spread — what taking liquidity costs, in probability."""
        gap = self.spread
        return None if gap is None else gap / 2.0

    @property
    def usable(self) -> bool:
        """A two-sided book inside the unit interval, which is what a price means."""
        mid = self.mid
        return (
            self.spread is not None
            and self.spread >= 0
            and mid is not None
            and MIN_PRICE < mid < MAX_PRICE
        )

    def depth(self, side: str, within: float) -> float:
        """Size available within `within` of the touch, on one side.

        Liquidity is the constraint an exchange has instead of a betting limit,
        so a signal that only exists in markets with no depth is not a signal
        anybody can act on.
        """
        if side == "bid":
            best = self.best_bid
            return sum(
                level.size
                for level in self.bids
                if best is not None and level.price >= best - within
            )
        best = self.best_ask
        return sum(
            level.size for level in self.asks if best is not None and level.price <= best + within
        )


def _levels(raw: object) -> tuple[Level, ...]:
    """Read one side of a book, dropping anything that is not a usable level."""
    if not isinstance(raw, list):
        return ()
    out = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            price = float(entry["price"])
            size = float(entry["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if MIN_PRICE <= price <= MAX_PRICE and size > 0:
            out.append(Level(price=price, size=size))
    return tuple(out)


def quote_from_book(payload: dict) -> Quote:
    """Build a Quote from a CLOB `/book` response."""
    if not isinstance(payload, dict):
        raise PolymarketError("A book response must be an object.")
    token = str(payload.get("asset_id") or payload.get("market") or "")
    if not token:
        raise PolymarketError("A book response carries no asset id.")
    try:
        timestamp = int(float(payload.get("timestamp") or 0))
    except (TypeError, ValueError):
        timestamp = 0
    last = payload.get("last_trade_price")
    try:
        last_trade = float(last) if last is not None else None
    except (TypeError, ValueError):
        last_trade = None
    return Quote(
        token=token,
        timestamp=timestamp,
        bids=_levels(payload.get("bids")),
        asks=_levels(payload.get("asks")),
        last_trade=last_trade,
    )


def series_from_history(payload: dict, token: str) -> list[Quote]:
    """Build a price series from a `/prices-history` response.

    History carries a traded price rather than a book, so these quotes have no
    spread and `usable` is False for all of them — deliberately. They are for
    measuring movement, never for costing a trade.
    """
    points = payload.get("history") if isinstance(payload, dict) else None
    if not isinstance(points, list):
        raise PolymarketError("A history response carries no 'history' list.")
    out = []
    for point in points:
        if not isinstance(point, dict):
            continue
        try:
            price = float(point["p"])
            stamp = int(float(point["t"]))
        except (KeyError, TypeError, ValueError):
            continue
        if MIN_PRICE <= price <= MAX_PRICE:
            out.append(Quote(token=token, timestamp=stamp, last_trade=price))
    out.sort(key=lambda q: q.timestamp)
    return out


def movements_from_series(
    series: Sequence[Quote],
    feature: float,
    *,
    market: str,
    selection: str = "yes",
    anchored: bool = False,
    settle_after: int = 0,
) -> list[movement.Movement]:
    """Turn one market's trajectory into a single Movement for the bench.

    ``opening`` is the first price in the series and ``closing`` the last one at
    or before ``settle_after`` (zero meaning the end of the series). One market
    yields one observation, because a market's own path is a single dependent
    thing however many points were sampled from it.
    """
    usable = [q for q in series if q.mid is not None]
    if len(usable) < 2:
        return []
    tail = [q for q in usable if q.timestamp <= settle_after] if settle_after else usable
    if len(tail) < 2:
        return []
    first, last = tail[0], tail[-1]
    return [
        movement.Movement(
            match=market,
            selection=selection,
            raw=feature,
            opening=first.mid,
            closing=last.mid,
            anchored=anchored,
            opening_price=1.0 / first.mid if first.mid else 0.0,
            closing_price=1.0 / last.mid if last.mid else 0.0,
        )
    ]


def worth_acting_on(fit: movement.Fit, cost_to_cross: float) -> float | None:
    """The predicted move as a multiple of what crossing the spread costs.

    The exchange analogue of `movement.Fit.value_per_sd`: on a bookmaker the
    hurdle is the overround, here it is the half-spread. Below 1 the signal
    cannot pay for the trade that captures it.
    """
    moved = fit.movement_per_sd
    if moved is None or cost_to_cross <= 0:
        return None
    return moved / cost_to_cross


# --- the part this environment cannot exercise ----------------------------


def fetch_json(url: str, *, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """GET one URL and parse it as JSON.

    UNVERIFIED HERE: Polymarket's hosts are blocked by this environment's egress
    policy, so this function has never received a real response. Run
    `check_polymarket` first anywhere it can reach the network.
    """
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        raise PolymarketError(f"{url}: {error}") from error


def book_url(token: str) -> str:
    return f"{CLOB_HOST}{BOOK_PATH}?token_id={token}"


def history_url(token: str, *, interval: str = "max", fidelity: int = 60) -> str:
    return (
        f"{CLOB_HOST}{PRICES_HISTORY_PATH}?market={token}&interval={interval}&fidelity={fidelity}"
    )


def markets_url(cursor: str = "") -> str:
    base = f"{CLOB_HOST}{MARKETS_PATH}"
    return f"{base}?next_cursor={cursor}" if cursor else base
