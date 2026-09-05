"""Parse ESPN's odds payloads into decimal prices per market and selection.

ESPN quotes moneylines in American format and exposes totals as a single
``overUnder`` line with separate over/under prices. This module converts those
into the market/selection vocabulary used by `apps.espn.markets`, so model
probabilities and bookmaker prices can be compared directly.

The shape handled here follows `docs/response_schemas.md`. ESPN's odds payloads
vary by sport and provider, so every field is read defensively and anything
unrecognised is skipped rather than guessed at.
"""

from __future__ import annotations

from typing import Any

from apps.espn.markets import (
    MARKET_BTTS,
    MARKET_MATCH_ODDS,
    MARKET_TOTALS,
    SELECTION_AWAY,
    SELECTION_DRAW,
    SELECTION_HOME,
)

# Prices outside this range are almost certainly a parsing error, not a real quote.
MIN_DECIMAL_ODDS = 1.001
MAX_DECIMAL_ODDS = 1000.0


def american_to_decimal(american: float) -> float | None:
    """Convert an American moneyline to decimal odds."""
    if american is None or american == 0:
        return None
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def _coerce_price(value: Any) -> float | None:
    """Read a price that may be an American moneyline, a decimal, or a string."""
    if isinstance(value, dict):
        for key in ("decimal", "decimalOdds", "value", "moneyLine", "american"):
            if key in value:
                return _coerce_price(value[key])
        return None

    if isinstance(value, str):
        cleaned = value.strip().replace("+", "")
        if not cleaned or cleaned.upper() in {"EVEN", "EV", "PK", "OFF", "N/A"}:
            return 2.0 if cleaned.upper() in {"EVEN", "EV"} else None
        try:
            value = float(cleaned)
        except ValueError:
            return None

    if not isinstance(value, int | float):
        return None

    # Decimal odds are always above 1; American moneylines are at or beyond ±100.
    decimal = float(value) if 1.0 < value < 100.0 else american_to_decimal(float(value))
    if decimal is None or not MIN_DECIMAL_ODDS <= decimal <= MAX_DECIMAL_ODDS:
        return None
    return round(decimal, 4)


def _moneyline_of(container: Any) -> float | None:
    if isinstance(container, dict):
        for key in ("moneyLine", "moneyline", "decimal", "value", "odds"):
            if key in container:
                price = _coerce_price(container[key])
                if price is not None:
                    return price
        return None
    return _coerce_price(container)


def _format_line(value: Any) -> str | None:
    try:
        line = float(value)
    except (TypeError, ValueError):
        return None
    # Whole-number totals can push, which the model's half-line markets do not
    # represent, so they are not stored.
    return f"{line}" if line % 1 else None


def parse_odds_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn one provider entry into decimal-priced market/selection rows."""
    rows: list[dict[str, Any]] = []

    def add(market: str, selection: str, price: float | None, line: str = "") -> None:
        if price is not None:
            rows.append(
                {"market": market, "selection": selection, "line": line, "decimal_odds": price}
            )

    add(MARKET_MATCH_ODDS, SELECTION_HOME, _moneyline_of(item.get("homeTeamOdds")))
    add(MARKET_MATCH_ODDS, SELECTION_AWAY, _moneyline_of(item.get("awayTeamOdds")))
    add(MARKET_MATCH_ODDS, SELECTION_DRAW, _moneyline_of(item.get("drawOdds")))

    line = _format_line(item.get("overUnder"))
    if line:
        add(MARKET_TOTALS, "over", _coerce_price(item.get("overOdds")), line)
        add(MARKET_TOTALS, "under", _coerce_price(item.get("underOdds")), line)

    both_score = item.get("bothTeamsToScoreOdds") or {}
    if isinstance(both_score, dict):
        add(MARKET_BTTS, "yes", _coerce_price(both_score.get("yes")))
        add(MARKET_BTTS, "no", _coerce_price(both_score.get("no")))

    return rows


def parse_odds_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse a full odds response into rows tagged with their provider."""
    rows: list[dict[str, Any]] = []
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        provider = item.get("provider") or {}
        provider_id = str(provider.get("id") or "")
        if not provider_id:
            continue
        for row in parse_odds_item(item):
            rows.append(
                {
                    **row,
                    "provider_espn_id": provider_id,
                    "provider_name": str(provider.get("name") or ""),
                    "raw_data": item,
                }
            )
    return rows
