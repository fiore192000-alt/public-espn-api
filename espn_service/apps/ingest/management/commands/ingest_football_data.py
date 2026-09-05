"""Load historical football results and bookmaker prices from a Football-Data CSV.

Football-Data.co.uk publishes results, match statistics and bookmaker prices for
dozens of leagues going back to the 1990s. This command reads **two** layouts of
that data and decides which one it is looking at from the header:

``football-data.co.uk`` (the original season files)
    ``Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,…`` with a wide block of odds
    columns. Crucially these carry **closing** prices as well as pre-match ones —
    a bookmaker abbreviation followed by ``C`` (``PSCH`` is Pinnacle's closing
    home price). Pinnacle's closing line is the sharpest widely published price in
    football, which makes it the benchmark worth being measured against and the
    only one that supports a closing-line-value calculation.

``club-football-match-data`` (the derived mirror)
    ``Division,MatchDate,…,OddHome,MaxHome,…`` with ClubElo ratings and
    pre-computed form alongside. Convenient and broad, but it keeps only the
    pre-match Bet365 quote and the best price across ~17 books: **no closing
    prices**, so it cannot answer whether a bet beat the line it was struck at.

Sources: https://www.football-data.co.uk/
         https://github.com/xgabora/Club-Football-Match-Data-2000-2025

Opening and closing quotes from the same bookmaker are stored as **separate
providers** (``fd-ps`` and ``fd-psc``) rather than as two rows of one provider.
The Odds model has no notion of when a price was taken, and every analysis in
this project already keys on the provider, so this keeps a closing line from
being silently devigged or settled as if it were the price you could have had.
"""

import argparse
import csv
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.espn.markets import (
    MARKET_MATCH_ODDS,
    MARKET_TOTALS,
    SELECTION_AWAY,
    SELECTION_DRAW,
    SELECTION_HOME,
)
from apps.espn.models import Competitor, Event, League, Odds, Sport, Team

# Football-Data division codes mapped to ESPN-style sport/league slugs, so a
# league loaded from here sits alongside anything ingested from ESPN.
DIVISIONS = {
    "I1": ("soccer", "ita.1", "Italian Serie A", "SERIE A"),
    "I2": ("soccer", "ita.2", "Italian Serie B", "SERIE B"),
    "E0": ("soccer", "eng.1", "English Premier League", "PREM"),
    "E1": ("soccer", "eng.2", "English Championship", "CHAMP"),
    "E2": ("soccer", "eng.3", "English League One", "LGE1"),
    "E3": ("soccer", "eng.4", "English League Two", "LGE2"),
    "EC": ("soccer", "eng.5", "English National League", "NATL"),
    "SP1": ("soccer", "esp.1", "Spanish La Liga", "LALIGA"),
    "SP2": ("soccer", "esp.2", "Spanish Segunda División", "LALIGA2"),
    "D1": ("soccer", "ger.1", "German Bundesliga", "BUND"),
    "D2": ("soccer", "ger.2", "German 2. Bundesliga", "BUND2"),
    "F1": ("soccer", "fra.1", "French Ligue 1", "LIGUE1"),
    "F2": ("soccer", "fra.2", "French Ligue 2", "LIGUE2"),
    "N1": ("soccer", "ned.1", "Dutch Eredivisie", "ERE"),
    "P1": ("soccer", "por.1", "Portuguese Primeira Liga", "PRIMEIRA"),
    "T1": ("soccer", "tur.1", "Turkish Süper Lig", "SUPERLIG"),
    "B1": ("soccer", "bel.1", "Belgian Pro League", "BELPRO"),
    "SC0": ("soccer", "sco.1", "Scottish Premiership", "SCOPREM"),
    "SC1": ("soccer", "sco.2", "Scottish Championship", "SCOCHAMP"),
    "SC2": ("soccer", "sco.3", "Scottish League One", "SCOLGE1"),
    "SC3": ("soccer", "sco.4", "Scottish League Two", "SCOLGE2"),
    "G1": ("soccer", "gre.1", "Greek Super League", "GRESL"),
}

# Provider ids. A closing quote gets its own id: it is not a price anyone could
# have taken when the model made its forecast, and treating it as one would turn
# hindsight into an edge.
PROVIDER_BET365 = ("fd-b365", "Bet365")
PROVIDER_BET365_CLOSE = ("fd-b365c", "Bet365 (closing)")
PROVIDER_PINNACLE = ("fd-ps", "Pinnacle")
PROVIDER_PINNACLE_CLOSE = ("fd-psc", "Pinnacle (closing)")
# Never a coherent book: a maximum taken across seventeen bookmakers has an
# overround near or below 1 and must not be devigged as if one book quoted it.
PROVIDER_MAXIMUM = ("fd-max", "Best of ~17 bookmakers")
PROVIDER_MAXIMUM_CLOSE = ("fd-maxc", "Best of ~17 bookmakers (closing)")
PROVIDER_AVERAGE = ("fd-avg", "Market average")
PROVIDER_AVERAGE_CLOSE = ("fd-avgc", "Market average (closing)")

TOTALS_LINE = "2.5"
OVER, UNDER = "over", "under"

SCHEMA_ORIGINAL = "football-data.co.uk"
SCHEMA_DERIVED = "club-football-match-data"


@dataclass(frozen=True)
class PriceSource:
    """One provider, and the columns its prices live in.

    Each entry maps a market to a tuple of candidate column names because the
    source renamed its aggregate columns partway through: the best-price column
    is ``BbMxH`` up to 2018/19 and ``MaxH`` afterwards. First non-empty wins, so a
    file from either era loads under the same provider id.
    """

    provider_id: str
    provider_name: str
    columns: dict[tuple[str, str, str], tuple[str, ...]]


def _match_odds(home: tuple[str, ...], draw: tuple[str, ...], away: tuple[str, ...]) -> dict:
    return {
        (MARKET_MATCH_ODDS, SELECTION_HOME, ""): home,
        (MARKET_MATCH_ODDS, SELECTION_DRAW, ""): draw,
        (MARKET_MATCH_ODDS, SELECTION_AWAY, ""): away,
    }


def _totals(over: tuple[str, ...], under: tuple[str, ...]) -> dict:
    return {
        (MARKET_TOTALS, OVER, TOTALS_LINE): over,
        (MARKET_TOTALS, UNDER, TOTALS_LINE): under,
    }


@dataclass(frozen=True)
class Schema:
    """Where a given CSV layout keeps each thing this command needs."""

    name: str
    division: str
    date: str
    time: str
    date_formats: tuple[str, ...]
    home_goals: str
    away_goals: str
    statistics: dict[str, tuple[str, str]]
    prices: tuple[PriceSource, ...]
    extras: dict[str, str] = field(default_factory=dict)
    home: str = "HomeTeam"
    away: str = "AwayTeam"

    @property
    def has_closing_prices(self) -> bool:
        return any(source.provider_id.endswith("c") for source in self.prices)


ORIGINAL = Schema(
    name=SCHEMA_ORIGINAL,
    division="Div",
    date="Date",
    time="Time",
    # Four-digit years appear from 2017/18; earlier files use two.
    date_formats=("%d/%m/%Y", "%d/%m/%y"),
    home_goals="FTHG",
    away_goals="FTAG",
    statistics={
        "shots": ("HS", "AS"),
        "shotsOnTarget": ("HST", "AST"),
        "corners": ("HC", "AC"),
        "fouls": ("HF", "AF"),
        "yellowCards": ("HY", "AY"),
        "redCards": ("HR", "AR"),
    },
    prices=(
        PriceSource(
            *PROVIDER_BET365,
            {
                **_match_odds(("B365H",), ("B365D",), ("B365A",)),
                **_totals(("B365>2.5",), ("B365<2.5",)),
            },
        ),
        PriceSource(
            *PROVIDER_BET365_CLOSE,
            {
                **_match_odds(("B365CH",), ("B365CD",), ("B365CA",)),
                **_totals(("B365C>2.5",), ("B365C<2.5",)),
            },
        ),
        PriceSource(
            *PROVIDER_PINNACLE,
            {
                **_match_odds(("PSH", "PH"), ("PSD", "PD"), ("PSA", "PA")),
                **_totals(("P>2.5",), ("P<2.5",)),
            },
        ),
        PriceSource(
            *PROVIDER_PINNACLE_CLOSE,
            {
                **_match_odds(("PSCH", "PCH"), ("PSCD", "PCD"), ("PSCA", "PCA")),
                **_totals(("PC>2.5",), ("PC<2.5",)),
            },
        ),
        PriceSource(
            *PROVIDER_MAXIMUM,
            {
                **_match_odds(("MaxH", "BbMxH"), ("MaxD", "BbMxD"), ("MaxA", "BbMxA")),
                **_totals(("Max>2.5", "BbMx>2.5"), ("Max<2.5", "BbMx<2.5")),
            },
        ),
        PriceSource(
            *PROVIDER_MAXIMUM_CLOSE,
            {
                **_match_odds(("MaxCH",), ("MaxCD",), ("MaxCA",)),
                **_totals(("MaxC>2.5",), ("MaxC<2.5",)),
            },
        ),
        PriceSource(
            *PROVIDER_AVERAGE,
            {
                **_match_odds(("AvgH", "BbAvH"), ("AvgD", "BbAvD"), ("AvgA", "BbAvA")),
                **_totals(("Avg>2.5", "BbAv>2.5"), ("Avg<2.5", "BbAv<2.5")),
            },
        ),
        PriceSource(
            *PROVIDER_AVERAGE_CLOSE,
            {
                **_match_odds(("AvgCH",), ("AvgCD",), ("AvgCA",)),
                **_totals(("AvgC>2.5",), ("AvgC<2.5",)),
            },
        ),
    ),
    extras={"referee": "Referee", "half_time_home": "HTHG", "half_time_away": "HTAG"},
)

DERIVED = Schema(
    name=SCHEMA_DERIVED,
    division="Division",
    date="MatchDate",
    time="MatchTime",
    date_formats=("%Y-%m-%d",),
    home_goals="FTHome",
    away_goals="FTAway",
    statistics={
        "shots": ("HomeShots", "AwayShots"),
        "shotsOnTarget": ("HomeTarget", "AwayTarget"),
        "corners": ("HomeCorners", "AwayCorners"),
        "fouls": ("HomeFouls", "AwayFouls"),
        "yellowCards": ("HomeYellow", "AwayYellow"),
        "redCards": ("HomeRed", "AwayRed"),
    },
    prices=(
        PriceSource(
            *PROVIDER_BET365,
            {
                **_match_odds(("OddHome",), ("OddDraw",), ("OddAway",)),
                **_totals(("Over25",), ("Under25",)),
            },
        ),
        PriceSource(
            *PROVIDER_MAXIMUM,
            {
                **_match_odds(("MaxHome",), ("MaxDraw",), ("MaxAway",)),
                **_totals(("MaxOver25",), ("MaxUnder25",)),
            },
        ),
    ),
    extras={
        "home_elo": "HomeElo",
        "away_elo": "AwayElo",
        "form3_home": "Form3Home",
        "form5_home": "Form5Home",
        "form3_away": "Form3Away",
        "form5_away": "Form5Away",
    },
)

SCHEMAS = (ORIGINAL, DERIVED)


def detect_schema(fieldnames: list[str] | None) -> Schema:
    """Decide which layout a file uses from its header alone.

    Both layouts are identified by the columns only they have, so a file that
    matches neither is rejected rather than half-read into empty results.
    """
    columns = set(fieldnames or ())
    for schema in SCHEMAS:
        if {schema.division, schema.date, schema.home_goals} <= columns:
            return schema
    raise CommandError(
        "Unrecognised CSV layout. Expected either the football-data.co.uk columns "
        "(Div, Date, FTHG) or the Club Football Match Data columns "
        f"(Division, MatchDate, FTHome). Found: {', '.join(sorted(columns)[:12])}…"
    )


def _league_for(division: str) -> tuple[str, str, str, str]:
    if division in DIVISIONS:
        return DIVISIONS[division]
    slug = f"fd.{division.lower()}"
    return "soccer", slug, f"Football-Data {division}", division.upper()[:10]


def _decimal(value: Any) -> float | None:
    """Read a decimal price, rejecting anything that is not a usable quote."""
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if 1.01 <= price <= 1000.0 else None


def _int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _first_price(row: dict[str, Any], columns: tuple[str, ...]) -> tuple[float, str] | None:
    """The first column of the candidate list that carries a usable price."""
    for column in columns:
        price = _decimal(row.get(column))
        if price is not None:
            return price, column
    return None


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}{digest}"


class Command(BaseCommand):
    help = "Load historical results and bookmaker prices from a Football-Data style CSV."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("file", help="Path to the CSV, in either supported layout.")
        parser.add_argument(
            "--division",
            required=True,
            help="Football-Data division code to load (e.g. I1 for Serie A).",
        )
        parser.add_argument(
            "--league-slug",
            help="Load into this league slug instead of the one the division maps to.",
        )
        parser.add_argument(
            "--date-from", help="Only load matches on or after this date (YYYY-MM-DD)."
        )
        parser.add_argument(
            "--date-to", help="Only load matches on or before this date (YYYY-MM-DD)."
        )
        parser.add_argument(
            "--no-odds",
            action="store_true",
            help="Load results only, skipping the bookmaker prices.",
        )

    def handle(self, *args, **options) -> None:
        division = options["division"].upper()
        date_from = _parse_date(options["date_from"], "--date-from")
        date_to = _parse_date(options["date_to"], "--date-to")

        sport_slug, league_slug, league_name, abbreviation = _league_for(division)
        league_slug = options["league_slug"] or league_slug
        sport, _ = Sport.objects.get_or_create(slug=sport_slug, defaults={"name": "Soccer"})
        league, _ = League.objects.get_or_create(
            sport=sport,
            slug=league_slug,
            defaults={"name": league_name, "abbreviation": abbreviation},
        )

        try:
            schema, rows = self._read(options["file"], division, date_from, date_to)
        except OSError as exc:
            raise CommandError(f"Could not read {options['file']}: {exc}") from exc

        if not rows:
            raise CommandError(f"No rows for division {division!r} in the requested date range.")

        events, odds_rows, providers, skipped = self._load(
            league, schema, rows, with_odds=not options["no_odds"]
        )

        self.stdout.write(
            self.style.SUCCESS(
                f"Loaded {events} matches into {sport_slug}/{league_slug} "
                f"({odds_rows} odds rows, {skipped} rows skipped)."
            )
        )
        self.stdout.write(f"Layout detected: {schema.name}.")
        if providers:
            self.stdout.write(f"Price series stored: {', '.join(sorted(providers))}.")
        closing = sorted(p for p in providers if p.endswith("c"))
        if closing:
            self.stdout.write(
                self.style.SUCCESS(
                    f"Closing prices present ({', '.join(closing)}) — closing-line value can be "
                    "measured on this league."
                )
            )
        else:
            self.stdout.write(
                self.style.WARNING(
                    "No closing prices in this file: these are pre-match quotes only, so they "
                    "support a market benchmark but not a closing-line-value calculation."
                )
            )

    def _read(
        self,
        path: str,
        division: str,
        date_from: datetime | None,
        date_to: datetime | None,
    ) -> tuple[Schema, list[dict[str, Any]]]:
        rows = []
        with open(path, newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            schema = detect_schema(reader.fieldnames)
            for row in reader:
                if (row.get(schema.division) or "").strip().upper() != division:
                    continue
                kickoff = _parse_kickoff(row, schema)
                if kickoff is None:
                    continue
                if date_from and kickoff < date_from:
                    continue
                if date_to and kickoff > date_to:
                    continue
                rows.append({**row, "_kickoff": kickoff})
        rows.sort(key=lambda row: row["_kickoff"])
        return schema, rows

    @transaction.atomic
    def _load(
        self,
        league: League,
        schema: Schema,
        rows: list[dict[str, Any]],
        with_odds: bool,
    ) -> tuple[int, int, set[str], int]:
        teams: dict[str, Team] = {}
        events = odds_rows = skipped = 0
        providers: set[str] = set()

        for row in rows:
            home_name = (row.get(schema.home) or "").strip()
            away_name = (row.get(schema.away) or "").strip()
            home_goals = _int(row.get(schema.home_goals))
            away_goals = _int(row.get(schema.away_goals))

            if not home_name or not away_name or home_goals is None or away_goals is None:
                skipped += 1
                continue

            home = self._team(teams, league, home_name)
            away = self._team(teams, league, away_name)
            kickoff = row["_kickoff"]

            event, _ = Event.objects.update_or_create(
                league=league,
                espn_id=_stable_id(
                    "fd", league.slug, kickoff.date().isoformat(), home_name, away_name
                ),
                defaults={
                    "date": kickoff,
                    "name": f"{away_name} at {home_name}",
                    "short_name": f"{away.abbreviation} @ {home.abbreviation}",
                    "season_year": _season_year(kickoff),
                    "season_type": 2,
                    "status": Event.STATUS_FINAL,
                    "status_detail": "Final",
                    # Whatever the layout carries beyond the result travels with the
                    # event, so later models can use it without a second load.
                    "raw_data": {
                        "source": "football-data",
                        "layout": schema.name,
                        "division": row.get(schema.division),
                        **{key: row.get(column) for key, column in schema.extras.items()},
                    },
                },
            )

            event.competitors.all().delete()
            for team, score, side, order in (
                (home, home_goals, Competitor.HOME, 1),
                (away, away_goals, Competitor.AWAY, 0),
            ):
                Competitor.objects.create(
                    event=event,
                    team=team,
                    home_away=side,
                    score=str(score),
                    winner=score > (away_goals if side == Competitor.HOME else home_goals),
                    order=order,
                    statistics=_statistics(row, schema, side),
                )
            events += 1

            if with_odds:
                written, seen = self._odds(event, schema, row)
                odds_rows += written
                providers |= seen

        return events, odds_rows, providers, skipped

    def _team(self, cache: dict[str, Team], league: League, name: str) -> Team:
        if name in cache:
            return cache[name]
        team, _ = Team.objects.get_or_create(
            league=league,
            espn_id=_stable_id("fdt", league.slug, name),
            defaults={
                "display_name": name,
                "abbreviation": name[:3].upper(),
                "name": name,
                "location": name,
                "raw_data": {"source": "football-data", "name": name},
            },
        )
        cache[name] = team
        return team

    def _odds(self, event: Event, schema: Schema, row: dict[str, Any]) -> tuple[int, set[str]]:
        written = 0
        providers: set[str] = set()
        for source in schema.prices:
            for (market, selection, line), columns in source.columns.items():
                found = _first_price(row, columns)
                if found is None:
                    continue
                price, column = found
                Odds.objects.update_or_create(
                    event=event,
                    provider_espn_id=source.provider_id,
                    market=market,
                    selection=selection,
                    line=line,
                    defaults={
                        "provider_name": source.provider_name,
                        "decimal_odds": price,
                        # The column is kept because two eras of the source use
                        # different names for the same aggregate, and a later
                        # reader should be able to tell which one this came from.
                        "raw_data": {"source": "football-data", "column": column},
                    },
                )
                written += 1
                providers.add(source.provider_id)
        return written, providers


def _statistics(row: dict[str, Any], schema: Schema, side: str) -> list[dict[str, Any]]:
    index = 0 if side == Competitor.HOME else 1
    return [
        {"name": name, "value": _int(row.get(columns[index]))}
        for name, columns in schema.statistics.items()
        if _int(row.get(columns[index])) is not None
    ]


def _season_year(kickoff: datetime) -> int:
    """European seasons straddle the new year; label them by the starting year."""
    return kickoff.year if kickoff.month >= 7 else kickoff.year - 1


def _parse_kickoff(row: dict[str, Any], schema: Schema) -> datetime | None:
    raw_date = (row.get(schema.date) or "").strip()
    if not raw_date:
        return None
    # Kick-off times only appear in the later files; noon keeps a dateless match
    # ordered within its own day without pretending to know when it started.
    raw_time = (row.get(schema.time) or "").strip()
    for date_format in schema.date_formats:
        if raw_time:
            for time_format in ("%H:%M", "%H:%M:%S"):
                try:
                    parsed = datetime.strptime(
                        f"{raw_date} {raw_time}", f"{date_format} {time_format}"
                    )
                    return parsed.replace(tzinfo=UTC)
                except ValueError:
                    continue
        try:
            return datetime.strptime(raw_date, date_format).replace(hour=12, tzinfo=UTC)
        except ValueError:
            continue
    return None


def _parse_date(value: str | None, flag: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise CommandError(f"{flag} must be YYYY-MM-DD, got {value!r}.") from exc
