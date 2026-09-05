"""Load historical football results and pre-match odds from a Football-Data CSV.

Football-Data.co.uk publishes results, match statistics and bookmaker prices for
dozens of leagues going back to the 1990s. This command reads that data in the
column layout used by the Club Football Match Data mirror, which adds ClubElo
ratings and pre-computed form alongside it.

Unlike the ESPN ingestion this is a bulk historical load: it exists so the model
can be measured against real results and, more importantly, against real prices.

Source: https://github.com/xgabora/Club-Football-Match-Data-2000-2025
        (results and odds from https://www.football-data.co.uk/)

The odds columns are Bet365's pre-match price and the best price across roughly
seventeen European bookmakers. They are *not* closing prices, so they support a
market benchmark but not a closing-line-value calculation.
"""

import argparse
import csv
import hashlib
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
    "G1": ("soccer", "gre.1", "Greek Super League", "GRESL"),
}

# Two price series per match. The first is one bookmaker's own quote; the second
# is the best price available anywhere, which is not a coherent book and must never
# be devigged as if it were — its overround sits near or below 1.
#
# Bet365 is a real book with a real margin, so it is the one the market benchmark
# uses. Calling it a "market average" would overstate what it represents.
PROVIDER_BET365 = ("fd-b365", "Bet365")
PROVIDER_MAXIMUM = ("fd-max", "Best of ~17 bookmakers")

TOTALS_LINE = "2.5"


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


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}{digest}"


class Command(BaseCommand):
    help = "Load historical results and pre-match odds from a Football-Data style CSV."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("file", help="Path to the Matches CSV.")
        parser.add_argument(
            "--division",
            required=True,
            help="Football-Data division code to load (e.g. I1 for Serie A).",
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
        sport, _ = Sport.objects.get_or_create(slug=sport_slug, defaults={"name": "Soccer"})
        league, _ = League.objects.get_or_create(
            sport=sport,
            slug=league_slug,
            defaults={"name": league_name, "abbreviation": abbreviation},
        )

        try:
            rows = self._read(options["file"], division, date_from, date_to)
        except OSError as exc:
            raise CommandError(f"Could not read {options['file']}: {exc}") from exc

        if not rows:
            raise CommandError(f"No rows for division {division!r} in the requested date range.")

        events, odds_rows, skipped = self._load(league, rows, with_odds=not options["no_odds"])

        self.stdout.write(
            self.style.SUCCESS(
                f"Loaded {events} matches into {sport_slug}/{league_slug} "
                f"({odds_rows} odds rows, {skipped} rows skipped)."
            )
        )
        self.stdout.write(
            "Odds are Bet365's pre-match price and the best of ~17 books, not closing prices."
        )

    def _read(
        self,
        path: str,
        division: str,
        date_from: datetime | None,
        date_to: datetime | None,
    ) -> list[dict[str, Any]]:
        rows = []
        with open(path, newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                if (row.get("Division") or "").upper() != division:
                    continue
                kickoff = _parse_kickoff(row)
                if kickoff is None:
                    continue
                if date_from and kickoff < date_from:
                    continue
                if date_to and kickoff > date_to:
                    continue
                rows.append({**row, "_kickoff": kickoff})
        rows.sort(key=lambda row: row["_kickoff"])
        return rows

    @transaction.atomic
    def _load(
        self,
        league: League,
        rows: list[dict[str, Any]],
        with_odds: bool,
    ) -> tuple[int, int, int]:
        teams: dict[str, Team] = {}
        events = odds_rows = skipped = 0

        for row in rows:
            home_name = (row.get("HomeTeam") or "").strip()
            away_name = (row.get("AwayTeam") or "").strip()
            home_goals = _int(row.get("FTHome"))
            away_goals = _int(row.get("FTAway"))

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
                    # Elo and form travel with the event so later models can use them.
                    "raw_data": {
                        "source": "football-data",
                        "division": row.get("Division"),
                        "home_elo": row.get("HomeElo"),
                        "away_elo": row.get("AwayElo"),
                        "form3_home": row.get("Form3Home"),
                        "form5_home": row.get("Form5Home"),
                        "form3_away": row.get("Form3Away"),
                        "form5_away": row.get("Form5Away"),
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
                    statistics=_statistics(row, side),
                )
            events += 1

            if with_odds:
                odds_rows += self._odds(event, row)

        return events, odds_rows, skipped

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

    def _odds(self, event: Event, row: dict[str, Any]) -> int:
        written = 0
        for provider_id, provider_name, columns in (
            (
                *PROVIDER_BET365,
                {
                    (MARKET_MATCH_ODDS, SELECTION_HOME, ""): "OddHome",
                    (MARKET_MATCH_ODDS, SELECTION_DRAW, ""): "OddDraw",
                    (MARKET_MATCH_ODDS, SELECTION_AWAY, ""): "OddAway",
                    (MARKET_TOTALS, "over", TOTALS_LINE): "Over25",
                    (MARKET_TOTALS, "under", TOTALS_LINE): "Under25",
                },
            ),
            (
                *PROVIDER_MAXIMUM,
                {
                    (MARKET_MATCH_ODDS, SELECTION_HOME, ""): "MaxHome",
                    (MARKET_MATCH_ODDS, SELECTION_DRAW, ""): "MaxDraw",
                    (MARKET_MATCH_ODDS, SELECTION_AWAY, ""): "MaxAway",
                    (MARKET_TOTALS, "over", TOTALS_LINE): "MaxOver25",
                    (MARKET_TOTALS, "under", TOTALS_LINE): "MaxUnder25",
                },
            ),
        ):
            for (market, selection, line), column in columns.items():
                price = _decimal(row.get(column))
                if price is None:
                    continue
                Odds.objects.update_or_create(
                    event=event,
                    provider_espn_id=provider_id,
                    market=market,
                    selection=selection,
                    line=line,
                    defaults={
                        "provider_name": provider_name,
                        "decimal_odds": price,
                        "raw_data": {"source": "football-data", "column": column},
                    },
                )
                written += 1
        return written


def _statistics(row: dict[str, Any], side: str) -> list[dict[str, Any]]:
    prefix = "Home" if side == Competitor.HOME else "Away"
    fields = {
        "shots": f"{prefix}Shots",
        "shotsOnTarget": f"{prefix}Target",
        "corners": f"{prefix}Corners",
        "fouls": f"{prefix}Fouls",
        "yellowCards": f"{prefix}Yellow",
        "redCards": f"{prefix}Red",
    }
    return [
        {"name": name, "value": _int(row.get(column))}
        for name, column in fields.items()
        if _int(row.get(column)) is not None
    ]


def _season_year(kickoff: datetime) -> int:
    """European seasons straddle the new year; label them by the starting year."""
    return kickoff.year if kickoff.month >= 7 else kickoff.year - 1


def _parse_kickoff(row: dict[str, Any]) -> datetime | None:
    raw_date = (row.get("MatchDate") or "").strip()
    if not raw_date:
        return None
    raw_time = (row.get("MatchTime") or "").strip() or "12:00"
    for time_format in ("%H:%M", "%H:%M:%S"):
        try:
            parsed = datetime.strptime(f"{raw_date} {raw_time}", f"%Y-%m-%d {time_format}")
            return parsed.replace(tzinfo=UTC)
        except ValueError:
            continue
    try:
        return datetime.strptime(raw_date, "%Y-%m-%d").replace(hour=12, tzinfo=UTC)
    except ValueError:
        return None


def _parse_date(value: str | None, flag: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise CommandError(f"{flag} must be YYYY-MM-DD, got {value!r}.") from exc
