"""Aggregate shot-by-shot expected goals onto matches already in the database.

Understat publishes every shot with a model's estimate of how often it is scored.
Summing those per side gives the match-level number this command stores: not what
happened, but what the chances created were worth.

The hard part is not the arithmetic, it is **identity**. Understat and
Football-Data name clubs differently ("Manchester United" against "Man United",
"Internazionale" against "Inter"), so a naive join loses a third of the fixtures
silently — the worst possible failure, because the survivors are not a random
sample of the whole. Matching therefore goes date-first: a fixture is identified
by its date and its two clubs, clubs are matched by a normalised name, and the
command **reports every unmatched match** rather than quietly dropping it.
"""

import argparse
import csv
import re
import unicodedata
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.espn.analysis import _sided_competitors
from apps.espn.models import Event, ExpectedGoals, League

SOURCE = "understat"
# A fixture may be recorded a day either side of the stored kick-off: Understat
# timestamps in local time, Football-Data in its own. Wider than this and
# distinct fixtures between the same clubs start colliding.
DATE_TOLERANCE_DAYS = 1

# Clubs whose two sources disagree by more than punctuation. Everything else is
# handled by _normalise; this list exists so the ones that are not are visible.
ALIASES = {
    "manchesterunited": "manunited",
    "manchestercity": "mancity",
    "newcastleunited": "newcastle",
    "wolverhamptonwanderers": "wolves",
    "sheffieldunited": "sheffieldutd",
    "westbromwichalbion": "westbrom",
    "westhamunited": "westham",
    "tottenham": "tottenham",
    "leedsunited": "leeds",
    "queensparkrangers": "qpr",
    "leicesterunited": "leicester",
    "brightonandhovealbion": "brighton",
    "huddersfieldtown": "huddersfield",
    "cardiffcity": "cardiff",
    "swanseacity": "swansea",
    "stokecity": "stoke",
    "hullcity": "hull",
    "norwichcity": "norwich",
    "internazionale": "inter",
    "parmacalcio": "parma",
    "acmilan": "milan",
    "hellasverona": "verona",
    "spal": "spal",
    "chievo": "chievo",
    # Spain. The database carries Football-Data's short forms, Understat the
    # full club names; neither is wrong, so every pair is written out.
    "athleticclub": "athbilbao",
    "atleticomadrid": "athmadrid",
    "celtavigo": "celta",
    "deportivolacoruna": "lacoruna",
    "espanyol": "espanol",
    "rayovallecano": "vallecano",
    "realbetis": "betis",
    "realsociedad": "sociedad",
    "realvalladolid": "valladolid",
    "sdhuesca": "huesca",
    "sportinggijon": "spgijon",
    # Germany.
    "arminiabielefeld": "bielefeld",
    "bayerleverkusen": "leverkusen",
    "borussiadortmund": "dortmund",
    "borussiamgladbach": "mgladbach",
    "eintrachtfrankfurt": "einfrankfurt",
    "cologne": "koln",
    "fortunaduesseldorf": "fortunadusseldorf",
    "greutherfuerth": "greutherfurth",
    "hamburgersv": "hamburg",
    "herthaberlin": "hertha",
    "nuernberg": "nurnberg",
    "rasenballsportleipzig": "rbleipzig",
    "vfbstuttgart": "stuttgart",
    # France.
    "clermontfoot": "clermont",
    "gfcajaccio": "ajacciogfco",
    "parissaintgermain": "parissg",
    "saintetienne": "stetienne",
    "scbastia": "bastia",
}


def _normalise(name: str) -> str:
    """Strip accents, punctuation, case and the noise words clubs collect."""
    folded = unicodedata.normalize("NFKD", name or "")
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    folded = re.sub(r"[^a-zA-Z]", "", folded).lower()
    for noise in ("fc", "afc", "calcio", "ssc", "ac", "as", "us", "uc"):
        if folded.startswith(noise) and len(folded) > len(noise) + 3:
            folded = folded[len(noise) :]
    return ALIASES.get(folded, folded)


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class Command(BaseCommand):
    help = "Load match-level expected goals from an Understat shots CSV."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("file", help="Path to a shots_*.csv from the Understat dataset.")
        parser.add_argument("--league", required=True, help="League slug to attach xG to.")
        parser.add_argument(
            "--report-unmatched",
            type=int,
            default=10,
            help="How many unmatched fixtures to print (default: 10).",
        )

    def handle(self, *args, **options) -> None:
        try:
            league = League.objects.get(slug__iexact=options["league"])
        except League.DoesNotExist as exc:
            raise CommandError(f"No league with slug {options['league']!r}.") from exc

        try:
            matches = self._aggregate(options["file"])
        except OSError as exc:
            raise CommandError(f"Could not read {options['file']}: {exc}") from exc
        if not matches:
            raise CommandError(
                f"No usable shots in {options['file']}. Expected the Understat columns "
                "(xG, h_team, a_team, date, h_a)."
            )

        index = self._index_events(league)
        written, unmatched = self._store(matches, index)

        self.stdout.write(
            self.style.SUCCESS(
                f"Stored xG for {written} of {len(matches)} matches in {league.slug}."
            )
        )
        if unmatched:
            self.stdout.write(
                self.style.WARNING(
                    f"{len(unmatched)} fixtures had no match in the database. Unmatched "
                    "fixtures are not a random sample — a club whose name never resolves "
                    "loses every one of its games — so these are listed rather than counted."
                )
            )
            for home, away, date in unmatched[: options["report_unmatched"]]:
                self.stdout.write(f"    {date}  {home} vs {away}")
            if len(unmatched) > options["report_unmatched"]:
                self.stdout.write(f"    … and {len(unmatched) - options['report_unmatched']} more")

    def _aggregate(self, path: str) -> dict[tuple[str, str, str], dict]:
        """Sum every shot's xG onto its match, keeping the two sides apart."""
        totals: dict[tuple[str, str, str], dict] = defaultdict(
            lambda: {"home": 0.0, "away": 0.0, "home_shots": 0, "away_shots": 0}
        )
        with open(path, newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                xg = _float(row.get("xG"))
                date = (row.get("date") or "")[:10]
                home, away, side = row.get("h_team"), row.get("a_team"), row.get("h_a")
                if xg is None or not date or not home or not away or side not in ("h", "a"):
                    continue
                entry = totals[(home, away, date)]
                if side == "h":
                    entry["home"] += xg
                    entry["home_shots"] += 1
                else:
                    entry["away"] += xg
                    entry["away_shots"] += 1
        return dict(totals)

    def _index_events(self, league: League) -> dict[tuple[str, str, str], Event]:
        """Every stored fixture, keyed by normalised clubs and date."""
        index: dict[tuple[str, str, str], Event] = {}
        for event in (
            Event.objects.filter(league=league, status=Event.STATUS_FINAL)
            .prefetch_related("competitors__team")
            .only("id", "date", "espn_id")
        ):
            sides = _sided_competitors(event)
            if sides is None:
                continue
            home, away = sides
            key = (
                _normalise(home.team.display_name),
                _normalise(away.team.display_name),
                event.date.date().isoformat(),
            )
            index[key] = event
        return index

    @transaction.atomic
    def _store(self, matches: dict, index: dict) -> tuple[int, list]:
        written = 0
        unmatched = []
        for (home, away, date), totals in matches.items():
            event = self._find(index, _normalise(home), _normalise(away), date)
            if event is None:
                unmatched.append((home, away, date))
                continue
            ExpectedGoals.objects.update_or_create(
                event=event,
                defaults={
                    "home": round(totals["home"], 4),
                    "away": round(totals["away"], 4),
                    "home_shots": totals["home_shots"],
                    "away_shots": totals["away_shots"],
                    "source": SOURCE,
                    "raw_data": {"home_team": home, "away_team": away, "date": date},
                },
            )
            written += 1
        return written, unmatched

    def _find(self, index: dict, home: str, away: str, date: str) -> Event | None:
        """Look the fixture up on its own date, then a day either side."""
        try:
            stamp = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            return None
        for offset in range(-DATE_TOLERANCE_DAYS, DATE_TOLERANCE_DAYS + 1):
            shifted = stamp.fromordinal(stamp.toordinal() + offset).date().isoformat()
            event = index.get((home, away, shifted))
            if event is not None:
                return event
        return None
