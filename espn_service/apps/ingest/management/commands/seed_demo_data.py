"""Seed a synthetic league so the service can be run and demoed without ESPN access.

The generated payloads use ESPN's scoreboard shape and are pushed through the
real ingestion service, so this exercises the same parsing path as live data.
Teams and results are fictional — never treat this data as real ESPN output.
"""

import math
import random
from datetime import datetime, timedelta
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from apps.ingest.services import ScoreboardIngestionService
from clients.espn_client import ESPNResponse

SOCCER_CLUBS = [
    ("Northgate Rovers", "NGR", "Northgate"),
    ("Selby Athletic", "SEL", "Selby"),
    ("Ardmore City", "ARD", "Ardmore"),
    ("Calder United", "CAL", "Calder"),
    ("Fairhaven FC", "FAI", "Fairhaven"),
    ("Ravensmoor", "RAV", "Ravensmoor"),
    ("Kingsport Town", "KGP", "Kingsport"),
    ("Westmere FC", "WES", "Westmere"),
]

BASKETBALL_CLUBS = [
    ("Northgate Comets", "NGC", "Northgate"),
    ("Selby Surge", "SES", "Selby"),
    ("Ardmore Ridge", "ARR", "Ardmore"),
    ("Calder Current", "CAC", "Calder"),
    ("Fairhaven Falcons", "FAF", "Fairhaven"),
    ("Ravensmoor Rise", "RAR", "Ravensmoor"),
]

PROFILES = {
    "soccer": {
        "sport": "soccer",
        "league": "demo.1",
        "clubs": SOCCER_CLUBS,
        "base_rate": 1.35,
        "home_edge": 1.25,
        "periods": 2,
    },
    "basketball": {
        "sport": "basketball",
        "league": "demo-nba",
        "clubs": BASKETBALL_CLUBS,
        "base_rate": 108.0,
        "home_edge": 1.03,
        "periods": 4,
    },
}


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth's sampler — adequate for the small rates used here."""
    limit = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


class _CannedClient:
    """Stands in for ESPNClient and replays a prepared scoreboard payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def get_scoreboard(self, sport: str, league: str, date=None, limit=None) -> ESPNResponse:  # noqa: ARG002
        return ESPNResponse(data=self._payload, status_code=200, url="seed://demo-scoreboard")


class Command(BaseCommand):
    help = "Seed a synthetic league (fictional teams) for offline demos and analysis testing."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--profile",
            choices=sorted(PROFILES),
            default="soccer",
            help="Scoring profile to generate (default: soccer).",
        )
        parser.add_argument(
            "--rounds",
            type=int,
            default=8,
            help="Round-robin rounds of completed games to generate (default: 8).",
        )
        parser.add_argument(
            "--upcoming",
            type=int,
            default=4,
            help="Number of scheduled (not yet played) fixtures to append (default: 4).",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed, so runs are reproducible (default: 42).",
        )

    def handle(self, *args, **options) -> None:
        profile_name = options["profile"]
        profile = PROFILES[profile_name]
        rounds = options["rounds"]
        upcoming = options["upcoming"]

        if rounds < 1:
            raise CommandError("--rounds must be at least 1")

        rng = random.Random(options["seed"])
        clubs = self._build_clubs(profile, rng)
        events = self._build_events(profile, clubs, rounds, upcoming, rng)

        service = ScoreboardIngestionService(client=_CannedClient({"events": events}))
        result = service.ingest_scoreboard(profile["sport"], profile["league"])

        self.stdout.write(
            self.style.WARNING(
                "Synthetic demo data — fictional teams and results, not real ESPN data."
            )
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"Seeded {profile['sport']}/{profile['league']}: "
                f"{result.created} created, {result.updated} updated, {result.errors} errors "
                f"({len(events)} events, {len(clubs)} teams)."
            )
        )

    def _build_clubs(self, profile: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
        clubs = []
        for index, (display_name, abbreviation, location) in enumerate(profile["clubs"]):
            clubs.append(
                {
                    "id": str(90000 + index),
                    "abbreviation": abbreviation,
                    "displayName": display_name,
                    "shortDisplayName": location,
                    "name": display_name.split()[-1],
                    "location": location,
                    "attack": rng.uniform(0.80, 1.25),
                    "defense": rng.uniform(0.80, 1.25),
                }
            )
        return clubs

    def _build_events(
        self,
        profile: dict[str, Any],
        clubs: list[dict[str, Any]],
        rounds: int,
        upcoming: int,
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        fixtures = self._round_robin(clubs, rounds)

        # Scheduled fixtures come last so the completed history sits before them.
        scheduled_pairs = [
            (clubs[i % len(clubs)], clubs[(i + 1) % len(clubs)]) for i in range(upcoming)
        ]

        now = timezone.now().replace(minute=0, second=0, microsecond=0)
        events = []
        total_played = len(fixtures)

        for index, (home, away) in enumerate(fixtures):
            kickoff = now - timedelta(days=(total_played - index) * 3, hours=rng.randint(0, 6))
            events.append(self._make_event(profile, home, away, kickoff, rng, completed=True))

        for index, (home, away) in enumerate(scheduled_pairs):
            kickoff = now + timedelta(days=(index + 1) * 3, hours=2)
            events.append(self._make_event(profile, home, away, kickoff, rng, completed=False))

        return events

    def _round_robin(
        self, clubs: list[dict[str, Any]], rounds: int
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Circle-method schedule; venues swap each cycle so both splits fill up."""
        rotation = list(clubs)
        half = len(rotation) // 2
        fixtures = []

        for round_index in range(rounds):
            for i in range(half):
                home, away = rotation[i], rotation[-(i + 1)]
                if (round_index // (len(clubs) - 1)) % 2:
                    home, away = away, home
                fixtures.append((home, away))
            rotation = [rotation[0], rotation[-1], *rotation[1:-1]]

        return fixtures

    def _make_event(
        self,
        profile: dict[str, Any],
        home: dict[str, Any],
        away: dict[str, Any],
        kickoff: datetime,
        rng: random.Random,
        completed: bool,
    ) -> dict[str, Any]:
        espn_id = (
            f"{profile['sport'][:3]}{int(kickoff.timestamp())}{home['id'][-2:]}{away['id'][-2:]}"
        )

        home_score = away_score = None
        if completed:
            base, edge = profile["base_rate"], profile["home_edge"]
            home_lambda = base * home["attack"] / away["defense"] * edge
            away_lambda = base * away["attack"] / home["defense"]
            if profile["base_rate"] > 20:
                home_score = max(0, round(rng.gauss(home_lambda, 9)))
                away_score = max(0, round(rng.gauss(away_lambda, 9)))
                if home_score == away_score:  # no draws in this profile
                    home_score += 1
            else:
                home_score = _poisson(rng, home_lambda)
                away_score = _poisson(rng, away_lambda)

        status = (
            {
                "type": {"state": "post", "completed": True, "detail": "Final"},
                "period": profile["periods"],
            }
            if completed
            else {"type": {"state": "pre", "completed": False, "detail": "Scheduled"}, "period": 0}
        )

        return {
            "id": espn_id,
            "uid": f"s:seed~e:{espn_id}",
            "date": kickoff.isoformat().replace("+00:00", "Z"),
            "name": f"{away['displayName']} at {home['displayName']}",
            "shortName": f"{away['abbreviation']} @ {home['abbreviation']}",
            "season": {"year": kickoff.year, "type": 2, "slug": "regular-season"},
            "status": status,
            "links": [],
            "competitions": [
                {
                    "attendance": rng.randint(12000, 62000) if completed else None,
                    "broadcasts": [],
                    "venue": {
                        "id": f"9{home['id']}",
                        "fullName": f"{home['location']} Stadium",
                        "address": {"city": home["location"], "country": "Demoland"},
                        "indoor": profile["sport"] == "basketball",
                        "capacity": 65000,
                    },
                    "competitors": [
                        self._make_competitor(home, "home", home_score, away_score),
                        self._make_competitor(away, "away", away_score, home_score),
                    ],
                }
            ],
        }

    def _make_competitor(
        self,
        club: dict[str, Any],
        home_away: str,
        score: int | None,
        opponent_score: int | None,
    ) -> dict[str, Any]:
        winner = None if score is None or opponent_score is None else score > opponent_score
        return {
            "homeAway": home_away,
            "score": "" if score is None else str(score),
            "winner": winner,
            "linescores": [],
            "records": [],
            "statistics": [],
            "leaders": [],
            "team": {
                "id": club["id"],
                "abbreviation": club["abbreviation"],
                "displayName": club["displayName"],
                "shortDisplayName": club["shortDisplayName"],
                "name": club["name"],
                "location": club["location"],
            },
        }
