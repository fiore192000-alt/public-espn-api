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

from apps.espn.dixon_coles import poisson_pmf
from apps.ingest.services import (
    IngestionResult,
    OddsIngestionService,
    ScoreboardIngestionService,
)
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


def _decimal_to_american(decimal_odds: float) -> float:
    """Invert the moneyline conversion so seeded prices look like ESPN's."""
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1.0) * 100.0)
    return round(-100.0 / (decimal_odds - 1.0))


def _true_probabilities(home_rate: float, away_rate: float) -> tuple[dict[str, float], float]:
    """Exact 1X2 probabilities and over-2.5 probability for independent Poisson rates."""
    max_goals = 12
    home = draw = away = over = 0.0
    for home_goals in range(max_goals + 1):
        home_pmf = poisson_pmf(home_goals, home_rate)
        for away_goals in range(max_goals + 1):
            probability = home_pmf * poisson_pmf(away_goals, away_rate)
            if home_goals > away_goals:
                home += probability
            elif home_goals == away_goals:
                draw += probability
            else:
                away += probability
            if home_goals + away_goals > 2.5:
                over += probability

    total = home + draw + away
    return {"home": home / total, "draw": draw / total, "away": away / total}, over / total


class _CannedClient:
    """Stands in for ESPNClient and replays a prepared scoreboard payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def get_scoreboard(self, sport: str, league: str, date=None, limit=None) -> ESPNResponse:  # noqa: ARG002
        return ESPNResponse(data=self._payload, status_code=200, url="seed://demo-scoreboard")


class _CannedOddsClient:
    """Stands in for ESPNClient and replays a prepared odds payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def get_odds(self, sport: str, league: str, event_id: str, competition_id=None) -> ESPNResponse:  # noqa: ARG002
        return ESPNResponse(data=self._payload, status_code=200, url="seed://demo-odds")


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
        parser.add_argument(
            "--with-odds",
            action="store_true",
            help="Also generate synthetic bookmaker prices for every fixture.",
        )
        parser.add_argument(
            "--odds-margin",
            type=float,
            default=1.06,
            help="Bookmaker overround applied to the synthetic prices (default: 1.06).",
        )
        parser.add_argument(
            "--odds-bias",
            type=float,
            default=0.0,
            help=(
                "Probability shifted from the away side to the home side before pricing, "
                "creating a deliberate inefficiency for value detection to find (default: 0.0)."
            ),
        )

    def handle(self, *args, **options) -> None:
        profile_name = options["profile"]
        profile = PROFILES[profile_name]
        rounds = options["rounds"]
        upcoming = options["upcoming"]

        if rounds < 1:
            raise CommandError("--rounds must be at least 1")
        if options["odds_margin"] < 1.0:
            raise CommandError("--odds-margin must be at least 1.0")

        self.true_rates: dict[str, tuple[float, float]] = {}
        rng = random.Random(options["seed"])
        clubs = self._build_clubs(profile, rng)
        events = self._build_events(profile, clubs, rounds, upcoming, rng)

        service = ScoreboardIngestionService(client=_CannedClient({"events": events}))
        result = service.ingest_scoreboard(profile["sport"], profile["league"])

        odds_result = None
        if options["with_odds"]:
            odds_result = self._seed_odds(
                profile, events, options["odds_margin"], options["odds_bias"]
            )

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
        if odds_result is not None:
            self.stdout.write(
                self.style.SUCCESS(
                    f"Seeded odds: {odds_result.created} created, "
                    f"{odds_result.updated} updated, {odds_result.errors} errors."
                )
            )

    def _seed_odds(
        self,
        profile: dict[str, Any],
        events: list[dict[str, Any]],
        margin: float,
        bias: float,
    ) -> IngestionResult:
        """Price every fixture off the rates that generated it, then add margin.

        Because the prices come from the true probabilities, a well-fitted model
        should find no value at all — which is the point. `--odds-bias` moves
        probability from the away side to the home side to plant an inefficiency
        and confirm the detector actually fires.
        """
        from apps.espn.models import Event

        total = IngestionResult(details=[])
        events_by_id = {
            event.espn_id: event for event in Event.objects.filter(league__slug=profile["league"])
        }

        for payload in events:
            espn_id = payload["id"]
            event = events_by_id.get(espn_id)
            rates = self.true_rates.get(espn_id)
            if event is None or rates is None:
                continue

            item = self._odds_item(rates, margin, bias)
            service = OddsIngestionService(client=_CannedOddsClient({"items": [item]}))
            result = service.ingest_event_odds(profile["sport"], profile["league"], event)
            total.created += result.created
            total.updated += result.updated
            total.errors += result.errors

        return total

    def _odds_item(
        self,
        rates: tuple[float, float],
        margin: float,
        bias: float,
    ) -> dict[str, Any]:
        """Build one ESPN-shaped provider entry, priced in American moneylines."""
        home_rate, away_rate = rates
        outcomes, over = _true_probabilities(home_rate, away_rate)

        if bias:
            shift = min(bias, outcomes["away"])
            outcomes = {
                **outcomes,
                "home": outcomes["home"] + shift,
                "away": outcomes["away"] - shift,
            }

        def price(probability: float) -> float:
            # Overround is applied by inflating each implied probability equally.
            return _decimal_to_american(1.0 / max(probability * margin, 1e-9))

        return {
            "provider": {"id": "9999", "name": "Demo Book"},
            "overUnder": 2.5,
            "homeTeamOdds": {"moneyLine": price(outcomes["home"])},
            "awayTeamOdds": {"moneyLine": price(outcomes["away"])},
            "drawOdds": {"moneyLine": price(outcomes["draw"])},
            "overOdds": price(over),
            "underOdds": price(1.0 - over),
        }

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
        total_rounds = max((round_index for round_index, _, _ in fixtures), default=0) + 1

        # One match round per week, with that round's fixtures spread over a
        # weekend — the schedule shape the time-decay weighting assumes.
        for round_index, home, away in fixtures:
            weeks_ago = total_rounds - round_index
            kickoff = (
                now
                - timedelta(weeks=weeks_ago)
                + timedelta(days=rng.randint(0, 1), hours=rng.randint(0, 6))
            )
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
                fixtures.append((round_index, home, away))
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

        base, edge = profile["base_rate"], profile["home_edge"]
        home_lambda = base * home["attack"] / away["defense"] * edge
        away_lambda = base * away["attack"] / home["defense"]
        # Remember the rates that generated this fixture so synthetic odds can be
        # priced off the truth rather than off the model being tested.
        self.true_rates[espn_id] = (home_lambda, away_lambda)

        home_score = away_score = None
        if completed:
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
