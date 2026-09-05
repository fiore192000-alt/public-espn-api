"""Print a match analysis for stored events."""

import json

from django.core.management.base import BaseCommand, CommandError

from apps.espn.analysis import DEFAULT_LOOKBACK, AnalysisNotAvailable, analyze_event
from apps.espn.models import Event


class Command(BaseCommand):
    help = "Analyse a stored event: recent form, head-to-head, and a projected result."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--event",
            help="ESPN event id to analyse. Omit to analyse upcoming fixtures instead.",
        )
        parser.add_argument("--league", help="League slug filter (e.g. 'nba', 'demo.1').")
        parser.add_argument(
            "--upcoming",
            type=int,
            default=3,
            help="How many scheduled events to analyse when --event is not given (default: 3).",
        )
        parser.add_argument(
            "--lookback",
            type=int,
            default=DEFAULT_LOOKBACK,
            help=f"Games of history per team (default: {DEFAULT_LOOKBACK}).",
        )
        parser.add_argument(
            "--json", action="store_true", help="Emit raw JSON instead of a report."
        )

    def handle(self, *args, **options) -> None:
        events = self._select_events(options)
        if not events:
            raise CommandError("No matching events found. Ingest a scoreboard first.")

        payloads = []
        for event in events:
            try:
                payloads.append(analyze_event(event, lookback=options["lookback"]))
            except AnalysisNotAvailable as exc:
                self.stderr.write(self.style.WARNING(str(exc)))

        if options["json"]:
            self.stdout.write(json.dumps(payloads, indent=2))
            return

        for payload in payloads:
            self._render(payload)

    def _select_events(self, options: dict) -> list[Event]:
        qs = Event.objects.select_related("league", "league__sport", "venue").prefetch_related(
            "competitors__team"
        )
        if options["league"]:
            qs = qs.filter(league__slug__iexact=options["league"])

        if options["event"]:
            return list(qs.filter(espn_id=options["event"])[:1])

        upcoming = list(
            qs.filter(status=Event.STATUS_SCHEDULED).order_by("date")[: options["upcoming"]]
        )
        if upcoming:
            return upcoming
        return list(qs.order_by("-date")[: options["upcoming"]])

    def _render(self, payload: dict) -> None:
        event = payload["event"]
        home, away = payload["home"], payload["away"]
        projection = payload["projection"]
        probabilities = projection["probabilities"]

        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING(f"{event['name']}"))
        self.stdout.write(
            f"  {event['date']} · {event['league'].upper()} · {event['status_detail'] or event['status']}"
            + (f" · {event['venue']}" if event["venue"] else "")
        )

        self.stdout.write("")
        self.stdout.write(f"  {'':22}{'HOME':>12}{'AWAY':>12}")
        self._row("team", home["team"]["abbreviation"], away["team"]["abbreviation"])
        self._row(
            "record (W-D-L)",
            self._record(home["form"]["overall"]),
            self._record(away["form"]["overall"]),
        )
        self._row("streak", home["form"]["streak"] or "-", away["form"]["streak"] or "-")
        self._row(
            "scored / game",
            home["form"]["overall"]["avg_scored"],
            away["form"]["overall"]["avg_scored"],
        )
        self._row(
            "conceded / game",
            home["form"]["overall"]["avg_conceded"],
            away["form"]["overall"]["avg_conceded"],
        )
        self._row(
            "home/away split",
            self._record(home["form"]["home"]),
            self._record(away["form"]["away"]),
        )
        self._row(
            "absences (weighted)",
            self._absences(home["injuries"]),
            self._absences(away["injuries"]),
        )
        self._row(
            "momentum (pts/game)",
            f"{home['form']['momentum']['points_delta']:+.2f}",
            f"{away['form']['momentum']['points_delta']:+.2f}",
        )
        self._row(
            "opponent strength",
            self._optional(home["form"]["opponent_strength"]),
            self._optional(away["form"]["opponent_strength"]),
        )
        self._row(
            "rest days",
            self._optional(home["context"]["rest_days"]),
            self._optional(away["context"]["rest_days"]),
        )
        self._row(
            "matches in 14 days",
            home["context"]["matches_in_last_14_days"],
            away["context"]["matches_in_last_14_days"],
        )

        self.stdout.write("")
        self.stdout.write("  Projection")
        self._row("expected score", projection["home_score"], projection["away_score"])
        self._row(
            "win probability",
            f"{probabilities['home_win']:.1%}",
            f"{probabilities['away_win']:.1%}",
        )
        if probabilities["draw"]:
            self.stdout.write(f"  {'draw probability':22}{probabilities['draw']:>12.1%}")
        self.stdout.write(f"  {'confidence':22}{payload['confidence']:>12}")

        self.stdout.write("")
        self.stdout.write("  Notes")
        for note in payload["insights"]:
            self.stdout.write(f"    - {note}")

    def _row(self, label: str, home_value, away_value) -> None:
        self.stdout.write(f"  {label:22}{str(home_value):>12}{str(away_value):>12}")

    @staticmethod
    def _record(split: dict) -> str:
        return f"{split['wins']}-{split['draws']}-{split['losses']}"

    @staticmethod
    def _absences(injuries: dict) -> str:
        if not injuries["count"]:
            return "0"
        return f"{injuries['count']} ({injuries['weighted']:.1f})"

    @staticmethod
    def _optional(value) -> str:
        return "-" if value is None else str(value)
