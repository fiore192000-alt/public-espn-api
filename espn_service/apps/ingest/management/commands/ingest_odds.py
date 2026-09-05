"""Management command to ingest ESPN betting odds for stored events."""

import argparse

from django.core.management.base import BaseCommand, CommandError

from apps.espn.models import Event
from apps.ingest.services import OddsIngestionService


class Command(BaseCommand):
    help = "Fetch ESPN odds for events already in the database."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("sport", help="Sport slug (e.g. soccer)")
        parser.add_argument("league", help="League slug (e.g. ita.1)")
        parser.add_argument(
            "--status",
            default=Event.STATUS_SCHEDULED,
            help=f"Only fetch odds for events in this status (default: {Event.STATUS_SCHEDULED}).",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=50,
            help="Maximum events to fetch odds for (default: 50).",
        )

    def handle(self, *args, **options) -> None:
        events = list(
            Event.objects.filter(
                league__slug__iexact=options["league"],
                league__sport__slug__iexact=options["sport"],
                status=options["status"],
            ).order_by("date")[: options["limit"]]
        )
        if not events:
            raise CommandError(
                f"No {options['status']} events stored for {options['sport']}/{options['league']}."
            )

        result = OddsIngestionService().ingest_league_odds(
            options["sport"], options["league"], events
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"Odds for {len(events)} events: {result.created} created, "
                f"{result.updated} updated, {result.errors} errors."
            )
        )
