"""Score the models against the closing line instead of against the result.

The result is one bit per match. The closing line is continuous, it is the
market's own final word, and beating it predicts profit on this data — so this
is the same question every other command asks, put in a form that a few thousand
matches can actually answer.
"""

import json

from django.core.management.base import BaseCommand, CommandError

from apps.espn import backtest, clv
from apps.espn.models import League

CANDIDATES = (
    ("dixon_coles", "probabilities"),
    ("elo", "elo_probabilities"),
    ("club_elo", "club_elo_probabilities"),
)


class Command(BaseCommand):
    help = "Measure whether a model anticipates the closing line, out of sample."

    def add_arguments(self, parser) -> None:
        parser.add_argument("league", help="League slug (e.g. 'eng.2').")
        parser.add_argument(
            "--refit-every",
            type=int,
            default=10,
            help="Refit after this many matches; higher is faster (default: 10).",
        )
        parser.add_argument(
            "--edge",
            type=float,
            default=0.0,
            help="Only count picks the model rates this far above the open (default: 0).",
        )
        parser.add_argument("--json", action="store_true", help="Emit the raw report as JSON.")

    def handle(self, *args, **options) -> None:
        try:
            league = League.objects.get(slug__iexact=options["league"])
        except League.DoesNotExist as exc:
            raise CommandError(f"No league with slug {options['league']!r}.") from exc

        report = backtest.run(league, refit_every=options["refit_every"])
        with_close = [record for record in report.forecasts if record.closing_probabilities]
        if not with_close:
            raise CommandError(
                f"No match in {league.slug} carries a closing line "
                f"({len(report.forecasts)} scored). Load the original football-data.co.uk "
                "season files, which have the closing columns — see ingest_football_data."
            )

        reports = []
        for name, source in CANDIDATES:
            observations = clv.observations_from(with_close, source)
            if observations:
                reports.append(clv.assess(name, observations, edge=options["edge"]))

        if not reports:
            raise CommandError("No model produced a forecast on a match that also has both lines.")

        payload = {
            "league": league.slug,
            "matches_with_closing_line": len(with_close),
            "edge_threshold": options["edge"],
            "models": [report.to_dict() for report in reports],
        }

        if options["json"]:
            self.stdout.write(json.dumps(payload, indent=2))
            return

        self._render(payload, reports)

    def _render(self, payload: dict, reports: list[clv.Report]) -> None:
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING(f"Closing-line value — {payload['league']}"))
        self.stdout.write(
            f"  {payload['matches_with_closing_line']} matches carry both an opening and a "
            "closing line."
        )

        self.stdout.write("")
        self.stdout.write("  Scored on the same fixtures (lower is better)")
        self.stdout.write(f"  {'':16}{'model':>10}{'opening':>10}{'closing':>10}")
        for report in reports:
            losses = report.losses
            self.stdout.write(
                f"  {report.name:16}{losses['model']:>10.4f}"
                f"{losses['opening']:>10.4f}{losses['closing']:>10.4f}"
            )
        self.stdout.write(
            "  The closing column is the wall. It is the market's last word, and no price\n"
            "  in it was available when these forecasts were made."
        )

        self.stdout.write("")
        self.stdout.write("  Anticipation: of the model's disagreement with the open,")
        self.stdout.write("  how much does the market itself go on to travel?")
        self.stdout.write(f"  {'':16}{'slope':>9}{'stderr':>9}{'t':>8}   95% interval")
        for report in reports:
            a = report.anticipation
            low, high = a.interval()
            self.stdout.write(
                f"  {report.name:16}{a.slope:>+9.4f}{a.stderr:>9.4f}{a.t_stat:>+8.2f}"
                f"   [{low:+.4f}, {high:+.4f}]"
            )
        self.stdout.write(
            "  Zero means the market never ratifies the disagreement. One would mean the\n"
            "  market ends up exactly where the model already was."
        )

        self.stdout.write("")
        self.stdout.write(
            f"  Closing-line value of the model's own picks"
            f"{f' (edge > {payload["edge_threshold"]})' if payload['edge_threshold'] else ''}"
        )
        self.stdout.write(f"  {'':16}{'matches':>9}{'picks':>8}{'CLV':>10}{'t':>8}")
        for report in reports:
            c = report.closing_line
            self.stdout.write(
                f"  {report.name:16}{c.matches:>9}{c.picks:>8}{c.mean:>+10.4f}{c.t_stat:>+8.2f}"
            )

        self._verdict(reports)

    def _verdict(self, reports: list[clv.Report]) -> None:
        self.stdout.write("")
        winners = [r for r in reports if r.anticipation.anticipates_the_market]
        beaters = [r for r in reports if r.closing_line.beats_the_close]

        if winners or beaters:
            named = ", ".join(sorted({r.name for r in winners + beaters}))
            best = max((r.closing_line.mean for r in reports), default=0.0)
            self.stdout.write(
                self.style.SUCCESS(
                    f"    {named} carries information the opening price does not. That is real, "
                    "and it is the only positive finding this project has produced."
                )
            )
            self.stdout.write(
                self.style.WARNING(
                    f"    It is not yet an edge. The best closing-line value here is "
                    f"{best:+.4f} per unit, against a bookmaker margin of roughly 2-3% on the "
                    "same fixtures: information an order of magnitude too small to pay for the "
                    "cost of trading. Settle these picks on results before believing otherwise "
                    "— a positive slope and a losing strategy are perfectly compatible."
                )
            )
        else:
            self.stdout.write(
                self.style.WARNING(
                    "    Neither model anticipates the market. Their disagreements with the "
                    "opening price are not where the line goes, so there is nothing here the "
                    "market has not already priced — measured against a target far sharper "
                    "than win/lose, on far fewer matches than a profit test would need."
                )
            )
        self.stdout.write(
            "    The closing price is a measuring instrument, never a signal: it exists only "
            "after\n    the moment a bet would have had to be struck."
        )
