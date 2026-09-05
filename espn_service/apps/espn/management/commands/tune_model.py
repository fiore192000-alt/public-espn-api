"""Choose the model's decay half-life from a league's own history."""

import json

from django.core.management.base import BaseCommand, CommandError

from apps.espn.backtest import sweep_half_life
from apps.espn.models import League

DEFAULT_HALF_LIVES = "30,60,90,120,180,365"
# Below this gap in log-loss, two half-lives are not meaningfully different on a
# few hundred matches; picking the winner is then picking noise.
MEANINGFUL_LOG_LOSS_GAP = 0.005


class Command(BaseCommand):
    help = "Sweep the decay half-life and rank it by out-of-sample forecast quality."

    def add_arguments(self, parser) -> None:
        parser.add_argument("league", help="League slug (e.g. 'ita.1').")
        parser.add_argument(
            "--half-lives",
            default=DEFAULT_HALF_LIVES,
            help=f"Comma-separated candidates in days (default: {DEFAULT_HALF_LIVES}).",
        )
        parser.add_argument(
            "--refit-every",
            type=int,
            default=5,
            help="Refit after this many matches; higher is faster (default: 5).",
        )
        parser.add_argument("--json", action="store_true", help="Emit raw JSON.")

    def handle(self, *args, **options) -> None:
        try:
            league = League.objects.get(slug__iexact=options["league"])
        except League.DoesNotExist as exc:
            raise CommandError(f"No league with slug {options['league']!r}.") from exc

        try:
            half_lives = [
                float(value) for value in options["half_lives"].split(",") if value.strip()
            ]
        except ValueError as exc:
            raise CommandError("--half-lives must be a comma-separated list of numbers.") from exc
        if not half_lives:
            raise CommandError("--half-lives must contain at least one value.")

        results = sweep_half_life(league, half_lives, refit_every=options["refit_every"])

        if options["json"]:
            self.stdout.write(json.dumps(results, indent=2))
            return

        self._render(league.slug, results)

    def _render(self, league_slug: str, results: list[dict]) -> None:
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING(f"Half-life sweep — {league_slug}"))

        scored = [row for row in results if row["log_loss"] is not None]
        if not scored:
            self.stdout.write(
                self.style.WARNING("  No matches had enough history to score at any half-life.")
            )
            return

        self.stdout.write("")
        self.stdout.write(
            f"  {'half-life':>10}{'matches':>10}{'log loss':>12}{'brier':>10}{'accuracy':>10}"
        )
        for row in results:
            if row["log_loss"] is None:
                continue
            self.stdout.write(
                f"  {row['half_life_days']:>10.0f}{row['forecasts']:>10}"
                f"{row['log_loss']:>12.4f}{row['brier']:>10.4f}{row['accuracy']:>10.4f}"
            )

        best, worst = scored[0], scored[-1]
        spread = worst["log_loss"] - best["log_loss"]
        baseline = best["baseline_log_loss"]

        self.stdout.write("")
        self.stdout.write(
            f"  Best half-life: {best['half_life_days']:.0f} days "
            f"(log loss {best['log_loss']:.4f} over {best['forecasts']} matches)."
        )

        if spread < MEANINGFUL_LOG_LOSS_GAP:
            self.stdout.write(
                self.style.WARNING(
                    f"    All candidates land within {spread:.4f} log loss of each other. On this "
                    "much data the choice is not distinguishable — keep the default rather than "
                    "tuning to noise."
                )
            )
        else:
            self.stdout.write(
                f"    Spread across candidates: {spread:.4f} log loss. Re-run on more history "
                "before treating the winner as settled."
            )

        if baseline is not None and best["log_loss"] > baseline:
            self.stdout.write(
                self.style.WARNING(
                    f"    Even the best half-life ({best['log_loss']:.4f}) does not beat the "
                    f"in-sample base-rate reference ({baseline:.4f}). The model is not yet adding "
                    "skill on this data."
                )
            )
