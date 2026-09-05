"""Walk-forward backtest of the scoreline model over a league's history."""

import json

from django.core.management.base import BaseCommand, CommandError
from django.utils.dateparse import parse_datetime

from apps.espn import backtest
from apps.espn.dixon_coles import DEFAULT_HALF_LIFE_DAYS, RELIABLE_MATCH_COUNT
from apps.espn.models import League
from apps.espn.value import (
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_KELLY_FRACTION,
    DEFAULT_MAX_STAKE_FRACTION,
)


class Command(BaseCommand):
    help = "Replay a league's history, scoring the model on matches it never saw."

    def add_arguments(self, parser) -> None:
        parser.add_argument("league", help="League slug (e.g. 'ita.1').")
        parser.add_argument("--date-from", help="Only score matches on or after this ISO datetime.")
        parser.add_argument("--date-to", help="Only score matches on or before this ISO datetime.")
        parser.add_argument(
            "--half-life",
            type=float,
            default=DEFAULT_HALF_LIFE_DAYS,
            help=f"Days after which a match counts half as much (default: {DEFAULT_HALF_LIFE_DAYS}).",
        )
        parser.add_argument(
            "--refit-every",
            type=int,
            default=1,
            help="Refit after this many matches; higher is faster and slightly staler (default: 1).",
        )
        parser.add_argument(
            "--edge",
            type=float,
            default=DEFAULT_EDGE_THRESHOLD,
            help=f"Minimum edge over the devigged price to bet (default: {DEFAULT_EDGE_THRESHOLD}).",
        )
        parser.add_argument(
            "--kelly",
            type=float,
            default=DEFAULT_KELLY_FRACTION,
            help=f"Fraction of full Kelly to stake (default: {DEFAULT_KELLY_FRACTION}).",
        )
        parser.add_argument(
            "--max-stake",
            type=float,
            default=DEFAULT_MAX_STAKE_FRACTION,
            help=f"Cap on any single stake, as a share of bankroll (default: {DEFAULT_MAX_STAKE_FRACTION}).",
        )
        parser.add_argument(
            "--min-history",
            type=float,
            default=RELIABLE_MATCH_COUNT,
            help=f"Weighted matches required before betting (default: {RELIABLE_MATCH_COUNT}).",
        )
        parser.add_argument("--json", action="store_true", help="Emit the raw report as JSON.")

    def handle(self, *args, **options) -> None:
        try:
            league = League.objects.get(slug__iexact=options["league"])
        except League.DoesNotExist as exc:
            raise CommandError(f"No league with slug {options['league']!r}.") from exc

        report = backtest.run(
            league,
            date_from=_parse(options["date_from"], "--date-from"),
            date_to=_parse(options["date_to"], "--date-to"),
            half_life_days=options["half_life"],
            refit_every=options["refit_every"],
            edge_threshold=options["edge"],
            kelly_multiplier=options["kelly"],
            max_stake_fraction=options["max_stake"],
            min_effective_matches=options["min_history"],
        )
        payload = report.to_dict()

        if options["json"]:
            self.stdout.write(json.dumps(payload, indent=2))
            return

        self._render(payload)

    def _render(self, payload: dict) -> None:
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING(f"Backtest — {payload['league']}"))
        self.stdout.write(
            f"  {payload['forecasts']} matches scored, {payload['refits']} refits, "
            f"skipped {payload['skipped']}"
        )

        if not payload["forecasts"]:
            self.stdout.write(self.style.WARNING("  No matches had enough history to score."))
            return

        model, baseline = payload["model"], payload["baseline"]
        self.stdout.write("")
        self.stdout.write("  Forecast quality")
        self.stdout.write(f"  {'':22}{'model':>12}{'baseline':>12}")
        self.stdout.write(f"  {'log loss':22}{model['log_loss']:>12}{baseline['log_loss']:>12}")
        self.stdout.write(f"  {'brier':22}{model['brier']:>12}{baseline['brier']:>12}")
        self.stdout.write(f"  {'accuracy':22}{payload['accuracy']:>12}")
        self.stdout.write(
            "  The baseline predicts the window's own outcome rates, measured on that same\n"
            "  window, so it flatters itself. Losing to it narrowly is not damning."
        )

        self._render_market(payload["market"])

        self.stdout.write("")
        self.stdout.write("  Calibration (predicted vs observed)")
        for bucket in payload["calibration"]:
            self.stdout.write(
                f"    {bucket['bucket']:10} n={bucket['predictions']:<5} "
                f"predicted={bucket['mean_predicted']:.3f}  observed={bucket['observed']:.3f}"
            )

        betting = payload["betting"]
        self.stdout.write("")
        self.stdout.write("  Betting simulation")
        if not betting["bets"]:
            self.stdout.write(
                "    No qualifying bets — no odds stored, or none cleared the filters."
            )
            return

        for label, key in (
            ("bets", "bets"),
            ("hit rate", "hit_rate"),
            ("yield (Kelly)", "yield"),
            ("yield (flat)", "flat_stake_yield"),
            ("yield std error", "flat_yield_stderr"),
            ("t statistic", "flat_yield_t_stat"),
            ("final bankroll", "final_bankroll"),
            ("max drawdown", "max_drawdown"),
        ):
            self.stdout.write(f"    {label:22}{str(betting[key]):>12}")

        if betting["distinguishable_from_zero"] and (betting["flat_stake_yield"] or 0) < 0:
            self.stdout.write(
                self.style.ERROR(
                    "    The yield is more than two standard errors BELOW zero. This is not noise: "
                    "the selection rule is reliably picking losing bets. Betting it would lose "
                    "money at a rate the sample size can already prove."
                )
            )
        elif betting["distinguishable_from_zero"]:
            self.stdout.write(
                self.style.WARNING(
                    "    The yield is more than two standard errors above zero. That is a signal "
                    "worth investigating, not a guarantee."
                )
            )
        else:
            self.stdout.write(
                self.style.WARNING(
                    "    This yield is within two standard errors of zero — it is indistinguishable "
                    "from no edge at all, whatever its sign."
                )
            )

    def _render_market(self, market: dict) -> None:
        """The comparison that actually decides whether this is bettable."""
        self.stdout.write("")
        self.stdout.write("  Against the market")
        if not market["matches"]:
            self.stdout.write("    No stored odds to compare against.")
            return

        model, prices = market["model"], market["market"]
        self.stdout.write(f"  {'':22}{'model':>12}{'market':>12}")
        self.stdout.write(f"  {'log loss':22}{model['log_loss']:>12}{prices['log_loss']:>12}")
        self.stdout.write(f"  {'brier':22}{model['brier']:>12}{prices['brier']:>12}")
        self.stdout.write(f"  {'matches compared':22}{market['matches']:>12}")

        delta = market["log_loss_delta"]
        if market["model_beats_market"]:
            self.stdout.write(
                self.style.SUCCESS(
                    f"    The model's log loss is {abs(delta):.4f} below the devigged price. "
                    "That is the only result here that would justify betting — verify it holds "
                    "out of sample before acting on it."
                )
            )
        else:
            self.stdout.write(
                self.style.WARNING(
                    f"    The market is better by {delta:.4f} log loss. The model carries no "
                    "information the price does not already have, so no edge found here is real: "
                    "any positive yield below is variance."
                )
            )


def _parse(value: str | None, flag: str):
    if not value:
        return None
    parsed = parse_datetime(value)
    if parsed is None:
        raise CommandError(f"{flag} must be an ISO datetime, got {value!r}.")
    return parsed
