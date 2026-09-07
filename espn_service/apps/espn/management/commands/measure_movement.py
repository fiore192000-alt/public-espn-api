"""Score candidate features on the market's own move from opening to closing price.

The question with the power. A bet settled on the result gives one bit per match;
the market's correction of its first guess is continuous, and the same signal is
visible in hundreds of matches rather than tens of thousands.

Every candidate is reported with the anchor's own contribution beside it, because
the target is ``closing - opening`` and any feature carrying the opening regresses
on itself. Two controls run alongside: the opening price, which is pure mean
reversion, and gaussian noise, which must score nothing.
"""

import json

from django.core.management.base import BaseCommand, CommandError

from apps.espn import backtest, movement
from apps.espn.models import League

# Margin against which a predicted move is judged worth having. Pinnacle's own,
# measured over the 13,657 matches this data covers.
DEFAULT_MARGIN = 0.0288

MODELS = (
    ("dixon-coles", "probabilities"),
    ("elo", "elo_probabilities"),
    ("club-elo", "club_elo_probabilities"),
    ("expected-goals", "expected_goals_probabilities"),
)


class Command(BaseCommand):
    help = "Measure which features predict the market's move from open to close."

    def add_arguments(self, parser) -> None:
        parser.add_argument("league", help="League slug (e.g. 'eng.2').")
        parser.add_argument(
            "--refit-every",
            type=int,
            default=25,
            help="Refit after this many matches; higher is faster (default: 25).",
        )
        parser.add_argument(
            "--margin",
            type=float,
            default=DEFAULT_MARGIN,
            help=f"Margin a predicted move must be worth clearing (default: {DEFAULT_MARGIN}).",
        )
        parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")

    def handle(self, *args, **options) -> None:
        try:
            league = League.objects.get(slug__iexact=options["league"])
        except League.DoesNotExist as exc:
            raise CommandError(f"No league with slug {options['league']!r}.") from exc

        report = backtest.run(league, refit_every=options["refit_every"])
        records = [
            record
            for record in report.forecasts
            if record.market_probabilities and record.closing_probabilities
        ]
        if len(records) < movement.MINIMUM_MATCHES:
            raise CommandError(
                f"{league.slug} has {len(records)} matches carrying both an opening and a "
                f"closing line; {movement.MINIMUM_MATCHES} are needed. Load the original "
                "football-data.co.uk season files, which have the closing columns."
            )

        candidates: list[tuple[str, movement.FeatureFn, bool]] = [
            (name, movement.model_probability(source), True)
            for name, source in MODELS
            if any(getattr(record, source, None) for record in records)
        ]
        candidates.append(("opening price (control)", movement.price_level(), False))
        candidates.append(("gaussian noise (control)", movement.gaussian_noise(), False))

        fits = []
        for name, feature, anchored in candidates:
            movements = movement.movements_from(records, feature, anchored=anchored)
            if not movements:
                continue
            fits.append(movement.fit_against_null(movements, name))

        payload = {
            "league": league.slug,
            "matches_with_both_lines": len(records),
            "margin": options["margin"],
            "candidates": [
                {
                    **entry.to_dict(),
                    "value_per_sd": (
                        None
                        if entry.value_per_sd(options["margin"]) is None
                        else round(entry.value_per_sd(options["margin"]), 4)
                    ),
                }
                for entry in fits
            ],
        }

        if options["json"]:
            self.stdout.write(json.dumps(payload, indent=2))
            return
        self._render(payload, fits, options["margin"])

    def _render(self, payload, fits, margin) -> None:
        write = self.stdout.write
        write("")
        write(f"MARKET MOVEMENT — {payload['league']}")
        write(
            f"  {payload['matches_with_both_lines']:,} matches carrying an opening "
            "and a closing line"
        )
        write("  target: closing probability minus opening probability, through the origin,")
        write("  errors clustered by match.")
        write("")
        write(
            f"  {'feature':<26} {'slope':>9} {'t':>7} {'null':>9} {'net':>9} "
            f"{'anchor':>8} {'per sd':>9} {'verdict':>10}"
        )
        for entry in fits:
            share = entry.anchor_share
            write(
                f"  {entry.name:<26} {entry.slope:>+9.4f} {entry.t_stat:>+7.2f} "
                f"{(entry.null_slope or 0.0):>+9.4f} {(entry.net_slope or 0.0):>+9.4f} "
                f"{('—' if share is None else f'{share:>7.0%}'):>8} "
                f"{(entry.movement_per_sd or 0.0):>+9.4f} "
                f"{('PREDICTS' if entry.predicts_the_market else 'no'):>10}"
            )
        write("")
        for entry in fits:
            if entry.anchor_driven:
                write(
                    f"  ⚠ {entry.name}: the shared opening price accounts for "
                    f"{entry.anchor_share:.0%} of this slope. Read the net column, not the slope."
                )
        write("")
        write(f"  Economic translation, against a {margin:.2%} margin:")
        for entry in fits:
            value = entry.value_per_sd(margin)
            if value is None:
                continue
            write(
                f"    {entry.name:<26} one standard deviation of the feature predicts "
                f"{(entry.movement_per_sd or 0.0):+.4f} of probability "
                f"= {value:.3f} of the margin"
            )
        write("")
        write(
            "  A feature has to clear its own anchor, exclude zero, and be worth more "
            "than a\n  fraction of the margin before it is worth acting on. "
            "H-0001 reached 0.042 and was set aside."
        )
        write("")
