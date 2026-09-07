"""Compare models against each other and, decisively, against the market price."""

import json

from django.core.management.base import BaseCommand, CommandError

from apps.espn import backtest, elo
from apps.espn.combination import Sample, assess
from apps.espn.models import League

MARKET = "market"
DIXON_COLES = "dixon_coles"
ELO = "elo"
CLUB_ELO = "club_elo"
EXPECTED_GOALS = "expected_goals"


class Command(BaseCommand):
    help = "Score every model against the market, then test each for information the price lacks."

    def add_arguments(self, parser) -> None:
        parser.add_argument("league", help="League slug (e.g. 'ita.1').")
        parser.add_argument(
            "--refit-every",
            type=int,
            default=10,
            help="Refit after this many matches; higher is faster (default: 10).",
        )
        parser.add_argument(
            "--train-fraction",
            type=float,
            default=0.5,
            help="Share of matches used to fit the pooling weights (default: 0.5).",
        )
        parser.add_argument(
            "--no-xg",
            action="store_true",
            help=(
                "Skip expected goals and compare the other models over the whole "
                "history instead of the seasons xG covers."
            ),
        )
        parser.add_argument("--json", action="store_true", help="Emit the raw report as JSON.")

    def handle(self, *args, **options) -> None:
        try:
            league = League.objects.get(slug__iexact=options["league"])
        except League.DoesNotExist as exc:
            raise CommandError(f"No league with slug {options['league']!r}.") from exc

        if not 0.0 < options["train_fraction"] < 1.0:
            raise CommandError("--train-fraction must be between 0 and 1.")

        report = backtest.run(league, refit_every=options["refit_every"])
        records = [
            record
            for record in report.forecasts
            if record.market_probabilities and record.elo_probabilities
        ]
        # ClubElo joins only where the league carries the ratings; a league loaded
        # from a source without them still compares the other two.
        rated = [record for record in records if record.club_elo_probabilities]
        if rated:
            records = rated

        # xG exists only from 2014/15 onward, so `records[0]` is the wrong probe:
        # it is the oldest match in the league and never carries it. Comparing
        # five sources also needs one common set of fixtures -- a log loss over
        # 2,829 matches cannot be set beside one over 7,888 -- and the pool
        # raises on a missing source, so when xG covers part of the history the
        # whole comparison narrows to that part.
        with_xg = [record for record in records if record.expected_goals_probabilities]
        narrowed_from = 0
        if with_xg and not options["no_xg"]:
            narrowed_from = len(records)
            records = with_xg
        if not records:
            with_odds = sum(1 for r in report.forecasts if r.market_probabilities)
            with_elo = sum(1 for r in report.forecasts if r.elo_probabilities)
            raise CommandError(
                f"No match has both stored odds and an Elo forecast "
                f"({len(report.forecasts)} scored, {with_odds} with odds, {with_elo} with Elo). "
                + (
                    "Load odds alongside the results — see ingest_football_data."
                    if not with_odds
                    else f"Elo needs at least {elo.MINIMUM_FIT_SAMPLES} completed matches "
                    "before it produces a forecast; load more history."
                )
            )

        # A source joins only where every remaining record carries it: the pool
        # raises on a missing one, and a log loss over a subset cannot be set
        # beside a log loss over the whole.
        candidates = [DIXON_COLES, ELO]
        if all(record.club_elo_probabilities for record in records):
            candidates.append(CLUB_ELO)
        if all(record.expected_goals_probabilities for record in records):
            candidates.append(EXPECTED_GOALS)

        samples = [
            Sample(
                probabilities={
                    MARKET: record.market_probabilities,
                    DIXON_COLES: record.probabilities,
                    ELO: record.elo_probabilities,
                    **({CLUB_ELO: record.club_elo_probabilities} if CLUB_ELO in candidates else {}),
                    **(
                        {EXPECTED_GOALS: record.expected_goals_probabilities}
                        if EXPECTED_GOALS in candidates
                        else {}
                    ),
                },
                actual=record.actual,
            )
            for record in records
        ]

        incremental = assess(
            samples,
            market=MARKET,
            candidates=candidates,
            train_fraction=options["train_fraction"],
        )
        standalone = {
            name: backtest._scores(records, source)
            for name, source in (
                (MARKET, "market_probabilities"),
                (DIXON_COLES, "probabilities"),
                (ELO, "elo_probabilities"),
                *([(CLUB_ELO, "club_elo_probabilities")] if CLUB_ELO in candidates else []),
                *(
                    [(EXPECTED_GOALS, "expected_goals_probabilities")]
                    if EXPECTED_GOALS in candidates
                    else []
                ),
            )
        }

        payload = {
            "league": league.slug,
            "matches": len(records),
            "narrowed_from": narrowed_from,
            "sources": [MARKET, *candidates],
            "standalone": standalone,
            "incremental": incremental.to_dict(),
        }

        if options["json"]:
            self.stdout.write(json.dumps(payload, indent=2))
            return

        self._render(payload)

    def _render(self, payload: dict) -> None:
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING(f"Model comparison — {payload['league']}"))
        self.stdout.write(f"  {payload['matches']} matches with both odds and an Elo forecast.")
        if payload["narrowed_from"]:
            self.stdout.write(
                f"  Narrowed from {payload['narrowed_from']} to the seasons expected goals "
                "covers, so\n  all five sources are scored on the same fixtures. "
                "Pass --no-xg for the full history."
            )

        self.stdout.write("")
        self.stdout.write("  Standalone forecast quality (lower is better)")
        self.stdout.write(f"  {'':22}{'log loss':>12}{'brier':>10}")
        for name, scores in payload["standalone"].items():
            self.stdout.write(f"  {name:22}{scores['log_loss']:>12}{scores['brier']:>10}")
        self.stdout.write(
            "  Predicting well is not the same as being useful. The test below is the one\n"
            "  that matters: does a model know anything the price does not already contain?"
        )

        incremental = payload["incremental"]
        self.stdout.write("")
        self.stdout.write(
            f"  Incremental information over the market "
            f"({incremental['holdout_matches']} held-out matches)"
        )
        self.stdout.write(f"  {'market alone':38}{incremental['market_only_log_loss']:>12}")

        for verdict in incremental["verdicts"]:
            label = " + ".join(verdict["candidates"])
            weights = " ".join(f"{name}={weight}" for name, weight in verdict["weights"].items())
            low, high = verdict["interval"]
            self.stdout.write(
                f"  {'market + ' + label:38}{verdict['holdout_log_loss']:>12}"
                f"   improvement {verdict['improvement_over_market']:+.4f}"
                f"  t {verdict['t']:+.2f}  95% [{low:+.4f}, {high:+.4f}]"
            )
            self.stdout.write(
                f"  {'':38}   same candidate, forecasts shuffled onto the wrong "
                f"matches: {verdict['null_improvement']:+.4f}"
            )
            self.stdout.write(f"  {'':38}   weights: {weights}")

        self.stdout.write(
            "  The shuffled figure is the floor. A pool given any extra source uses it as a\n"
            "  temperature knob on the price, so a candidate that knows nothing still shows a\n"
            "  small gain — an improvement has to clear that, and its own standard error."
        )

        useful = [v for v in incremental["verdicts"] if v["adds_information"]]
        self.stdout.write("")
        if useful:
            names = ", ".join(" + ".join(v["candidates"]) for v in useful)
            self.stdout.write(
                self.style.SUCCESS(
                    f"    Adds information beyond the price: {names}. This is the only result "
                    "that justifies promoting a model into a betting decision — confirm it on "
                    "another league and another period before trusting it."
                )
            )
        else:
            self.stdout.write(
                self.style.WARNING(
                    "    No candidate improves on the market once its weights are fitted out of "
                    "sample. Everything these models know is already in the price, so no edge "
                    "derived from them is real."
                )
            )
