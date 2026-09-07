"""Measure what this project's own rule search finds when there is nothing to find.

Every result the project has produced is the winner of a search. A winner's
yield is only meaningful against the yield a winner would have had on a market
with no edge at all — and that number cannot be read off real data, because real
data does not come with its truth attached. This command simulates markets whose
truth it sets, runs the same sweep `find_market_bias` runs, and reports how good
the best rule looked anyway.

Two controls, in one run:

**negative** — a market where the book believes the truth exactly, so every bet
has the same expected return and nothing is beatable. Whatever the search finds
here is manufactured by the search.

**positive** — the same market with a favourite-longshot bias planted in it. A
search that cannot find a bias we put there is not a search that should be
trusted with one we did not.
"""

import json

from django.core.management.base import BaseCommand, CommandError

from apps.espn import market_bias, synthetic

DEFAULT_LONGSHOT = 0.08
# The real sweep settles rules at the best price across ~17 books, whose pooled
# overround is about 1.003 — not at the single book's 1.05. Simulating the null
# at the book price would compare the search against a market it never searched.
DEFAULT_BEST_PRICE_EDGE = 0.0469


class Command(BaseCommand):
    help = "Report what the rule search finds on markets with a known truth."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--matches",
            type=int,
            default=synthetic.DEFAULT_MATCHES,
            help=f"Matches per simulated market (default: {synthetic.DEFAULT_MATCHES}).",
        )
        parser.add_argument(
            "--trials",
            type=int,
            default=synthetic.DEFAULT_TRIALS,
            help=f"Independent markets to search (default: {synthetic.DEFAULT_TRIALS}).",
        )
        parser.add_argument(
            "--margin",
            type=float,
            default=synthetic.DEFAULT_MARGIN,
            help=f"Bookmaker overround minus one (default: {synthetic.DEFAULT_MARGIN}).",
        )
        parser.add_argument(
            "--longshot",
            type=float,
            default=DEFAULT_LONGSHOT,
            help=(
                "Size of the favourite-longshot bias planted for the positive "
                f"control; 0 skips it (default: {DEFAULT_LONGSHOT})."
            ),
        )
        parser.add_argument(
            "--observed",
            type=float,
            default=None,
            help="A real yield to test against the null (e.g. 0.0497 for +4.97%%).",
        )
        parser.add_argument(
            "--best-price-edge",
            type=float,
            default=DEFAULT_BEST_PRICE_EDGE,
            help=(
                "How much better the best-of-N-books line is than the book. The "
                "real search settles at the best price, whose overround is near "
                f"1.003, so this defaults to {DEFAULT_BEST_PRICE_EDGE} — the "
                "multiplier that turns a 1.05 book into a nearly fair line."
            ),
        )
        parser.add_argument(
            "--observed-t",
            type=float,
            default=None,
            help="A real t-statistic to test against the null (e.g. 2.42).",
        )
        parser.add_argument("--seed", type=int, default=synthetic.DEFAULT_SEED)
        parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")

    def handle(self, *args, **options) -> None:
        try:
            null_spec = synthetic.MarketSpec(
                matches=options["matches"],
                margin=options["margin"],
                best_price_edge=options["best_price_edge"],
                seed=options["seed"],
            )
        except synthetic.SpecError as error:
            raise CommandError(str(error)) from error

        if options["trials"] <= 0:
            raise CommandError("--trials must be positive.")

        # Confirm the generator built the market it was asked for before
        # drawing any conclusion from searching it.
        sample = synthetic.simulate(null_spec)
        realised = synthetic.realised_house_edge(sample)
        spread = synthetic.edge_spread(sample)

        null = synthetic.search_under_the_null(null_spec, trials=options["trials"])

        planted = None
        power = None
        if options["longshot"] > 0:
            planted = synthetic.MarketSpec(
                matches=options["matches"],
                margin=options["margin"],
                best_price_edge=options["best_price_edge"],
                longshot=options["longshot"],
                seed=options["seed"],
            )
            power = synthetic.power_against(planted, null)

        observed = options["observed"]
        report = {
            "market": {
                "matches": null_spec.matches,
                "margin": null_spec.margin,
                "efficient": null_spec.efficient,
                "best_price_edge": null_spec.best_price_edge,
                "best_price_overround": (1 + null_spec.margin) / (1 + null_spec.best_price_edge),
                "house_edge_theoretical": null_spec.house_edge,
                "house_edge_realised": realised,
                "edge_spread": spread,
            },
            "null": {
                "trials": null.trials,
                "rules_per_search": null.rules,
                "mean_best_yield": null.mean_yield,
                "p50_best_yield": null.yield_at(0.50),
                "p95_best_yield": null.yield_at(0.95),
                "p99_best_yield": null.yield_at(0.99),
                "mean_best_t": null.mean_t,
                "p95_best_t": null.t_at(0.95),
                "share_clearing_t2": null.clears_conventional_t,
                "winner_was_a_selection_rule": null.favourite_share,
            },
            "positive_control": (
                None
                if planted is None
                else {
                    "longshot": planted.longshot,
                    "edge_spread": synthetic.edge_spread(synthetic.simulate(planted)),
                    "power_at_95th_percentile": power,
                }
            ),
            "observed": (
                None
                if observed is None
                else {"yield": observed, "p_value_against_null": null.beats(observed)}
            ),
            "observed_t": (
                None
                if options["observed_t"] is None
                else {
                    "t": options["observed_t"],
                    "p_value_against_null": null.beats_t(options["observed_t"]),
                }
            ),
        }

        if options["json"]:
            self.stdout.write(json.dumps(report, indent=2))
            return
        self._report(report, null_spec)

    def _report(self, report, spec) -> None:
        write = self.stdout.write
        market, null = report["market"], report["null"]

        write("")
        write("NEGATIVE CONTROL — a market with nothing to find")
        write(
            f"  {market['matches']:,} matches per search, book overround "
            f"{1 + market['margin']:.4f}, best-price overround "
            f"{market['best_price_overround']:.4f}, {null['trials']} independent markets"
        )
        write(
            f"  every bet's true expected value: {market['house_edge_theoretical']:+.4%} "
            f"(realised {market['house_edge_realised']:+.4%})"
        )
        write(
            f"  spread of expected value across selections: {market['edge_spread']:.6f} "
            "— zero means no bet is better than any other"
        )
        write("")
        write(f"  The search tried {null['rules_per_search']} rules and kept the best.")
        write("  Best rule's yield, on a market where no rule is better than another:")
        write(
            f"    mean {null['mean_best_yield']:+.2%}   "
            f"median {null['p50_best_yield']:+.2%}   "
            f"95th {null['p95_best_yield']:+.2%}   "
            f"99th {null['p99_best_yield']:+.2%}"
        )
        write(
            f"    its t-statistic: mean {null['mean_best_t']:+.2f}, "
            f"95th percentile {null['p95_best_t']:+.2f}"
        )
        write(
            f"    cleared the conventional |t| >= {market_bias.SIGNIFICANT_T:.0f} bar in "
            f"{null['share_clearing_t2']:.0%} of searches"
        )
        write(
            f"    the winner was a whole-selection rule "
            f"{null['winner_was_a_selection_rule']:.0%} of the time"
        )

        positive = report["positive_control"]
        if positive is not None:
            write("")
            write("POSITIVE CONTROL — the same market with a bias planted in it")
            write(
                f"  favourite-longshot loading {positive['longshot']:.2f}; spread of "
                f"expected value now {positive['edge_spread']:.6f}"
            )
            write(
                f"  the search beat the null's 95th-percentile t in "
                f"{positive['power_at_95th_percentile']:.0%} of runs"
            )

        observed = report["observed"]
        observed_t = report["observed_t"]
        if observed is not None or observed_t is not None:
            write("")
            write("AGAINST A REAL RESULT")
        if observed is not None:
            write(
                f"  observed yield {observed['yield']:+.2%} — reached by "
                f"{observed['p_value_against_null']:.1%} of searches on a market "
                "with no edge at all"
            )
        if observed_t is not None:
            write(
                f"  observed t = {observed_t['t']:+.2f} — reached by "
                f"{observed_t['p_value_against_null']:.1%} of searches on a market "
                "with no edge at all"
            )

        write("")
        write(
            f"  On {market['matches']:,} matches this search returns t = "
            f"{null['p95_best_t']:+.2f} from pure noise 5% of the time. A discovery "
            "below that bar is not evidence of anything."
        )
        write("")
