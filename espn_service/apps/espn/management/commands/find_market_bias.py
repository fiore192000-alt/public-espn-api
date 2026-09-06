"""Search bookmaker prices for an exploitable bias, and refuse to overclaim one.

The search is the easy half. What this command exists for is the second half: a
rule that wins a search over dozens of alternatives has to survive later matches
of the same league, whole leagues the search never read, and a settlement at a
single bookmaker rather than at the best price available anywhere. Most rules do
not, and the command says so in as many words.
"""

import json

from django.core.management.base import BaseCommand, CommandError

from apps.espn import devig, market_bias
from apps.espn.markets import SELECTION_AWAY, SELECTION_DRAW, SELECTION_HOME
from apps.espn.models import League

SELECTIONS = (SELECTION_AWAY, SELECTION_DRAW, SELECTION_HOME)
SINGLE_BOOK = "fd-b365"
BEST_PRICE = "fd-max"


class Command(BaseCommand):
    help = "Search the market for a price bias, then test it out of sample before believing it."

    def add_arguments(self, parser) -> None:
        parser.add_argument("league", help="League the search runs on (e.g. 'ita.1').")
        parser.add_argument(
            "--split-year",
            type=int,
            default=2019,
            help="Matches before this year are searched; the rest validate (default: 2019).",
        )
        parser.add_argument(
            "--validate-on",
            nargs="*",
            default=None,
            help="Extra league slugs used only for validation (default: every other league).",
        )
        parser.add_argument(
            "--provider",
            default=SINGLE_BOOK,
            help=f"Provider whose single book is settled and devigged (default: {SINGLE_BOOK}).",
        )
        parser.add_argument(
            "--best-provider",
            default=BEST_PRICE,
            help=f"Provider holding the best available price (default: {BEST_PRICE}).",
        )
        parser.add_argument(
            "--devig",
            default=devig.SHIN,
            choices=list(devig.METHODS),
            help=f"Method used to recover what the book believes (default: {devig.SHIN}).",
        )
        parser.add_argument(
            "--minimum-bets",
            type=int,
            default=market_bias.MINIMUM_BETS,
            help=f"Smallest sample a rule may be ranked on (default: {market_bias.MINIMUM_BETS}).",
        )
        parser.add_argument(
            "--carry",
            type=int,
            default=market_bias.CANDIDATES_CARRIED,
            help=(
                "How many of the best discovery rules face validation "
                f"(default: {market_bias.CANDIDATES_CARRIED})."
            ),
        )
        parser.add_argument("--json", action="store_true", help="Emit the raw report as JSON.")

    def handle(self, *args, **options) -> None:
        discovery_league = self._league(options["league"])
        loader = {
            "provider": options["provider"],
            "best_provider": options["best_provider"],
            "devig_method": options["devig"],
            "selections": SELECTIONS,
        }

        matches = market_bias.collect_matches(discovery_league, **loader)
        if not matches:
            raise CommandError(
                f"No complete 1X2 book stored for provider {options['provider']!r} in "
                f"{discovery_league.slug}. Load results with odds first — see "
                "ingest_football_data."
            )

        split = options["split_year"]
        discovery = [match for match in matches if match.date.year < split]
        held_out = [match for match in matches if match.date.year >= split]
        if not discovery:
            raise CommandError(
                f"No {discovery_league.slug} matches before {split}, so there is nothing to "
                "search. Lower --split-year or load earlier seasons."
            )

        validation: list[tuple[str, list[market_bias.PricedMatch]]] = []
        if held_out:
            validation.append((f"{discovery_league.slug} {split}+", held_out))
        for league in self._validation_leagues(discovery_league, options["validate_on"]):
            unseen = market_bias.collect_matches(league, **loader)
            if unseen:
                validation.append((f"{league.slug} (all years)", unseen))

        if not validation:
            raise CommandError(
                "Nothing to validate on: no matches after the split year and no other league "
                "with odds. A search with no held-out data cannot conclude anything, so this "
                "command will not pretend otherwise."
            )

        report = market_bias.investigate(
            discovery,
            validation,
            selections=SELECTIONS,
            discovery_league=discovery_league.slug,
            provider=options["provider"],
            best_provider=options["best_provider"],
            devig_method=options["devig"],
            minimum_bets=options["minimum_bets"],
            carried=options["carry"],
        )

        if options["json"]:
            self.stdout.write(json.dumps(report.to_dict(), indent=2))
            return

        self._render(report)

    def _league(self, slug: str) -> League:
        try:
            return League.objects.get(slug__iexact=slug)
        except League.DoesNotExist as exc:
            raise CommandError(f"No league with slug {slug!r}.") from exc

    def _validation_leagues(self, discovery: League, requested: list[str] | None) -> list[League]:
        if requested is None:
            return list(League.objects.exclude(pk=discovery.pk).order_by("slug"))
        chosen = [self._league(slug) for slug in requested]
        return [league for league in chosen if league.pk != discovery.pk]

    # -- rendering ----------------------------------------------------------

    def _render(self, report: market_bias.Report) -> None:
        payload = report.to_dict()
        self._render_setup(payload)
        self._render_calibration(payload["discovery"])
        self._render_search(payload["discovery"])
        self._render_validation(payload)
        self._render_conclusion(report)

    def _render_setup(self, payload: dict) -> None:
        discovery = payload["discovery"]
        self.stdout.write("")
        self.stdout.write(
            self.style.MIGRATE_HEADING(f"Market bias search — discovery on {discovery['league']}")
        )
        self.stdout.write(
            f"  {discovery['matches']} matches searched. Single book {payload['provider']!r}, "
            f"best price {payload['best_provider']!r}, devigged with {payload['devig_method']}."
        )

    def _render_calibration(self, discovery: dict) -> None:
        self.stdout.write("")
        self.stdout.write("  1. Is the market calibrated, in the period being searched?")
        self.stdout.write(
            f"    {'odds':<14}{'legs':>8}{'implies':>10}{'happens':>10}{'gap':>9}{'z':>8}"
        )
        for band in discovery["calibration"]:
            self.stdout.write(
                f"    {band['band']:<14}{band['legs']:>8}{band['market_implies']:>10.4f}"
                f"{band['actually_happens']:>10.4f}{band['gap']:>+9.4f}{band['z']:>+8.1f}"
            )
        self.stdout.write(
            "    A gap here is not money. The margin has to be cleared before a gap in the\n"
            "    market's favour becomes a gap in yours, which is what step 2 measures."
        )

    def _render_search(self, discovery: dict) -> None:
        self.stdout.write("")
        self.stdout.write("  2. Every rule, settled as real bets over the discovery period")
        self.stdout.write(
            f"    {'rule':<22}{'matches':>9}{'at book':>10}{'at best':>10}{'t (best)':>10}"
        )
        for rule in discovery["rules"]:
            self.stdout.write(
                f"    {rule['rule']:<22}{rule['matches']:>9}"
                f"{rule['at_book']['yield']:>+10.4f}{rule['at_best']['yield']:>+10.4f}"
                f"{rule['t_at_best']:>+10.2f}"
            )
        self.stdout.write("")
        self.stdout.write(
            self.style.WARNING(
                f"    {discovery['hypotheses_tested']} hypotheses were eligible to be picked. "
                f"About {discovery['expected_false_positives']} of them would clear |t| >= "
                f"{market_bias.SIGNIFICANT_T:.0f} on pure noise, so the best t-statistic above "
                "is not a finding — it is a lottery ticket, and step 3 cashes it."
            )
        )

    def _render_validation(self, payload: dict) -> None:
        self.stdout.write("")
        self.stdout.write("  3. The winners, on data the search never touched")
        for verdict in payload["verdicts"]:
            self.stdout.write("")
            self.stdout.write(
                f"    RULE  {verdict['rule']}   (discovery t={verdict['discovery_t']:+.2f})"
            )
            for entry in verdict["sets"]:
                self.stdout.write(
                    f"      {entry['set']:<24}n={entry['matches']:<6} "
                    f"yield={entry['yield']:+.4f}  se={entry['stderr']:.4f}  t={entry['t']:+.2f}"
                )
            pooled = verdict["pooled"]
            low, high = verdict["interval"]
            self.stdout.write(
                f"      {'POOLED':<24}n={pooled['matches']:<6} "
                f"yield={pooled['at_best']['yield']:+.4f}  "
                f"95% [{low:+.4f}, {high:+.4f}]  "
                f"({verdict['positive_sets']}/{verdict['total_sets']} sets positive)"
            )
            self.stdout.write(
                f"      {'same rule, single book':<24}yield={pooled['at_book']['yield']:+.4f}  "
                f"— price selection is worth {pooled['price_selection_value']:+.4f} of that"
            )
            self._render_outcome(verdict)

    def _render_outcome(self, verdict: dict) -> None:
        if verdict["outcome"] == market_bias.ESTABLISHED:
            self.stdout.write(
                self.style.SUCCESS("      PASSES every gate. Worth a paper-traded forward test.")
            )
            return

        style = (
            self.style.ERROR if verdict["outcome"] == market_bias.REJECTED else self.style.WARNING
        )
        label = "REJECTED" if verdict["outcome"] == market_bias.REJECTED else "NOT ESTABLISHED"
        self.stdout.write(style(f"      {label}:"))
        for reason in verdict["reasons"]:
            self.stdout.write(style(f"        - {reason}"))

    def _render_conclusion(self, report: market_bias.Report) -> None:
        self.stdout.write("")
        if report.established:
            names = ", ".join(verdict.rule.label for verdict in report.established)
            self.stdout.write(
                self.style.SUCCESS(
                    f"  Survived out of sample: {names}. Before staking anything on it, run the "
                    "same rule forward on matches that had not been played when this command "
                    "was written — a held-out season still sits inside a dataset someone chose."
                )
            )
        else:
            self.stdout.write(
                self.style.WARNING(
                    "  No rule survived. That is the expected result: the pre-match 1X2 market "
                    "is priced by people doing this full time, and a search over price bands "
                    "and outcomes is the first thing anyone tries. An edge, if it exists, is "
                    "more likely to be in information the price does not yet contain — team "
                    "news, closing-line movement, a less-watched market — than in the shape of "
                    "the prices themselves."
                )
            )
