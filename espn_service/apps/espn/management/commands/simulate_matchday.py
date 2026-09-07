"""Replay one day's card under every staking rule and report the bankroll.

Answers a question with a very tempting wrong answer: *what would this have bet
yesterday, and what would that have done to the money?* The right answer names
the rules in advance, settles them against the real results, and then says how
much of the outcome was luck — which is what the exact day-distribution at the
bottom of the output is for.
"""

import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.espn import matchday

DEFAULT_BANKROLL = 10000.0
CARD_DIRECTORY = Path(settings.BASE_DIR) / "research" / "cards"


class Command(BaseCommand):
    help = "Settle one day's fixtures under each staking rule and report the bankroll."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "card",
            help=(
                "Path to a card JSON file, or the name of one in research/cards "
                "(e.g. '2026-09-06-serie-a')."
            ),
        )
        parser.add_argument(
            "--bankroll",
            type=float,
            default=DEFAULT_BANKROLL,
            help=f"Starting bankroll in euros (default: {DEFAULT_BANKROLL:.0f}).",
        )
        parser.add_argument(
            "--rule",
            action="append",
            dest="rules",
            help="Only run this rule; repeatable. Default: every rule.",
        )
        parser.add_argument(
            "--source",
            choices=(matchday.MARKET, matchday.MODEL),
            default=matchday.MARKET,
            help=(
                "Whose probabilities the luck calculation uses. 'market' is the "
                "honest default: it asks how the day would have gone if the book "
                "were right (default: market)."
            ),
        )
        parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")

    # --- loading ---------------------------------------------------------

    def _resolve(self, given: str) -> Path:
        candidates = [Path(given)]
        if not given.endswith(".json"):
            candidates.append(CARD_DIRECTORY / f"{given}.json")
        candidates.append(CARD_DIRECTORY / given)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        known = sorted(path.stem for path in CARD_DIRECTORY.glob("*.json"))
        raise CommandError(
            f"No card at {given!r}. Available in research/cards: {', '.join(known) or 'none'}."
        )

    def handle(self, *args, **options) -> None:
        path = self._resolve(options["card"])
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise CommandError(f"Cannot read {path}: {error}") from error

        try:
            fixtures = matchday.card_from_dict(document)
        except matchday.CardError as error:
            raise CommandError(str(error)) from error

        bankroll = options["bankroll"]
        if bankroll <= 0:
            raise CommandError("The bankroll must be positive.")

        names = options["rules"] or list(matchday.RULES)
        unknown = [name for name in names if name not in matchday.RULES]
        if unknown:
            raise CommandError(
                f"Unknown rule(s): {', '.join(unknown)}. Known: {', '.join(matchday.RULES)}."
            )

        ledgers = [matchday.settle(fixtures, name, bankroll) for name in names]
        outcomes = {
            ledger.rule: matchday.summarise(ledger.bets, options["source"]) for ledger in ledgers
        }

        if options["json"]:
            self.stdout.write(
                json.dumps(
                    self._as_dict(document, fixtures, ledgers, outcomes, bankroll),
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return
        self._report(document, fixtures, ledgers, outcomes, bankroll)

    # --- reporting -------------------------------------------------------

    def _report(self, document, fixtures, ledgers, outcomes, bankroll) -> None:
        write = self.stdout.write
        write("")
        write(f"CARD  {document.get('round') or document.get('date', '')}")
        write(f"      bankroll EUR {bankroll:,.2f}")
        write("")
        write(f"{'match':<24} {'1':>6} {'X':>6} {'2':>6} {'margin':>7}  model 1/X/2        result")
        for fixture in fixtures:
            model = "/".join(f"{fixture.model[key]:.3f}" for key in matchday.SELECTIONS)
            result = fixture.result or "pending"
            write(
                f"{fixture.name:<24} "
                f"{fixture.price('home'):>6.2f} {fixture.price('draw'):>6.2f} "
                f"{fixture.price('away'):>6.2f} {fixture.margin:>7.2%}  {model}  {result}"
            )

        for ledger in ledgers:
            write("")
            write(f"RULE  {ledger.rule} — {matchday.DESCRIPTIONS[ledger.rule]}")
            if not ledger.bets:
                write("      no bet met the rule on this card.")
                continue
            write(
                f"      {'selection':<44} {'price':>6} {'model':>7} {'fair':>7} "
                f"{'edge':>7} {'stake':>9} {'return':>10}"
            )
            for bet in ledger.bets:
                landed = "void" if not bet.settled else ("won" if bet.won else "lost")
                write(
                    f"      {bet.label:<44} {bet.price:>6.2f} "
                    f"{bet.model_probability:>7.3f} {bet.fair_probability:>7.3f} "
                    f"{bet.model_probability - bet.fair_probability:>+7.3f} "
                    f"{bet.stake:>9,.2f} {bet.returned:>10,.2f}  {landed}"
                )
            write(
                f"      staked {ledger.staked:,.2f} ({ledger.exposure:.2%} of bankroll)  "
                f"returned {ledger.returned:,.2f}  "
                f"profit {ledger.profit:+,.2f}  ROI {ledger.roi:+.2%}"
            )
            write(
                f"      bankroll EUR {bankroll:,.2f} -> EUR {ledger.closing:,.2f} "
                f"({ledger.growth:+.2%})"
            )
            write(
                f"      the book's own edge on these stakes: "
                f"EUR {-ledger.market_expected_profit:,.2f} expected to the house"
            )
            outcome = outcomes.get(ledger.rule)
            if outcome is None:
                continue
            write(
                f"      luck check ({outcome.source} probabilities): "
                f"mean EUR {outcome.mean:+,.2f}, sd EUR {outcome.deviation:,.2f}, "
                f"range {outcome.worst:+,.2f} to {outcome.best:+,.2f}"
            )
            if outcome.percentile is not None:
                write(
                    f"      the day landed at the {outcome.percentile:.0%} percentile "
                    f"of days this rule could have had; "
                    f"P(any profit) was {outcome.profitable:.0%}"
                )

        write("")
        write(f"{'rule':<20} {'staked':>10} {'profit':>11} {'closing':>13} {'ROI':>9} {'pct':>6}")
        for ledger in ledgers:
            outcome = outcomes.get(ledger.rule)
            percentile = (
                f"{outcome.percentile:.0%}"
                if outcome is not None and outcome.percentile is not None
                else "-"
            )
            write(
                f"{ledger.rule:<20} {ledger.staked:>10,.2f} {ledger.profit:>+11,.2f} "
                f"{ledger.closing:>13,.2f} {ledger.roi:>+9.2%} {percentile:>6}"
            )
        write("")
        write(
            "One card is one draw. Nothing above distinguishes a rule that works "
            "from one that does not; the sample sizes that can are in "
            "find_market_bias and measure_clv."
        )
        write("")

    def _as_dict(self, document, fixtures, ledgers, outcomes, bankroll) -> dict:
        return {
            "card": {
                "date": document.get("date"),
                "round": document.get("round"),
                "model": document.get("model"),
                "notes": document.get("notes", []),
            },
            "bankroll": bankroll,
            "fixtures": [
                {
                    "match": fixture.name,
                    "kickoff": fixture.kickoff,
                    "book": fixture.book,
                    "prices": fixture.prices,
                    "margin": fixture.margin,
                    "fair": fixture.fair,
                    "model": fixture.model,
                    "result": fixture.result,
                }
                for fixture in fixtures
            ],
            "rules": [
                {
                    "rule": ledger.rule,
                    "description": matchday.DESCRIPTIONS[ledger.rule],
                    "bets": [
                        {
                            "selection": bet.label,
                            "price": bet.price,
                            "model_probability": bet.model_probability,
                            "fair_probability": bet.fair_probability,
                            "edge": bet.model_probability - bet.fair_probability,
                            "expected_value": bet.expected_value,
                            "market_expected_value": bet.market_expected_value,
                            "stake": bet.stake,
                            "returned": bet.returned,
                            "profit": bet.profit,
                            "settled": bet.settled,
                            "won": bet.won,
                        }
                        for bet in ledger.bets
                    ],
                    "staked": ledger.staked,
                    "returned": ledger.returned,
                    "profit": ledger.profit,
                    "closing": ledger.closing,
                    "roi": ledger.roi,
                    "growth": ledger.growth,
                    "model_expected_profit": ledger.model_expected_profit,
                    "market_expected_profit": ledger.market_expected_profit,
                    "luck": (
                        None
                        if outcomes.get(ledger.rule) is None
                        else {
                            "source": outcomes[ledger.rule].source,
                            "mean": outcomes[ledger.rule].mean,
                            "deviation": outcomes[ledger.rule].deviation,
                            "realised": outcomes[ledger.rule].realised,
                            "percentile": outcomes[ledger.rule].percentile,
                            "probability_of_profit": outcomes[ledger.rule].profitable,
                            "best": outcomes[ledger.rule].best,
                            "worst": outcomes[ledger.rule].worst,
                            "z": outcomes[ledger.rule].z,
                        }
                    ),
                }
                for ledger in ledgers
            ],
        }
