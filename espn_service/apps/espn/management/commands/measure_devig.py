"""Measure which devig method best recovers what the market believed.

Also reports the market's microstructure: how much margin a single book charges,
and how much of it disappears when you take the best price available anywhere.
"""

import json
import statistics

from django.core.management.base import BaseCommand, CommandError

from apps.espn import devig, market_structure
from apps.espn.analysis import _sided_competitors
from apps.espn.backtest import outcome_of
from apps.espn.markets import MARKET_MATCH_ODDS, SELECTION_AWAY, SELECTION_DRAW, SELECTION_HOME
from apps.espn.models import Event, League

OUTCOMES = {SELECTION_HOME, SELECTION_DRAW, SELECTION_AWAY}
SINGLE_BOOK = "fd-b365"
BEST_PRICE = "fd-max"


class Command(BaseCommand):
    help = "Score devig methods against real results and report the market's margin structure."

    def add_arguments(self, parser) -> None:
        parser.add_argument("league", help="League slug (e.g. 'ita.1').")
        parser.add_argument(
            "--provider",
            default=SINGLE_BOOK,
            help=f"Provider whose book is devigged (default: {SINGLE_BOOK}).",
        )
        parser.add_argument("--json", action="store_true", help="Emit raw JSON.")

    def handle(self, *args, **options) -> None:
        try:
            league = League.objects.get(slug__iexact=options["league"])
        except League.DoesNotExist as exc:
            raise CommandError(f"No league with slug {options['league']!r}.") from exc

        books, single_overrounds, best_overrounds, best_gains = [], [], [], []
        priced_pairs: list[tuple[dict[str, float], dict[str, float], str]] = []

        for event in (
            Event.objects.filter(league=league, status=Event.STATUS_FINAL)
            .prefetch_related("competitors__team", "odds")
            .order_by("date")
        ):
            sides = _sided_competitors(event)
            if sides is None:
                continue
            home, away = sides
            if home.score_int is None or away.score_int is None:
                continue

            quotes: dict[str, dict[str, float]] = {}
            for odds in event.odds.all():
                if odds.market == MARKET_MATCH_ODDS and not odds.line:
                    quotes.setdefault(odds.provider_espn_id, {})[odds.selection] = odds.decimal_odds

            book = quotes.get(options["provider"])
            if not book or set(book) != OUTCOMES:
                continue

            books.append((book, outcome_of(home.score_int, away.score_int)))
            single_overrounds.append(devig.overround(book))

            best = quotes.get(BEST_PRICE)
            if best and set(best) == OUTCOMES:
                best_overrounds.append(devig.overround(best))
                best_gains.append(max(best[side] / book[side] for side in OUTCOMES))
                priced_pairs.append((book, best, outcome_of(home.score_int, away.score_int)))

        if not books:
            raise CommandError(
                f"No complete 1X2 books stored for provider {options['provider']!r} in "
                f"{league.slug}. Load odds first — see ingest_football_data."
            )

        scores = devig.measure_devig_methods(books)
        payload = {
            "league": league.slug,
            "provider": options["provider"],
            "books": len(books),
            "methods": [score.to_dict() for score in scores],
            "microstructure": _microstructure(single_overrounds, best_overrounds, best_gains),
            "price_selection": market_structure.summarise(
                market_structure.compare_prices(priced_pairs, tuple(sorted(OUTCOMES)))
            )
            if priced_pairs
            else None,
        }

        if options["json"]:
            self.stdout.write(json.dumps(payload, indent=2))
            return

        self._render(payload)

    _implied_from_overround: float | None = None

    def _render(self, payload: dict) -> None:
        self.stdout.write("")
        self.stdout.write(
            self.style.MIGRATE_HEADING(
                f"Devig comparison — {payload['league']} ({payload['provider']})"
            )
        )
        self.stdout.write(f"  {payload['books']} complete 1X2 books scored against real results.")

        self.stdout.write("")
        self.stdout.write(f"  {'method':16}{'log loss':>12}{'brier':>10}")
        for method in payload["methods"]:
            self.stdout.write(
                f"  {method['method']:16}{method['log_loss']:>12}{method['brier']:>10}"
            )
        self.stdout.write(
            "  How the margin is spread across a book is not cosmetic: it moves a longshot's\n"
            "  implied probability far more than a favourite's, which is exactly where a model\n"
            "  tends to disagree with the price."
        )

        structure = payload["microstructure"]
        self.stdout.write("")
        self.stdout.write("  Market structure")
        self.stdout.write(f"    {'book overround':28}{structure['book_overround']:>10}")
        self.stdout.write(f"    {'book margin':28}{structure['book_margin_pct']:>9}%")
        self._implied_from_overround = (
            1.0 / structure["best_price_overround"] - 1.0
            if structure["best_price_overround"]
            else None
        )
        if structure["best_price_overround"] is not None:
            self.stdout.write(
                f"    {'best-price overround':28}{structure['best_price_overround']:>10}"
            )
            self.stdout.write(
                f"    {'best vs book, best leg':28}{structure['best_price_gain_pct']:>9}%"
            )
        self._render_price_selection(payload.get("price_selection"))

    def _render_price_selection(self, selection: dict | None) -> None:
        """What flat-staking at each price level actually returned."""
        if not selection:
            return

        self.stdout.write("")
        self.stdout.write("  Flat-staking every selection, settled on real results")
        self.stdout.write(f"    {'':10}{'at book':>12}{'at best':>12}{'recovered':>12}")
        for row in selection["selections"]:
            self.stdout.write(
                f"    {row['selection']:10}{row['at_book']['yield']:>12}"
                f"{row['at_best']['yield']:>12}{row['recovered']:>12}"
            )

        pooled = selection["pooled"]
        self.stdout.write(
            f"    {'pooled':10}{pooled['at_book']['yield']:>12}"
            f"{pooled['at_best']['yield']:>12}{pooled['recovered']:>12}"
        )
        self.stdout.write(
            f"    (pooled over {pooled['matches']} matches, not legs: the three outcomes of "
            f"one match are a single dependent event. Yield standard error "
            f"{pooled['at_best']['stderr']} at the best price.)"
        )

        implied = self._implied_from_overround
        if implied is not None:
            realised = pooled["at_best"]["yield"]
            gap = implied - realised
            self.stdout.write("")
            self.stdout.write(
                f"    Backing all three at the best price: overround implies {implied:+.4f}, "
                f"actually returned {realised:+.4f}."
            )
            if gap > 0.005:
                self.stdout.write(
                    self.style.ERROR(
                        f"    The overround overstates price selection by {gap:.4f} — "
                        f"{gap * 100:.1f} percentage points. Three maxima taken across different "
                        "books are not one coherent book: the best price tends to be highest on "
                        "the outcomes that go on to lose, so the near-1.0 overround is not a "
                        "return you can collect. Trust the realised figure, not the summary."
                    )
                )

        self.stdout.write("")
        if pooled["profitable_at_best"]:
            self.stdout.write(
                self.style.SUCCESS(
                    "    Taking the best price is profitable on its own here, by more than two "
                    "standard errors. That is unusual enough to double-check the data before "
                    "believing it."
                )
            )
        else:
            self.stdout.write(
                self.style.WARNING(
                    "    Taking the best price recovers most of the bookmaker's margin, but it "
                    "does NOT create an edge: flat-staking at the best price is still not "
                    "profitable. Price selection is what makes a genuine edge survivable — it is "
                    "not a substitute for having one."
                )
            )


def _microstructure(
    single: list[float],
    best: list[float],
    gains: list[float],
) -> dict:
    book_overround = statistics.mean(single)
    return {
        "matches": len(single),
        "book_overround": round(book_overround, 4),
        "book_margin_pct": round(100.0 * (book_overround - 1.0) / book_overround, 2),
        "best_price_overround": round(statistics.mean(best), 4) if best else None,
        "best_price_gain_pct": round(100.0 * (statistics.mean(gains) - 1.0), 2) if gains else None,
    }
