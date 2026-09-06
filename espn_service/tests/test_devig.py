"""Tests for margin removal and the devig measurement command."""

import json
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.espn import devig, market_structure, value
from apps.espn.models import League
from tests.test_football_data import HEADER
from tests.test_football_data import row as fd_row

# A book with a real margin: implied probabilities sum to about 1.056.
MARGINED = {"home": 1.80, "draw": 3.60, "away": 4.50}
# A book with no margin at all.
FAIR = {"home": 2.0, "draw": 4.0, "away": 4.0}


class TestImpliedAndOverround:
    def test_overround_measures_the_margin(self):
        assert devig.overround(FAIR) == pytest.approx(1.0)
        assert devig.overround(MARGINED) == pytest.approx(1.0556, abs=1e-4)

    def test_margin_is_the_share_of_the_book(self):
        assert devig.margin(FAIR) == pytest.approx(0.0, abs=1e-9)
        assert devig.margin(MARGINED) == pytest.approx(0.0526, abs=1e-4)

    def test_a_single_price_cannot_be_devigged(self):
        with pytest.raises(devig.DevigError):
            devig.implied({"home": 1.8})

    def test_nonsense_prices_are_ignored(self):
        with pytest.raises(devig.DevigError):
            devig.implied({"home": 0.5, "draw": 1.0, "away": -3.0})


class TestDevigMethods:
    @pytest.mark.parametrize("method", devig.METHODS)
    def test_every_method_returns_a_distribution(self, method):
        fair = devig.remove_margin(MARGINED, method)
        assert sum(fair.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(0.0 < probability < 1.0 for probability in fair.values())

    @pytest.mark.parametrize("method", devig.METHODS)
    def test_a_margin_free_book_is_left_alone(self, method):
        fair = devig.remove_margin(FAIR, method)
        assert fair["home"] == pytest.approx(0.5, abs=1e-6)
        assert fair["draw"] == pytest.approx(0.25, abs=1e-6)

    @pytest.mark.parametrize("method", devig.METHODS)
    def test_ordering_is_preserved(self, method):
        fair = devig.remove_margin(MARGINED, method)
        assert fair["home"] > fair["draw"] > fair["away"]

    def test_proportional_scales_everything_equally(self):
        fair = devig.proportional(MARGINED)
        ratios = [fair[side] * MARGINED[side] for side in MARGINED]
        assert ratios[0] == pytest.approx(ratios[1]) == pytest.approx(ratios[2])

    def test_shin_and_power_take_more_from_the_longshot(self):
        # The whole reason the choice matters: books load margin onto longshots,
        # so a method that removes it evenly overstates their true chance.
        flat = devig.proportional(MARGINED)
        for method in (devig.SHIN, devig.POWER):
            adjusted = devig.remove_margin(MARGINED, method)
            assert adjusted["away"] < flat["away"]
            assert adjusted["home"] > flat["home"]

    def test_shin_falls_back_when_the_book_has_no_margin(self):
        assert devig.shin(FAIR)["home"] == pytest.approx(0.5, abs=1e-6)

    def test_unknown_method_is_rejected(self):
        with pytest.raises(devig.DevigError):
            devig.remove_margin(MARGINED, "wishful-thinking")

    def test_heavier_margins_are_still_normalised(self):
        heavy = {"home": 1.5, "draw": 3.0, "away": 3.5}
        assert devig.overround(heavy) > 1.15
        for method in devig.METHODS:
            assert sum(devig.remove_margin(heavy, method).values()) == pytest.approx(1.0, abs=1e-6)


class TestMeasurement:
    def test_ranks_methods_by_log_loss(self):
        books = [(MARGINED, "home")] * 50 + [(MARGINED, "away")] * 10
        scores = devig.measure_devig_methods(books)

        assert [score.method for score in scores] == sorted(
            [score.method for score in scores],
            key=lambda method: next(s.log_loss for s in scores if s.method == method),
        )
        assert all(score.matches == 60 for score in scores)

    def test_skips_books_it_cannot_devig(self):
        books = [(MARGINED, "home"), ({"home": 1.8}, "home")]
        scores = devig.measure_devig_methods(books)
        assert all(score.matches == 1 for score in scores)

    def test_skips_outcomes_missing_from_the_book(self):
        assert devig.measure_devig_methods([(MARGINED, "over")]) == []

    def test_no_books_produces_no_scores(self):
        assert devig.measure_devig_methods([]) == []

    def test_scores_serialise(self):
        payload = devig.measure_devig_methods([(MARGINED, "home")])[0].to_dict()
        assert set(payload) == {"method", "matches", "log_loss", "brier"}


class TestValueUsesTheChosenMethod:
    def test_default_is_shin(self):
        assert value.DEFAULT_DEVIG_METHOD == devig.SHIN

    def test_remove_margin_delegates(self):
        assert value.remove_margin(MARGINED) == devig.remove_margin(MARGINED, devig.SHIN)

    def test_unusable_market_returns_empty_rather_than_raising(self):
        assert value.remove_margin({"home": 1.8}) == {}

    def test_method_is_configurable_through_find_value_bets(self):
        model_markets = {
            "1x2": {
                "home": {"probability": 0.60},
                "draw": {"probability": 0.25},
                "away": {"probability": 0.15},
            }
        }
        prices = [
            value.PricedSelection("1x2", "home", "", 2.5, "1"),
            value.PricedSelection("1x2", "draw", "", 3.4, "1"),
            value.PricedSelection("1x2", "away", "", 3.0, "1"),
        ]
        for method in devig.METHODS:
            bets = value.find_value_bets(model_markets, prices, devig_method=method)
            assert [bet.selection for bet in bets] == ["home"]


@pytest.mark.django_db
class TestMeasureDevigCommand:
    @pytest.fixture
    def loaded(self, tmp_path):
        teams = ["Inter", "Milan", "Roma", "Lazio"]
        rows, day = [], 1
        for cycle in range(20):
            for index in range(0, len(teams), 2):
                rows.append(
                    fd_row(
                        date=f"2024-{1 + day // 28:02d}-{1 + day % 28:02d}",
                        home=teams[index],
                        away=teams[index + 1],
                        home_goals=str(cycle % 3),
                        away_goals=str((cycle + 1) % 2),
                    )
                )
                day += 3
            teams = [teams[0], teams[-1], *teams[1:-1]]

        path = tmp_path / "matches.csv"
        path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
        call_command("ingest_football_data", str(path), division="I1", stdout=StringIO())
        return League.objects.get(slug="ita.1")

    def test_renders_methods_and_structure(self, loaded):
        out = StringIO()
        call_command("measure_devig", "ita.1", stdout=out)
        output = out.getvalue()

        assert "Devig comparison" in output
        assert "book overround" in output
        assert "best-price overround" in output
        for method in devig.METHODS:
            assert method in output

    def test_json_output(self, loaded):
        out = StringIO()
        call_command("measure_devig", "ita.1", json=True, stdout=out)
        payload = json.loads(out.getvalue())

        assert payload["provider"] == "fd-b365"
        assert payload["books"] > 0
        assert len(payload["methods"]) == len(devig.METHODS)
        assert payload["microstructure"]["book_overround"] > 1.0
        # The best price across books is close to margin-free.
        assert (
            payload["microstructure"]["best_price_overround"]
            < payload["microstructure"]["book_overround"]
        )

    def test_unknown_league_raises(self, db):
        with pytest.raises(CommandError):
            call_command("measure_devig", "nope", stdout=StringIO())

    def test_unknown_provider_raises(self, loaded):
        with pytest.raises(CommandError):
            call_command("measure_devig", "ita.1", provider="not-a-book", stdout=StringIO())


class TestPriceComparison:
    """Realised returns of flat-staking at two price levels."""

    BOOK = {"home": 1.90, "draw": 3.30, "away": 4.00}
    BEST = {"home": 2.05, "draw": 3.60, "away": 4.40}
    SIDES = ("away", "draw", "home")

    def books(self, outcomes: list[str]):
        return [(self.BOOK, self.BEST, outcome) for outcome in outcomes]

    def test_a_winning_selection_returns_its_odds(self):
        comparison = market_structure.compare_prices(self.books(["home"]), self.SIDES)
        home = next(c for c in comparison.selections if c.selection == "home")

        assert home.at_book.mean == pytest.approx(0.90)
        assert home.at_best.mean == pytest.approx(1.05)
        assert home.recovered == pytest.approx(0.15)

    def test_a_losing_selection_loses_its_stake(self):
        comparison = market_structure.compare_prices(self.books(["home"]), self.SIDES)
        away = next(c for c in comparison.selections if c.selection == "away")

        assert away.at_book.mean == pytest.approx(-1.0)
        assert away.at_best.mean == pytest.approx(-1.0)
        # A losing leg recovers nothing: the better price never paid out.
        assert away.recovered == pytest.approx(0.0)

    def test_pooling_counts_matches_not_legs(self):
        comparison = market_structure.compare_prices(self.books(["home", "draw"]), self.SIDES)

        assert comparison.pooled_at_book.bets == 2
        assert sum(c.at_book.bets for c in comparison.selections) == 6

    def test_pooled_return_is_the_winning_odds_over_the_stake(self):
        comparison = market_structure.compare_prices(self.books(["home"]), self.SIDES)

        # Three units staked, 1.90 returned at the book.
        assert comparison.pooled_at_book.mean == pytest.approx((1.90 - 3) / 3)
        assert comparison.pooled_at_best.mean == pytest.approx((2.05 - 3) / 3)

    def test_pooled_standard_error_is_smaller_than_per_leg(self):
        # Backing all three outcomes is nearly a hedge, so the pooled spread is
        # far tighter than any single leg's. Treating legs as independent bets
        # would misstate it.
        outcomes = ["home", "draw", "away"] * 20
        comparison = market_structure.compare_prices(self.books(outcomes), self.SIDES)
        per_leg = next(c for c in comparison.selections if c.selection == "home")

        assert comparison.pooled_at_book.stderr < per_leg.at_book.stderr

    def test_a_genuinely_profitable_price_is_recognised(self):
        generous = {"home": 5.0, "draw": 5.0, "away": 5.0}
        books = [(self.BOOK, generous, "home")] * 100
        comparison = market_structure.compare_prices(books, self.SIDES)

        assert comparison.profitable_at_best is True

    def test_an_efficient_book_is_not_called_profitable(self):
        # Outcomes drawn in proportion to the book's own devigged probabilities:
        # the realistic case, where the price is right and the margin is the only
        # thing left. Uniform outcomes would instead make any book with a skewed
        # line profitable, which says nothing about price selection.
        priced = {"home": 1.95, "draw": 3.50, "away": 4.10}
        fair = devig.proportional(priced)
        outcomes = [
            selection
            for selection, probability in fair.items()
            for _ in range(round(probability * 1000))
        ]

        comparison = market_structure.compare_prices(
            [(self.BOOK, priced, outcome) for outcome in outcomes], self.SIDES
        )

        assert comparison.pooled_at_best.mean < 0
        assert comparison.profitable_at_best is False

    def test_too_few_bets_is_never_called_profitable(self):
        generous = {"home": 5.0, "draw": 5.0, "away": 5.0}
        comparison = market_structure.compare_prices([(self.BOOK, generous, "home")], self.SIDES)

        assert comparison.pooled_at_best.mean > 0
        assert comparison.profitable_at_best is False

    def test_selections_missing_from_a_book_are_skipped(self):
        partial = {"home": 2.0}
        comparison = market_structure.compare_prices([(partial, partial, "home")], self.SIDES)

        assert [c.selection for c in comparison.selections] == ["home"]
        assert comparison.pooled_at_book.bets == 1

    def test_summarise_serialises_both_levels(self):
        payload = market_structure.summarise(
            market_structure.compare_prices(self.books(["home", "away"]), self.SIDES)
        )

        assert payload["pooled"]["matches"] == 2
        assert set(payload["pooled"]) >= {"at_book", "at_best", "recovered", "profitable_at_best"}
        assert len(payload["selections"]) == 3

    def test_no_books_produces_nothing(self):
        comparison = market_structure.compare_prices([], self.SIDES)
        assert comparison.selections == []
        assert comparison.pooled_at_book.bets == 0
