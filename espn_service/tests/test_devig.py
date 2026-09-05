"""Tests for margin removal and the devig measurement command."""

import json
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.espn import devig, value
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
        assert payload["microstructure"]["best_price_overround"] < payload["microstructure"][
            "book_overround"
        ]

    def test_unknown_league_raises(self, db):
        with pytest.raises(CommandError):
            call_command("measure_devig", "nope", stdout=StringIO())

    def test_unknown_provider_raises(self, loaded):
        with pytest.raises(CommandError):
            call_command("measure_devig", "ita.1", provider="not-a-book", stdout=StringIO())
