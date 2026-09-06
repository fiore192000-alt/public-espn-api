"""Tests for closing-line value.

The instrument is tested in both directions. A model handed the closing line has
to come back with a slope near one, and a model that is pure noise has to come
back with nothing — otherwise a slope of zero on the real data would mean only
that the measurement cannot see anything at all.
"""

import math
import random
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.espn import clv
from apps.espn.backtest import ForecastRecord
from apps.espn.clv import Observation

HOME, DRAW, AWAY = "home", "draw", "away"
SIDES = (HOME, DRAW, AWAY)


def observation(
    match: str,
    selection: str = HOME,
    *,
    model: float = 0.50,
    opening: float = 0.45,
    closing: float = 0.45,
    won: bool = False,
) -> Observation:
    return Observation(
        match=match,
        selection=selection,
        model=model,
        opening=opening,
        closing=closing,
        opening_price=1.0 / opening,
        closing_price=1.0 / closing,
        won=won,
    )


def market(seed: int, count: int, *, ratified: float, noise: float = 0.0) -> list[Observation]:
    """A synthetic market where the close travels ``ratified`` of the model's gap.

    ``ratified`` is the answer the fit should recover, so a test can state the
    truth it is looking for rather than eyeballing a number.
    """
    rng = random.Random(seed)
    built = []
    for index in range(count):
        opening = rng.uniform(0.20, 0.60)
        gap = rng.uniform(-0.08, 0.08)
        model_probability = opening + gap
        closing = opening + ratified * gap + rng.gauss(0.0, noise)
        built.append(
            observation(
                match=f"m{index}",
                model=model_probability,
                opening=opening,
                closing=max(min(closing, 0.98), 0.02),
                won=rng.random() < closing,
            )
        )
    return built


class TestObservation:
    def test_disagreement_and_movement_are_signed_the_same_way(self):
        entry = observation("m", model=0.55, opening=0.50, closing=0.52)

        assert entry.disagreement == pytest.approx(0.05)
        assert entry.movement == pytest.approx(0.02)

    def test_closing_line_value_is_positive_when_the_price_shortened(self):
        """Taking 2.00 on something that closed at 1.80 beat the close by 11%."""
        entry = Observation(
            match="m",
            selection=HOME,
            model=0.5,
            opening=0.5,
            closing=0.5,
            opening_price=2.00,
            closing_price=1.80,
            won=True,
        )

        assert entry.closing_line_value == pytest.approx(2.00 / 1.80 - 1.0)

    def test_closing_line_value_is_negative_when_the_price_drifted(self):
        entry = Observation(
            match="m",
            selection=HOME,
            model=0.5,
            opening=0.5,
            closing=0.5,
            opening_price=1.80,
            closing_price=2.00,
            won=False,
        )

        assert entry.closing_line_value < 0


class TestAnticipation:
    def test_a_model_the_market_fully_ratifies_scores_a_slope_of_one(self):
        fit = clv.fit_anticipation(market(1, 400, ratified=1.0))

        assert fit.slope == pytest.approx(1.0, abs=0.01)
        assert fit.anticipates_the_market

    def test_a_model_the_market_half_ratifies_scores_a_half(self):
        fit = clv.fit_anticipation(market(2, 400, ratified=0.5, noise=0.005))

        assert fit.slope == pytest.approx(0.5, abs=0.05)
        assert fit.anticipates_the_market

    def test_disagreements_the_market_never_ratifies_score_zero(self):
        fit = clv.fit_anticipation(market(3, 400, ratified=0.0, noise=0.02))

        assert fit.slope == pytest.approx(0.0, abs=0.1)
        assert not fit.anticipates_the_market

    def test_a_model_the_market_moves_against_scores_negative(self):
        fit = clv.fit_anticipation(market(4, 400, ratified=-0.5, noise=0.005))

        assert fit.slope < 0
        assert not fit.anticipates_the_market

    def test_a_model_that_agrees_with_the_open_leaves_nothing_to_fit(self):
        flat = [observation(f"m{i}", model=0.45, opening=0.45, closing=0.50) for i in range(300)]

        fit = clv.fit_anticipation(flat)

        assert fit.slope == 0.0
        assert fit.matches == 0

    def test_a_small_sample_cannot_establish_anticipation(self):
        fit = clv.fit_anticipation(market(5, 20, ratified=1.0))

        assert fit.slope == pytest.approx(1.0, abs=0.01)
        assert not fit.anticipates_the_market

    def test_errors_are_clustered_by_match_not_by_leg(self):
        """Three legs of one fixture are one piece of evidence, not three.

        The same data relabelled as one match per leg must look *more* certain;
        if it does not, the clustering is not doing anything.
        """
        legs = []
        for index in range(300):
            for offset, side in enumerate(SIDES):
                entry = market(6 + offset, 300, ratified=0.5, noise=0.01)[index]
                legs.append(
                    observation(
                        match=f"m{index}",
                        selection=side,
                        model=entry.model,
                        opening=entry.opening,
                        closing=entry.closing,
                    )
                )
        independent = [
            observation(
                match=f"{leg.match}-{leg.selection}",
                selection=leg.selection,
                model=leg.model,
                opening=leg.opening,
                closing=leg.closing,
            )
            for leg in legs
        ]

        clustered = clv.fit_anticipation(legs)
        unclustered = clv.fit_anticipation(independent)

        assert clustered.slope == pytest.approx(unclustered.slope)
        assert clustered.matches == 300
        assert unclustered.matches == 900
        assert clustered.stderr > unclustered.stderr


class TestBeatingTheClose:
    def test_picks_that_shortened_show_positive_value(self):
        picks = [
            Observation(
                match=f"m{i}",
                selection=HOME,
                model=0.60,
                opening=0.50,
                closing=0.55,
                opening_price=2.00,
                closing_price=1.80,
                won=i % 2 == 0,
            )
            for i in range(300)
        ]

        result = clv.beat_the_close(picks)

        assert result.matches == 300
        assert result.mean == pytest.approx(2.00 / 1.80 - 1.0)
        assert result.beats_the_close

    def test_only_selections_the_model_rates_above_the_open_are_taken(self):
        liked = observation("m1", model=0.60, opening=0.50)
        disliked = observation("m2", model=0.40, opening=0.50)

        result = clv.beat_the_close([liked, disliked])

        assert result.picks == 1

    def test_the_edge_threshold_narrows_the_selection(self):
        marginal = observation("m1", model=0.51, opening=0.50)
        strong = observation("m2", model=0.62, opening=0.50)

        assert clv.beat_the_close([marginal, strong], edge=0.05).picks == 1

    def test_several_picks_in_one_fixture_count_once(self):
        """Backing two outcomes of the same match is one dependent event."""
        picks = [
            observation("same", selection=side, model=0.60, opening=0.50) for side in (HOME, DRAW)
        ]

        result = clv.beat_the_close(picks)

        assert (result.picks, result.matches) == (2, 1)

    def test_a_model_that_likes_nothing_returns_an_empty_result(self):
        result = clv.beat_the_close([observation("m1", model=0.30, opening=0.50)])

        assert result.matches == 0
        assert not result.beats_the_close


class TestLogLoss:
    def test_scores_only_the_outcome_that_happened(self):
        entries = [
            observation("m1", model=0.60, opening=0.50, closing=0.55, won=True),
            observation("m1", selection=DRAW, model=0.20, opening=0.25, closing=0.25, won=False),
        ]

        assert clv.log_loss(entries, "model") == pytest.approx(-math.log(0.60))
        assert clv.log_loss(entries, "closing") == pytest.approx(-math.log(0.55))

    def test_no_winner_scores_nothing_rather_than_dividing_by_zero(self):
        assert clv.log_loss([observation("m1")], "model") == 0.0


class TestObservationsFromRecords:
    def record(self, **overrides) -> ForecastRecord:
        base = {
            "event_espn_id": "e1",
            "date": None,
            "home": "H",
            "away": "A",
            "probabilities": {HOME: 0.5, DRAW: 0.3, AWAY: 0.2},
            "actual": HOME,
            "home_goals": 2,
            "away_goals": 1,
            "market_probabilities": {HOME: 0.45, DRAW: 0.30, AWAY: 0.25},
            "closing_probabilities": {HOME: 0.47, DRAW: 0.29, AWAY: 0.24},
            "opening_prices": {HOME: 2.10, DRAW: 3.20, AWAY: 3.80},
            "closing_prices": {HOME: 2.02, DRAW: 3.25, AWAY: 3.95},
        }
        return ForecastRecord(**{**base, **overrides})

    def test_builds_one_observation_per_selection(self):
        built = clv.observations_from([self.record()], "probabilities")

        assert len(built) == 3
        assert {o.selection for o in built} == set(SIDES)
        assert [o.won for o in built if o.selection == HOME] == [True]

    def test_a_record_with_no_closing_line_is_dropped(self):
        built = clv.observations_from([self.record(closing_probabilities=None)], "probabilities")

        assert built == []

    def test_a_record_with_no_closing_prices_is_dropped(self):
        built = clv.observations_from([self.record(closing_prices=None)], "probabilities")

        assert built == []

    def test_a_model_that_did_not_forecast_is_dropped(self):
        built = clv.observations_from([self.record()], "elo_probabilities")

        assert built == []

    def test_both_models_go_through_identical_arithmetic(self):
        record = self.record(elo_probabilities={HOME: 0.5, DRAW: 0.3, AWAY: 0.2})

        dixon = clv.observations_from([record], "probabilities")
        elo = clv.observations_from([record], "elo_probabilities")

        assert [o.disagreement for o in dixon] == [o.disagreement for o in elo]


class TestAssess:
    def test_reports_all_three_measurements_together(self):
        report = clv.assess("test", market(7, 400, ratified=0.8, noise=0.005))
        payload = report.to_dict()

        assert payload["model"] == "test"
        assert payload["anticipation"]["slope"] == pytest.approx(0.8, abs=0.05)
        assert set(payload["log_loss"]) == {"model", "opening", "closing"}
        assert "beats_the_close" in payload["beating_the_close"]


@pytest.mark.django_db
class TestMeasureClvCommand:
    def test_a_league_with_no_closing_line_says_where_to_get_one(self, league):
        with pytest.raises(CommandError, match="ingest_football_data"):
            call_command("measure_clv", league.slug, stdout=StringIO())

    def test_an_unknown_league_is_named_in_the_error(self, db):
        with pytest.raises(CommandError, match="No league with slug 'nowhere'"):
            call_command("measure_clv", "nowhere", stdout=StringIO())
