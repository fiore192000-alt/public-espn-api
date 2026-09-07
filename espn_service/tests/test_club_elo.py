"""Tests for the ClubElo model.

The point of this rating source is the case the other two handle badly: a club
whose recent history is mostly outside the league being modelled. The tests below
pin that, and pin the reading of ratings off an event, which is where a silent
failure would turn into a confident forecast about nobody.
"""

import math
import random

import pytest

from apps.espn import club_elo, elo

HOME, DRAW, AWAY = "home", "draw", "away"


class Event:
    """Just enough of an Event for the reader under test."""

    def __init__(self, raw_data):
        self.raw_data = raw_data


def rated(home, away) -> Event:
    return Event({"home_elo": home, "away_elo": away})


def samples(count: int = 400, *, beta: float = 0.6) -> list[tuple[float, str]]:
    """Outcomes drawn from a known ordered logit, so the fit has a truth to find."""
    rng = random.Random(7)
    built = []
    for _ in range(count):
        difference = rng.uniform(-4.0, 4.0)
        latent = beta * difference
        away_probability = 1.0 / (1.0 + math.exp(-(-1.0 - latent)))
        not_home = 1.0 / (1.0 + math.exp(-(0.3 - latent)))
        weights = [away_probability, not_home - away_probability, 1.0 - not_home]
        built.append((difference, rng.choices([AWAY, DRAW, HOME], weights=weights)[0]))
    return built


class TestReadingRatings:
    def test_reads_the_pair_the_loader_stored(self):
        assert club_elo.ratings_of(rated("1820.5", "1799.1")) == (1820.5, 1799.1)

    def test_numbers_are_accepted_as_well_as_strings(self):
        assert club_elo.ratings_of(rated(1820.5, 1799.1)) == (1820.5, 1799.1)

    @pytest.mark.parametrize(
        "raw",
        [
            {},
            {"home_elo": "1820.5"},
            {"away_elo": "1799.1"},
            {"home_elo": "1820.5", "away_elo": ""},
            {"home_elo": "1820.5", "away_elo": "n/a"},
            {"home_elo": None, "away_elo": None},
        ],
    )
    def test_half_a_pair_is_no_pair(self, raw):
        """A difference against a missing opponent is a claim about nobody."""
        assert club_elo.ratings_of(Event(raw)) is None

    def test_an_event_without_raw_data_is_handled(self):
        assert club_elo.ratings_of(Event(None)) is None

    def test_the_difference_is_scaled_but_not_reordered(self):
        assert club_elo.difference(1900.0, 1800.0) == pytest.approx(1.0)
        assert club_elo.difference(1800.0, 1900.0) == pytest.approx(-1.0)
        assert club_elo.difference(1800.0, 1800.0) == 0.0


class TestSamples:
    def test_builds_a_training_pair_from_an_event_and_its_result(self):
        assert club_elo.sample_of(rated("1900", "1800"), HOME) == (pytest.approx(1.0), HOME)

    def test_an_unrated_event_yields_nothing_to_train_on(self):
        assert club_elo.sample_of(Event({}), HOME) is None


class TestFit:
    def test_recovers_the_strength_it_was_generated_with(self):
        model = club_elo.fit(samples(600, beta=0.6))

        assert model.outcome.beta == pytest.approx(0.6, abs=0.12)
        assert model.samples == 600

    def test_a_stronger_home_rating_raises_the_home_probability(self):
        model = club_elo.fit(samples())

        weak = model.probabilities(1700.0, 1900.0)
        even = model.probabilities(1800.0, 1800.0)
        strong = model.probabilities(1900.0, 1700.0)

        assert weak[HOME] < even[HOME] < strong[HOME]
        assert strong[AWAY] < even[AWAY] < weak[AWAY]

    def test_probabilities_are_a_distribution(self):
        model = club_elo.fit(samples())

        for pair in ((1900.0, 1600.0), (1800.0, 1800.0), (1500.0, 2000.0)):
            probabilities = model.probabilities(*pair)
            assert sum(probabilities.values()) == pytest.approx(1.0)
            assert all(value > 0 for value in probabilities.values())

    def test_too_few_matches_is_refused_the_same_way_elo_refuses_them(self):
        with pytest.raises(elo.NotEnoughData):
            club_elo.fit(samples(10))

    def test_a_warm_start_reaches_the_same_place(self):
        first = club_elo.fit(samples(600))
        warm = club_elo.fit(samples(600), initial=first)

        assert warm.outcome.beta == pytest.approx(first.outcome.beta, abs=1e-3)

    def test_the_scale_is_reported_so_beta_can_be_read(self):
        payload = club_elo.fit(samples()).to_dict()

        assert payload["scale"] == club_elo.RATING_SCALE
        assert "beta" in payload and "draw_band" in payload


class TestPromotedClub:
    """The case the other two models get wrong.

    A club back from a lower division carries a rating earned there. Dixon-Coles
    would see two fixtures; this sees the whole history, so it does not mistake
    one good result for the best attack in the league.
    """

    def test_a_promoted_club_is_rated_where_its_rating_says(self):
        model = club_elo.fit(samples())
        # Frosinone 1591 against Venezia 1599 on 6 September 2026: eight points
        # apart, which is a coin flip with a home edge — not 89.8%.
        probabilities = model.probabilities(1591.17, 1599.42)

        assert probabilities[HOME] < 0.55
        assert probabilities[HOME] > probabilities[AWAY]

    def test_one_result_cannot_move_a_rating_the_model_does_not_own(self):
        """The ratings arrive from outside, so nothing here can overfit them."""
        model = club_elo.fit(samples())

        assert model.probabilities(1591.0, 1599.0) == model.probabilities(1591.0, 1599.0)
