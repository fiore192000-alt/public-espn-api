"""Tests for the Elo model and the incremental-information test."""

import json
import math
import random
from datetime import UTC, datetime, timedelta
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.espn import combination, elo
from apps.espn.dixon_coles import MatchObservation
from apps.espn.models import League
from tests.test_football_data import row as fd_row

START = datetime(2024, 1, 1, tzinfo=UTC)


def match(home, away, home_goals, away_goals, day=0) -> MatchObservation:
    return MatchObservation(home, away, home_goals, away_goals, START + timedelta(days=day))


class TestEloRatings:
    def test_starts_everyone_equal(self):
        ratings = elo.EloRatings()
        assert ratings.rating(1) == elo.DEFAULT_INITIAL_RATING
        assert ratings.rating(99) == elo.DEFAULT_INITIAL_RATING

    def test_home_advantage_is_in_the_difference(self):
        ratings = elo.EloRatings()
        assert ratings.difference(1, 2) == pytest.approx(elo.DEFAULT_HOME_ADVANTAGE)
        assert ratings.expected_score(1, 2) > 0.5

    def test_a_win_moves_both_ratings_in_opposite_directions(self):
        ratings = elo.EloRatings()
        ratings.update(match(1, 2, 2, 0))

        assert ratings.rating(1) > elo.DEFAULT_INITIAL_RATING
        assert ratings.rating(2) < elo.DEFAULT_INITIAL_RATING
        assert ratings.rating(1) - elo.DEFAULT_INITIAL_RATING == pytest.approx(
            elo.DEFAULT_INITIAL_RATING - ratings.rating(2)
        )

    def test_bigger_margins_move_ratings_further(self):
        narrow, wide = elo.EloRatings(), elo.EloRatings()
        narrow.update(match(1, 2, 1, 0))
        wide.update(match(1, 2, 5, 0))

        assert wide.rating(1) > narrow.rating(1)

    def test_margin_scaling_can_be_disabled(self):
        flat = elo.EloRatings(config=elo.EloConfig(margin_of_victory=False))
        flat.update(match(1, 2, 5, 0))
        one_goal = elo.EloRatings(config=elo.EloConfig(margin_of_victory=False))
        one_goal.update(match(1, 2, 1, 0))

        assert flat.rating(1) == pytest.approx(one_goal.rating(1))

    def test_an_expected_home_win_moves_ratings_less_than_an_upset(self):
        ratings = elo.EloRatings()
        ratings.ratings[1] = 1800.0
        ratings.ratings[2] = 1400.0

        favourite_wins = elo.EloRatings()
        favourite_wins.ratings.update({1: 1800.0, 2: 1400.0})
        favourite_wins.update(match(1, 2, 1, 0))

        upset = elo.EloRatings()
        upset.ratings.update({1: 1800.0, 2: 1400.0})
        upset.update(match(1, 2, 0, 1))

        assert abs(upset.rating(1) - 1800.0) > abs(favourite_wins.rating(1) - 1800.0)

    def test_records_a_sample_per_match(self):
        ratings = elo.EloRatings()
        ratings.update(match(1, 2, 2, 0))
        ratings.update(match(2, 1, 1, 1))
        ratings.update(match(1, 2, 0, 3))

        assert [outcome for _, outcome in ratings.samples] == ["home", "draw", "away"]
        assert ratings.matches == 3

    def test_build_ratings_applies_matches_in_date_order(self):
        shuffled = [match(1, 2, 3, 0, day=5), match(1, 2, 0, 3, day=1)]
        ordered = [match(1, 2, 0, 3, day=1), match(1, 2, 3, 0, day=5)]

        assert elo.build_ratings(shuffled).ratings == pytest.approx(
            elo.build_ratings(ordered).ratings
        )


def synthetic_samples(count: int = 400) -> list[tuple[float, str]]:
    """Rating differences with outcomes that follow the gap but are not separable.

    Every third match is deliberately given the "wrong" result. Perfectly separable
    data has no maximum likelihood estimate — beta simply runs to infinity — so a
    test built on it would be measuring the optimiser's stopping rule rather than
    the fit.
    """
    samples = []
    for index in range(count):
        difference = -1.0 + 2.0 * (index / count)
        if difference < -0.3:
            outcome = "away"
        elif difference > 0.3:
            outcome = "home"
        else:
            outcome = "draw"
        if index % 3 == 0:
            outcome = {"home": "draw", "draw": "away", "away": "home"}[outcome]
        samples.append((difference, outcome))
    return samples


class TestOutcomeModel:
    def test_needs_a_minimum_sample(self):
        with pytest.raises(elo.NotEnoughData):
            elo.fit_outcome_model(synthetic_samples(10))

    def test_probabilities_sum_to_one(self):
        model = elo.fit_outcome_model(synthetic_samples())
        for difference in (-2.0, -0.5, 0.0, 0.5, 2.0):
            probabilities = model.probabilities(difference)
            assert sum(probabilities.values()) == pytest.approx(1.0)
            assert all(value >= 0 for value in probabilities.values())

    def test_stronger_home_side_gets_a_higher_home_probability(self):
        model = elo.fit_outcome_model(synthetic_samples())
        weak = model.probabilities(-1.0)
        strong = model.probabilities(1.0)

        assert strong["home"] > weak["home"]
        assert strong["away"] < weak["away"]

    def test_the_draw_peaks_on_even_fixtures(self):
        model = elo.fit_outcome_model(synthetic_samples())
        even = model.probabilities(0.0)["draw"]

        assert even > model.probabilities(2.0)["draw"]
        assert even > model.probabilities(-2.0)["draw"]

    def test_converges_and_beats_its_starting_point(self):
        samples = synthetic_samples()
        model = elo.fit_outcome_model(samples)

        start = elo._log_likelihood(samples, 1.0, -0.5, 0.5)
        assert model.converged
        assert model.log_likelihood > start

    def test_warm_start_reaches_the_same_optimum(self):
        samples = synthetic_samples()
        cold = elo.fit_outcome_model(samples)
        warm = elo.fit_outcome_model(samples, initial=cold)

        assert warm.log_likelihood == pytest.approx(cold.log_likelihood, abs=1e-6)
        assert warm.beta == pytest.approx(cold.beta, abs=1e-3)

    def test_thresholds_stay_ordered(self):
        model = elo.fit_outcome_model(synthetic_samples())
        assert model.lower < model.upper


class TestEloFit:
    def test_recovers_relative_strength(self):
        # Team 1 beats 2 repeatedly; 2 beats 3 repeatedly.
        observations = []
        for day in range(60):
            observations.append(match(1, 2, 2, 0, day=day))
            observations.append(match(2, 3, 2, 0, day=day))
        model = elo.fit(observations)

        assert model.ratings.rating(1) > model.ratings.rating(2) > model.ratings.rating(3)

    def test_probabilities_differ_between_fixtures(self):
        observations = [match(1, 2, 3, 0, day=d) for d in range(40)]
        observations += [match(2, 3, 1, 1, day=d) for d in range(40)]
        model = elo.fit(observations)

        assert model.probabilities(1, 3) != model.probabilities(3, 1)

    def test_reports_its_configuration(self):
        observations = [match(1, 2, 1, 0, day=d) for d in range(60)]
        payload = elo.fit(observations).to_dict()

        assert payload["ratings"]["matches"] == 60
        assert payload["outcome"]["samples"] == 60
        assert "beta" in payload["outcome"]


def sample(market, candidate, actual="home") -> combination.Sample:
    return combination.Sample(
        probabilities={"market": market, "candidate": candidate}, actual=actual
    )


class TestPooling:
    EVEN = {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}
    SHARP = {"home": 0.7, "draw": 0.2, "away": 0.1}

    def test_pooling_a_single_source_at_weight_one_returns_it(self):
        pooled = combination.pooled_probabilities(sample(self.SHARP, self.EVEN), {"market": 1.0})
        assert pooled["home"] == pytest.approx(0.7)

    def test_pooled_probabilities_always_normalise(self):
        pooled = combination.pooled_probabilities(
            sample(self.SHARP, self.EVEN), {"market": 0.6, "candidate": 0.9}
        )
        assert sum(pooled.values()) == pytest.approx(1.0)

    def test_log_loss_rewards_the_better_forecaster(self):
        samples = [sample(self.SHARP, self.EVEN) for _ in range(10)]
        assert combination.log_loss(samples, {"market": 1.0}) < combination.log_loss(
            samples, {"candidate": 1.0}
        )

    def test_split_is_chronological(self):
        samples = [sample(self.SHARP, self.EVEN) for _ in range(10)]
        train, holdout = combination.split(samples, 0.7)
        assert (len(train), len(holdout)) == (7, 3)


class TestIncrementalInformation:
    MARKET = {"home": 0.5, "draw": 0.25, "away": 0.25}

    def test_a_useless_candidate_gets_no_credit(self):
        # The candidate is pure noise: identical on every match, so it cannot
        # explain anything the market does not.
        flat = {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}
        samples = [
            sample(self.MARKET, flat, actual="home" if index % 2 else "away")
            for index in range(200)
        ]
        report = combination.assess(samples, "market", ["candidate"])

        verdict = report.verdicts[0]
        assert verdict["adds_information"] is False
        assert verdict["improvement_over_market"] <= combination.MEANINGFUL_IMPROVEMENT

    def test_a_candidate_that_knows_the_answer_is_credited(self):
        # The candidate leans towards whatever actually happened, so the pool
        # should give it real weight and improve out of sample.
        samples = []
        for index in range(200):
            actual = "home" if index % 2 else "away"
            informed = {"home": 0.2, "draw": 0.2, "away": 0.2}
            informed[actual] = 0.6
            samples.append(sample(self.MARKET, informed, actual=actual))

        report = combination.assess(samples, "market", ["candidate"])
        verdict = report.verdicts[0]

        assert verdict["adds_information"] is True
        assert verdict["improvement_over_market"] > 0
        assert verdict["weights"]["candidate"] > 0.1

    def test_reports_market_alone_as_the_reference(self):
        samples = [sample(self.MARKET, self.MARKET) for _ in range(50)]
        report = combination.assess(samples, "market", ["candidate"])

        assert report.market_log_loss is not None
        assert report.holdout_matches == 25

    def test_multiple_candidates_get_a_joint_verdict(self):
        samples = [
            combination.Sample(
                probabilities={"market": self.MARKET, "a": self.MARKET, "b": self.MARKET},
                actual="home",
            )
            for _ in range(50)
        ]
        report = combination.assess(samples, "market", ["a", "b"])

        assert [v["candidates"] for v in report.verdicts] == [["a"], ["b"], ["a", "b"]]

    def test_too_few_samples_returns_an_empty_report(self):
        report = combination.assess([sample(self.MARKET, self.MARKET)], "market", ["candidate"])
        assert report.holdout_matches == 0
        assert report.verdicts == []

    def test_weights_are_finite(self):
        samples = [sample(self.MARKET, self.MARKET) for _ in range(50)]
        report = combination.assess(samples, "market", ["candidate"])
        assert all(math.isfinite(w) for w in report.combinations[-1].weights.values())


@pytest.mark.django_db
class TestCompareModelsCommand:
    @pytest.fixture
    def loaded(self, tmp_path):
        from tests.test_football_data import HEADER

        teams = ["Inter", "Milan", "Roma", "Lazio", "Napoli", "Juventus"]
        rows, day = [], 1
        for cycle in range(30):
            for index in range(0, len(teams), 2):
                home, away = teams[index], teams[index + 1]
                if cycle % 2:
                    home, away = away, home
                rows.append(
                    fd_row(
                        date=f"2024-{1 + day // 28:02d}-{1 + day % 28:02d}",
                        home=home,
                        away=away,
                        home_goals=str((cycle + index) % 4),
                        away_goals=str((cycle + index + 1) % 3),
                    )
                )
                day += 2
            teams = [teams[0], teams[-1], *teams[1:-1]]

        path = tmp_path / "matches.csv"
        path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
        call_command("ingest_football_data", str(path), division="I1", stdout=StringIO())
        return League.objects.get(slug="ita.1")

    def test_renders_the_comparison(self, loaded):
        out = StringIO()
        call_command("compare_models", "ita.1", refit_every=5, stdout=out)
        output = out.getvalue()

        assert "Standalone forecast quality" in output
        assert "Incremental information over the market" in output
        assert "market alone" in output

    def test_json_output_has_both_sections(self, loaded):
        out = StringIO()
        call_command("compare_models", "ita.1", refit_every=5, json=True, stdout=out)
        payload = json.loads(out.getvalue())

        assert set(payload["standalone"]) == {"market", "dixon_coles", "elo"}
        assert payload["incremental"]["market_only_log_loss"] is not None
        assert payload["matches"] > 0

    def test_rejects_an_unknown_league(self, db):
        with pytest.raises(CommandError):
            call_command("compare_models", "nope", stdout=StringIO())

    def test_rejects_a_bad_train_fraction(self, loaded):
        with pytest.raises(CommandError):
            call_command("compare_models", "ita.1", train_fraction=1.5, stdout=StringIO())

    def test_requires_odds_and_elo(self, db):
        call_command("seed_demo_data", rounds=20, upcoming=0, stdout=StringIO())
        with pytest.raises(CommandError):
            call_command("compare_models", "demo.1", stdout=StringIO())


class TestNullControl:
    """The gate is only worth having if a source that knows nothing fails it.

    A pure-noise candidate was observed improving the holdout by +0.0001 with
    t = +3.14 under the old rule, because the pool uses any extra source as a
    temperature knob on the market's probabilities. These tests pin that shut.
    """

    def market(self, index: int) -> dict[str, float]:
        # A market that is right on average but varies match to match.
        tilt = 0.10 * math.sin(index)
        return {"home": 0.45 + tilt, "draw": 0.28, "away": 0.27 - tilt}

    def samples(self, count: int = 600, *, informed: bool) -> list[combination.Sample]:
        """A candidate that leans towards a signal, which may or may not be real.

        The signal agrees with the outcome more often than chance when
        ``informed``, and is drawn independently when not. Both candidates look
        identical in shape and confidence; only one correlates with what
        happened, which is exactly the distinction the gate has to make.
        """
        rng = random.Random(4242)
        built = []
        for index in range(count):
            market = self.market(index)
            outcomes = list(market)
            actual = rng.choices(outcomes, weights=list(market.values()))[0]
            drawn = rng.choices(outcomes, weights=list(market.values()))[0]
            signal = actual if (informed and rng.random() < 0.55) else drawn
            candidate = {k: v * (1.8 if k == signal else 1.0) for k, v in market.items()}
            total = sum(candidate.values())
            built.append(
                combination.Sample(
                    probabilities={
                        "market": market,
                        "candidate": {k: v / total for k, v in candidate.items()},
                    },
                    actual=actual,
                )
            )
        return built

    def verdict(self, samples: list[combination.Sample]) -> dict:
        return combination.assess(samples, market="market", candidates=["candidate"]).verdicts[0]

    def test_a_candidate_that_knows_something_still_passes(self):
        verdict = self.verdict(self.samples(informed=True))

        assert verdict["adds_information"]
        assert verdict["improvement_over_market"] > verdict["null_improvement"]
        assert verdict["interval"][0] > 0

    def test_a_candidate_that_knows_nothing_is_refused(self):
        verdict = self.verdict(self.samples(informed=False))

        assert not verdict["adds_information"]

    def test_the_verdict_carries_an_error_and_an_interval(self):
        verdict = self.verdict(self.samples(informed=True))

        assert verdict["stderr"] > 0
        assert verdict["interval"][0] < verdict["improvement_over_market"] < verdict["interval"][1]
        assert abs(verdict["t"]) > 0

    def test_the_null_is_the_same_forecasts_on_the_wrong_matches(self):
        samples = self.samples(60, informed=True)

        permuted = combination.permute(samples, ["candidate"])

        assert len(permuted) == len(samples)
        assert [s.actual for s in permuted] == [s.actual for s in samples]
        # The market is untouched; only the candidate is scrambled.
        assert [s.probabilities["market"] for s in permuted] == [
            s.probabilities["market"] for s in samples
        ]
        donors = sorted(tuple(sorted(s.probabilities["candidate"].items())) for s in permuted)
        originals = sorted(tuple(sorted(s.probabilities["candidate"].items())) for s in samples)
        assert donors == originals

    def test_an_improvement_indistinguishable_from_zero_is_refused(self):
        """Significance is required on top of size, not instead of it.

        Thirty matches of a real signal give a large point estimate the sample
        cannot stand behind. Under the old rule its size alone would have passed.
        """
        verdict = self.verdict(self.samples(30, informed=True))

        assert verdict["improvement_over_market"] > combination.MEANINGFUL_IMPROVEMENT
        assert verdict["interval"][0] <= 0
        assert not verdict["adds_information"]

    def test_the_noise_candidate_clears_the_old_size_threshold(self):
        """The exact failure this gate was added for.

        A candidate that knows nothing still improves the holdout by more than
        MEANINGFUL_IMPROVEMENT, because the pool uses any extra source as a
        temperature knob. Only the interval catches it.
        """
        verdict = self.verdict(self.samples(informed=False))

        assert verdict["improvement_over_market"] > combination.MEANINGFUL_IMPROVEMENT
        assert verdict["interval"][0] <= 0
        assert not verdict["adds_information"]


class TestPairedImprovement:
    def test_measures_the_reduction_per_match(self):
        # Differences 0.1, 0.2, 0.1 — the mean is a third of 0.4, not 0.2.
        mean, stderr = combination.paired_improvement([1.0, 1.2, 0.8], [0.9, 1.0, 0.7])

        assert mean == pytest.approx(0.4 / 3, abs=1e-9)
        assert stderr > 0

    def test_a_single_match_has_no_spread(self):
        assert combination.paired_improvement([1.0], [0.5]) == (0.5, 0.0)
