"""Tests for the scoreline model, markets, odds parsing, value detection and backtest."""

import math
import random
from datetime import UTC, datetime, timedelta

import pytest

from apps.espn import backtest, markets, odds_parsing, value
from apps.espn.dixon_coles import (
    MatchObservation,
    NotEnoughData,
    ScoreGrid,
    fit,
    poisson_pmf,
    score_grid,
    tau,
)
from apps.espn.models import Event, League
from tests.test_analysis import make_game

START = datetime(2024, 8, 1, tzinfo=UTC)


def synthetic_matches(
    *,
    teams: int = 12,
    rounds: int = 2,
    home_advantage: float = 1.35,
    base_rate: float = 1.3,
    seed: int = 7,
) -> tuple[list[MatchObservation], dict[int, float], dict[int, float]]:
    """Generate independent-Poisson matches from known team strengths."""
    rng = random.Random(seed)
    attack = {i: math.exp(rng.gauss(0, 0.35)) for i in range(teams)}
    scale = math.exp(sum(math.log(v) for v in attack.values()) / teams)
    attack = {k: v / scale for k, v in attack.items()}
    defence = {i: math.exp(rng.gauss(0, 0.30)) for i in range(teams)}

    def sample(rate: float) -> int:
        limit, k, p = math.exp(-rate), 0, 1.0
        while True:
            p *= rng.random()
            if p <= limit:
                return k
            k += 1

    observations = []
    day = 0
    for round_index in range(rounds):
        for i in range(teams):
            for j in range(teams):
                if i == j:
                    continue
                home, away = (i, j) if round_index % 2 == 0 else (j, i)
                observations.append(
                    MatchObservation(
                        home_id=home,
                        away_id=away,
                        home_goals=sample(
                            base_rate * attack[home] * defence[away] * home_advantage
                        ),
                        away_goals=sample(base_rate * attack[away] * defence[home]),
                        date=START + timedelta(days=day // 6),
                    )
                )
                day += 1
    return observations, attack, defence


def correlation(left: list[float], right: list[float]) -> float:
    n = len(left)
    mean_left, mean_right = sum(left) / n, sum(right) / n
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True))
    spread_left = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    spread_right = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    return covariance / (spread_left * spread_right)


class TestPoissonAndTau:
    def test_poisson_pmf_is_a_distribution(self):
        total = sum(poisson_pmf(k, 1.7) for k in range(60))
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_poisson_pmf_matches_known_values(self):
        assert poisson_pmf(0, 2.0) == pytest.approx(math.exp(-2.0))
        assert poisson_pmf(3, 1.0) == pytest.approx(math.exp(-1.0) / 6.0)

    def test_tau_only_touches_low_scores(self):
        for home_goals, away_goals in [(2, 0), (0, 2), (1, 2), (3, 3)]:
            assert tau(home_goals, away_goals, 1.4, 1.1, 0.1) == 1.0

    def test_tau_directions(self):
        rho = 0.1
        assert tau(0, 0, 1.4, 1.1, rho) < 1.0
        assert tau(1, 1, 1.4, 1.1, rho) < 1.0
        assert tau(0, 1, 1.4, 1.1, rho) > 1.0
        assert tau(1, 0, 1.4, 1.1, rho) > 1.0

    def test_tau_is_neutral_at_zero_rho(self):
        for home_goals in range(2):
            for away_goals in range(2):
                assert tau(home_goals, away_goals, 1.4, 1.1, 0.0) == 1.0


class TestFit:
    def test_recovers_known_strengths(self):
        observations, attack, defence = synthetic_matches()
        model = fit(observations, half_life_days=0)

        team_ids = sorted(attack)
        assert model.converged
        assert (
            correlation([attack[t] for t in team_ids], [model.ratings[t].attack for t in team_ids])
            > 0.9
        )
        assert (
            correlation(
                [defence[t] for t in team_ids], [model.ratings[t].defence for t in team_ids]
            )
            > 0.9
        )

    def test_recovers_home_advantage(self):
        observations, _, _ = synthetic_matches(home_advantage=1.4)
        model = fit(observations, half_life_days=0)

        # The fitted term absorbs the base scoring rate, so compare the ratio of
        # league-wide home to away goals it implies rather than the raw factor.
        home_goals = sum(m.home_goals for m in observations)
        away_goals = sum(m.away_goals for m in observations)
        assert model.home_advantage == pytest.approx(home_goals / away_goals, rel=0.15)

    def test_finds_no_correlation_in_independent_data(self):
        # Rho is estimated from low-scoring matches only, so its scatter is wide on
        # small samples: measured across 40 seeds it centres near -0.01 with a
        # standard deviation of 0.06 at 528 matches, narrowing to 0.04 at 1584.
        # The bound below is loose enough not to flake and tight enough to catch a
        # genuinely broken estimator.
        observations, _, _ = synthetic_matches(rounds=6)
        assert abs(fit(observations, half_life_days=0).rho) < 0.05

    def test_attack_ratings_are_normalised(self):
        observations, _, _ = synthetic_matches()
        model = fit(observations, half_life_days=0)
        log_mean = sum(math.log(r.attack) for r in model.ratings.values()) / len(model.ratings)
        assert math.exp(log_mean) == pytest.approx(1.0, abs=1e-6)

    def test_decay_reduces_effective_sample(self):
        observations, _, _ = synthetic_matches()
        undecayed = fit(observations, half_life_days=0)
        decayed = fit(observations, half_life_days=10)

        assert undecayed.effective_matches == len(observations)
        assert decayed.effective_matches < undecayed.effective_matches

    def test_reliability_reflects_effective_sample(self):
        observations, _, _ = synthetic_matches()
        assert fit(observations, half_life_days=0).is_reliable
        assert not fit(observations[:10], half_life_days=0).is_reliable

    def test_rejects_empty_and_single_team_input(self):
        with pytest.raises(NotEnoughData):
            fit([])
        with pytest.raises(NotEnoughData):
            fit([MatchObservation(1, 1, 1, 1, START)])


class TestScoreGrid:
    @pytest.fixture
    def grid(self) -> ScoreGrid:
        observations, _, _ = synthetic_matches()
        return score_grid(fit(observations, half_life_days=0), 0, 1)

    def test_matrix_is_a_normalised_distribution(self, grid):
        total = sum(sum(row) for row in grid.matrix)
        assert total == pytest.approx(1.0, abs=1e-9)
        assert all(probability >= 0 for row in grid.matrix for probability in row)

    def test_unknown_team_is_rejected(self):
        observations, _, _ = synthetic_matches()
        with pytest.raises(NotEnoughData):
            score_grid(fit(observations, half_life_days=0), 999, 0)


class TestMarkets:
    @pytest.fixture
    def grid(self) -> ScoreGrid:
        observations, _, _ = synthetic_matches()
        return score_grid(fit(observations, half_life_days=0), 0, 1)

    def test_match_odds_sum_to_one(self, grid):
        assert sum(markets.match_odds(grid).values()) == pytest.approx(1.0, abs=1e-9)

    def test_double_chance_is_consistent_with_1x2(self, grid):
        outcomes = markets.match_odds(grid)
        chances = markets.double_chance(outcomes)
        assert chances["home_or_draw"] == pytest.approx(outcomes["home"] + outcomes["draw"])
        assert sum(chances.values()) == pytest.approx(2.0, abs=1e-9)

    def test_totals_complement_each_other(self, grid):
        for sides in markets.totals(grid).values():
            assert sides["over"] + sides["under"] == pytest.approx(1.0, abs=1e-9)

    def test_higher_lines_are_harder_to_pass(self, grid):
        lines = markets.totals(grid)
        assert lines["0.5"]["over"] > lines["2.5"]["over"] > lines["4.5"]["over"]

    def test_btts_complements(self, grid):
        both = markets.both_teams_to_score(grid)
        assert both["yes"] + both["no"] == pytest.approx(1.0, abs=1e-9)

    def test_correct_scores_are_ranked(self, grid):
        scores = markets.correct_scores(grid, limit=5)
        probabilities = [entry["probability"] for entry in scores]
        assert len(scores) == 5
        assert probabilities == sorted(probabilities, reverse=True)

    def test_fair_odds_invert_probability(self):
        assert markets.fair_odds(0.25) == 4.0
        assert markets.fair_odds(0.0) is None

    def test_summarise_covers_every_market(self, grid):
        summary = markets.summarise(grid)
        for key in ("expected", "1x2", "double_chance", "totals", "btts", "correct_score"):
            assert key in summary


class TestOddsParsing:
    def test_american_to_decimal(self):
        assert odds_parsing.american_to_decimal(100) == pytest.approx(2.0)
        assert odds_parsing.american_to_decimal(-200) == pytest.approx(1.5)
        assert odds_parsing.american_to_decimal(150) == pytest.approx(2.5)
        assert odds_parsing.american_to_decimal(0) is None

    def test_parses_moneylines_and_totals(self):
        rows = odds_parsing.parse_odds_payload(
            {
                "items": [
                    {
                        "provider": {"id": "41", "name": "DraftKings"},
                        "overUnder": 2.5,
                        "overOdds": -110,
                        "underOdds": -110,
                        "homeTeamOdds": {"moneyLine": -165},
                        "awayTeamOdds": {"moneyLine": 140},
                        "drawOdds": {"moneyLine": 260},
                    }
                ]
            }
        )
        by_key = {(row["market"], row["selection"]): row for row in rows}

        assert by_key[("1x2", "home")]["decimal_odds"] == pytest.approx(1.6061, abs=1e-3)
        assert by_key[("1x2", "away")]["decimal_odds"] == pytest.approx(2.4, abs=1e-3)
        assert by_key[("1x2", "draw")]["decimal_odds"] == pytest.approx(3.6, abs=1e-3)
        assert by_key[("totals", "over")]["line"] == "2.5"
        assert all(row["provider_espn_id"] == "41" for row in rows)

    def test_whole_number_totals_are_skipped(self):
        rows = odds_parsing.parse_odds_payload(
            {
                "items": [
                    {
                        "provider": {"id": "1"},
                        "overUnder": 3.0,
                        "overOdds": -110,
                        "underOdds": -110,
                        "homeTeamOdds": {"moneyLine": -120},
                        "awayTeamOdds": {"moneyLine": 100},
                    }
                ]
            }
        )
        assert not [row for row in rows if row["market"] == "totals"]

    def test_entries_without_a_provider_are_ignored(self):
        assert (
            odds_parsing.parse_odds_payload({"items": [{"homeTeamOdds": {"moneyLine": -120}}]})
            == []
        )

    def test_unparseable_prices_are_dropped(self):
        rows = odds_parsing.parse_odds_payload(
            {
                "items": [
                    {
                        "provider": {"id": "1"},
                        "homeTeamOdds": {"moneyLine": "OFF"},
                        "awayTeamOdds": {"moneyLine": -110},
                    }
                ]
            }
        )
        assert [(row["selection"]) for row in rows] == ["away"]


class TestValue:
    def test_remove_margin_normalises(self):
        fair = value.remove_margin({"home": 2.0, "draw": 4.0, "away": 4.0})
        assert sum(fair.values()) == pytest.approx(1.0)
        assert fair["home"] == pytest.approx(0.5)

    def test_overround_detects_the_margin(self):
        assert value.overround([2.0, 2.0]) == pytest.approx(1.0)
        assert value.overround([1.9, 1.9]) > 1.0

    def test_expected_value_and_kelly(self):
        assert value.expected_value(0.5, 2.0) == pytest.approx(0.0)
        assert value.expected_value(0.6, 2.0) == pytest.approx(0.2)
        assert value.kelly_fraction(0.6, 2.0) == pytest.approx(0.2)
        assert value.kelly_fraction(0.4, 2.0) == 0.0

    def test_finds_value_when_the_model_disagrees(self):
        model_markets = {
            "1x2": {
                "home": {"probability": 0.60},
                "draw": {"probability": 0.25},
                "away": {"probability": 0.15},
            }
        }
        prices = [
            value.PricedSelection("1x2", "home", "", 2.5, "1", "Book"),
            value.PricedSelection("1x2", "draw", "", 3.4, "1", "Book"),
            value.PricedSelection("1x2", "away", "", 3.0, "1", "Book"),
        ]

        bets = value.find_value_bets(model_markets, prices)

        assert [bet.selection for bet in bets] == ["home"]
        bet = bets[0]
        assert bet.edge > 0.05
        assert bet.expected_value == pytest.approx(0.5, abs=1e-4)
        assert 0 < bet.stake_fraction <= value.DEFAULT_MAX_STAKE_FRACTION

    def test_finds_nothing_when_prices_match_the_model(self):
        model_markets = {
            "1x2": {
                "home": {"probability": 0.5},
                "draw": {"probability": 0.25},
                "away": {"probability": 0.25},
            }
        }
        # Fair prices plus a margin: the model agrees exactly, so there is no edge.
        prices = [
            value.PricedSelection("1x2", "home", "", 1.9, "1"),
            value.PricedSelection("1x2", "draw", "", 3.8, "1"),
            value.PricedSelection("1x2", "away", "", 3.8, "1"),
        ]
        assert value.find_value_bets(model_markets, prices) == []

    def test_single_sided_market_is_skipped(self):
        model_markets = {"1x2": {"home": {"probability": 0.9}}}
        prices = [value.PricedSelection("1x2", "home", "", 5.0, "1")]
        assert value.find_value_bets(model_markets, prices) == []

    def test_stake_is_capped(self):
        model_markets = {"1x2": {"home": {"probability": 0.95}, "away": {"probability": 0.05}}}
        prices = [
            value.PricedSelection("1x2", "home", "", 5.0, "1"),
            value.PricedSelection("1x2", "away", "", 1.2, "1"),
        ]
        bets = value.find_value_bets(model_markets, prices, max_stake_fraction=0.02)
        assert bets[0].stake_fraction == 0.02


class TestSettlement:
    @pytest.mark.parametrize(
        ("home_goals", "away_goals", "expected"),
        [(2, 1, "home"), (1, 2, "away"), (1, 1, "draw"), (0, 0, "draw")],
    )
    def test_outcome_of(self, home_goals, away_goals, expected):
        assert backtest.outcome_of(home_goals, away_goals) == expected

    def test_settles_match_odds(self):
        assert backtest.settle("1x2", "home", "", 2, 1) is True
        assert backtest.settle("1x2", "away", "", 2, 1) is False

    def test_settles_totals(self):
        assert backtest.settle("totals", "over", "2.5", 2, 1) is True
        assert backtest.settle("totals", "under", "2.5", 2, 1) is False
        assert backtest.settle("totals", "under", "3.5", 2, 1) is True

    def test_settles_btts(self):
        assert backtest.settle("btts", "yes", "", 1, 1) is True
        assert backtest.settle("btts", "yes", "", 2, 0) is False
        assert backtest.settle("btts", "no", "", 2, 0) is True

    def test_settles_double_chance(self):
        assert backtest.settle("double_chance", "home_or_draw", "", 1, 1) is True
        assert backtest.settle("double_chance", "draw_or_away", "", 2, 1) is False

    def test_unknown_market_returns_none(self):
        assert backtest.settle("asian_handicap", "home", "-0.5", 2, 1) is None
        assert backtest.settle("totals", "over", "not-a-line", 2, 1) is None


class TestBettingLedger:
    def test_tracks_bankroll_and_drawdown(self):
        ledger = backtest.BettingLedger()
        ledger.record(0.1, 3.0, won=True)
        assert ledger.bankroll == pytest.approx(1.2)

        ledger.record(0.5, 2.0, won=False)
        assert ledger.bankroll == pytest.approx(0.6)
        assert ledger.max_drawdown == pytest.approx(0.5)

    def test_reports_uncertainty_on_the_yield(self):
        ledger = backtest.BettingLedger()
        for _ in range(10):
            ledger.record(0.01, 2.0, won=True)
            ledger.record(0.01, 2.0, won=False)

        report = ledger.to_dict()
        assert report["bets"] == 20
        assert report["flat_stake_yield"] == pytest.approx(0.0)
        assert report["flat_yield_stderr"] > 0
        assert report["distinguishable_from_zero"] is False

    def test_empty_ledger_reports_nothing(self):
        report = backtest.BettingLedger().to_dict()
        assert report["bets"] == 0
        assert report["flat_stake_yield"] is None
        assert report["distinguishable_from_zero"] is None


@pytest.mark.django_db
class TestForecastAgainstStoredEvents:
    @pytest.fixture
    def seeded(self):
        from io import StringIO

        from django.core.management import call_command

        call_command("seed_demo_data", rounds=20, upcoming=2, with_odds=True, stdout=StringIO())
        return League.objects.get(slug="demo.1")

    def test_forecast_endpoint_returns_markets(self, api_client, seeded):
        event = Event.objects.filter(status=Event.STATUS_SCHEDULED).order_by("date").first()

        response = api_client.get(f"/api/v1/events/{event.pk}/forecast/")

        assert response.status_code == 200
        payload = response.json()
        assert payload["model"]["matches"] > 0
        assert sum(
            entry["probability"] for entry in payload["markets"]["1x2"].values()
        ) == pytest.approx(1.0, abs=1e-4)
        assert payload["odds_available"] > 0
        assert isinstance(payload["value_bets"], list)

    def test_forecast_needs_history(self, api_client, db, league, team, team2):
        event = make_game(league, team, team2, None, None, START, status=Event.STATUS_SCHEDULED)

        response = api_client.get(f"/api/v1/events/{event.pk}/forecast/")

        assert response.status_code == 400
        assert "error" in response.json()

    def test_backtest_scores_only_out_of_sample_matches(self, seeded):
        report = backtest.run(seeded, refit_every=5).to_dict()

        assert report["forecasts"] > 0
        assert report["model"]["log_loss"] > 0
        assert report["baseline"]["log_loss"] > 0
        assert sum(bucket["predictions"] for bucket in report["calibration"]) == (
            report["forecasts"] * 3
        )

    def test_backtest_gate_blocks_bets_on_thin_history(self, seeded):
        gated = backtest.run(seeded, min_effective_matches=10_000).to_dict()
        assert gated["betting"]["bets"] == 0
        assert gated["skipped"]["unreliable_fit_not_bet"] > 0


@pytest.mark.django_db
class TestBacktestCommand:
    @pytest.fixture
    def seeded(self):
        from io import StringIO

        from django.core.management import call_command

        call_command("seed_demo_data", rounds=20, upcoming=2, with_odds=True, stdout=StringIO())

    def run(self, **kwargs) -> str:
        from io import StringIO

        from django.core.management import call_command

        out = StringIO()
        call_command("backtest_model", "demo.1", stdout=out, **kwargs)
        return out.getvalue()

    def test_renders_a_report(self, seeded):
        output = self.run(refit_every=5)

        assert "Forecast quality" in output
        assert "Calibration" in output
        assert "Betting simulation" in output

    def test_reports_uncertainty_alongside_any_yield(self, seeded):
        # min_history=0 lifts the reliability gate so bets are certain to be placed.
        output = self.run(refit_every=5, min_history=0)

        assert "yield std error" in output
        assert "standard error" in output

    def test_json_output_is_parseable(self, seeded):
        import json

        payload = json.loads(self.run(refit_every=5, json=True))

        assert payload["league"] == "demo.1"
        assert payload["forecasts"] > 0
        assert "calibration" in payload
        assert "betting" in payload

    def test_unknown_league_raises(self, db):
        from django.core.management.base import CommandError

        with pytest.raises(CommandError):
            self.run()

    def test_invalid_date_raises(self, seeded):
        from django.core.management.base import CommandError

        with pytest.raises(CommandError):
            self.run(date_from="not-a-date")

    def test_reports_nothing_scored_when_history_is_too_short(self, db):
        from io import StringIO

        from django.core.management import call_command

        call_command("seed_demo_data", rounds=2, upcoming=0, stdout=StringIO())
        assert "No matches had enough history" in self.run()
