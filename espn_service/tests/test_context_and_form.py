"""Tests for match context, time-weighted form, momentum and half-life tuning."""

import json
from datetime import UTC, datetime, timedelta
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.espn import context
from apps.espn.analysis import (
    FORM_HALF_LIFE_DAYS,
    analyze_event,
    build_league_baseline,
    build_team_form,
)
from apps.espn.models import AthleteSeasonStats, Event, Injury, League, Team
from tests.test_analysis import make_game

BASE = datetime(2024, 12, 20, 19, 0, tzinfo=UTC)


def at(days_ago: int) -> datetime:
    return BASE - timedelta(days=days_ago)


@pytest.fixture
def team3(db, league: League) -> Team:
    return Team.objects.create(
        league=league, espn_id="3", abbreviation="THR", display_name="Third Team"
    )


@pytest.fixture
def schedule(db, league: League, team: Team, team2: Team, team3: Team) -> dict:
    """Team plays four matches in the fortnight before an upcoming fixture."""
    games = [
        make_game(league, team, team3, 1, 3, at(20)),
        make_game(league, team3, team, 0, 2, at(12)),
        make_game(league, team, team2, 3, 0, at(8)),
        make_game(league, team2, team, 1, 2, at(4)),
        make_game(league, team, team3, 2, 0, at(2)),
    ]
    upcoming = make_game(league, team, team2, None, None, BASE, status=Event.STATUS_SCHEDULED)
    return {"games": games, "upcoming": upcoming}


class TestRestAndCongestion:
    def test_rest_days_counts_from_the_last_match(self, schedule, team):
        assert context.rest_days(team, BASE) == 2

    def test_rest_days_is_none_without_history(self, db, team3):
        assert context.rest_days(team3, BASE) is None

    def test_matches_in_window_counts_only_the_window(self, schedule, team):
        assert context.matches_in_window(team, BASE, window_days=14) == 4
        assert context.matches_in_window(team, BASE, window_days=5) == 2

    def test_congestion_flag(self, schedule, team, team2):
        home_context = context.build_context(
            schedule["upcoming"], schedule["upcoming"].competitors.get(team=team)
        )
        assert home_context.matches_in_window == 4
        assert home_context.congested is True

        away_context = context.build_context(
            schedule["upcoming"], schedule["upcoming"].competitors.get(team=team2)
        )
        assert away_context.congested is False


class TestInjuryBurden:
    def test_counts_and_weights_by_severity(self, db, league, team):
        Injury.objects.create(
            league=league, team=team, athlete_name="A Player", status=Injury.STATUS_OUT
        )
        Injury.objects.create(
            league=league, team=team, athlete_name="B Player", status=Injury.STATUS_QUESTIONABLE
        )

        burden = context.injury_burden(league, team)

        assert burden.count == 2
        assert burden.weighted == pytest.approx(1.4)
        assert burden.by_status == {Injury.STATUS_OUT: 1, Injury.STATUS_QUESTIONABLE: 1}
        assert burden.importance_known is False

    def test_players_are_ranked_by_weight(self, db, league, team):
        Injury.objects.create(
            league=league, team=team, athlete_name="Minor", status=Injury.STATUS_DAY_TO_DAY
        )
        Injury.objects.create(
            league=league, team=team, athlete_name="Major", status=Injury.STATUS_OUT
        )

        burden = context.injury_burden(league, team)

        assert [player.name for player in burden.players] == ["Major", "Minor"]

    def test_appearances_scale_importance_when_known(self, db, league, team):
        AthleteSeasonStats.objects.create(
            league=league,
            athlete_espn_id="100",
            athlete_name="Regular",
            season_year=2024,
            stats={"appearances": 30},
        )
        AthleteSeasonStats.objects.create(
            league=league,
            athlete_espn_id="200",
            athlete_name="Fringe",
            season_year=2024,
            stats={"appearances": 6},
        )
        Injury.objects.create(
            league=league,
            team=team,
            athlete_name="Regular",
            athlete_espn_id="100",
            status=Injury.STATUS_OUT,
        )
        Injury.objects.create(
            league=league,
            team=team,
            athlete_name="Fringe",
            athlete_espn_id="200",
            status=Injury.STATUS_OUT,
        )

        burden = context.injury_burden(league, team)

        assert burden.importance_known is True
        # The regular starter counts fully; the fringe player at 6/30 of that.
        assert burden.weighted == pytest.approx(1.0 + 0.2)
        assert burden.players[0].name == "Regular"

    def test_unknown_appearances_fall_back_to_full_importance(self, db, league, team):
        Injury.objects.create(
            league=league,
            team=team,
            athlete_name="Unknown",
            athlete_espn_id="999",
            status=Injury.STATUS_OUT,
        )

        burden = context.injury_burden(league, team)

        assert burden.importance_known is False
        assert burden.weighted == pytest.approx(1.0)

    def test_no_injuries_is_empty(self, db, league, team):
        burden = context.injury_burden(league, team)
        assert burden.count == 0
        assert burden.weighted == 0.0
        assert burden.players == []


class TestWeightedForm:
    def test_recent_games_carry_more_weight(self, schedule, team):
        form = build_team_form(team, before=BASE)

        # Unweighted the side is 3-0-2; the two losses are the oldest games, so
        # the decayed points rate must sit above the flat one.
        assert form.weighted.points_per_game > form.points_per_game
        assert form.weighted.half_life_days == FORM_HALF_LIFE_DAYS
        assert 0 < form.weighted.effective_games <= form.overall.played

    def test_no_decay_matches_the_flat_average(self, schedule, team):
        from apps.espn.analysis import _weighted_form

        form = build_team_form(team, before=BASE)
        flat = _weighted_form(form.games, reference=BASE, half_life_days=0)

        assert flat.points_per_game == pytest.approx(form.points_per_game)
        assert flat.effective_games == pytest.approx(form.overall.played)

    def test_empty_form_is_zeroed(self, db, team3):
        form = build_team_form(team3)
        assert form.weighted.effective_games == 0.0
        assert form.weighted.points_per_game == 0.0


class TestMomentum:
    def test_compares_recent_games_to_the_window(self, schedule, team):
        form = build_team_form(team, before=BASE)
        momentum = form.momentum

        assert momentum.games == 5
        assert momentum.window == 5
        # With exactly window-many games the recent slice is the whole window.
        assert momentum.points_delta == pytest.approx(0.0)

    def test_shorter_slice_detects_an_upturn(self, schedule, team):
        from apps.espn.analysis import _momentum_of

        form = build_team_form(team, before=BASE)
        momentum = _momentum_of(form.games, window=2)

        assert momentum.games == 2
        assert momentum.points_per_game == pytest.approx(3.0)
        assert momentum.points_delta > 0

    def test_empty_form_has_no_momentum(self, db, team3):
        assert build_team_form(team3).momentum.games == 0


class TestOpponentStrength:
    def test_reports_average_opponent_goal_difference(self, schedule, team):
        strength = build_team_form(team, before=BASE).opponent_strength
        assert strength is not None
        assert isinstance(strength, float)

    def test_none_without_games(self, db, team3):
        assert build_team_form(team3).opponent_strength is None


class TestBaselineWindow:
    def test_window_restricts_the_sample(self, schedule, league):
        whole = build_league_baseline(league, before=BASE)
        recent = build_league_baseline(league, before=BASE, window_days=5)

        assert whole.sample == 5
        assert recent.sample == 2

    def test_window_beyond_history_matches_the_full_sample(self, schedule, league):
        assert build_league_baseline(league, before=BASE, window_days=3650).sample == 5


@pytest.mark.django_db
class TestContextInAnalysis:
    def test_analysis_carries_context_for_both_sides(self, schedule):
        payload = analyze_event(schedule["upcoming"])

        assert payload["home"]["context"]["rest_days"] == 2
        assert payload["home"]["context"]["congested"] is True
        assert payload["away"]["context"]["matches_in_last_14_days"] >= 0
        assert "injuries" in payload["home"]["context"]

    def test_congestion_and_absences_reach_the_notes(self, schedule, league, team):
        Injury.objects.create(
            league=league, team=team, athlete_name="Out Player", status=Injury.STATUS_OUT
        )

        notes = " ".join(analyze_event(schedule["upcoming"])["insights"])

        assert "matches in 14 days" in notes
        assert "listed absence" in notes
        assert "playing time unknown" in notes

    def test_baseline_window_is_honoured(self, schedule):
        payload = analyze_event(schedule["upcoming"], baseline_window_days=5)
        assert payload["league_baseline"]["sample"] == 2


@pytest.mark.django_db
class TestHalfLifeTuning:
    @pytest.fixture
    def seeded(self) -> League:
        call_command("seed_demo_data", rounds=20, upcoming=2, stdout=StringIO())
        return League.objects.get(slug="demo.1")

    def test_sweep_ranks_candidates_by_log_loss(self, seeded):
        from apps.espn.backtest import sweep_half_life

        results = sweep_half_life(seeded, [30, 120], refit_every=10)

        assert len(results) == 2
        assert all(row["forecasts"] > 0 for row in results)
        assert results[0]["log_loss"] <= results[1]["log_loss"]
        assert {row["half_life_days"] for row in results} == {30, 120}

    def test_every_candidate_scores_the_same_matches(self, seeded):
        from apps.espn.backtest import sweep_half_life

        results = sweep_half_life(seeded, [30, 90, 365], refit_every=10)
        assert len({row["forecasts"] for row in results}) == 1

    def test_command_renders_a_table(self, seeded):
        out = StringIO()
        call_command("tune_model", "demo.1", half_lives="60,180", refit_every=10, stdout=out)
        output = out.getvalue()

        assert "Half-life sweep" in output
        assert "Best half-life" in output
        assert "log loss" in output

    def test_command_json_output(self, seeded):
        out = StringIO()
        call_command(
            "tune_model", "demo.1", half_lives="60,180", refit_every=10, json=True, stdout=out
        )
        payload = json.loads(out.getvalue())

        assert len(payload) == 2
        assert "half_life_days" in payload[0]

    def test_command_rejects_unknown_league(self, db):
        with pytest.raises(CommandError):
            call_command("tune_model", "nope", stdout=StringIO())

    def test_command_rejects_bad_half_lives(self, seeded):
        with pytest.raises(CommandError):
            call_command("tune_model", "demo.1", half_lives="fast,slow", stdout=StringIO())

    def test_command_warns_when_history_is_too_short(self, db):
        call_command("seed_demo_data", rounds=2, upcoming=0, stdout=StringIO())
        out = StringIO()
        call_command("tune_model", "demo.1", half_lives="60", refit_every=10, stdout=out)

        assert "No matches had enough history" in out.getvalue()


@pytest.mark.django_db
class TestProjectionSource:
    def test_uses_the_scoreline_model_when_history_allows(self):
        call_command("seed_demo_data", rounds=20, upcoming=2, stdout=StringIO())
        event = Event.objects.filter(status=Event.STATUS_SCHEDULED).order_by("date").first()

        projection = analyze_event(event)["projection"]

        assert projection["source"] == "dixon_coles"
        assert projection["model"]["matches"] > 0

    def test_agrees_with_the_forecast_endpoint(self):
        from apps.espn import forecast

        call_command("seed_demo_data", rounds=20, upcoming=2, stdout=StringIO())
        event = Event.objects.filter(status=Event.STATUS_SCHEDULED).order_by("date").first()

        analysed = analyze_event(event)["projection"]["probabilities"]
        forecast_markets = forecast.forecast_event(event)["markets"]["1x2"]

        assert analysed["home_win"] == pytest.approx(
            forecast_markets["home"]["probability"], abs=1e-3
        )
        assert analysed["draw"] == pytest.approx(forecast_markets["draw"]["probability"], abs=1e-3)
        assert analysed["away_win"] == pytest.approx(
            forecast_markets["away"]["probability"], abs=1e-3
        )

    def test_draw_probability_varies_between_fixtures(self):
        call_command("seed_demo_data", rounds=30, upcoming=4, stdout=StringIO())
        events = Event.objects.filter(status=Event.STATUS_SCHEDULED).order_by("date")[:4]

        draws = {analyze_event(event)["projection"]["probabilities"]["draw"] for event in events}

        # The old form projection priced every fixture in a league identically.
        assert len(draws) > 1

    def test_falls_back_and_says_so_without_enough_history(self, schedule):
        payload = analyze_event(schedule["upcoming"])

        assert payload["projection"]["source"] == "fallback_form"
        assert "model" not in payload["projection"]
        assert any("fallback" in note for note in payload["insights"])

    def test_probabilities_still_sum_to_one_on_both_paths(self, schedule):
        fallback = analyze_event(schedule["upcoming"])["projection"]["probabilities"]
        assert sum(fallback.values()) == pytest.approx(1.0)

        call_command("seed_demo_data", rounds=20, upcoming=2, stdout=StringIO())
        event = Event.objects.filter(league__slug="demo.1", status=Event.STATUS_SCHEDULED).first()
        modelled = analyze_event(event)["projection"]["probabilities"]
        assert sum(modelled.values()) == pytest.approx(1.0)

    def test_confidence_follows_the_model_when_it_is_used(self):
        call_command("seed_demo_data", rounds=30, upcoming=2, stdout=StringIO())
        event = Event.objects.filter(status=Event.STATUS_SCHEDULED).order_by("date").first()

        payload = analyze_event(event)
        assert payload["projection"]["source"] == "dixon_coles"
        assert payload["confidence"] in {"medium", "high"}

    def test_fallback_confidence_is_capped(self, schedule):
        payload = analyze_event(schedule["upcoming"])
        assert payload["projection"]["source"] == "fallback_form"
        assert payload["confidence"] in {"none", "low", "medium"}
