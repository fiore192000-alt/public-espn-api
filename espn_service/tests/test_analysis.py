"""Tests for the match analysis engine and its API surface."""

from datetime import UTC, datetime

import pytest

from apps.espn.analysis import (
    AnalysisNotAvailable,
    analyze_event,
    build_head_to_head,
    build_league_baseline,
    build_team_form,
)
from apps.espn.models import Competitor, Event, League, Team


def make_game(
    league: League,
    home: Team,
    away: Team,
    home_score: int | None,
    away_score: int | None,
    date: datetime,
    status: str = Event.STATUS_FINAL,
) -> Event:
    """Create a two-sided event with scores already attached."""
    event = Event.objects.create(
        league=league,
        espn_id=f"g{int(date.timestamp())}{home.pk}{away.pk}",
        date=date,
        name=f"{away.display_name} at {home.display_name}",
        short_name=f"{away.abbreviation} @ {home.abbreviation}",
        season_year=date.year,
        status=status,
        status_detail="Final" if status == Event.STATUS_FINAL else "Scheduled",
    )
    for team, score, side, order in (
        (home, home_score, Competitor.HOME, 1),
        (away, away_score, Competitor.AWAY, 0),
    ):
        Competitor.objects.create(
            event=event,
            team=team,
            home_away=side,
            score="" if score is None else str(score),
            order=order,
        )
    return event


@pytest.fixture
def team3(db, league: League) -> Team:
    return Team.objects.create(
        league=league,
        espn_id="3",
        abbreviation="THR",
        display_name="Third Team",
        location="Third City",
    )


@pytest.fixture
def history(db, league: League, team: Team, team2: Team, team3: Team) -> dict:
    """Five completed games plus one upcoming fixture between team and team2.

    From team's point of view, most recent first: W W D L W.
    """

    def at(day: int) -> datetime:
        return datetime(2024, 12, day, 19, 0, tzinfo=UTC)

    games = [
        make_game(league, team, team2, 110, 100, at(1)),
        make_game(league, team2, team, 105, 95, at(3)),
        make_game(league, team, team2, 100, 100, at(5)),
        make_game(league, team3, team, 90, 120, at(7)),
        make_game(league, team, team3, 115, 90, at(9)),
    ]
    upcoming = make_game(league, team, team2, None, None, at(15), status=Event.STATUS_SCHEDULED)
    return {"games": games, "upcoming": upcoming}


class TestTeamForm:
    def test_counts_results_and_totals(self, history, team):
        form = build_team_form(team)

        assert form.overall.played == 5
        assert (form.overall.wins, form.overall.draws, form.overall.losses) == (3, 1, 1)
        assert form.overall.scored == 540
        assert form.overall.conceded == 485
        assert form.point_differential == 55

    def test_splits_home_and_away(self, history, team):
        form = build_team_form(team)

        assert form.home.played == 3
        assert (form.home.wins, form.home.draws, form.home.losses) == (2, 1, 0)
        assert form.away.played == 2
        assert (form.away.wins, form.away.draws, form.away.losses) == (1, 0, 1)
        assert form.home.avg_scored == pytest.approx(108.33, abs=0.01)

    def test_streak_uses_most_recent_run(self, history, team):
        assert build_team_form(team).streak == "W2"

    def test_win_pct_counts_draws_as_half(self, history, team):
        assert build_team_form(team).overall.win_pct == pytest.approx(0.7)

    def test_lookback_limits_games(self, history, team):
        assert build_team_form(team, lookback=2).overall.played == 2

    def test_before_excludes_later_games(self, history, team):
        cutoff = datetime(2024, 12, 6, tzinfo=UTC)
        assert build_team_form(team, before=cutoff).overall.played == 3

    def test_team_without_history_is_empty(self, db, team3):
        form = build_team_form(team3)

        assert form.overall.played == 0
        assert form.streak == ""
        assert form.overall.avg_scored == 0.0
        assert form.overall.win_pct == 0.0

    def test_scheduled_games_are_ignored(self, history, team):
        assert all(g.result for g in build_team_form(team).games)
        assert len(build_team_form(team).games) == 5


class TestHeadToHead:
    def test_records_meetings_from_first_team_perspective(self, history, team, team2):
        h2h = build_head_to_head(team, team2)

        assert h2h.played == 3
        assert (h2h.team_a_wins, h2h.team_b_wins, h2h.draws) == (1, 1, 1)
        assert h2h.team_a_points == 305
        assert h2h.team_b_points == 305
        assert h2h.avg_total_points == pytest.approx(203.33, abs=0.01)

    def test_perspective_is_mirrored(self, history, team, team2):
        forward = build_head_to_head(team, team2)
        reverse = build_head_to_head(team2, team)

        assert forward.team_a_wins == reverse.team_b_wins
        assert forward.team_b_wins == reverse.team_a_wins
        assert forward.draws == reverse.draws

    def test_no_meetings(self, history, team2, team3):
        h2h = build_head_to_head(team2, team3)

        assert h2h.played == 0
        assert h2h.avg_total_points == 0.0


class TestLeagueBaseline:
    def test_derives_home_advantage_and_draw_rate(self, history, league):
        baseline = build_league_baseline(league)

        assert baseline.sample == 5
        assert baseline.avg_home_score == pytest.approx(104.0)
        assert baseline.avg_away_score == pytest.approx(101.0)
        assert baseline.home_advantage == pytest.approx(3.0)
        assert baseline.draw_rate == pytest.approx(0.2)

    def test_empty_league_falls_back(self, db, league):
        baseline = build_league_baseline(league)

        assert baseline.sample == 0
        assert baseline.margin_sigma > 0


class TestAnalyzeEvent:
    def test_uses_only_history_before_the_event(self, history, team, team2):
        analysis = analyze_event(history["upcoming"])

        assert analysis["home"]["team"]["abbreviation"] == team.abbreviation
        assert analysis["away"]["team"]["abbreviation"] == team2.abbreviation
        assert analysis["home"]["form"]["overall"]["played"] == 5
        assert analysis["away"]["form"]["overall"]["played"] == 3
        assert analysis["head_to_head"]["played"] == 3

    def test_probabilities_sum_to_one(self, history):
        probabilities = analyze_event(history["upcoming"])["projection"]["probabilities"]

        assert sum(probabilities.values()) == pytest.approx(1.0)
        assert all(0.0 <= p <= 1.0 for p in probabilities.values())

    def test_stronger_home_side_is_favoured(self, history):
        projection = analyze_event(history["upcoming"])["projection"]
        probabilities = projection["probabilities"]

        assert projection["margin"] > 0
        assert probabilities["home_win"] > probabilities["away_win"]
        assert projection["total"] == pytest.approx(
            projection["home_score"] + projection["away_score"], abs=0.01
        )

    def test_draw_probability_tracks_league_rate(self, history):
        analysis = analyze_event(history["upcoming"])

        assert analysis["projection"]["probabilities"]["draw"] == pytest.approx(
            analysis["league_baseline"]["draw_rate"]
        )

    def test_confidence_reflects_sample_size(self, history, db, league, team3):
        rich = analyze_event(history["upcoming"])
        assert rich["confidence"] in {"low", "medium", "high"}

        blank = make_game(
            league,
            team3,
            Team.objects.create(
                league=league, espn_id="4", abbreviation="FOU", display_name="Fourth"
            ),
            None,
            None,
            datetime(2020, 1, 1, tzinfo=UTC),
            status=Event.STATUS_SCHEDULED,
        )
        assert analyze_event(blank)["confidence"] == "none"

    def test_insights_are_generated(self, history):
        insights = analyze_event(history["upcoming"])["insights"]

        assert insights
        assert any("Head-to-head" in note for note in insights)
        assert any("Model leans" in note for note in insights)

    def test_event_without_two_competitors_is_rejected(self, db, league, team):
        event = Event.objects.create(
            league=league,
            espn_id="solo-1",
            date=datetime(2024, 12, 20, tzinfo=UTC),
            name="Solo event",
            season_year=2024,
        )
        Competitor.objects.create(event=event, team=team, home_away=Competitor.HOME, order=0)

        with pytest.raises(AnalysisNotAvailable):
            analyze_event(event)


@pytest.mark.django_db
class TestAnalysisEndpoints:
    def test_event_analysis_endpoint(self, api_client, history):
        response = api_client.get(f"/api/v1/events/{history['upcoming'].pk}/analysis/")

        assert response.status_code == 200
        payload = response.json()
        assert payload["event"]["espn_id"] == history["upcoming"].espn_id
        assert payload["projection"]["probabilities"]
        assert payload["lookback"] == 10

    def test_lookback_is_clamped(self, api_client, history):
        response = api_client.get(
            f"/api/v1/events/{history['upcoming'].pk}/analysis/", {"lookback": "999"}
        )

        assert response.status_code == 200
        assert response.json()["lookback"] == 50

    def test_invalid_lookback_falls_back_to_default(self, api_client, history):
        response = api_client.get(
            f"/api/v1/events/{history['upcoming'].pk}/analysis/", {"lookback": "abc"}
        )

        assert response.status_code == 200
        assert response.json()["lookback"] == 10

    def test_non_two_sided_event_returns_400(self, api_client, db, league, team):
        event = Event.objects.create(
            league=league,
            espn_id="solo-2",
            date=datetime(2024, 12, 20, tzinfo=UTC),
            name="Solo event",
            season_year=2024,
        )
        Competitor.objects.create(event=event, team=team, home_away=Competitor.HOME, order=0)

        response = api_client.get(f"/api/v1/events/{event.pk}/analysis/")

        assert response.status_code == 400
        assert "error" in response.json()

    def test_team_form_endpoint(self, api_client, history, team):
        response = api_client.get(f"/api/v1/teams/{team.pk}/form/", {"lookback": "3"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["abbreviation"] == team.abbreviation
        assert payload["overall"]["played"] == 3
        assert len(payload["games"]) == 3
