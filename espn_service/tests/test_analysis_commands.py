"""Tests for the seed_demo_data and analyze_match management commands."""

import json
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.espn.models import Competitor, Event, Team


def seed(**kwargs) -> str:
    out = StringIO()
    call_command("seed_demo_data", stdout=out, **kwargs)
    return out.getvalue()


@pytest.mark.django_db
class TestSeedDemoData:
    def test_creates_completed_and_scheduled_events(self):
        output = seed(rounds=4, upcoming=2)

        assert Team.objects.filter(league__slug="demo.1").count() == 8
        assert Event.objects.filter(status=Event.STATUS_FINAL).count() == 16
        assert Event.objects.filter(status=Event.STATUS_SCHEDULED).count() == 2
        assert "Synthetic demo data" in output

    def test_completed_events_carry_scores_and_scheduled_ones_do_not(self):
        seed(rounds=2, upcoming=1)

        final = Event.objects.filter(status=Event.STATUS_FINAL).first()
        assert all(c.score_int is not None for c in final.competitors.all())

        scheduled = Event.objects.get(status=Event.STATUS_SCHEDULED)
        assert all(c.score_int is None for c in scheduled.competitors.all())

    def test_every_event_is_two_sided_with_one_home_and_one_away(self):
        seed(rounds=3, upcoming=2)

        for event in Event.objects.prefetch_related("competitors"):
            sides = [c.home_away for c in event.competitors.all()]
            assert sorted(sides) == [Competitor.AWAY, Competitor.HOME]

    def test_same_seed_reproduces_same_scores(self):
        seed(rounds=2, upcoming=0, seed=7)
        first = sorted(Event.objects.values_list("espn_id", "competitors__score"))

        Event.objects.all().delete()
        seed(rounds=2, upcoming=0, seed=7)
        second = sorted(Event.objects.values_list("espn_id", "competitors__score"))

        assert first == second

    def test_basketball_profile_scores_high_and_avoids_draws(self):
        seed(profile="basketball", rounds=3, upcoming=0)

        totals = []
        for event in Event.objects.prefetch_related("competitors"):
            scores = [c.score_int for c in event.competitors.all()]
            assert scores[0] != scores[1]
            totals.append(sum(scores))

        assert min(totals) > 100

    def test_rejects_zero_rounds(self):
        with pytest.raises(CommandError):
            seed(rounds=0)


@pytest.mark.django_db
class TestAnalyzeMatch:
    @pytest.fixture(autouse=True)
    def seeded(self):
        seed(rounds=6, upcoming=2)

    def run(self, **kwargs) -> str:
        out = StringIO()
        call_command("analyze_match", stdout=out, **kwargs)
        return out.getvalue()

    def test_reports_upcoming_fixtures(self):
        output = self.run(upcoming=1)

        assert "Projection" in output
        assert "win probability" in output
        assert "Notes" in output

    def test_json_output_is_parseable(self):
        payloads = json.loads(self.run(upcoming=2, json=True))

        assert len(payloads) == 2
        for payload in payloads:
            assert payload["projection"]["probabilities"]
            assert payload["home"]["form"]["overall"]["played"] >= 0

    def test_selects_a_specific_event(self):
        event = Event.objects.filter(status=Event.STATUS_SCHEDULED).first()

        payloads = json.loads(self.run(event=event.espn_id, json=True))

        assert len(payloads) == 1
        assert payloads[0]["event"]["espn_id"] == event.espn_id

    def test_falls_back_to_recent_events_when_none_scheduled(self):
        Event.objects.filter(status=Event.STATUS_SCHEDULED).delete()

        payloads = json.loads(self.run(upcoming=1, json=True))

        assert payloads[0]["event"]["status"] == Event.STATUS_FINAL

    def test_unknown_league_raises(self):
        with pytest.raises(CommandError):
            self.run(league="not-a-league")

    def test_lookback_limits_history(self):
        payloads = json.loads(self.run(upcoming=1, lookback=2, json=True))

        assert payloads[0]["home"]["form"]["overall"]["played"] <= 2
        assert payloads[0]["lookback"] == 2
