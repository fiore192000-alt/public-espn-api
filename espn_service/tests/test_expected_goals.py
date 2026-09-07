"""Tests for the expected-goals forecaster and its loader.

Two things matter here beyond the arithmetic. The shrinkage, because the failure
it exists to prevent is on record: Dixon-Coles gave a promoted club 89.8% on two
fixtures. And the loader's name matching, because a club whose name never
resolves loses *every* one of its matches — an unmatched fixture is not a random
sample, so a silent join is worse than a loud failure.
"""

from datetime import UTC, datetime, timedelta
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.espn import expected_goals
from apps.espn.dixon_coles import NotEnoughData
from apps.espn.models import ExpectedGoals
from apps.ingest.management.commands.ingest_understat_xg import _normalise

START = datetime(2024, 1, 1, tzinfo=UTC)


def match(home, away, home_xg, away_xg, day=0) -> expected_goals.XgObservation:
    return expected_goals.XgObservation(home, away, home_xg, away_xg, START + timedelta(days=day))


def league(strong: int = 1, weak: int = 2, count: int = 60) -> list:
    """A league where 1 creates twice what 2 does, and concedes half."""
    built = []
    for day in range(count):
        built.append(match(strong, weak, 2.4, 0.6, day))
        built.append(match(weak, strong, 0.6, 2.4, day))
    return built


class TestFit:
    def test_needs_a_minimum_of_matches(self):
        with pytest.raises(NotEnoughData):
            expected_goals.fit(league(count=5), reference_date=START + timedelta(days=100))

    def test_recovers_which_side_creates_more(self):
        model = expected_goals.fit(league(), reference_date=START + timedelta(days=61))

        assert model.ratings[1].attack > model.ratings[2].attack
        assert model.ratings[1].defence < model.ratings[2].defence

    def test_expected_goals_follow_the_ratings(self):
        model = expected_goals.fit(league(), reference_date=START + timedelta(days=61))

        strong_home, weak_away = model.rates(1, 2)
        weak_home, strong_away = model.rates(2, 1)

        assert strong_home > weak_away
        assert strong_away > weak_home

    def test_probabilities_are_a_distribution(self):
        model = expected_goals.fit(league(), reference_date=START + timedelta(days=61))

        for pair in ((1, 2), (2, 1)):
            probabilities = model.probabilities(*pair)
            assert sum(probabilities.values()) == pytest.approx(1.0)
            assert all(value > 0 for value in probabilities.values())

    def test_the_stronger_side_is_favoured(self):
        model = expected_goals.fit(league(), reference_date=START + timedelta(days=61))

        assert model.probabilities(1, 2)["home"] > 0.6
        assert model.probabilities(2, 1)["home"] < 0.3

    def test_an_unrated_team_is_refused_rather_than_guessed(self):
        model = expected_goals.fit(league(), reference_date=START + timedelta(days=61))

        with pytest.raises(NotEnoughData):
            model.probabilities(1, 999)

    def test_home_advantage_is_divided_out_of_the_attack(self):
        """A side that has played only at home must not be rated stronger for it."""
        observations = league()
        # Team 3 plays only at home, at exactly the league's home scoring rate.
        for day in range(40):
            observations.append(match(3, 2, 1.5, 1.5, day))
        model = expected_goals.fit(observations, reference_date=START + timedelta(days=61))

        assert model.ratings[3].attack < 1.35

    def test_reports_its_configuration(self):
        payload = expected_goals.fit(league(), reference_date=START + timedelta(days=61)).to_dict()

        assert payload["teams"] == 2
        assert payload["matches"] == 120
        assert payload["league_mean_xg"] > 0


class TestShrinkage:
    """The Frosinone failure, prevented at the source.

    A club with two matches must sit near the league average whatever those two
    matches looked like, because two matches is not evidence of being the best
    attack in the division.
    """

    def test_a_club_with_two_matches_stays_near_average(self):
        observations = league()
        observations.append(match(9, 2, 6.0, 0.1, 58))
        observations.append(match(2, 9, 0.1, 6.0, 59))

        model = expected_goals.fit(observations, reference_date=START + timedelta(days=61))

        assert model.ratings[9].attack < 3.0
        assert model.ratings[9].weight < 3.0

    def test_more_history_carries_more_weight_and_so_shrinks_less(self):
        """Kept mild on purpose: extreme extra fixtures move the league mean too."""
        few, many = league(), league()
        for day in range(2):
            few.append(match(9, 2, 2.4, 0.6, day))
            few.append(match(2, 9, 0.6, 2.4, day))
        for day in range(20):
            many.append(match(9, 2, 2.4, 0.6, day))
            many.append(match(2, 9, 0.6, 2.4, day))

        thin = expected_goals.fit(few, reference_date=START + timedelta(days=61))
        thick = expected_goals.fit(many, reference_date=START + timedelta(days=61))

        assert thick.ratings[9].weight > thin.ratings[9].weight
        assert abs(thick.ratings[9].attack - 1.0) > abs(thin.ratings[9].attack - 1.0)

    def test_shrinkage_is_symmetric_for_a_weak_newcomer(self):
        observations = league()
        observations.append(match(9, 1, 0.05, 5.0, 58))
        observations.append(match(1, 9, 5.0, 0.05, 59))

        model = expected_goals.fit(observations, reference_date=START + timedelta(days=61))

        assert model.ratings[9].attack > 0.4


class TestDecay:
    def test_recent_matches_weigh_more(self):
        """Balanced home and away, or the home-advantage estimate is meaningless.

        With one side always at home its form and the league's home advantage
        are the same number, and dividing one out of the other is nonsense.
        """
        old_form = []
        new_form = []
        for day in range(50):
            old_form += [match(1, 2, 3.0, 0.3, day), match(2, 1, 0.3, 3.0, day)]
        for day in range(50, 100):
            new_form += [match(1, 2, 0.3, 3.0, day), match(2, 1, 3.0, 0.3, day)]

        model = expected_goals.fit(
            old_form + new_form, reference_date=START + timedelta(days=101), half_life_days=20
        )

        assert model.ratings[1].attack < 1.0

    def test_no_decay_weighs_everything_alike(self):
        observations = []
        for day in range(50):
            observations += [match(1, 2, 3.0, 0.3, day), match(2, 1, 0.3, 3.0, day)]
        for day in range(50, 100):
            observations += [match(1, 2, 0.3, 3.0, day), match(2, 1, 3.0, 0.3, day)]

        model = expected_goals.fit(
            observations, reference_date=START + timedelta(days=101), half_life_days=0
        )

        assert model.ratings[1].attack == pytest.approx(1.0, abs=0.15)


class TestNameNormalisation:
    @pytest.mark.parametrize(
        ("understat", "football_data"),
        [
            ("Manchester United", "Man United"),
            ("Manchester City", "Man City"),
            ("Wolverhampton Wanderers", "Wolves"),
            ("Internazionale", "Inter"),
            ("Parma Calcio 1913", "Parma"),
            ("Queens Park Rangers", "QPR"),
            ("Atlético Madrid", "Atletico Madrid"),
            ("Bayern Munich", "Bayern Munich"),
        ],
    )
    def test_the_two_sources_resolve_to_one_club(self, understat, football_data):
        assert _normalise(understat) == _normalise(football_data)

    def test_different_clubs_do_not_collide(self):
        assert _normalise("Manchester United") != _normalise("Manchester City")
        assert _normalise("Inter") != _normalise("Milan")

    def test_empty_names_are_handled(self):
        assert _normalise("") == ""
        assert _normalise(None) == ""


HEADER = "id,minute,result,X,Y,xG,player,h_a,player_id,situation,season,shotType,match_id,h_team,a_team,h_goals,a_goals,date,player_assisted,lastAction"


def shot(*, home, away, side, xg, date="2024-08-17") -> str:
    return ",".join(
        [
            "1",
            "10",
            "Goal",
            "0.8",
            "0.5",
            str(xg),
            "Player",
            side,
            "1",
            "OpenPlay",
            "2024",
            "RightFoot",
            "99",
            home,
            away,
            "2",
            "0",
            f"{date} 18:30",
            "",
            "Pass",
        ]
    )


@pytest.mark.django_db
class TestLoader:
    @pytest.fixture
    def shots_csv(self, tmp_path):
        def write(rows: list[str]) -> str:
            path = tmp_path / "shots.csv"
            path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
            return str(path)

        return write

    @pytest.fixture
    def loaded(self, tmp_path):
        from tests.test_football_data import HEADER as FD_HEADER
        from tests.test_football_data import row as fd_row

        path = tmp_path / "matches.csv"
        path.write_text(
            "\n".join([FD_HEADER, fd_row(date="2024-08-17", home="Inter", away="Genoa")]) + "\n",
            encoding="utf-8",
        )
        call_command("ingest_football_data", str(path), division="I1", stdout=StringIO())

    def load(self, path, **kwargs) -> str:
        out = StringIO()
        call_command("ingest_understat_xg", path, stdout=out, **kwargs)
        return out.getvalue()

    def test_sums_every_shot_onto_its_match(self, loaded, shots_csv):
        path = shots_csv(
            [
                shot(home="Inter", away="Genoa", side="h", xg=0.4),
                shot(home="Inter", away="Genoa", side="h", xg=0.35),
                shot(home="Inter", away="Genoa", side="a", xg=0.1),
            ]
        )

        self.load(path, league="ita.1")

        entry = ExpectedGoals.objects.get()
        assert entry.home == pytest.approx(0.75)
        assert entry.away == pytest.approx(0.10)
        assert (entry.home_shots, entry.away_shots) == (2, 1)

    def test_a_club_named_differently_still_matches(self, loaded, shots_csv):
        """Football-Data says Inter; Understat says Internazionale."""
        path = shots_csv([shot(home="Internazionale", away="Genoa", side="h", xg=0.4)])

        self.load(path, league="ita.1")

        assert ExpectedGoals.objects.count() == 1

    def test_a_fixture_recorded_a_day_out_still_matches(self, loaded, shots_csv):
        path = shots_csv([shot(home="Inter", away="Genoa", side="h", xg=0.4, date="2024-08-18")])

        self.load(path, league="ita.1")

        assert ExpectedGoals.objects.count() == 1

    def test_an_unmatched_fixture_is_named_not_counted(self, loaded, shots_csv):
        path = shots_csv([shot(home="Nowhere United", away="Elsewhere", side="h", xg=0.4)])

        output = self.load(path, league="ita.1")

        assert ExpectedGoals.objects.count() == 0
        assert "Nowhere United" in output
        assert "not a random sample" in output

    def test_reloading_updates_rather_than_duplicates(self, loaded, shots_csv):
        self.load(shots_csv([shot(home="Inter", away="Genoa", side="h", xg=0.4)]), league="ita.1")
        self.load(shots_csv([shot(home="Inter", away="Genoa", side="h", xg=0.9)]), league="ita.1")

        entry = ExpectedGoals.objects.get()
        assert entry.home == pytest.approx(0.9)

    def test_an_unknown_league_is_named_in_the_error(self, db, shots_csv):
        with pytest.raises(CommandError, match="No league with slug 'nowhere'"):
            self.load(
                shots_csv([shot(home="Inter", away="Genoa", side="h", xg=0.4)]), league="nowhere"
            )

    def test_a_file_with_no_usable_shots_says_so(self, loaded, shots_csv):
        with pytest.raises(CommandError, match="No usable shots"):
            self.load(shots_csv([]), league="ita.1")

    def test_the_stored_row_exposes_total_and_difference(self, loaded, shots_csv):
        self.load(
            shots_csv(
                [
                    shot(home="Inter", away="Genoa", side="h", xg=1.5),
                    shot(home="Inter", away="Genoa", side="a", xg=0.5),
                ]
            ),
            league="ita.1",
        )

        entry = ExpectedGoals.objects.get()
        assert entry.total == pytest.approx(2.0)
        assert entry.difference == pytest.approx(1.0)
        assert "xG" in str(entry)
