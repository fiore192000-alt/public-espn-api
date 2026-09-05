"""Tests for the Football-Data loader and the market benchmark."""

import json
from datetime import UTC, datetime
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.espn import backtest
from apps.espn.backtest import ForecastRecord, _market_probabilities
from apps.espn.models import Competitor, Event, League, Odds, Team
from apps.espn.value import PricedSelection
from apps.ingest.management.commands.ingest_football_data import detect_schema

HEADER = (
    "Division,MatchDate,MatchTime,HomeTeam,AwayTeam,HomeElo,AwayElo,"
    "Form3Home,Form5Home,Form3Away,Form5Away,FTHome,FTAway,FTResult,"
    "HomeShots,AwayShots,HomeTarget,AwayTarget,HomeCorners,AwayCorners,"
    "HomeYellow,AwayYellow,HomeRed,AwayRed,"
    "OddHome,OddDraw,OddAway,MaxHome,MaxDraw,MaxAway,"
    "Over25,Under25,MaxOver25,MaxUnder25"
)


def row(
    *,
    division="I1",
    date="2024-08-17",
    time="18:30",
    home="Inter",
    away="Genoa",
    home_goals="2",
    away_goals="0",
    odds=("1.40", "5.00", "8.00"),
    max_odds=("1.45", "5.20", "8.60"),
    totals=("1.80", "2.05"),
) -> str:
    return ",".join(
        [
            division,
            date,
            time,
            home,
            away,
            "1900.5",
            "1600.2",
            "1.0",
            "1.0",
            "0.0",
            "0.0",
            home_goals,
            away_goals,
            "H",
            "15",
            "6",
            "7",
            "2",
            "8",
            "3",
            "1",
            "2",
            "0",
            "0",
            *odds,
            *max_odds,
            *totals,
            "1.85",
            "2.10",
        ]
    )


@pytest.fixture
def csv_file(tmp_path):
    def write(rows: list[str]) -> str:
        path = tmp_path / "matches.csv"
        path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
        return str(path)

    return write


def load(path: str, **kwargs) -> str:
    out = StringIO()
    call_command("ingest_football_data", path, stdout=out, **kwargs)
    return out.getvalue()


@pytest.mark.django_db
class TestFootballDataLoader:
    def test_creates_league_teams_and_event(self, csv_file):
        output = load(csv_file([row()]), division="I1")

        league = League.objects.get(slug="ita.1")
        assert league.sport.slug == "soccer"
        assert league.name == "Italian Serie A"
        assert Team.objects.filter(league=league).count() == 2
        assert Event.objects.count() == 1
        assert "Loaded 1 matches" in output

    def test_records_the_result_on_both_sides(self, csv_file):
        load(csv_file([row(home_goals="2", away_goals="0")]), division="I1")

        event = Event.objects.get()
        assert event.status == Event.STATUS_FINAL
        home = event.competitors.get(home_away=Competitor.HOME)
        away = event.competitors.get(home_away=Competitor.AWAY)
        assert (home.score_int, away.score_int) == (2, 0)
        assert home.winner is True
        assert away.winner is False

    def test_stores_odds_for_both_aggregates(self, csv_file):
        load(csv_file([row()]), division="I1")

        event = Event.objects.get()
        average = {
            o.selection: o.decimal_odds
            for o in event.odds.filter(provider_espn_id="fd-b365", market="1x2")
        }
        maximum = {
            o.selection: o.decimal_odds
            for o in event.odds.filter(provider_espn_id="fd-max", market="1x2")
        }

        assert average == {"home": 1.40, "draw": 5.00, "away": 8.00}
        assert maximum == {"home": 1.45, "draw": 5.20, "away": 8.60}
        assert event.odds.filter(market="totals", line="2.5").count() == 4

    def test_no_odds_flag_skips_prices(self, csv_file):
        load(csv_file([row()]), division="I1", no_odds=True)
        assert Odds.objects.count() == 0

    def test_keeps_elo_and_stats(self, csv_file):
        load(csv_file([row()]), division="I1")

        event = Event.objects.get()
        assert event.raw_data["home_elo"] == "1900.5"
        assert event.raw_data["source"] == "football-data"
        home = event.competitors.get(home_away=Competitor.HOME)
        stats = {entry["name"]: entry["value"] for entry in home.statistics}
        assert stats["shots"] == 15
        assert stats["shotsOnTarget"] == 7

    def test_filters_by_division_and_date(self, csv_file):
        path = csv_file(
            [
                row(division="I1", date="2024-08-17", home="Inter", away="Genoa"),
                row(division="E0", date="2024-08-17", home="Arsenal", away="Wolves"),
                row(division="I1", date="2020-01-05", home="Roma", away="Lazio"),
            ]
        )
        load(path, division="I1", date_from="2024-01-01")

        assert Event.objects.count() == 1
        assert Event.objects.get().name == "Genoa at Inter"

    def test_season_year_follows_the_european_calendar(self, csv_file):
        load(
            csv_file(
                [
                    row(date="2024-08-17", home="Inter", away="Genoa"),
                    row(date="2025-03-02", home="Roma", away="Lazio"),
                ]
            ),
            division="I1",
        )
        assert set(Event.objects.values_list("season_year", flat=True)) == {2024}

    def test_reloading_is_idempotent(self, csv_file):
        path = csv_file([row()])
        load(path, division="I1")
        load(path, division="I1")

        assert Event.objects.count() == 1
        assert Odds.objects.filter(market="1x2").count() == 6

    def test_rows_without_a_result_are_skipped(self, csv_file):
        output = load(csv_file([row(home_goals="", away_goals="")]), division="I1")
        assert Event.objects.count() == 0
        assert "1 rows skipped" in output

    def test_unusable_prices_are_dropped(self, csv_file):
        load(csv_file([row(odds=("0.5", "", "abc"))]), division="I1")
        assert not Odds.objects.filter(provider_espn_id="fd-b365", market="1x2").exists()

    def test_unknown_division_gets_a_generated_slug(self, csv_file):
        load(csv_file([row(division="ZZ9")]), division="ZZ9")
        assert League.objects.filter(slug="fd.zz9").exists()

    def test_empty_selection_raises(self, csv_file):
        with pytest.raises(CommandError):
            load(csv_file([row(division="I1")]), division="E0")

    def test_missing_file_raises(self):
        with pytest.raises(CommandError):
            load("/nonexistent/matches.csv", division="I1")

    def test_bad_date_flag_raises(self, csv_file):
        with pytest.raises(CommandError):
            load(csv_file([row()]), division="I1", date_from="17-08-2024")


class TestMarketProbabilities:
    def test_devigs_the_average_provider(self):
        prices = [
            PricedSelection("1x2", "home", "", 2.0, "fd-b365"),
            PricedSelection("1x2", "draw", "", 4.0, "fd-b365"),
            PricedSelection("1x2", "away", "", 4.0, "fd-b365"),
        ]
        probabilities = _market_probabilities(prices)

        assert sum(probabilities.values()) == pytest.approx(1.0)
        assert probabilities["home"] == pytest.approx(0.5)

    def test_prefers_the_average_over_the_maximum(self):
        prices = [
            PricedSelection("1x2", "home", "", 2.0, "fd-b365"),
            PricedSelection("1x2", "draw", "", 4.0, "fd-b365"),
            PricedSelection("1x2", "away", "", 4.0, "fd-b365"),
            PricedSelection("1x2", "home", "", 2.6, "fd-max"),
            PricedSelection("1x2", "draw", "", 4.4, "fd-max"),
            PricedSelection("1x2", "away", "", 4.4, "fd-max"),
        ]
        # Devigging a best-price line would understate the margin and read as an edge.
        assert _market_probabilities(prices)["home"] == pytest.approx(0.5)

    def test_incomplete_market_is_ignored(self):
        prices = [
            PricedSelection("1x2", "home", "", 2.0, "fd-b365"),
            PricedSelection("1x2", "draw", "", 4.0, "fd-b365"),
        ]
        assert _market_probabilities(prices) is None

    def test_totals_do_not_stand_in_for_match_odds(self):
        prices = [
            PricedSelection("totals", "over", "2.5", 1.9, "fd-b365"),
            PricedSelection("totals", "under", "2.5", 1.9, "fd-b365"),
        ]
        assert _market_probabilities(prices) is None

    def test_no_prices_at_all(self):
        assert _market_probabilities([]) is None


def make_record(probabilities, market, actual="home") -> ForecastRecord:
    return ForecastRecord(
        event_espn_id="x",
        date=datetime(2024, 1, 1, tzinfo=UTC),
        home="A",
        away="B",
        probabilities=probabilities,
        actual=actual,
        home_goals=1,
        away_goals=0,
        market_probabilities=market,
    )


class TestMarketComparison:
    CONFIDENT = {"home": 0.7, "draw": 0.2, "away": 0.1}
    VAGUE = {"home": 0.34, "draw": 0.33, "away": 0.33}

    def test_reports_nothing_without_odds(self):
        comparison = backtest._market_comparison([make_record(self.CONFIDENT, None)])
        assert comparison["matches"] == 0
        assert comparison["model"] is None

    def test_scores_both_on_the_same_subset(self):
        records = [
            make_record(self.CONFIDENT, self.VAGUE),
            make_record(self.CONFIDENT, None),
        ]
        comparison = backtest._market_comparison(records)

        assert comparison["matches"] == 1
        assert comparison["model"]["log_loss"] is not None
        assert comparison["market"]["log_loss"] is not None

    def test_detects_when_the_model_is_better(self):
        records = [make_record(self.CONFIDENT, self.VAGUE) for _ in range(5)]
        comparison = backtest._market_comparison(records)

        assert comparison["log_loss_delta"] < 0
        assert comparison["model_beats_market"] is True

    def test_detects_when_the_market_is_better(self):
        records = [make_record(self.VAGUE, self.CONFIDENT) for _ in range(5)]
        comparison = backtest._market_comparison(records)

        assert comparison["log_loss_delta"] > 0
        assert comparison["model_beats_market"] is False

    def test_includes_market_calibration(self):
        records = [make_record(self.CONFIDENT, self.VAGUE) for _ in range(5)]
        buckets = backtest._market_comparison(records)["market_calibration"]

        assert buckets
        assert sum(bucket["predictions"] for bucket in buckets) == 15


@pytest.mark.django_db
class TestBacktestAgainstRealShapedData:
    @pytest.fixture
    def loaded(self, csv_file):
        rows = []
        teams = ["Inter", "Milan", "Roma", "Lazio", "Napoli", "Juventus"]
        day = 1
        for cycle in range(9):
            for index in range(0, len(teams), 2):
                home, away = teams[index], teams[index + 1]
                if cycle % 2:
                    home, away = away, home
                rows.append(
                    row(
                        date=f"2024-{1 + day // 28:02d}-{1 + day % 28:02d}",
                        home=home,
                        away=away,
                        home_goals=str((cycle + index) % 4),
                        away_goals=str((cycle + index + 1) % 3),
                    )
                )
                day += 3
            teams = [teams[0], teams[-1], *teams[1:-1]]
        load(csv_file(rows), division="I1")
        return League.objects.get(slug="ita.1")

    def test_market_comparison_is_populated(self, loaded):
        report = backtest.run(loaded, refit_every=5).to_dict()

        assert report["forecasts"] > 0
        assert report["market"]["matches"] > 0
        assert report["market"]["market"]["log_loss"] > 0
        assert report["market"]["model"]["log_loss"] > 0
        assert isinstance(report["market"]["model_beats_market"], bool)

    def test_command_renders_the_market_section(self, loaded):
        out = StringIO()
        call_command("backtest_model", "ita.1", refit_every=5, stdout=out)
        output = out.getvalue()

        assert "Against the market" in output
        assert "matches compared" in output

    def test_json_output_carries_the_market_block(self, loaded):
        out = StringIO()
        call_command("backtest_model", "ita.1", refit_every=5, json=True, stdout=out)
        payload = json.loads(out.getvalue())

        assert "market" in payload
        assert payload["market"]["matches"] > 0


# --- the original football-data.co.uk layout -------------------------------------
#
# The columns that matter here are the closing ones: a bookmaker abbreviation
# followed by C. They are the only prices in this project that let a bet be
# judged against the line it was struck at rather than against the result.

UK_HEADER = (
    "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,"
    "HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,"
    "B365H,B365D,B365A,PSH,PSD,PSA,MaxH,MaxD,MaxA,AvgH,AvgD,AvgA,"
    "B365CH,B365CD,B365CA,PSCH,PSCD,PSCA,MaxCH,MaxCD,MaxCA,AvgCH,AvgCD,AvgCA,"
    "B365>2.5,B365<2.5,PC>2.5,PC<2.5"
)

# The pre-2019 layout: no closing prices at all, and the aggregates are Betbrain's.
UK_HEADER_OLD = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,"
    "HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,"
    "B365H,B365D,B365A,PSH,PSD,PSA,BbMxH,BbAvH,BbMxD,BbAvD,BbMxA,BbAvA"
)


def uk_row(
    *,
    division="E0",
    date="10/08/2018",
    time="20:00",
    home="Man United",
    away="Leicester",
    home_goals="2",
    away_goals="1",
    opening=("1.58", "3.93", "7.50"),
    closing=("1.55", "4.07", "7.69"),
) -> str:
    result = (
        "H"
        if int(home_goals) > int(away_goals)
        else "A"
        if int(away_goals) > int(home_goals)
        else "D"
    )
    return ",".join(
        [
            division,
            date,
            time,
            home,
            away,
            home_goals,
            away_goals,
            result,
            "1",
            "0",
            "H",
            "A Marriner",
            "8",
            "13",
            "6",
            "4",
            "11",
            "8",
            "2",
            "5",
            "2",
            "1",
            "0",
            "0",
            "1.57",
            "3.90",
            "7.50",  # B365 pre-match
            *opening,  # PS pre-match
            "1.60",
            "4.20",
            "8.05",  # Max pre-match
            "1.56",
            "3.92",
            "7.06",  # Avg pre-match
            "1.55",
            "4.00",
            "7.60",  # B365 closing
            *closing,  # PS closing
            "1.58",
            "4.20",
            "8.10",  # Max closing
            "1.54",
            "4.05",
            "7.40",  # Avg closing
            "1.80",
            "2.05",  # B365 over/under 2.5
            "1.85",
            "2.10",  # Pinnacle closing over/under 2.5
        ]
    )


def uk_old_row(*, division="E0", date="17/08/13", home="Arsenal", away="Villa") -> str:
    return ",".join(
        [
            division,
            date,
            home,
            away,
            "1",
            "3",
            "A",
            "1",
            "1",
            "D",
            "A Taylor",
            "20",
            "8",
            "6",
            "4",
            "12",
            "9",
            "9",
            "2",
            "1",
            "2",
            "0",
            "0",
            "1.30",
            "5.50",
            "12.0",  # B365
            "1.32",
            "5.60",
            "12.5",  # PS
            "1.35",
            "1.31",
            "5.80",
            "5.55",
            "13.0",
            "12.2",  # BbMx/BbAv interleaved H,D,A
        ]
    )


@pytest.fixture
def uk_csv(tmp_path):
    def write(rows: list[str], header: str = UK_HEADER) -> str:
        path = tmp_path / "E0.csv"
        path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
        return str(path)

    return write


class TestSchemaDetection:
    def test_original_layout_is_recognised(self):
        schema = detect_schema(UK_HEADER.split(","))

        assert schema.name == "football-data.co.uk"
        assert schema.has_closing_prices

    def test_derived_layout_is_recognised(self):
        schema = detect_schema(HEADER.split(","))

        assert schema.name == "club-football-match-data"
        assert not schema.has_closing_prices

    def test_an_unknown_layout_is_refused_rather_than_half_read(self):
        with pytest.raises(CommandError, match="Unrecognised CSV layout"):
            detect_schema(["kickoff", "team_a", "team_b", "score"])


@pytest.mark.django_db
class TestOriginalFootballDataLayout:
    def test_loads_results_from_the_original_columns(self, uk_csv):
        load(uk_csv([uk_row()]), division="E0")

        event = Event.objects.get()
        assert event.status == Event.STATUS_FINAL
        assert event.date.year == 2018 and event.date.hour == 20
        home = event.competitors.get(home_away=Competitor.HOME)
        away = event.competitors.get(home_away=Competitor.AWAY)
        assert (home.score_int, away.score_int) == (2, 1)
        assert home.winner is True

    def test_stores_opening_and_closing_as_separate_providers(self, uk_csv):
        """A closing price is not one anybody could have taken at forecast time.

        Keeping them apart is what stops hindsight being settled as an edge.
        """
        load(uk_csv([uk_row()]), division="E0")
        event = Event.objects.get()

        opening = {
            o.selection: o.decimal_odds
            for o in event.odds.filter(provider_espn_id="fd-ps", market="1x2")
        }
        closing = {
            o.selection: o.decimal_odds
            for o in event.odds.filter(provider_espn_id="fd-psc", market="1x2")
        }

        assert opening == {"home": 1.58, "draw": 3.93, "away": 7.50}
        assert closing == {"home": 1.55, "draw": 4.07, "away": 7.69}

    def test_every_price_series_in_the_file_is_stored(self, uk_csv):
        load(uk_csv([uk_row()]), division="E0")

        stored = set(Odds.objects.values_list("provider_espn_id", flat=True))

        assert stored == {
            "fd-b365",
            "fd-b365c",
            "fd-ps",
            "fd-psc",
            "fd-max",
            "fd-maxc",
            "fd-avg",
            "fd-avgc",
        }

    def test_reports_that_closing_prices_are_available(self, uk_csv):
        output = load(uk_csv([uk_row()]), division="E0")

        assert "Layout detected: football-data.co.uk" in output
        assert "closing-line value can be measured" in output

    def test_warns_when_a_file_has_no_closing_prices(self, csv_file):
        output = load(csv_file([row()]), division="I1")

        assert "not a closing-line-value calculation" in output

    def test_keeps_match_statistics_and_the_referee(self, uk_csv):
        load(uk_csv([uk_row()]), division="E0")

        event = Event.objects.get()
        stats = {s["name"]: s["value"] for s in event.competitors.get(home_away="home").statistics}
        assert stats == {
            "shots": 8,
            "shotsOnTarget": 6,
            "corners": 2,
            "fouls": 11,
            "yellowCards": 2,
            "redCards": 0,
        }
        assert event.raw_data["referee"] == "A Marriner"
        assert event.raw_data["layout"] == "football-data.co.uk"

    def test_stores_totals_from_both_eras_of_column_naming(self, uk_csv):
        load(uk_csv([uk_row()]), division="E0")

        assert event_price("fd-b365", "totals", "over") == 1.80
        assert event_price("fd-psc", "totals", "over") == 1.85


@pytest.mark.django_db
class TestLegacyColumnNames:
    def test_betbrain_aggregates_load_under_the_same_provider(self, uk_csv):
        """The source renamed BbMxH to MaxH in 2019 without changing its meaning.

        Both eras land on fd-max so an analysis spanning the rename does not see
        the aggregate simply vanish halfway through.
        """
        load(uk_csv([uk_old_row()], header=UK_HEADER_OLD), division="E0")

        assert event_price("fd-max", "1x2", "home") == 1.35
        assert event_price("fd-avg", "1x2", "home") == 1.31

    def test_the_column_a_price_came_from_is_recorded(self, uk_csv):
        load(uk_csv([uk_old_row()], header=UK_HEADER_OLD), division="E0")

        odds = Odds.objects.get(provider_espn_id="fd-max", market="1x2", selection="home")
        assert odds.raw_data["column"] == "BbMxH"

    def test_two_digit_years_are_read_as_this_century(self, uk_csv):
        load(uk_csv([uk_old_row(date="17/08/13")], header=UK_HEADER_OLD), division="E0")

        assert Event.objects.get().date.year == 2013

    def test_a_file_with_no_closing_columns_stores_none(self, uk_csv):
        load(uk_csv([uk_old_row()], header=UK_HEADER_OLD), division="E0")

        assert not Odds.objects.filter(provider_espn_id__endswith="c").exists()


@pytest.mark.django_db
class TestLeagueSlugOverride:
    def test_a_load_can_be_directed_at_an_explicit_league(self, uk_csv):
        load(uk_csv([uk_row()]), division="E0", league_slug="eng.1.closing")

        assert League.objects.filter(slug="eng.1.closing").exists()
        assert not League.objects.filter(slug="eng.1").exists()

    def test_lower_english_divisions_have_their_own_slugs(self, uk_csv):
        load(uk_csv([uk_row(division="E2")]), division="E2")

        league = League.objects.get(slug="eng.3")
        assert league.name == "English League One"


def event_price(provider: str, market: str, selection: str) -> float:
    return Odds.objects.get(
        provider_espn_id=provider, market=market, selection=selection
    ).decimal_odds
