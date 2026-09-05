"""Tests for the market-bias search and the gates that stop it overclaiming.

The gates are tested in both directions on purpose. A search that can only ever
say "no edge" is not a search, it is a slogan — so a fabricated market with a
genuine, large, consistent bias has to come back ESTABLISHED, and a fairly priced
one has to come back with nothing.
"""

from datetime import UTC, datetime
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.espn import devig, market_bias
from apps.espn.market_bias import (
    ESTABLISHED,
    NOT_ESTABLISHED,
    REJECTED,
    PricedMatch,
    Rule,
    Series,
    Verdict,
)
from apps.espn.market_structure import Returns

HOME, DRAW, AWAY = "home", "draw", "away"
SELECTIONS = (AWAY, DRAW, HOME)

# One selection per band, so a bucket rule and a selection rule pick out the same
# bets and the arithmetic below stays checkable by hand.
BOOK = {HOME: 1.80, DRAW: 3.60, AWAY: 4.50}
BEST = {HOME: 1.85, DRAW: 3.70, AWAY: 4.70}


def priced(actual: str, *, book=BOOK, best=BEST, year: int = 2010) -> PricedMatch:
    return PricedMatch(
        date=datetime(year, 5, 1, tzinfo=UTC),
        book=book,
        best=best,
        actual=actual,
        fair=devig.remove_margin(book, devig.SHIN),
    )


def sequence(count: int, pattern: str, **kwargs) -> list[PricedMatch]:
    """Deterministic outcomes cycling through ``pattern``.

    Randomness would make a gate's verdict depend on a seed, which is exactly the
    kind of fragility these tests exist to catch elsewhere.
    """
    letters = {"h": HOME, "d": DRAW, "a": AWAY}
    return [priced(letters[pattern[index % len(pattern)]], **kwargs) for index in range(count)]


def returns_of(profits: list[float]) -> Returns:
    """A Returns carrying exactly these profits per unit staked."""
    accumulated = Returns()
    for profit in profits:
        accumulated.add_portfolio(returned=1.0 + profit, staked=1.0)
    return accumulated


def verdict_of(
    profits: list[float],
    *,
    sets: list[list[float]] | None = None,
    book_profits: list[float] | None = None,
) -> Verdict:
    rule = Rule(kind=market_bias.BUCKET_RULE, key=(1.5, 2.0), label="price 1.50-2.00")
    pooled = market_bias.RuleScore(
        rule=rule,
        at_book=returns_of(book_profits if book_profits is not None else profits),
        at_best=returns_of(profits),
    )
    return Verdict(
        rule=rule,
        discovery_t=3.0,
        sets=[
            market_bias.ValidationSet(
                name=f"set {index}",
                score=market_bias.RuleScore(
                    rule=rule, at_book=returns_of(chunk), at_best=returns_of(chunk)
                ),
            )
            for index, chunk in enumerate(sets or [profits])
        ],
        pooled=pooled,
    )


class TestSeries:
    def test_mean_and_standard_error(self):
        series = Series()
        for value in (1.0, 2.0, 3.0, 4.0):
            series.add(value)

        assert series.mean == pytest.approx(2.5)
        # sd = 1.2910, se = sd / sqrt(4)
        assert series.stderr == pytest.approx(0.6455, abs=1e-4)
        assert series.t_stat == pytest.approx(3.873, abs=1e-3)

    def test_a_single_observation_has_no_spread_to_report(self):
        series = Series()
        series.add(1.0)

        assert series.stderr == 0.0
        assert series.t_stat == 0.0


class TestRules:
    def test_a_bucket_rule_picks_every_selection_in_the_band(self):
        rule = Rule(kind=market_bias.BUCKET_RULE, key=(3.0, 5.0), label="band")

        legs = rule.legs(priced(HOME), use_best=False)

        assert sorted(legs) == [(AWAY, 4.50), (DRAW, 3.60)]

    def test_a_selection_rule_picks_one_side(self):
        rule = Rule(kind=market_bias.SELECTION_RULE, key=HOME, label="home")

        assert rule.legs(priced(DRAW), use_best=False) == [(HOME, 1.80)]

    def test_the_best_price_is_used_when_asked_for(self):
        rule = Rule(kind=market_bias.SELECTION_RULE, key=HOME, label="home")

        assert rule.legs(priced(HOME), use_best=True) == [(HOME, 1.85)]

    def test_a_match_with_no_best_line_falls_back_to_the_book(self):
        rule = Rule(kind=market_bias.SELECTION_RULE, key=HOME, label="home")

        assert rule.legs(priced(HOME, best=None), use_best=True) == [(HOME, 1.80)]

    def test_price_rules_cover_every_band(self):
        assert len(market_bias.price_rules()) == len(market_bias.PRICE_BUCKETS)
        assert market_bias.price_rules()[-1].label.endswith("10.00+")


class TestSettlement:
    def test_a_winning_single_leg_returns_the_price(self):
        result = market_bias.settle(
            [priced(HOME)],
            Rule(kind=market_bias.SELECTION_RULE, key=HOME, label="home"),
            use_best=False,
        )

        assert result.bets == 1
        assert result.mean == pytest.approx(0.80)

    def test_a_losing_single_leg_loses_the_stake(self):
        result = market_bias.settle(
            [priced(AWAY)],
            Rule(kind=market_bias.SELECTION_RULE, key=HOME, label="home"),
            use_best=False,
        )

        assert result.mean == pytest.approx(-1.0)

    def test_two_legs_of_one_match_are_a_single_observation(self):
        """Backing two outcomes of the same fixture is one dependent result.

        Counting them as two independent bets would shrink the standard error by
        roughly the square root of two and make any yield look surer than it is.
        """
        rule = Rule(kind=market_bias.BUCKET_RULE, key=(3.0, 5.0), label="band")

        result = market_bias.settle([priced(DRAW)], rule, use_best=False)

        assert result.bets == 1
        # Two units staked, 3.60 returned.
        assert result.mean == pytest.approx((3.60 - 2.0) / 2.0)

    def test_a_rule_that_never_fires_settles_nothing(self):
        rule = Rule(kind=market_bias.BUCKET_RULE, key=(20.0, 30.0), label="band")

        assert market_bias.settle(sequence(10, "hda"), rule, use_best=False).bets == 0


class TestCalibration:
    def test_a_market_that_is_right_shows_no_gap(self):
        """Outcomes drawn to match the devigged book leave nothing to find."""
        fair = devig.remove_margin(BOOK, devig.SHIN)
        # 1.80/3.60/4.50 devigs to roughly 53/26/21 percent.
        pattern = "h" * 53 + "d" * 26 + "a" * 21
        bands = {band.label: band for band in market_bias.calibrate(sequence(100, pattern))}

        for label, expected in (
            ("1.50-2.00", fair[HOME]),
            ("3.00-4.00", fair[DRAW]),
            ("4.00-6.00", fair[AWAY]),
        ):
            assert bands[label].observed == pytest.approx(expected, abs=0.02)
            assert abs(bands[label].gap) < 0.02

    def test_an_underpriced_band_shows_a_positive_gap(self):
        bands = {band.label: band for band in market_bias.calibrate(sequence(100, "h" * 8 + "da"))}

        assert bands["1.50-2.00"].observed == pytest.approx(0.80)
        assert bands["1.50-2.00"].gap > 0.20
        assert bands["1.50-2.00"].residual.t_stat > 2.0

    def test_bands_nobody_was_priced_into_are_omitted(self):
        labels = {band.label for band in market_bias.calibrate(sequence(20, "hda"))}

        assert "10.00+" not in labels


class TestSearchBurden:
    def test_both_price_levels_count_as_hypotheses(self):
        """Choosing the better of two settlements of a rule is itself a search."""
        discovery = market_bias.Discovery(
            league="test",
            matches=300,
            scores=market_bias.score(sequence(300, "hda"), market_bias.price_rules()),
            calibration=[],
            minimum_bets=200,
        )

        # Three bands are populated, each settled at a book and at a best price.
        assert discovery.hypotheses == 6
        assert discovery.expected_false_positives == pytest.approx(0.3)

    def test_rules_below_the_floor_are_neither_counted_nor_ranked(self):
        discovery = market_bias.Discovery(
            league="test",
            matches=10,
            scores=market_bias.score(sequence(10, "hda"), market_bias.price_rules()),
            calibration=[],
            minimum_bets=200,
        )

        assert discovery.hypotheses == 0
        assert discovery.ranked() == []


class TestGates:
    def test_a_large_consistent_profit_is_established(self):
        verdict = market_bias.judge(verdict_of([0.9, -1.0] * 150 + [0.9] * 100))

        assert verdict.outcome == ESTABLISHED
        assert verdict.reasons == []

    def test_an_interval_containing_zero_is_not_established(self):
        verdict = market_bias.judge(verdict_of([0.82, -1.0] * 150))

        assert verdict.outcome == NOT_ESTABLISHED
        assert any("contains zero" in reason for reason in verdict.reasons)

    def test_a_reliable_loss_is_rejected(self):
        verdict = market_bias.judge(verdict_of([-1.0, -1.0, 0.8] * 150))

        assert verdict.outcome == REJECTED
        assert any("reliably losing" in reason for reason in verdict.reasons)

    def test_too_small_a_sample_cannot_establish_anything(self):
        verdict = market_bias.judge(verdict_of([5.0] * 10))

        assert verdict.outcome == NOT_ESTABLISHED
        assert any("floor for a claim" in reason for reason in verdict.reasons)

    def test_one_set_carrying_the_others_fails_the_consistency_gate(self):
        """Profitable pooled, but only because a single validation set is huge."""
        winners = [3.0, -1.0] * 150
        losers = [-1.0, -1.0, 0.8] * 60
        verdict = market_bias.judge(
            verdict_of(winners + losers * 3, sets=[winners, losers, losers, losers])
        )

        assert verdict.outcome == NOT_ESTABLISHED
        assert any("only 1 of 4 validation sets" in reason for reason in verdict.reasons)

    def test_an_edge_that_exists_only_at_the_best_price_is_flagged(self):
        verdict = market_bias.judge(
            verdict_of([0.9, -1.0] * 150 + [0.9] * 100, book_profits=[-0.05] * 400)
        )

        assert verdict.outcome == NOT_ESTABLISHED
        assert any("recovered margin" in reason for reason in verdict.reasons)


class TestInvestigation:
    def test_a_real_bias_survives_every_gate(self):
        """The gates have to be able to say yes, or they say nothing at all.

        Home is priced at 1.80 and wins 70% of the time: a bias far larger than
        any real market would leave lying around, and consistent across sets.
        """
        pattern = "hhhhhhhdda"
        report = market_bias.investigate(
            sequence(600, pattern),
            [(f"set {index}", sequence(400, pattern)) for index in range(3)],
            selections=SELECTIONS,
        )

        assert [verdict.rule.label for verdict in report.established]
        established = report.established[0]
        assert established.pooled.at_best.mean > 0.2
        assert established.positive_sets == 3

    def test_a_fairly_priced_market_yields_nothing(self):
        pattern = "h" * 53 + "d" * 26 + "a" * 21
        report = market_bias.investigate(
            sequence(600, pattern),
            [(f"set {index}", sequence(400, pattern)) for index in range(3)],
            selections=SELECTIONS,
        )

        assert report.established == []
        assert report.discovery.hypotheses > 0

    def test_the_report_serialises_for_json_output(self):
        report = market_bias.investigate(
            sequence(600, "hhda"),
            [("later", sequence(400, "hhda"))],
            selections=SELECTIONS,
            discovery_league="test.1",
            provider="book",
            best_provider="best",
        )
        payload = report.to_dict()

        assert payload["discovery"]["league"] == "test.1"
        assert payload["provider"] == "book"
        assert payload["verdicts"]
        assert set(payload["verdicts"][0]) >= {"rule", "sets", "pooled", "outcome", "reasons"}


HEADER = (
    "Division,MatchDate,MatchTime,HomeTeam,AwayTeam,HomeElo,AwayElo,"
    "Form3Home,Form5Home,Form3Away,Form5Away,FTHome,FTAway,FTResult,"
    "HomeShots,AwayShots,HomeTarget,AwayTarget,HomeCorners,AwayCorners,"
    "HomeYellow,AwayYellow,HomeRed,AwayRed,"
    "OddHome,OddDraw,OddAway,MaxHome,MaxDraw,MaxAway,"
    "Over25,Under25,MaxOver25,MaxUnder25"
)


def csv_row(*, date: str, home: str, away: str, home_goals: int, away_goals: int) -> str:
    result = "H" if home_goals > away_goals else "A" if away_goals > home_goals else "D"
    return ",".join(
        [
            "I1",
            date,
            "18:30",
            home,
            away,
            "1900",
            "1600",
            *(["1.0"] * 4),
            str(home_goals),
            str(away_goals),
            result,
            *(["5"] * 10),
            "1.80",
            "3.60",
            "4.50",
            "1.85",
            "3.70",
            "4.70",
            "1.80",
            "2.05",
            "1.85",
            "2.10",
        ]
    )


@pytest.fixture
def league_csv(tmp_path):
    """Two seasons of a small league, so the command has something to split."""
    rows = []
    for index in range(120):
        year = 2016 if index < 80 else 2020
        home_goals, away_goals = (2, 0) if index % 10 < 7 else (1, 1) if index % 10 < 9 else (0, 1)
        rows.append(
            csv_row(
                date=f"{year}-0{index % 9 + 1}-15",
                home=f"Team {index % 8}",
                away=f"Team {(index + 3) % 8}",
                home_goals=home_goals,
                away_goals=away_goals,
            )
        )
    path = tmp_path / "matches.csv"
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return str(path)


@pytest.mark.django_db
class TestFindMarketBiasCommand:
    def run(self, *args, **options) -> str:
        out = StringIO()
        call_command("find_market_bias", *args, stdout=out, **options)
        return out.getvalue()

    def test_reports_the_search_burden_and_a_verdict(self, league_csv):
        call_command("ingest_football_data", league_csv, division="I1", stdout=StringIO())

        output = self.run("ita.1", "--split-year", "2019", "--minimum-bets", "20")

        assert "Market bias search" in output
        assert "hypotheses were eligible to be picked" in output
        assert "the search never touched" in output

    def test_json_output_is_machine_readable(self, league_csv):
        import json

        call_command("ingest_football_data", league_csv, division="I1", stdout=StringIO())

        payload = json.loads(
            self.run("ita.1", "--split-year", "2019", "--minimum-bets", "20", "--json")
        )

        assert payload["discovery"]["league"] == "ita.1"
        assert payload["discovery"]["hypotheses_tested"] > 0

    def test_an_unknown_league_is_named_in_the_error(self, db):
        with pytest.raises(CommandError, match="No league with slug 'nowhere'"):
            self.run("nowhere")

    def test_a_league_with_no_stored_book_says_so(self, league_csv):
        call_command(
            "ingest_football_data", league_csv, division="I1", no_odds=True, stdout=StringIO()
        )

        with pytest.raises(CommandError, match="ingest_football_data"):
            self.run("ita.1")

    def test_refuses_to_conclude_without_held_out_data(self, league_csv):
        call_command("ingest_football_data", league_csv, division="I1", stdout=StringIO())

        with pytest.raises(CommandError, match="cannot conclude anything"):
            self.run("ita.1", "--split-year", "2030", "--minimum-bets", "20")
