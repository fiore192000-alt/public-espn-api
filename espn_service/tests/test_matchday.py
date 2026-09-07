"""Tests for settling a day's card under fixed staking rules.

The interesting failures here are silent ones: a rule that peeks at the result,
a multiple priced as if its margin were charged once, a "luck" distribution whose
probabilities do not sum to one. Each has a test.
"""

import json
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.espn import matchday

BANKROLL = 10000.0


def fixture(
    home="Home",
    away="Away",
    prices=(2.00, 3.40, 4.00),
    model=(0.50, 0.30, 0.20),
    result=None,
):
    keys = matchday.SELECTIONS
    return matchday.Fixture(
        home=home,
        away=away,
        prices=dict(zip(keys, prices, strict=True)),
        model=dict(zip(keys, model, strict=True)),
        result=result,
    )


# --- the fixture itself ---------------------------------------------------


def test_a_fixture_normalises_its_model_probabilities():
    match = fixture(model=(1.0, 1.0, 2.0))
    assert match.model == pytest.approx({"home": 0.25, "draw": 0.25, "away": 0.5})


def test_a_fixture_without_a_full_book_is_rejected():
    with pytest.raises(matchday.CardError, match="no price for away"):
        matchday.Fixture(
            home="Home",
            away="Away",
            prices={"home": 2.0, "draw": 3.4},
            model={"home": 0.5, "draw": 0.3, "away": 0.2},
        )


def test_an_impossible_result_is_rejected():
    with pytest.raises(matchday.CardError, match="not a 1X2 result"):
        fixture(result="penalties")


def test_model_probabilities_that_sum_to_nothing_are_rejected():
    with pytest.raises(matchday.CardError, match="sum to something positive"):
        fixture(model=(0.0, 0.0, 0.0))


def test_the_margin_is_the_book_over_one_hundred_percent():
    match = fixture(prices=(2.00, 4.00, 4.00))
    assert match.overround == pytest.approx(1.0)
    assert match.margin == pytest.approx(0.0)


def test_fair_probabilities_sum_to_one_and_shrink_the_book():
    match = fixture(prices=(1.95, 3.60, 4.00))
    assert sum(match.fair.values()) == pytest.approx(1.0)
    # Devigging can only move an implied probability down, never up.
    for selection in matchday.SELECTIONS:
        assert match.fair[selection] < 1.0 / match.price(selection)


def test_edge_is_the_model_minus_the_devigged_price():
    match = fixture()
    assert match.edge("home") == pytest.approx(match.model["home"] - match.fair["home"])


def test_expected_value_is_zero_at_the_fair_price():
    match = fixture(prices=(2.00, 4.00, 4.00), model=(0.50, 0.25, 0.25))
    assert match.expected_value("home") == pytest.approx(0.0)


def test_probabilities_needs_a_known_source():
    with pytest.raises(matchday.CardError, match="not a probability source"):
        fixture().probabilities("vibes")


# --- bets -----------------------------------------------------------------


def test_a_multiple_multiplies_price_and_probability():
    first, second = fixture(home="A"), fixture(home="B")
    bet = matchday.Bet("test", (matchday.Leg(first, "home"), matchday.Leg(second, "home")), 100.0)
    assert bet.price == pytest.approx(4.0)
    assert bet.model_probability == pytest.approx(0.25)


def test_a_multiple_pays_the_margin_twice():
    """The whole case against accumulators, as arithmetic."""
    first, second = fixture(home="A"), fixture(home="B")
    single = matchday.Bet("test", (matchday.Leg(first, "home"),), 100.0)
    double = matchday.Bet(
        "test", (matchday.Leg(first, "home"), matchday.Leg(second, "home")), 100.0
    )
    assert double.market_expected_value < single.market_expected_value


def test_two_legs_from_one_match_are_rejected():
    match = fixture()
    with pytest.raises(matchday.CardError, match="two legs from the same match"):
        matchday.Bet("test", (matchday.Leg(match, "home"), matchday.Leg(match, "draw")), 10.0)


def test_a_bet_needs_a_leg():
    with pytest.raises(matchday.CardError, match="at least one leg"):
        matchday.Bet("test", (), 10.0)


def test_an_unsettled_bet_returns_and_profits_nothing():
    bet = matchday.Bet("test", (matchday.Leg(fixture(), "home"),), 100.0)
    assert bet.settled is False
    assert bet.won is None
    assert bet.returned == 0.0
    assert bet.profit == 0.0


def test_a_winning_bet_returns_stake_times_price():
    bet = matchday.Bet("test", (matchday.Leg(fixture(result="home"), "home"),), 100.0)
    assert bet.won is True
    assert bet.returned == pytest.approx(200.0)
    assert bet.profit == pytest.approx(100.0)


def test_a_multiple_needs_every_leg():
    won = fixture(home="A", result="home")
    lost = fixture(home="B", result="away")
    bet = matchday.Bet("test", (matchday.Leg(won, "home"), matchday.Leg(lost, "home")), 100.0)
    assert bet.won is False
    assert bet.profit == pytest.approx(-100.0)


def test_profit_if_agrees_with_the_settled_profit():
    match = fixture(result="draw")
    bet = matchday.Bet("test", (matchday.Leg(match, "home"),), 100.0)
    assert bet.profit_if({match.name: "draw"}) == pytest.approx(bet.profit)
    assert bet.profit_if({match.name: "home"}) == pytest.approx(100.0)


# --- the rules ------------------------------------------------------------


def test_no_rule_reads_the_result():
    """The rules must stake identically whether or not the card has settled."""
    blind = [fixture(home="A", model=(0.60, 0.25, 0.15)), fixture(home="B")]
    settled = [
        fixture(home="A", model=(0.60, 0.25, 0.15), result="away"),
        fixture(home="B", result="away"),
    ]
    for name in matchday.RULES:
        before = matchday.settle(blind, name, BANKROLL)
        after = matchday.settle(settled, name, BANKROLL)
        assert [(bet.label, bet.stake) for bet in before.bets] == [
            (bet.label, bet.stake) for bet in after.bets
        ], name


def test_the_value_rule_ignores_selections_below_the_threshold():
    # A model that agrees with the book has no edge anywhere.
    match = fixture(prices=(2.00, 4.00, 4.00), model=(0.50, 0.25, 0.25))
    assert matchday.settle([match], "value", BANKROLL).bets == []


def test_the_value_rule_stakes_quarter_kelly():
    match = fixture(prices=(3.00, 3.40, 2.40), model=(0.40, 0.35, 0.25))
    bets = matchday.settle([match], "value", BANKROLL).bets
    home = next(bet for bet in bets if bet.legs[0].selection == "home")
    full = (0.40 * 3.00 - 1.0) / 2.00
    assert home.stake == pytest.approx(round(BANKROLL * full * 0.25, 2))


def test_the_value_rule_respects_the_cap():
    """A huge claimed edge must not become a huge stake."""
    match = fixture(prices=(3.00, 3.40, 2.40), model=(0.80, 0.15, 0.05))
    [bet] = matchday.settle([match], "value", BANKROLL).bets
    assert bet.stake == pytest.approx(BANKROLL * matchday.MAX_STAKE_FRACTION)


def test_the_value_rule_takes_the_biggest_edge_first():
    small = fixture(home="A", prices=(2.50, 3.40, 3.00), model=(0.45, 0.30, 0.25))
    large = fixture(home="B", prices=(2.50, 3.40, 3.00), model=(0.60, 0.25, 0.15))
    bets = matchday.settle([small, large], "value", BANKROLL).bets
    edges = [bet.legs[0].edge for bet in bets]
    assert edges == sorted(edges, reverse=True)


def test_best_edge_picks_exactly_one_selection():
    card = [fixture(home="A", model=(0.60, 0.25, 0.15)), fixture(home="B")]
    ledger = matchday.settle(card, "best-edge", BANKROLL)
    assert len(ledger.bets) == 1
    assert ledger.bets[0].legs[0].fixture.home == "A"


def test_shortest_price_ignores_the_model_entirely():
    short = fixture(home="A", prices=(1.20, 6.00, 12.00), model=(0.10, 0.30, 0.60))
    long = fixture(home="B", prices=(2.00, 3.40, 4.00), model=(0.90, 0.05, 0.05))
    [bet] = matchday.settle([short, long], "shortest-price", BANKROLL).bets
    assert bet.legs[0].fixture.home == "A"
    assert bet.price == pytest.approx(1.20)


def test_market_favourites_bets_every_match_once():
    card = [fixture(home="A"), fixture(home="B"), fixture(home="C")]
    ledger = matchday.settle(card, "market-favourites", BANKROLL)
    assert len(ledger.bets) == 3
    assert all(len(bet.legs) == 1 for bet in ledger.bets)


def test_the_accumulator_needs_two_distinct_matches():
    assert matchday.settle([fixture()], "accumulator", BANKROLL).bets == []
    card = [fixture(home="A"), fixture(home="B")]
    [bet] = matchday.settle(card, "accumulator", BANKROLL).bets
    assert len({leg.fixture.name for leg in bet.legs}) == 2


def test_an_unknown_rule_is_rejected():
    with pytest.raises(matchday.CardError, match="not a known rule"):
        matchday.settle([fixture()], "martingale", BANKROLL)


def test_every_rule_is_described():
    assert set(matchday.DESCRIPTIONS) == set(matchday.RULES)


def test_settle_all_covers_every_rule():
    ledgers = matchday.settle_all([fixture(home="A"), fixture(home="B")], BANKROLL)
    assert {ledger.rule for ledger in ledgers} == set(matchday.RULES)


# --- the ledger -----------------------------------------------------------


def test_the_ledger_adds_up():
    won = fixture(home="A", result="home")
    lost = fixture(home="B", result="away")
    ledger = matchday.Ledger(
        rule="test",
        bankroll=BANKROLL,
        bets=[
            matchday.Bet("test", (matchday.Leg(won, "home"),), 100.0),
            matchday.Bet("test", (matchday.Leg(lost, "home"),), 100.0),
        ],
    )
    assert ledger.staked == pytest.approx(200.0)
    assert ledger.returned == pytest.approx(200.0)
    assert ledger.profit == pytest.approx(0.0)
    assert ledger.closing == pytest.approx(BANKROLL)
    assert ledger.roi == pytest.approx(0.0)
    assert ledger.exposure == pytest.approx(0.02)


def test_roi_counts_only_settled_stakes():
    settled = fixture(home="A", result="home")
    pending = fixture(home="B")
    ledger = matchday.Ledger(
        rule="test",
        bankroll=BANKROLL,
        bets=[
            matchday.Bet("test", (matchday.Leg(settled, "home"),), 100.0),
            matchday.Bet("test", (matchday.Leg(pending, "home"),), 100.0),
        ],
    )
    assert ledger.settled_stake == pytest.approx(100.0)
    assert ledger.roi == pytest.approx(1.0)


def test_an_empty_ledger_has_no_roi():
    assert matchday.Ledger(rule="test", bankroll=BANKROLL).roi == 0.0


def test_the_house_expects_to_keep_the_margin():
    """Under the book's own probabilities every stake loses money on average."""
    card = [fixture(home="A", prices=(1.95, 3.60, 4.00))]
    for name in matchday.RULES:
        ledger = matchday.settle(card, name, BANKROLL)
        assert ledger.market_expected_profit <= 0.0, name


# --- the luck check -------------------------------------------------------


def test_the_distribution_is_a_probability_distribution():
    card = [fixture(home="A"), fixture(home="B")]
    bets = matchday.settle(card, "market-favourites", BANKROLL).bets
    outcomes = matchday.distribution(bets)
    assert len(outcomes) == 9
    assert sum(probability for _, probability in outcomes) == pytest.approx(1.0)


def test_the_distribution_of_nothing_is_empty():
    assert matchday.distribution([]) == []
    assert matchday.summarise([]) is None


def test_the_market_mean_is_the_book_s_edge():
    card = [fixture(home="A", prices=(1.95, 3.60, 4.00))]
    bets = matchday.settle(card, "market-favourites", BANKROLL).bets
    outcome = matchday.summarise(bets, matchday.MARKET)
    assert outcome.mean == pytest.approx(sum(bet.stake * bet.market_expected_value for bet in bets))
    assert outcome.mean < 0


def test_the_model_mean_is_the_model_s_claimed_edge():
    card = [fixture(home="A", prices=(2.50, 3.40, 3.00), model=(0.60, 0.25, 0.15))]
    bets = matchday.settle(card, "value", BANKROLL).bets
    outcome = matchday.summarise(bets, matchday.MODEL)
    assert outcome.mean == pytest.approx(sum(bet.stake * bet.expected_value for bet in bets))
    assert outcome.mean > 0


def test_the_percentile_places_the_realised_day():
    won = fixture(home="A", prices=(2.00, 4.00, 4.00), result="home")
    lost = fixture(home="B", prices=(2.00, 4.00, 4.00), result="away")
    best = matchday.summarise([matchday.Bet("test", (matchday.Leg(won, "home"),), 100.0)])
    worst = matchday.summarise([matchday.Bet("test", (matchday.Leg(lost, "home"),), 100.0)])
    assert best.percentile > worst.percentile
    assert best.realised == pytest.approx(100.0)
    assert worst.realised == pytest.approx(-100.0)


def test_an_unsettled_day_has_no_realised_outcome():
    outcome = matchday.summarise([matchday.Bet("test", (matchday.Leg(fixture(), "home"),), 100.0)])
    assert outcome.realised is None
    assert outcome.percentile is None
    assert outcome.z is None


def test_z_measures_the_day_in_standard_deviations():
    match = fixture(prices=(2.00, 4.00, 4.00), result="home")
    outcome = matchday.summarise([matchday.Bet("test", (matchday.Leg(match, "home"),), 100.0)])
    assert outcome.z == pytest.approx((outcome.realised - outcome.mean) / outcome.deviation)


def test_a_bigger_card_widens_the_spread_less_than_it_raises_the_stake():
    """Diversification: four flat bets swing less per euro staked than one."""
    one = matchday.settle([fixture(home="A")], "market-favourites", BANKROLL)
    four = matchday.settle([fixture(home=name) for name in "ABCD"], "market-favourites", BANKROLL)
    single = matchday.summarise(one.bets)
    spread = matchday.summarise(four.bets)
    assert spread.deviation / four.staked < single.deviation / one.staked
    assert spread.deviation > single.deviation


# --- loading a card -------------------------------------------------------


def test_a_card_round_trips_from_json():
    document = {
        "fixtures": [
            {
                "home": "A",
                "away": "B",
                "kickoff": "2026-09-06T15:00",
                "book": "test",
                "prices": {"home": 2.0, "draw": 3.4, "away": 4.0},
                "model": {"home": 0.5, "draw": 0.3, "away": 0.2},
                "result": "draw",
            }
        ]
    }
    [match] = matchday.card_from_dict(document)
    assert match.name == "A-B"
    assert match.result == "draw"
    assert match.book == "test"


def test_an_empty_card_is_rejected():
    with pytest.raises(matchday.CardError, match="non-empty"):
        matchday.card_from_dict({"fixtures": []})


def test_a_malformed_fixture_is_rejected():
    with pytest.raises(matchday.CardError, match="Unusable fixture entry"):
        matchday.fixture_from_dict({"home": "A", "away": "B"})


# --- the command ----------------------------------------------------------


def run(*args, **options) -> str:
    out = StringIO()
    call_command("simulate_matchday", *args, stdout=out, **options)
    return out.getvalue()


def test_the_command_settles_the_shipped_card():
    output = run("2026-09-06-serie-a")
    assert "Bologna-Sassuolo" in output
    assert "value" in output
    assert "One card is one draw." in output


def test_the_command_emits_json():
    report = json.loads(run("2026-09-06-serie-a", "--json"))
    assert report["bankroll"] == pytest.approx(10000.0)
    assert len(report["fixtures"]) == 4
    assert {entry["rule"] for entry in report["rules"]} == set(matchday.RULES)


def test_every_shipped_fixture_carries_a_result_and_a_full_book():
    report = json.loads(run("2026-09-06-serie-a", "--json"))
    for entry in report["fixtures"]:
        assert entry["result"] in matchday.SELECTIONS
        assert sum(entry["fair"].values()) == pytest.approx(1.0)
        assert 0.0 < entry["margin"] < 0.12


def test_the_shipped_card_loses_money_under_every_rule():
    """Not a claim about the rules — a record of what 6 September actually did."""
    report = json.loads(run("2026-09-06-serie-a", "--json"))
    for entry in report["rules"]:
        assert entry["profit"] < 0, entry["rule"]
        assert entry["closing"] < 10000.0


def test_the_bankroll_scales_the_stakes_linearly():
    small = json.loads(run("2026-09-06-serie-a", "--json", bankroll=1000.0))
    large = json.loads(run("2026-09-06-serie-a", "--json", bankroll=10000.0))
    for lean, fat in zip(small["rules"], large["rules"], strict=True):
        assert fat["staked"] == pytest.approx(lean["staked"] * 10.0, abs=0.5)


def test_a_single_rule_can_be_selected():
    report = json.loads(run("2026-09-06-serie-a", "--json", rules=["value"]))
    assert [entry["rule"] for entry in report["rules"]] == ["value"]


def test_an_unknown_rule_is_refused():
    with pytest.raises(CommandError, match="Unknown rule"):
        run("2026-09-06-serie-a", rules=["martingale"])


def test_a_missing_card_lists_what_exists():
    with pytest.raises(CommandError, match="Available in research/cards"):
        run("no-such-card")


def test_a_negative_bankroll_is_refused():
    with pytest.raises(CommandError, match="bankroll must be positive"):
        run("2026-09-06-serie-a", bankroll=-1.0)


def test_the_model_source_can_be_chosen():
    report = json.loads(run("2026-09-06-serie-a", "--json", source=matchday.MODEL))
    for entry in report["rules"]:
        if entry["luck"] is not None:
            assert entry["luck"]["source"] == matchday.MODEL


def test_a_card_file_can_be_given_by_path(tmp_path):
    path = tmp_path / "card.json"
    path.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "home": "A",
                        "away": "B",
                        "prices": {"home": 2.0, "draw": 3.4, "away": 4.0},
                        "model": {"home": 0.6, "draw": 0.25, "away": 0.15},
                        "result": "home",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert "A-B" in run(str(path))


def test_an_unreadable_card_is_refused(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CommandError, match="Cannot read"):
        run(str(path))
