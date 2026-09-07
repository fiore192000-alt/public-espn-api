"""Tests for the market whose truth is known.

The generator is only useful if it builds the market it was asked for, so most
of these tests check that property directly: on an efficient market every bet
must be equally bad, and on a biased one the bias must be where it was planted.
A generator that quietly leaves an edge in its "no edge" market would certify
noise as signal for the rest of the project.
"""

import json
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.espn import market_bias, synthetic

SMALL = 400


def spec(**overrides) -> synthetic.MarketSpec:
    base = {"matches": SMALL, "seed": 11}
    base.update(overrides)
    return synthetic.MarketSpec(**base)


# --- the spec -------------------------------------------------------------


def test_a_market_with_no_bias_and_no_longshot_is_efficient():
    assert spec().efficient is True


def test_any_planted_bias_makes_it_inefficient():
    assert spec(longshot=0.1).efficient is False
    assert spec(bias={"home": 0.05}).efficient is False


def test_a_zero_bias_entry_still_counts_as_efficient():
    assert spec(bias={"home": 0.0, "away": 0.0}).efficient is True


def test_the_house_edge_is_the_margin_as_a_share_of_the_book():
    assert spec(margin=0.05).house_edge == pytest.approx(-0.05 / 1.05)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"matches": 0}, "at least one match"),
        ({"margin": -0.01}, "not a bookmaker"),
        ({"concentration": 0.0}, "must be positive"),
        ({"best_price_edge": -0.1}, "cannot be worse"),
        ({"bias": {"corner": 0.1}}, "Unknown selection"),
    ],
)
def test_impossible_specs_are_rejected(overrides, message):
    with pytest.raises(synthetic.SpecError, match=message):
        spec(**overrides)


# --- the generator --------------------------------------------------------


def test_a_market_is_reproducible_from_its_seed():
    first = synthetic.simulate(spec())
    second = synthetic.simulate(spec())
    assert [m.priced.actual for m in first] == [m.priced.actual for m in second]
    assert [m.truth for m in first] == [m.truth for m in second]


def test_a_different_seed_gives_a_different_market():
    first = synthetic.simulate(spec(seed=1))
    second = synthetic.simulate(spec(seed=2))
    assert [m.priced.actual for m in first] != [m.priced.actual for m in second]


def test_every_match_has_a_full_book_and_a_result():
    for match in synthetic.simulate(spec()):
        assert set(match.priced.book) == set(synthetic.SELECTIONS)
        assert match.priced.actual in synthetic.SELECTIONS
        assert all(price > 1.0 for price in match.priced.book.values())


def test_truth_and_belief_are_probability_distributions():
    for match in synthetic.simulate(spec()):
        assert sum(match.truth.values()) == pytest.approx(1.0)
        assert sum(match.belief.values()) == pytest.approx(1.0)
        assert all(value >= synthetic.MINIMUM_PROBABILITY * 0.99 for value in match.truth.values())


def test_the_book_carries_exactly_the_requested_margin():
    for match in synthetic.simulate(spec(margin=0.07)):
        overround = sum(1.0 / price for price in match.priced.book.values())
        assert overround == pytest.approx(1.07)


def test_an_efficient_book_believes_the_truth():
    for match in synthetic.simulate(spec()):
        assert match.belief == pytest.approx(match.truth)


# --- the property the whole module rests on -------------------------------


def test_on_an_efficient_market_every_single_bet_is_equally_bad():
    """If this fails, the 'no edge' market has an edge and the null is void."""
    market = spec(margin=0.05)
    sample = synthetic.simulate(market)
    for match in sample:
        for selection in synthetic.SELECTIONS:
            assert match.expected_value(selection) == pytest.approx(market.house_edge)


def test_the_realised_house_edge_matches_the_theory():
    market = spec(matches=2000, margin=0.06)
    sample = synthetic.simulate(market)
    assert synthetic.realised_house_edge(sample) == pytest.approx(market.house_edge, abs=1e-9)


def test_an_efficient_market_has_no_spread_of_expected_value():
    assert synthetic.edge_spread(synthetic.simulate(spec())) == pytest.approx(0.0, abs=1e-9)


def test_a_planted_longshot_bias_creates_a_real_spread():
    biased = synthetic.edge_spread(synthetic.simulate(spec(longshot=0.10)))
    assert biased > 0.01


def test_the_longshot_bias_favours_favourites():
    """The planted effect must point the way the literature says it does."""
    sample = synthetic.simulate(spec(matches=1500, longshot=0.10))
    short, long = [], []
    for match in sample:
        for selection in synthetic.SELECTIONS:
            (short if match.priced.book[selection] < 2.5 else long).append(
                match.expected_value(selection)
            )
    assert sum(short) / len(short) > sum(long) / len(long)


def test_a_selection_bias_lands_on_the_selection_it_was_planted_on():
    sample = synthetic.simulate(spec(matches=1500, bias={"home": 0.12}))
    # The book over-rates the home side, so backing it must be worse than average.
    home = [m.expected_value("home") for m in sample]
    away = [m.expected_value("away") for m in sample]
    assert sum(home) / len(home) < sum(away) / len(away)


def test_the_best_price_is_better_than_the_book_and_only_by_what_was_asked():
    sample = synthetic.simulate(spec(best_price_edge=0.02))
    for match in sample:
        assert match.priced.best is not None
        for selection in synthetic.SELECTIONS:
            assert match.priced.best[selection] == pytest.approx(
                match.priced.book[selection] * 1.02
            )


def test_without_a_best_price_edge_there_is_no_best_line():
    assert all(match.priced.best is None for match in synthetic.simulate(spec()))


def test_the_analyst_only_ever_sees_the_priced_view():
    sample = synthetic.simulate(spec())
    exposed = synthetic.priced(sample)
    assert len(exposed) == len(sample)
    assert all(isinstance(match, market_bias.PricedMatch) for match in exposed)
    # PricedMatch carries no field that could leak the truth.
    assert not hasattr(exposed[0], "truth")


def test_the_devigged_fair_line_recovers_the_belief_closely():
    """The analyst's reconstruction should land near what the book believed."""
    sample = synthetic.simulate(spec(matches=800))
    errors = [
        abs(match.priced.fair[selection] - match.belief[selection])
        for match in sample
        for selection in synthetic.SELECTIONS
    ]
    assert sum(errors) / len(errors) < 0.02


# --- the search -----------------------------------------------------------


def test_the_sweep_returns_the_rule_the_real_search_would_carry():
    """Ranked by t-statistic, exactly as market_bias.Discovery.ranked does."""
    sample = synthetic.simulate(spec(matches=3000))
    winner = synthetic.sweep(sample)
    assert winner is not None
    rules = market_bias.price_rules() + list(market_bias.selection_rules(synthetic.SELECTIONS))
    scores = market_bias.score(synthetic.priced(sample), rules)
    eligible = [s for s in scores if s.at_best.bets >= market_bias.MINIMUM_BETS]
    assert winner.at_best.t_stat == max(s.at_best.t_stat for s in eligible)


def test_a_market_too_small_for_the_minimum_has_no_winner():
    assert synthetic.sweep(synthetic.simulate(spec(matches=50))) is None


def test_the_null_search_finds_something_positive_even_though_nothing_is_there():
    """The point of the whole module, as a test."""
    outcome = synthetic.search_under_the_null(spec(matches=1200), trials=12)
    assert outcome.trials > 0
    assert outcome.mean_yield > 0.0
    assert outcome.yield_at(0.95) > outcome.yield_at(0.50)


def test_the_null_needs_trials():
    with pytest.raises(synthetic.SpecError, match="at least one trial"):
        synthetic.search_under_the_null(spec(), trials=0)


def test_an_empty_outcome_reports_zeroes_rather_than_dividing_by_zero():
    empty = synthetic.SearchOutcome(trials=0, matches=0, rules=0, yields=[], t_stats=[], labels=[])
    assert empty.mean_yield == 0.0
    assert empty.mean_t == 0.0
    assert empty.yield_at(0.95) == 0.0
    assert empty.clears_conventional_t == 0.0
    assert empty.favourite_share == 0.0
    assert empty.beats(0.05) == 1.0
    assert empty.beats_t(2.0) == 1.0


def test_a_p_value_falls_as_the_claimed_result_grows():
    outcome = synthetic.SearchOutcome(
        trials=4,
        matches=0,
        rules=0,
        yields=[0.01, 0.03, 0.05, 0.07],
        t_stats=[0.5, 1.0, 1.5, 2.5],
        labels=["price 1.00-1.50"] * 4,
    )
    assert outcome.beats(0.00) == 1.0
    assert outcome.beats(0.04) == 0.5
    assert outcome.beats(0.10) == 0.0
    assert outcome.beats_t(1.2) == 0.5


def test_percentiles_are_ordered():
    outcome = synthetic.SearchOutcome(
        trials=5,
        matches=0,
        rules=0,
        yields=[0.05, 0.01, 0.09, 0.03, 0.07],
        t_stats=[1, 2, 3, 4, 5],
        labels=[],
    )
    assert outcome.yield_at(0.0) <= outcome.yield_at(0.5) <= outcome.yield_at(0.99)
    assert outcome.t_at(0.99) == 5


def test_the_share_clearing_the_conventional_bar_is_counted():
    outcome = synthetic.SearchOutcome(
        trials=4,
        matches=0,
        rules=0,
        yields=[0.0] * 4,
        t_stats=[0.5, 1.9, 2.1, 3.0],
        labels=[],
    )
    assert outcome.clears_conventional_t == 0.5


def test_the_positive_control_fires_more_often_than_the_negative_one():
    """A planted bias must be found more often than noise clears the same bar."""
    null = synthetic.search_under_the_null(spec(matches=1200), trials=12)
    strong = synthetic.power_against(spec(matches=1200, longshot=0.30), null, trials=12)
    assert strong > 0.05


# --- the command ----------------------------------------------------------


def run(*args, **options) -> str:
    out = StringIO()
    call_command("verify_search", *args, stdout=out, **options)
    return out.getvalue()


def test_the_command_reports_both_controls():
    output = run("--matches", "600", "--trials", "4")
    assert "NEGATIVE CONTROL" in output
    assert "POSITIVE CONTROL" in output


def test_the_command_emits_json():
    report = json.loads(run("--matches", "600", "--trials", "4", "--json"))
    assert report["market"]["efficient"] is True
    assert report["market"]["house_edge_realised"] == pytest.approx(
        report["market"]["house_edge_theoretical"], abs=1e-6
    )
    assert report["null"]["trials"] > 0


def test_the_positive_control_can_be_skipped():
    report = json.loads(run("--matches", "600", "--trials", "3", "--longshot", "0", "--json"))
    assert report["positive_control"] is None


def test_an_observed_result_gets_a_p_value():
    report = json.loads(
        run(
            "--matches",
            "600",
            "--trials",
            "4",
            "--observed",
            "0.0497",
            "--observed-t",
            "2.42",
            "--json",
        )
    )
    assert 0.0 <= report["observed"]["p_value_against_null"] <= 1.0
    assert 0.0 <= report["observed_t"]["p_value_against_null"] <= 1.0


def test_zero_trials_is_refused():
    with pytest.raises(CommandError, match="--trials must be positive"):
        run("--matches", "600", "--trials", "0")


def test_an_impossible_market_is_refused():
    with pytest.raises(CommandError, match="at least one match"):
        run("--matches", "0", "--trials", "2")
