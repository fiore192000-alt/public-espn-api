"""Tests for the open-to-close movement bench.

The bench exists to refuse one specific mistake — a feature that contains the
opening price, regressed on a target that also contains it — so most of these
tests are about that refusal rather than about the arithmetic.
"""

import random
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.espn import clv, movement

SELECTIONS = ("home", "draw", "away")


def move(match, selection="home", raw=0.0, opening=0.4, closing=0.4, anchored=False, price=2.5):
    return movement.Movement(
        match=match,
        selection=selection,
        raw=raw,
        opening=opening,
        closing=closing,
        anchored=anchored,
        opening_price=price,
        closing_price=price,
    )


def card(n, slope, noise=0.0, seed=5):
    """n matches whose market moves `slope` times a clean, independent feature."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        for selection in SELECTIONS:
            x = rng.gauss(0.0, 0.05)
            opening = rng.uniform(0.2, 0.6)
            out.append(
                move(
                    f"m{i}",
                    selection,
                    raw=x,
                    opening=opening,
                    closing=opening + slope * x + rng.gauss(0.0, noise),
                )
            )
    return out


# --- the arithmetic -------------------------------------------------------


def test_the_target_is_what_the_market_did():
    assert move("a", opening=0.40, closing=0.46).travelled == pytest.approx(0.06)


def test_a_clean_feature_recovers_its_own_slope():
    got = movement.fit(card(400, slope=0.5), "clean")
    assert got.slope == pytest.approx(0.5, abs=0.02)


def test_a_feature_that_does_nothing_has_a_slope_near_zero():
    got = movement.fit(card(400, slope=0.0, noise=0.01), "flat")
    assert abs(got.slope) < 0.1


def test_an_all_zero_feature_is_reported_rather_than_dividing_by_zero():
    got = movement.fit([move("a", raw=0.0), move("b", raw=0.0)], "zero")
    assert got.slope == 0.0
    assert got.stderr == 0.0
    assert got.t_stat == 0.0


def test_a_fit_needs_observations():
    with pytest.raises(movement.MovementError, match="at least one observation"):
        movement.fit([], "empty")


def test_errors_are_clustered_by_match_not_by_leg():
    """Three legs of one fixture must not count as three observations."""
    legs = [move("m1", s, raw=0.05, opening=0.3, closing=0.34) for s in SELECTIONS] + [
        move("m2", s, raw=-0.05, opening=0.3, closing=0.26) for s in SELECTIONS
    ]
    got = movement.fit(legs, "clustered")
    assert got.observations == 6
    assert got.matches == 2


def test_the_interval_widens_with_the_standard_error():
    got = movement.fit(card(300, slope=0.4, noise=0.02), "noisy")
    low, high = got.interval()
    assert low < got.slope < high
    assert high - low == pytest.approx(2 * movement.CONFIDENCE_Z * got.stderr)


# --- the trap -------------------------------------------------------------


CENTRE = 0.35


def contaminated(n, seed=7):
    """H-0001's shape: a mean-reverting market and a feature that knows nothing.

    The closing pulls back toward a centre, so ``closing - opening`` is a
    function of the opening alone. The candidate's own value is a CONSTANT — it
    carries no information whatsoever — but anchored it becomes
    ``CENTRE - opening`` and regresses on the target beautifully, through the
    shared term and nothing else. Permuting the constant changes nothing, so the
    null recovers the same slope and the anchor share goes to 1.
    """
    rng = random.Random(seed)
    out = []
    for i in range(n):
        for selection in SELECTIONS:
            opening = rng.uniform(0.15, 0.65)
            out.append(
                move(
                    f"m{i}",
                    selection,
                    raw=CENTRE,
                    opening=opening,
                    closing=opening + 0.10 * (CENTRE - opening) + rng.gauss(0.0, 0.004),
                    anchored=True,
                )
            )
    return out


def test_a_contaminated_feature_is_flagged_by_its_anchor_share():
    """A constant, anchored, on a mean-reverting market: pure shared term."""
    got = movement.fit_against_null(contaminated(500), "anchor only")
    assert got.t_stat > 5  # it looks magnificent
    assert got.anchor_share == pytest.approx(1.0, abs=0.1)
    assert got.anchor_driven is True
    assert got.predicts_the_market is False


def test_a_clean_feature_is_not_flagged():
    got = movement.fit_against_null(card(400, slope=0.5, noise=0.005), "clean")
    assert got.anchor_driven is False
    assert got.predicts_the_market is True


def test_the_null_keeps_each_match_s_own_opening_and_closing():
    original = card(50, slope=0.3)
    permuted = movement.anchor_permuted(original)
    by_key = {(m.match, m.selection): m for m in original}
    for m in permuted:
        own = by_key[(m.match, m.selection)]
        assert m.opening == own.opening
        assert m.closing == own.closing


def test_the_null_lends_a_different_match_s_feature():
    permuted = movement.anchor_permuted(card(50, slope=0.3))
    by_key = {(m.match, m.selection): m for m in card(50, slope=0.3)}
    borrowed = sum(1 for m in permuted if m.feature != by_key[(m.match, m.selection)].feature)
    assert borrowed > len(permuted) * 0.9


def test_the_null_never_lends_across_selections():
    original = card(60, slope=0.2)
    values = {s: {m.raw for m in original if m.selection == s} for s in SELECTIONS}
    for m in movement.anchor_permuted(original):
        assert m.raw in values[m.selection]


def test_a_single_observation_per_selection_cannot_be_permuted():
    assert movement.anchor_permuted([move("only")]) == []


# --- the verdict ----------------------------------------------------------


def test_nothing_certifies_without_a_null():
    got = movement.fit(card(400, slope=0.5, noise=0.005), "no null")
    assert got.null_slope is None
    assert got.predicts_the_market is False


def test_a_thin_sample_cannot_certify():
    got = movement.fit_against_null(card(20, slope=0.5, noise=0.005), "thin")
    assert got.matches < movement.MINIMUM_MATCHES
    assert got.predicts_the_market is False


def test_a_negative_slope_never_certifies():
    got = movement.fit_against_null(card(400, slope=-0.5, noise=0.005), "wrong way")
    assert got.slope < 0
    assert got.predicts_the_market is False


def test_anchor_share_is_undefined_for_a_zero_slope():
    got = movement.Fit(
        name="x",
        slope=0.0,
        stderr=0.0,
        observations=10,
        matches=5,
        feature_sd=0.1,
        mean_opening=0.4,
        null_slope=0.0,
    )
    assert got.anchor_share is None
    assert got.anchor_driven is False


# --- the economics --------------------------------------------------------


def test_movement_per_sd_scales_with_the_feature_s_spread():
    got = movement.fit_against_null(card(400, slope=0.5, noise=0.005), "clean")
    assert got.movement_per_sd == pytest.approx(got.net_slope * got.feature_sd)


def test_value_is_measured_against_the_margin_it_must_clear():
    got = movement.fit_against_null(card(400, slope=0.5, noise=0.005), "clean")
    tight = got.value_per_sd(0.01)
    wide = got.value_per_sd(0.05)
    assert tight > wide  # the same signal is worth more against a thinner margin


def test_value_is_undefined_without_a_margin():
    got = movement.fit_against_null(card(300, slope=0.4, noise=0.01), "clean")
    assert got.value_per_sd(0.0) is None


# --- agreement with the module this generalises ---------------------------


def test_it_reproduces_clv_s_slope_on_clv_s_own_feature():
    """`clv.py` is the special case where the feature is model minus opening."""
    rng = random.Random(11)
    observations, movements = [], []
    for i in range(300):
        for selection in SELECTIONS:
            opening = rng.uniform(0.2, 0.6)
            model = opening + rng.gauss(0.0, 0.04)
            closing = opening + 0.3 * (model - opening) + rng.gauss(0.0, 0.005)
            observations.append(
                clv.Observation(
                    match=f"m{i}",
                    selection=selection,
                    model=model,
                    opening=opening,
                    closing=closing,
                    opening_price=1 / opening,
                    closing_price=1 / closing,
                    won=False,
                )
            )
            movements.append(
                move(f"m{i}", selection, raw=model, opening=opening, closing=closing, anchored=True)
            )
    assert movement.fit(movements).slope == pytest.approx(
        clv.fit_anticipation(observations).slope, abs=1e-9
    )


# --- feature builders -----------------------------------------------------


class FakeRecord:
    def __init__(self, i, opening, closing, model=None):
        self.event_espn_id = f"e{i}"
        self.market_probabilities = opening
        self.closing_probabilities = closing
        self.probabilities = model or {}
        self.opening_prices = {s: 1 / v for s, v in opening.items()}
        self.closing_prices = {s: 1 / v for s, v in closing.items()}


def records(n=5):
    base = {"home": 0.45, "draw": 0.28, "away": 0.27}
    return [
        FakeRecord(i, dict(base), {k: v + 0.01 for k, v in base.items()}, dict(base))
        for i in range(n)
    ]


def test_records_without_both_lines_are_dropped():
    good = records(3)
    bad = records(1)[0]
    bad.closing_probabilities = None
    built = movement.movements_from([*good, bad], movement.price_level())
    assert {m.match for m in built} == {"e0", "e1", "e2"}


def test_the_price_level_control_is_the_opening_itself():
    [first, *_] = movement.movements_from(records(1), movement.price_level())
    assert first.feature == pytest.approx(first.opening)


def test_an_anchored_model_feature_is_model_minus_opening():
    built = movement.movements_from(
        records(1), movement.model_probability("probabilities"), anchored=True
    )
    assert all(m.feature == pytest.approx(0.0) for m in built)


def test_the_same_model_unanchored_is_the_probability_itself():
    built = movement.movements_from(records(1), movement.model_probability("probabilities"))
    assert built[0].feature == pytest.approx(built[0].opening)


def test_a_missing_model_contributes_nothing():
    assert movement.movements_from(records(3), movement.model_probability("nope")) == []


def test_noise_is_reproducible_from_its_seed():
    first = movement.movements_from(records(4), movement.gaussian_noise(3))
    second = movement.movements_from(records(4), movement.gaussian_noise(3))
    assert [m.feature for m in first] == [m.feature for m in second]


# --- the command ----------------------------------------------------------


def test_the_command_refuses_an_unknown_league(db):
    with pytest.raises(CommandError, match="No league with slug"):
        call_command("measure_movement", "nope", stdout=StringIO())


def test_the_command_refuses_a_league_without_closing_lines(db):
    call_command("seed_demo_data", rounds=6, upcoming=0, stdout=StringIO())
    with pytest.raises(CommandError, match="closing line"):
        call_command("measure_movement", "demo.1", stdout=StringIO())
