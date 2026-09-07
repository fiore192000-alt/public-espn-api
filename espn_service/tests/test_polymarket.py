"""Tests for reading Polymarket as a price source.

The HTTP layer cannot be exercised here — the hosts are blocked by this
environment's egress policy — so everything below tests the pure functions that
turn a payload into a price and a price into an observation the movement bench
can use. Those are where the mistakes would be, and the payload shapes come from
Polymarket's own client rather than from memory.
"""

import pytest

from apps.espn import movement, polymarket


def book(bids=((0.48, 500),), asks=((0.52, 500),), token="t1", stamp=1000, last=None):
    return {
        "asset_id": token,
        "timestamp": str(stamp),
        "bids": [{"price": str(p), "size": str(s)} for p, s in bids],
        "asks": [{"price": str(p), "size": str(s)} for p, s in asks],
        **({"last_trade_price": str(last)} if last is not None else {}),
    }


# --- reading a book -------------------------------------------------------


def test_a_two_sided_book_gives_a_midpoint_and_a_spread():
    q = polymarket.quote_from_book(book())
    assert q.best_bid == pytest.approx(0.48)
    assert q.best_ask == pytest.approx(0.52)
    assert q.mid == pytest.approx(0.50)
    assert q.spread == pytest.approx(0.04)
    assert q.cost_to_cross == pytest.approx(0.02)
    assert q.usable is True


def test_the_best_bid_is_the_highest_and_the_best_ask_the_lowest():
    q = polymarket.quote_from_book(
        book(bids=((0.40, 10), (0.47, 10), (0.45, 10)), asks=((0.60, 10), (0.53, 10)))
    )
    assert q.best_bid == pytest.approx(0.47)
    assert q.best_ask == pytest.approx(0.53)


def test_a_one_sided_book_falls_back_to_the_side_that_exists():
    only_bids = polymarket.quote_from_book(book(asks=()))
    assert only_bids.mid == pytest.approx(0.48)
    assert only_bids.spread is None
    assert only_bids.usable is False


def test_an_empty_book_falls_back_to_the_last_trade():
    q = polymarket.quote_from_book(book(bids=(), asks=(), last=0.61))
    assert q.mid == pytest.approx(0.61)
    assert q.usable is False


def test_an_empty_book_with_no_trade_has_no_price():
    assert polymarket.quote_from_book(book(bids=(), asks=())).mid is None


def test_levels_outside_the_unit_interval_are_dropped():
    q = polymarket.quote_from_book(book(bids=((1.4, 10), (0.30, 10)), asks=((0.55, 10),)))
    assert q.best_bid == pytest.approx(0.30)


def test_zero_sized_and_malformed_levels_are_dropped():
    payload = book()
    payload["bids"] = [
        {"price": "0.49", "size": "0"},
        {"price": "nope", "size": "10"},
        "not a dict",
        {"price": "0.45", "size": "20"},
    ]
    assert polymarket.quote_from_book(payload).best_bid == pytest.approx(0.45)


def test_a_book_without_an_asset_id_is_refused():
    payload = book()
    del payload["asset_id"]
    with pytest.raises(polymarket.PolymarketError, match="no asset id"):
        polymarket.quote_from_book(payload)


def test_a_non_object_response_is_refused():
    with pytest.raises(polymarket.PolymarketError, match="must be an object"):
        polymarket.quote_from_book([])


def test_an_unreadable_timestamp_becomes_zero_rather_than_raising():
    payload = book()
    payload["timestamp"] = "later"
    assert polymarket.quote_from_book(payload).timestamp == 0


# --- depth ----------------------------------------------------------------


def test_depth_counts_only_levels_near_the_touch():
    q = polymarket.quote_from_book(
        book(bids=((0.50, 100), (0.49, 200), (0.30, 900)), asks=((0.52, 50),))
    )
    assert q.depth("bid", 0.02) == pytest.approx(300)
    assert q.depth("bid", 0.001) == pytest.approx(100)
    assert q.depth("ask", 0.02) == pytest.approx(50)


def test_depth_of_an_empty_side_is_zero():
    assert polymarket.quote_from_book(book(bids=())).depth("bid", 0.05) == 0


# --- history --------------------------------------------------------------


def test_a_history_becomes_a_sorted_series():
    payload = {"history": [{"t": 30, "p": "0.6"}, {"t": 10, "p": "0.4"}, {"t": 20, "p": "0.5"}]}
    series = polymarket.series_from_history(payload, "t1")
    assert [q.timestamp for q in series] == [10, 20, 30]
    assert [q.mid for q in series] == [0.4, 0.5, 0.6]


def test_history_points_have_no_spread_and_are_never_usable():
    """A traded price cannot cost a trade, and must not be mistaken for a book."""
    payload = {"history": [{"t": 1, "p": "0.5"}]}
    [q] = polymarket.series_from_history(payload, "t1")
    assert q.spread is None
    assert q.usable is False


def test_malformed_history_points_are_dropped():
    payload = {"history": [{"t": 1, "p": "0.5"}, {"p": "0.6"}, {"t": 3, "p": "9"}, "x"]}
    assert len(polymarket.series_from_history(payload, "t1")) == 1


def test_a_history_without_the_list_is_refused():
    with pytest.raises(polymarket.PolymarketError, match="no 'history' list"):
        polymarket.series_from_history({}, "t1")


# --- into the movement bench ---------------------------------------------


def series(prices, start=100, step=60):
    return [
        polymarket.Quote(token="t", timestamp=start + i * step, last_trade=p)
        for i, p in enumerate(prices)
    ]


def test_a_market_yields_exactly_one_observation():
    """However many points were sampled, a market's path is one dependent thing."""
    built = polymarket.movements_from_series(
        series([0.40, 0.44, 0.47, 0.52]), feature=0.1, market="m1"
    )
    assert len(built) == 1
    assert built[0].opening == pytest.approx(0.40)
    assert built[0].closing == pytest.approx(0.52)
    assert built[0].travelled == pytest.approx(0.12)


def test_a_series_too_short_to_move_yields_nothing():
    assert polymarket.movements_from_series(series([0.5]), feature=0.1, market="m") == []


def test_settle_after_truncates_the_series():
    built = polymarket.movements_from_series(
        series([0.40, 0.44, 0.47, 0.52]), feature=0.1, market="m1", settle_after=160
    )
    # Timestamps run 100, 160, 220, 280; cutting at 160 keeps the first two.
    assert built[0].closing == pytest.approx(0.44)


def test_settling_before_the_second_point_yields_nothing():
    assert (
        polymarket.movements_from_series(
            series([0.4, 0.5]), feature=0.1, market="m", settle_after=100
        )
        == []
    )


def test_the_anchored_flag_reaches_the_movement():
    [m] = polymarket.movements_from_series(
        series([0.40, 0.50]), feature=0.55, market="m1", anchored=True
    )
    assert m.anchored is True
    assert m.feature == pytest.approx(0.55 - 0.40)


def test_prices_become_decimal_odds_for_the_record():
    [m] = polymarket.movements_from_series(series([0.40, 0.50]), feature=0.1, market="m1")
    assert m.opening_price == pytest.approx(2.5)
    assert m.closing_price == pytest.approx(2.0)


def test_a_real_signal_is_recovered_through_the_bench():
    """End to end: markets whose feature predicts their own move."""
    import random

    rng = random.Random(4)
    built = []
    for i in range(400):
        x = rng.gauss(0.0, 0.08)
        start = rng.uniform(0.25, 0.65)
        built += polymarket.movements_from_series(
            series([start, start + 0.5 * x]), feature=x, market=f"m{i}"
        )
    fit = movement.fit_against_null(built, "polymarket")
    assert fit.slope == pytest.approx(0.5, abs=0.05)
    assert fit.predicts_the_market is True


# --- the economics --------------------------------------------------------


def test_a_signal_is_measured_against_the_spread_it_must_cross():
    fit = movement.Fit(
        name="x",
        slope=0.5,
        stderr=0.05,
        observations=400,
        matches=400,
        feature_sd=0.08,
        mean_opening=0.45,
        null_slope=0.0,
    )
    assert fit.movement_per_sd == pytest.approx(0.04)
    assert polymarket.worth_acting_on(fit, 0.02) == pytest.approx(2.0)
    assert polymarket.worth_acting_on(fit, 0.08) == pytest.approx(0.5)


def test_a_signal_smaller_than_the_spread_scores_below_one():
    fit = movement.Fit(
        name="x",
        slope=0.1,
        stderr=0.05,
        observations=400,
        matches=400,
        feature_sd=0.05,
        mean_opening=0.5,
        null_slope=0.0,
    )
    assert polymarket.worth_acting_on(fit, 0.02) < 1.0


def test_no_spread_means_no_verdict():
    fit = movement.Fit(
        name="x",
        slope=0.5,
        stderr=0.05,
        observations=10,
        matches=10,
        feature_sd=0.08,
        mean_opening=0.45,
        null_slope=0.0,
    )
    assert polymarket.worth_acting_on(fit, 0.0) is None


def test_a_fit_without_a_null_has_no_economic_value():
    fit = movement.Fit(
        name="x",
        slope=0.5,
        stderr=0.05,
        observations=400,
        matches=400,
        feature_sd=0.08,
        mean_opening=0.45,
    )
    assert polymarket.worth_acting_on(fit, 0.02) is None


# --- urls -----------------------------------------------------------------


def test_the_urls_match_polymarket_s_own_client():
    assert polymarket.book_url("abc") == "https://clob.polymarket.com/book?token_id=abc"
    assert "prices-history?market=abc" in polymarket.history_url("abc")
    assert polymarket.markets_url() == "https://clob.polymarket.com/markets"
    assert polymarket.markets_url("MTA=").endswith("?next_cursor=MTA=")


def test_fetch_reports_the_url_it_failed_on():
    with pytest.raises(polymarket.PolymarketError, match="not-a-host"):
        polymarket.fetch_json("https://not-a-host.invalid/x", timeout=0.5)
