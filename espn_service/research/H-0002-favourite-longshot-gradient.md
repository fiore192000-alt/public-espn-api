# H-0002 — The favourite-longshot gradient

**Status: real, and economically dead.** Registered after the fact, which is
stated here rather than hidden: this was found by searching, not predicted
before looking. Everything below is therefore a *result about the past*, and the
decay test at the end is the only part with a bearing on tomorrow.

## The observation

Flat-staking every selection whose best price falls in a band, across 36,283
matches of Serie A, Premier League, La Liga, Bundesliga and Ligue 1
(2005–2026), settled at the best of ~17 books:

| band | bets | yield |
|---|---|---|
| 1.00–1.50 | 5,722 | **+1.28%** |
| 1.50–2.00 | 11,602 | **+1.38%** |
| 2.00–3.00 | 18,953 | +0.51% |
| 3.00–4.00 | 24,409 | −0.91% |
| 4.00–6.00 | 17,615 | −3.43% |
| 6.00–10.00 | 8,988 | −3.47% |
| 10.00+ | 4,176 | **−7.30%** |

Spearman rank correlation between a band's place in the price ladder and its
yield: **−0.964**.

## Test 1 — is a single band's yield distinguishable from noise? **No.**

The best band returns +1.38% at `t = +1.73`. Against 60 simulated efficient
markets of the same size (`synthetic.MarketSpec`, margin 0.05, best-price
overround 1.003), searched with this project's own sweep:

- the best of ten rules returns **+2.00% on average** and **+8.01%** at the 95th
  percentile, from a market where nothing is beatable;
- it clears `|t| ≥ 2` in **8.3%** of searches.

**A yield of +1.38% is reached by 38.3% of searches on a market with no edge.
Its `t = +1.73` by 20.0%.** On its own, the band establishes nothing.

## Test 2 — is the *ordering* distinguishable from noise? **Yes.**

The gradient uses all seven bands at once, and the max-of-ten null above does
not test it. Against **200** efficient markets of the same size:

- mean Spearman **+0.009**, 5th percentile **−0.786**;
- a perfectly decreasing gradient arose **0 times out of 200**;
- **`p(Spearman ≤ −0.964) = 0.000`**.

The shape is not something this search manufactures from an efficient market.

## Test 3 — is it a mispricing, or the bookmaker's margin structure?

Books load more margin onto longshots by construction. That alone would produce
a yield gradient without any mistake in the price, so the two must be separated:
devig each book and compare what it *believed* against what happened.

| band | legs | implied | observed | residual | t |
|---|---|---|---|---|---|
| 1.00–1.50 | 6,785 | 0.7438 | 0.7574 | **+0.0136** | **+2.64** |
| 1.50–2.00 | 12,389 | 0.5595 | 0.5622 | +0.0027 | +0.61 |
| 2.00–3.00 | 23,011 | 0.3966 | 0.3973 | +0.0007 | +0.87 |
| 3.00–4.00 | 37,577 | 0.2779 | 0.2764 | −0.0016 | −0.36 |
| 4.00–6.00 | 17,513 | 0.2006 | 0.1972 | −0.0033 | −1.33 |
| 6.00–10.00 | 8,128 | 0.1244 | 0.1227 | −0.0017 | −0.43 |
| 10.00+ | 3,446 | 0.0612 | 0.0577 | −0.0035 | −0.83 |

**The gradient survives devigging.** Favourites happen more often than the
devigged price says, longshots less. It is a miscalibration, not just margin.
(The one individually significant residual, `t = +2.64` on the shortest band, is
one of seven tests and falls short of the Bonferroni bar of `|t| > 2.69`.)

## Test 4 — does it persist? **No. It is gone.**

The literature's finding is not that the effect never existed but that rules
built on it are short-lived. Split by era:

| era | matches | Spearman |
|---|---|---|
| 2005–2011 | 11,421 | **−0.750** |
| 2012–2017 | 11,014 | **−0.429** |
| 2018–2026 | 13,848 | **+0.000** |

**The gradient decays monotonically to exactly zero.**

The pooled −0.964 was carried by the extreme longshots, and that is where the
decay is starkest — the `10.00+` band returned **−25.08%** in 2005–2011 and
**−20.82%** in 2012–2017, then **+18.85%** in 2018–2026. The mispricing that
made the ladder monotone has not merely shrunk; it has reversed sign.

The two short-price bands, era by era, never reach significance in any of them:

| band | 2005–2011 | 2012–2017 | 2018–2026 |
|---|---|---|---|
| 1.00–1.50 | +1.16% (t +0.77) | +2.02% (t +1.59) | +0.80% (t +0.69) |
| 1.50–2.00 | +0.09% (t +0.07) | +2.21% (t +1.51) | +1.77% (t +1.40) |

## Verdict

The favourite-longshot bias is **real in this data and absent from the last
eight years of it**. On 13,848 modern matches the rank correlation between price
and yield is zero to three decimal places.

Nothing here supports a bet. The finding is worth recording because it is the
first thing in this project to clear its own null control, and because the way
it dies — visible in the pooled number, invisible until the data is cut by era —
is the failure mode every backtest in this repository is built to catch.

## What would change this verdict

A pre-registered test on data none of the above touched: a different market
(totals, both-teams-to-score), a different competition tier, or the 2026/27
season as it happens. Post-hoc slicing of these same 36,283 matches would not.

## Reproduction

    python manage.py verify_search --matches 36283 --trials 60 \
        --observed 0.0138 --observed-t 1.73

The gradient, calibration and decay tables come from the pooled five-league
`market_bias.collect_matches` set at `fd-b365` / `fd-max` with Shin devig; the
Spearman null uses `synthetic.simulate` at the same sample size with
`best_price_edge=0.0469`.
