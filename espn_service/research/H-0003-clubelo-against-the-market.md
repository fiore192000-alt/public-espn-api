# H-0003 — Where does ClubElo beat the market?

**Answer: nowhere, and the way it fails is more useful than the answer.**

Post-hoc, and said plainly: this is a subgroup search, which is the single most
reliable way to manufacture a finding. It is reported here because it came back
empty and because the *shape* of the emptiness explains an earlier result.

## The question

ClubElo is the best forecaster in this repository on three of five leagues. It
still fails the incremental-information gate everywhere. So: is there a
**subset** of matches — a price band, a strength gap, a league, an era — where
it beats the market?

## The design, fixed before looking

**Metric.** Paired per-match difference in log loss, `market − ClubElo`.
Positive means ClubElo is better. Errors are the ordinary paired standard error;
each match contributes one observation.

**Subgroups.** Twenty-three, enumerated in advance: five bands of the market
favourite's probability, five of ClubElo's own home-minus-away gap, five of the
model/market disagreement, five leagues, three eras. Minimum 200 matches each.

**Null.** Two hundred simulated worlds in which the market believes the *truth*
and the model is that truth degraded by multiplicative noise — so the model
cannot beat the market anywhere, by construction. The same 23-subgroup search
runs on each. Whatever the best subgroup reaches there is what the search
manufactures from nothing.

**Data.** 36,464 matches carrying both a ClubElo forecast and a devigged market
price, across Serie A, Premier League, La Liga, Bundesliga and Ligue 1.

## The result

**Not one of the 23 subgroups is positive.** The best is the band where ClubElo
and the market barely disagree, and there it merely ties:

| subgroup | matches | ClubElo − market | t |
|---|---|---|---|
| model/market gap 0.00–0.02 | 5,264 | −0.00025 | −0.52 |
| model/market gap 0.02–0.04 | 8,939 | −0.00072 | −0.93 |
| model/market gap 0.04–0.06 | 7,203 | −0.00656 | −4.84 |
| model/market gap 0.06–0.09 | 7,115 | −0.01656 | −8.52 |
| **model/market gap 0.09+** | 7,103 | **−0.05286** | **−14.65** |

The null had the power to find something and found nothing: across 200 searches
with no edge to find, the best subgroup's `t` averaged **−1.12**, reached
**+0.62** at the 95th percentile, and cleared `|t| ≥ 2` in **0%** of searches.
The real best, `t = −0.52`, is reached by 26.5% of them.

## What the shape says

The disagreement gradient is monotone across all five bands and spans a factor
of **200**. It is not a statement about *when* ClubElo is right. It is a
statement about what its disagreement means:

> **ClubElo's distance from the price measures ClubElo being wrong, not the
> price being wrong.**

That is the mechanism behind a result measured earlier and left unexplained:
the rule "back the home side when the model's edge exceeds +3%" returned
**−7.2% over 3,004 Serie A matches, `t = −3.31`**. Betting on model
disagreement is betting on the model's error. The gradient above is that
sentence, measured.

Two secondary readings, both in the same direction:

- Every league is negative, between −0.0133 and −0.0165. There is no country
  where the model is closer.
- By era: **−0.0132 (2005–2011), −0.0149 (2012–2017), −0.0175 (2018–2027).**
  The market is pulling away, not being caught.

## What would change this verdict

Nothing in these 36,464 matches. A subgroup found by slicing them further is a
smaller version of the same search. The verdict changes only with an input the
market prices slowly — lineups, in-window team news, order flow — tested on data
that did not produce the hypothesis, and tested against the market's own
movement rather than against results, because that is where the statistical
power is.

## Reproduction

Per-match ClubElo and market forecasts come from `backtest.run(league,
refit_every=25)` over the five leagues; the subgroup grid and the simulated null
are the script recorded with this note's commit.
