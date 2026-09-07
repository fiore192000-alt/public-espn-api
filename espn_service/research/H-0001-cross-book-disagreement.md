# H-0001 — Does disagreement between books predict where the line goes?

**Status: PRE-REGISTERED**
**Registered: 2026-09-07, before any result was computed. The commit that adds this
file contains no results; the commit that adds them comes after.**

---

## Why this experiment exists

A larger project has been proposed: record order books across bookmakers,
exchanges and prediction markets minute by minute, and ask whether *flow* carries
information about future price beyond what the current price already contains.
It needs three months of recording before the first question can be asked.

This is the cheapest possible version of that question, answerable today, on data
already in the repository. If the coarse version shows nothing, the fine version
is not obviously worth three months. If it shows something, the sign and the
order of magnitude tell us what to look for.

## The question

Not "who will win" and not "is the closing line sharper" — that is already
established (+0.00306 log loss, `t = +5.20`) and is merely the market improving
itself.

The question is:

> At the moment the market opens, does the **disagreement between two books**
> predict the **direction the market subsequently moves**, beyond what the price
> already says?

This is the coarsest possible flow signal: a cross-section of opinion available
*before* the move, rather than the move itself. It is not circular — the feature
is measured at time *t* and the target is the change from *t* to close.

## Hypotheses

Three, and no more. The count is fixed here so the correction below is not chosen
after seeing results.

**H-0001a (primary).** Where Pinnacle's opening probability exceeds Bet365's on a
selection, the closing line moves further in that direction.

- feature `d = p_pinnacle_open[s] − p_bet365[s]`, Shin-devigged, per selection
- target `m = p_pinnacle_close[s] − p_pinnacle_open[s]`
- estimator: `m = b·d + ε`, fitted through the origin, standard errors
  **clustered by match** (three selections of one fixture move together)
- prediction: `b > 0` — the sharp book leads and the market follows it

**H-0001b.** The same, with the market average (`fd-avg`) in place of Bet365.

**H-0001c (economic).** If either slope is positive, selections chosen by the sign
of `d` beat the closing line by more than the round-trip cost of taking them.

## Gates, all three required

1. **Size.** `|b| ≥ 0.02`. Below this the effect cannot survive any realistic cost.
2. **Significance under the multiple-testing budget.** Three hypotheses, Bonferroni
   at α = 0.05 → **α = 0.0167 → |t| ≥ 2.39**. Not 2.0.
3. **Beats its own null.** The feature is permuted across matches within the same
   division and season, preserving the marginal distribution and destroying the
   link to any particular fixture. The real slope must exceed the permuted one.

Note on the null: permutation is appropriate *here* because matches are close to
independent events. It would **not** be appropriate for the high-frequency project
this is a proxy for, where autocorrelation and time-of-day effects survive
permutation and a stationary block bootstrap or circular shift is required
instead. That distinction is recorded now so it is not forgotten later.

## Data

- `eng.1`–`eng.5`, English football 2015/16–2020/21
- 13,655 matches carrying Pinnacle opening, Pinnacle closing and a second
  contemporaneous book
- Full sample is the primary test. Nothing is being tuned — no thresholds, no
  feature selection, no market selection — so a hold-out would protect against
  nothing that is not already prevented by fixing the hypotheses in advance. The
  two halves (2015–2018, 2018–2021) are reported as a **stability check**, not as
  a discovery/validation split.

## Known limitation, stated before the result

Football-Data records one pre-match snapshot per book without a timestamp. The two
opening prices are therefore only *approximately* contemporaneous. If Bet365's
snapshot is systematically later than Pinnacle's, part of any measured slope is
Bet365 having already absorbed a move rather than disagreeing with it. **This
experiment cannot rule that out**, and a positive result must be read as
suggestive rather than established. Resolving it is precisely what timestamped
recording would buy.

## What each outcome means for the larger project

| result | reading |
|---|---|
| `b` clears all three gates | flow-like information exists at the coarsest resolution; the recorder is worth building |
| `b ≈ 0`, tight interval | the cross-section of opinion carries nothing the price lacks, at this resolution |
| `b < 0` | books disagree because the *soft* one leads, which would be a genuine surprise |
| `b` positive but under the size gate | real but small; the recorder is worth building only if the fine resolution can plausibly multiply it |

A null result here does **not** kill the larger project — a minute-by-minute order
book is a far richer object than two price snapshots. It does mean the project
would be starting without a measured anchor, and should be sized accordingly.
