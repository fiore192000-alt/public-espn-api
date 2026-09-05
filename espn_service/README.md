# ESPN Service API

A production-ready Django REST API for ingesting and querying ESPN sports data.

## Features

- **Data Ingestion**: Fetch and persist data from ESPN's public/undocumented API endpoints
- **REST API**: Clean, paginated endpoints for querying teams, events, and games
- **Match Analysis**: Team form, head-to-head, projected scores and win probabilities from stored history
- **Scoreline Model**: Dixon-Coles fit over league history, reduced to 1X2 / totals / BTTS / correct-score markets, with value detection against bookmaker odds and a walk-forward backtest
- **Background Jobs**: Celery tasks for scheduled data refresh
- **Multi-Sport Support**: All 17 ESPN sports — NFL, NBA, MLB, NHL, WNBA, MLS, UFC, PGA, F1, NRL, and more
- **Production-Ready**: Docker, PostgreSQL, Redis, structured logging, health checks

## Quick Start

### Using Docker (Recommended)

```bash
cd espn_service
cp .env.example .env
docker compose up --build

# API: http://localhost:8000
# Docs: http://localhost:8000/api/docs/
```

### Local Development

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -e ".[dev]"
pre-commit install
python manage.py migrate
python manage.py runserver
```

---

## Service API Endpoints

### Health Check

```bash
GET /healthz
```

### Data Ingestion

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/ingest/teams/` | POST | Ingest teams from ESPN |
| `/api/v1/ingest/scoreboard/` | POST | Ingest events/games |

**Request Body:**
```json
{
    "sport": "basketball",
    "league": "nba",
    "date": "20241215"  // Optional for scoreboard
}
```

### Query Data

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/teams/` | GET | List teams (with filters) |
| `/api/v1/teams/{id}/` | GET | Team details |
| `/api/v1/teams/espn/{espn_id}/` | GET | Team by ESPN ID |
| `/api/v1/events/` | GET | List events (with filters) |
| `/api/v1/events/{id}/` | GET | Event details |
| `/api/v1/events/espn/{espn_id}/` | GET | Event by ESPN ID |

**Filter Parameters:**
- `sport` - Filter by sport slug
- `league` - Filter by league slug
- `search` - Search teams by name
- `date` - Filter events by date (YYYY-MM-DD)
- `team` - Filter events by team abbreviation
- `status` - Filter events by status

### Match Analysis

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/events/{id}/analysis/` | GET | Full match analysis for an event |
| `/api/v1/teams/{id}/form/` | GET | Recent form for a team |

Both accept `?lookback=N` (1–50, default 10) to set how many past games per team are used.

Analysis is computed entirely from events already stored in the database — no ESPN
call is made — and only history dated **before** the event is used, so past games can
be analysed as they would have looked beforehand.

The analysis payload contains:

| Section | Contents |
|---------|----------|
| `home` / `away` | Team, current score, `form`, `context` and weighted absences |
| `head_to_head` | Previous meetings, wins per side, combined points per game |
| `league_baseline` | League scoring level, empirical home advantage, margin spread, draw rate |
| `projection` | Expected score per side, margin, `home_win` / `draw` / `away_win` probabilities, and the `source` that produced them |
| `confidence` | `none` / `low` / `medium` / `high`, from the available sample size |
| `insights` | Plain-language notes summarising the above |

Each side's `form` carries, beyond the plain record:

| Field | Meaning |
|-------|---------|
| `weighted` | Points and goals per game with older matches discounted (60-day half-life) |
| `momentum` | The last 5 games against the side's own window average — `points_delta` above zero means it has been picking up more than usual lately |
| `opponent_strength` | Mean goal difference per game of the opponents actually faced, so a good run against weak sides reads as one |

And `context` carries the situational picture:

| Field | Meaning |
|-------|---------|
| `rest_days` | Days since that side's previous completed match |
| `matches_in_last_14_days` / `congested` | Fixture pile-up |
| `injuries` | Absences weighted by status severity **and**, where season appearances are stored, by how much the player actually plays. `importance_known` says whether that playing-time weighting was available — ESPN's injury feed does not mark starters, so without stored stats every absence counts the same |

**None of these feed the projection.** They are reported next to it, because none has
been validated against real results yet, and an unvalidated adjustment makes a model
worse while looking more sophisticated. Wire one in only after the backtest says it
helps.

#### Where the projection comes from

`projection.source` says which method produced the numbers:

| Source | When | Draw probability |
|--------|------|------------------|
| `dixon_coles` | Default, whenever the league has enough history to fit | Derived per fixture from the scoreline distribution |
| `fallback_form` | Only when the league has fewer than 20 completed matches | The league's observed draw rate — **the same constant for every fixture** |

The model path is the same fit that backs `/forecast/`, so the two endpoints agree
rather than offering contradictory probabilities for one match. When the fallback is
used, `insights` says so explicitly.

The fallback blends each side's scoring rate with the opponent's concession rate, using
home/away splits once they hold at least three games. It cannot tell a tight match from
a mismatch as far as the draw is concerned, which is exactly why it is a fallback: with
20 matches stored the model takes over.

```bash
curl "http://localhost:8000/api/v1/events/44/analysis/?lookback=10"
curl "http://localhost:8000/api/v1/teams/7/form/?lookback=5"
```

### Scoreline Model & Betting Markets (football)

`GET /api/v1/events/{id}/forecast/` fits a **Dixon-Coles** model to the league
history preceding the event and returns a full scoreline distribution, reduced to
markets:

| Market | Selections |
|--------|------------|
| `1x2` | home, draw, away |
| `double_chance` | home_or_draw, home_or_away, draw_or_away |
| `totals` | over/under at 0.5, 1.5, 2.5, 3.5, 4.5 |
| `btts` | yes, no |
| `correct_score` | most likely exact scorelines |

Every market is a different sum over the *same* probability grid, so they are
mutually consistent by construction. Each carries `fair_odds` (the break-even
decimal price).

| Parameter | Meaning |
|-----------|---------|
| `half_life` | Days after which a past match counts half as much (default 120) |
| `edge` | Minimum edge over the devigged market price to flag a bet (default 0.05) |
| `kelly` | Fraction of full Kelly to stake (default 0.25) |

**Model.** Goals are Poisson with per-team attack and defence strengths and a home
advantage term, plus the Dixon-Coles low-score correction. Attack, defence and home
advantage are fitted by weighted maximum likelihood with exponential time decay;
the correlation term is fitted conditionally. No numpy or scipy — the fit is a few
hundred lines of plain Python and takes ~0.03s for a 760-match history.

Against simulated leagues with known parameters, the fit recovers attack and
defence ratings at r > 0.98. `model.reliable` reports whether enough weighted
history backs the fit; below that, treat the numbers as decoration.

**Value bets.** Where odds are stored for the event, the response also lists
selections whose model probability beats the bookmaker's *devigged* price by at
least `edge`, with a fractional Kelly stake. Markets are devigged within their own
provider and complement — a one-sided quote is skipped, because its overround is
indistinguishable from an edge.

### Odds

```bash
# Fetch odds for scheduled events already stored for a league
python manage.py ingest_odds soccer ita.1 --limit 20
```

Prices are normalised to decimal on the way in, whatever format the source quotes.
The ESPN odds parser follows `docs/response_schemas.md` and skips anything it does
not recognise rather than guessing — **it has not been exercised against the live
ESPN endpoint**, so expect to adjust it on first real contact.

### Historical data with real odds

ESPN does not serve deep history, and the model needs seasons, not weeks. The
`ingest_football_data` command loads results, match statistics, pre-match odds and
ClubElo ratings from a Football-Data-derived CSV:

```bash
# Matches.csv from https://github.com/xgabora/Club-Football-Match-Data-2000-2025
# (results and odds originally from https://www.football-data.co.uk/)
python manage.py ingest_football_data Matches.csv --division I1 --date-from 2015-07-01
```

Division codes map onto ESPN-style league slugs (`I1` → `ita.1`, `E0` → `eng.1`,
`SP1` → `esp.1`, …), so a league loaded this way sits alongside anything ingested
from ESPN and works with every command above. Odds are stored as two providers:
`fd-b365` (Bet365's own price) and `fd-max` (best price across ~17 books).

The dataset is **not** committed to this repository — download it separately. Note
that these are Bet365's pre-match price and a best-of-~17 maximum, **not** closing
odds; for those, load the original football-data.co.uk season files instead — see
below.

### Backtesting

```bash
python manage.py backtest_model ita.1 --refit-every 5
python manage.py backtest_model ita.1 --edge 0.08 --kelly 0.25 --json
```

#### Choosing the half-life from the data

The decay half-life decides how fast old matches stop counting — that is, how quickly
the league turns over. Guessing it is guessing that. `tune_model` runs the same
walk-forward backtest at each candidate and ranks them by out-of-sample log-loss:

```bash
python manage.py tune_model ita.1 --half-lives 30,60,90,120,180,365 --refit-every 5
```

Every candidate scores the same matches, so the numbers are comparable. The command
also reports the spread across candidates and says so when they are too close to
separate — on a few hundred matches, neighbouring half-lives usually differ by noise,
and picking the winner then means tuning to noise. It also warns when even the best
candidate fails to beat the base-rate reference, which means the model is not yet
adding skill on that data at any setting.

The model is refitted for every match on only the matches that finished before it,
so no result informs its own prediction. The report separates two questions:

- **Is it calibrated?** Log-loss, Brier score and a predicted-vs-observed
  calibration table, shown next to a base-rate baseline. That baseline is measured
  on the same window it scores, so it flatters itself — losing to it narrowly is
  not damning, clearly beating it is meaningful.
- **Would betting have made money?** Yield under both Kelly and flat staking, hit
  rate, max drawdown — **and the standard error and t-statistic of the yield**. A
  yield within two standard errors of zero is indistinguishable from no edge, and
  the command says so out loud. On a few hundred bets that covers most results.

#### Against the market — the result that decides everything

The report scores the model against the **devigged bookmaker price** on the same
fixtures. This, not accuracy, is the question that matters: a model only has value
if it knows something the price does not.

Measured on **3,765 real Serie A matches (2015–2025)**, walk-forward:

| | model | market | base rate |
|---|---|---|---|
| log loss | 1.0052 | **0.9469** | 1.0767 |
| Brier | 0.5823 | **0.5610** | 0.6519 |

The model clearly beats a naive base rate — it has learned real football. It is
just as clearly **worse than the price**, by 0.058 log loss over 3,764 matches.
Betting it flat lost **6.8% per bet** (standard error 1.4%, t = −4.96): a loss
large enough that the sample proves it, not variance.

That is the expected outcome, and the reason this comparison exists. Any "value"
the edge filter finds against a better-informed price is the model's own error
being mistaken for an opportunity.

Two things the same run showed that are worth acting on:

- **Calibration is good in the middle, overconfident at the top.** In the 0.9–1.0
  band the model predicted 0.944 and observed 0.743; at 0.8–0.9 it predicted 0.841
  and observed 0.793. It is most wrong exactly where it would stake most.
- **The value filter fires far too often** — 8,369 bets over 3,765 matches. Against
  a superior price, a permissive edge threshold is a machine for finding your own
  mistakes.

### A second model, and the rule for promoting one

`apps/espn/elo.py` adds Elo as a deliberately **independent** second opinion.
Dixon-Coles models goals; Elo models only who is stronger. They fail differently,
which is the whole point — a model that agreed with Dixon-Coles by construction
could not add anything to it.

Elo alone gives an expected score, not three probabilities. The rating difference
is mapped to 1X2 through an **ordered logit** with two thresholds fitted by
maximum likelihood, so the draw is a per-fixture estimate: tight matches get a
higher draw probability than mismatches.

```bash
python manage.py compare_models ita.1 --refit-every 10
```

#### The promotion rule

> **A model enters the betting decision only if it shows incremental information
> over the market, out of sample.**

Predicting well and being useful are different things. A model can be accurate and
still worthless if everything it knows is already in the price. `combination.py`
settles it with a **logarithmic opinion pool**: the market's probabilities are
pooled with the candidate's, the weights are fitted on the earlier half of the
matches, and the pool is scored on the later half. Read the weights directly — a
candidate whose weight lands near zero is being told by the data that it adds
nothing.

Weights only, no per-outcome intercepts: an intercept would let the pool correct a
global bias in the market and show an "improvement" that has nothing to do with the
candidate's information.

#### What it says today

Measured on **3,734 real Serie A matches (2015–2025)**, walk-forward:

| standalone | log loss | Brier |
|---|---|---|
| market | **0.9466** | **0.5606** |
| Elo | 0.9729 | 0.5786 |
| Dixon-Coles | 0.9892 | 0.5800 |

Elo is the better of the two models — the simpler one wins. Both lose to the price.

| incremental (1,867 held-out matches) | log loss | vs market |
|---|---|---|
| market alone | 0.9642 | — |
| market + Dixon-Coles | 0.9642 | +0.0000 |
| market + Elo | 0.9644 | −0.0003 |
| market + both | 0.9644 | −0.0002 |

**Neither adds information.** The fitted weights on the candidates come out
*negative* (Dixon-Coles −0.09, Elo −0.28) while the market's own weight exceeds 1 —
the pool wants to sharpen the price and subtract the models, not blend them in.

Under the promotion rule, neither model is eligible for a betting decision. That is
the correct outcome, and the reason the rule exists.

### Removing the bookmaker's margin

A quoted book always implies more than 100%. Recovering what it actually believes
means deciding *how* that excess sits across the selections — and the choice is not
cosmetic: it moves a longshot's implied probability far more than a favourite's,
which is exactly where a model tends to disagree with the price.

```bash
python manage.py measure_devig ita.1
```

Three methods, scored against real results rather than assumed:

| method | log loss | Brier |
|---|---|---|
| power | **0.9456** | **0.5603** |
| Shin | 0.9459 | 0.5605 |
| proportional | 0.9470 | 0.5611 |

*(Bet365's 1X2 book, 3,799 Serie A matches.)*

Paired tests settle it: proportional is worse than both by a distinguishable margin
(t = −3.6 against Shin, −3.2 against power), while **Shin and power cannot be told
apart** (t = −1.9). The default is Shin — it wins the tie on grounds the data cannot
settle, by modelling *why* the margin sits where it does rather than fitting an
exponent that makes the book add up.

Switching off proportional makes the market benchmark **sharper**, which is the
honest direction: the wall the models have to clear gets higher, not lower.

### What the market's structure actually says

The same command reports the microstructure, and this is the most useful number in
the project so far:

| | |
|---|---|
| Bet365 overround | 1.0490 (**4.67%** margin) |
| Best price across ~17 books | **1.0003** |
| Best price vs Bet365, best leg | **+9.82%** |

That near-1.0 overround looks like it removes the entire margin. **It does not**,
and the difference matters enough to measure directly rather than infer.

Flat-staking every selection and settling on real results, over 3,799 matches:

| | at Bet365 | at best price | recovered |
|---|---|---|---|
| home | −11.56% | −7.42% | +4.14pp |
| draw | −4.18% | +1.25% | +5.42pp |
| away | −8.63% | −2.27% | +6.36pp |
| **pooled** | **−8.12%** | **−2.81%** | **+5.31pp** |

Price selection is worth **5.3 percentage points of yield** — large, real, and it
requires predicting nothing. But two things must be said plainly:

- **It does not make you profitable.** At the best price the yield is still −2.81%,
  with a standard error of 0.91% over 3,799 matches: reliably losing, not merely
  unproven. Line shopping is what makes a genuine edge survivable; it is not a
  substitute for having one.
- **The overround overstates it by 2.8 percentage points.** An overround of 1.0003
  implies backing all three outcomes returns −0.03%; it actually returned −2.81%.
  Three maxima taken across different books are not one coherent book — the best
  price tends to be highest on the outcomes that go on to lose. The realised figure
  is the trustworthy one.

The pooled figure is accumulated **per match, not per leg**: the three outcomes of
one match are a single dependent event, so treating them as three independent bets
misstates the spread.

#### Honest limits

- A backtest on `seed_demo_data` measures **nothing** about profitability. The data
  is synthetic and so are the prices. It checks that the machinery works.
- CLV cannot be computed from this data. Football-Data publishes the market's
  pre-match average and maximum, not opening-versus-closing prices, so there is no
  line movement to measure. Real closing odds would be needed.
- Verified on synthetic data: with prices generated from the true probabilities
  plus a 6% margin, the model finds no edge (yield −3.7%, t = −0.5). With a
  deliberate 12-point inefficiency injected via `--odds-bias`, it finds it (yield
  +11.9%). The detector responds to real mispricing and does not manufacture it.
- Devigging one bookmaker recovers roughly that bookmaker's own opinion. Beating it
  consistently is the entire difficulty; a positive edge is a hypothesis about a
  price, not a forecast of profit.
- Kelly assumes the model's probabilities are correct. They are not, which is why
  the stake defaults to a quarter of Kelly with a hard cap.

```bash
# End-to-end, offline: synthetic league with fair prices, then with an injected edge
python manage.py seed_demo_data --rounds 60 --with-odds
python manage.py backtest_model demo.1 --refit-every 5

python manage.py seed_demo_data --rounds 60 --with-odds --odds-bias 0.12
python manage.py backtest_model demo.1 --refit-every 5
```

### Closing odds, and why they change the question

Every verdict above is measured against a **pre-match** price. That is the noisier
of the two benchmarks available, and it flatters the models. The original
Football-Data.co.uk season files also carry **closing** prices — a bookmaker
abbreviation followed by `C`, so `PSCH` is Pinnacle's closing home price — and
`ingest_football_data` now reads that layout as well as the derived mirror,
detecting which one it is looking at from the header.

```bash
# The original football-data.co.uk season files, which carry closing prices
python manage.py ingest_football_data "2018-2019/Premier.csv" --division E0

# The derived Club Football Match Data mirror, which does not
python manage.py ingest_football_data Matches.csv --division I1
```

Opening and closing quotes are stored as **separate providers** (`fd-ps` and
`fd-psc`), never as two rows of one provider. The Odds model has no notion of when
a price was taken, and every analysis here keys on the provider — so keeping them
apart is what stops a closing line being devigged or settled as if it were a price
anybody could have taken when the forecast was made.

| provider | what it is |
|---|---|
| `fd-b365` / `fd-b365c` | Bet365, pre-match / closing |
| `fd-ps` / `fd-psc` | **Pinnacle**, pre-match / closing |
| `fd-avg` / `fd-avgc` | Market average |
| `fd-max` / `fd-maxc` | Best price across ~17 books (never a coherent book) |

#### What 13,657 matches with both prices say

English football, 2015/16–2020/21, Premier League down to the National League.

**Pinnacle's margin is less than half of Bet365's.** Overround 1.0297 against
Bet365's 1.0403 here, and 1.0490 on the Serie A data. The wall every model in this
project has been failing to clear was the *soft* one.

**The closing line is sharper than the opening line, beyond doubt.** Paired on the
same matches, closing beats opening by **0.00306 of log loss**, `t = +5.20`, 95% CI
[+0.0019, +0.0042].

| division | matches | close beats open by | t | open overround | close overround |
|---|---|---|---|---|---|
| Premier League | 2,084 | +0.00250 | 1.69 | 1.0231 | 1.0228 |
| Championship | 3,048 | +0.00306 | 2.77 | 1.0270 | 1.0250 |
| League One | 2,853 | +0.00521 | 4.10 | 1.0316 | 1.0293 |
| League Two | 2,899 | +0.00226 | 1.87 | 1.0316 | 1.0292 |
| National League | 2,773 | +0.00213 | 1.37 | 1.0336 | 1.0312 |

A hypothesis worth recording as **not confirmed**: the opening price in the
National League is not conspicuously worse than in the Premier League, so "less
watched means more beatable" does not show up here. What does scale with the
division is the **margin** — 2.31% up to 3.36%. The bookmaker charges more where it
knows less rather than pricing worse.

#### Closing-line value is a valid feedback metric — measured, not assumed

Taking the opening price only on selections that went on to shorten by the close:

| beat the close by | matches | yield | t |
|---|---|---|---|
| any amount | 13,447 | +2.45% | +2.34 |
| ≥ 1% | 12,645 | +2.21% | +1.92 |
| ≥ 2% | 11,160 | +3.22% | +2.47 |
| ≥ 3% | 9,490 | +4.61% | +3.12 |
| ≥ 5% | 6,619 | **+7.96%** | +4.15 |

Profitable at Pinnacle's own opening price, margin included, and rising with the
size of the beat. (The 1% row dipping below the 0% row is noise, not a pattern.)

**This is not a strategy, and reading it as one is the trap.** Selecting bets by
whether they beat the close requires knowing the close — which exists only after
the moment you would have had to bet. It is pure hindsight, the same error as
devigging a best-price line.

What it does establish is the thing worth having: **beating the close predicts
profit on this data**. So a model that picks in advance and systematically beats
the closing line is producing real edge, and that can be measured on a few thousand
matches against a continuous target instead of thirty thousand against a coin flip.
That is the feedback loop every negative result above was missing.

### Searching the price for a bias, and refusing to overclaim one

`find_market_bias` runs the search everybody runs — is some band of prices, or some
outcome, systematically wrong? — and then applies the gates that decide whether the
answer means anything. The gates are the point. A search over enough rules always
produces a winner, and a rule that won a search is not evidence.

```bash
python manage.py find_market_bias ita.1 --split-year 2019
python manage.py find_market_bias ita.1 --validate-on eng.1 esp.1 --json
```

Five gates, in the order they bite:

1. **Discovery and validation are separated.** Rules are ranked on Serie A before
   2019 only, then re-measured on Serie A from 2019 and on four leagues the search
   never read.
2. **The search burden is stated.** Every eligible rule is counted, along with how
   many would clear `|t| ≥ 2` on pure noise.
3. **Effects are measured in money.** A calibration gap is not an edge until it
   clears the margin, so every rule is settled as real flat-staked bets.
4. **Consistency is required.** Each validation set is reported separately, so a
   rule that pools positive because one set carries it is visible as such.
5. **Price selection is separated from the bias.** Every rule is settled twice, at
   one bookmaker and at the best price. Line shopping recovers margin on *any*
   selection, so an edge that exists only at the best price is a discount on the
   fee, not a mispricing.

Bets are pooled **per match, not per leg** — a rule that fires on two outcomes of
one fixture has one dependent result, not two independent ones.

#### What it finds on 35,883 real matches

The market is not perfectly calibrated. Over the Serie A discovery period, short
prices come in slightly more often than they imply and long prices slightly less —
the classic favourite–longshot bias:

| odds | legs | market implies | actually happens | gap |
|---|---|---|---|---|
| 1.00–1.50 | 938 | 0.7362 | 0.7623 | **+0.0261** |
| 1.50–2.00 | 1,812 | 0.5596 | 0.5822 | **+0.0226** |
| 2.00–3.00 | 3,223 | 0.3952 | 0.4083 | +0.0171 |
| 3.00–4.00 | 5,302 | 0.2809 | 0.2697 | −0.0106 |
| 4.00–6.00 | 2,390 | 0.1992 | 0.1862 | −0.0134 |
| 6.00–10.00 | 1,237 | 0.1236 | 0.1188 | −0.0043 |
| 10.00+ | 473 | 0.0610 | 0.0359 | **−0.0247** |

The best rule the search found — back anything priced 1.50–2.00, at the best
available price — returned **+4.97%** over the discovery period, `t = +2.42`. With
20 eligible hypotheses, **about 1 was expected to clear `|t| ≥ 2` by chance alone.**

Then the gates:

| validation set | matches | yield | t |
|---|---|---|---|
| ita.1 2019+ | 886 | +6.22% | +2.17 |
| eng.1 (all years) | 2,473 | +3.20% | +1.87 |
| esp.1 (all years) | 2,215 | +1.37% | +0.75 |
| fra.1 (all years) | 2,234 | −0.94% | −0.51 |
| ger.1 (all years) | 1,955 | −3.51% | −1.80 |
| **pooled** | **9,763** | **+0.77%** | 95% CI **[−0.93%, +2.48%]** |

**Not established**, on three counts: the interval contains zero, only 3 of 5 sets
are positive, and the same rule at a single bookmaker returns **−3.75%** — the
entire positive figure is the +4.52pp that price selection recovers, not the bias.

The favourite–longshot bias is real and visible. It is also smaller than the margin,
which is presumably why the bookmaker leaves it there.

The gates are tested in both directions: a fabricated market where a 1.80 shot wins
70% of the time comes back `ESTABLISHED`, and a fairly priced one comes back empty.
A search that can only ever say "no edge" is a slogan, not a search.

---

## ESPN API Endpoints Reference

This service consumes ESPN's undocumented public APIs. Below is a reference of available endpoints.

### Base URLs

| Domain | Purpose |
|--------|---------|
| `site.api.espn.com` | Scores, news, teams, standings |
| `sports.core.api.espn.com` | Athletes, stats, odds |
| `cdn.espn.com` | CDN-optimized live data |

### Supported Sports & Leagues

| Sport | League | Sport Slug | League Slug |
|-------|--------|------------|-------------|
| Football | NFL | `football` | `nfl` |
| Football | College | `football` | `college-football` |
| Football | CFL | `football` | `cfl` |
| Football | UFL | `football` | `ufl` |
| Basketball | NBA | `basketball` | `nba` |
| Basketball | WNBA | `basketball` | `wnba` |
| Basketball | NCAAM | `basketball` | `mens-college-basketball` |
| Basketball | NCAAW | `basketball` | `womens-college-basketball` |
| Baseball | MLB | `baseball` | `mlb` |
| Hockey | NHL | `hockey` | `nhl` |
| Soccer | EPL | `soccer` | `eng.1` |
| Soccer | MLS | `soccer` | `usa.1` |
| Soccer | UCL | `soccer` | `uefa.champions` |
| Soccer | 260+ leagues | `soccer` | See [soccer.md](../docs/sports/soccer.md) |
| MMA | UFC | `mma` | `ufc` |
| Golf | PGA | `golf` | `pga` |
| Golf | LPGA | `golf` | `lpga` |
| Golf | LIV | `golf` | `liv` |
| Tennis | ATP | `tennis` | `atp` |
| Tennis | WTA | `tennis` | `wta` |
| Racing | F1 | `racing` | `f1` |
| Racing | IndyCar | `racing` | `irl` |
| Racing | NASCAR Cup | `racing` | `nascar-premier` |
| Rugby Union | World Cup | `rugby` | `164205` |
| Rugby Union | Six Nations | `rugby` | `180659` |
| Rugby League | NRL / Super League | `rugby-league` | `3` |
| Lacrosse | PLL | `lacrosse` | `pll` |
| Lacrosse | NLL | `lacrosse` | `nll` |
| Australian Football | AFL | `australian-football` | `afl` |
| Cricket | ICC T20 | `cricket` | `icc.t20` |
| Cricket | IPL | `cricket` | `ipl` |
| Volleyball | FIVB Women | `volleyball` | `fivb.w` |
| Volleyball | FIVB Men | `volleyball` | `fivb.m` |

### Soccer League Codes

| League | Code |
|--------|------|
| Premier League | `eng.1` |
| La Liga | `esp.1` |
| Bundesliga | `ger.1` |
| Serie A | `ita.1` |
| Ligue 1 | `fra.1` |
| MLS | `usa.1` |
| Champions League | `uefa.champions` |

### ESPN Endpoint Patterns

**Site API (General Data):**
```
https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/{resource}
```

| Resource | Path |
|----------|------|
| Scoreboard | `/scoreboard` |
| Teams | `/teams` |
| Team Detail | `/teams/{id}` |
| Standings | `/standings` |
| News | `/news` |
| Game Summary | `/summary?event={id}` |

**Core API (Detailed Data):**
```
https://sports.core.api.espn.com/v2/sports/{sport}/leagues/{league}/{resource}
```

| Resource | Path |
|----------|------|
| Athletes | `/athletes?limit=1000` |
| Seasons | `/seasons` |
| Events | `/events?dates=2024` |
| Odds | `/events/{id}/competitions/{id}/odds` |

### ESPN Client Configuration

```python
ESPN_CLIENT = {
    "SITE_API_BASE_URL": "https://site.api.espn.com",
    "CORE_API_BASE_URL": "https://sports.core.api.espn.com",
    "TIMEOUT": 30.0,
    "MAX_RETRIES": 3,
    "RETRY_BACKOFF": 1.0,
}
```

---

## Example Commands

### curl Examples

```bash
# Ingest NBA teams
curl -X POST http://localhost:8000/api/v1/ingest/teams/ \
  -H "Content-Type: application/json" \
  -d '{"sport": "basketball", "league": "nba"}'

# Ingest NFL scoreboard
curl -X POST http://localhost:8000/api/v1/ingest/scoreboard/ \
  -H "Content-Type: application/json" \
  -d '{"sport": "football", "league": "nfl"}'

# Query teams
curl "http://localhost:8000/api/v1/teams/?league=nba"
curl "http://localhost:8000/api/v1/teams/?search=Lakers"

# Query events
curl "http://localhost:8000/api/v1/events/?league=nba&date=2024-12-15"
curl "http://localhost:8000/api/v1/events/?team=LAL&status=final"

# Health check
curl http://localhost:8000/healthz
```

### Management Commands

```bash
# Ingest teams for a single league
python manage.py ingest_teams basketball nba

# Ingest scoreboard for a single league
python manage.py ingest_scoreboard basketball nba --date=20241215

# Ingest teams for ALL 17 sports (40+ leagues)
python manage.py ingest_all_teams

# Filter to a single sport
python manage.py ingest_all_teams --sport soccer

# Preview what would run without ingesting
python manage.py ingest_all_teams --dry-run

# Analyse the next scheduled fixtures (or add --event <espn_id> for one match)
python manage.py analyze_match --league nba --upcoming 3
python manage.py analyze_match --event 401584666 --json

# Search the market for a price bias, with the out-of-sample gates applied
python manage.py find_market_bias ita.1 --split-year 2019
```

### Running without ESPN access

Where ESPN is unreachable (offline work, restricted egress, CI), `seed_demo_data`
generates a **synthetic** league of fictional teams and pushes it through the real
ingestion service, so the API and the analysis endpoints have something to work on:

```bash
python manage.py migrate
python manage.py seed_demo_data --rounds 10          # soccer-style, draws included
python manage.py seed_demo_data --profile basketball # high-scoring, no draws
python manage.py analyze_match --league demo.1 --upcoming 1
```

The generated teams and results are invented and reproducible from `--seed`. They are
never real ESPN data — keep them out of any environment where that distinction matters.

---

## Celery Background Jobs

```bash
# Start worker
celery -A config worker -l INFO

# Start scheduler
celery -A config beat -l INFO
```

### Scheduled Tasks

| Task | Schedule | Description |
|------|----------|-------------|
| `refresh_scoreboard_task` | On-demand | Refresh scoreboard for a specific sport/league/date |
| `refresh_teams_task` | On-demand | Refresh teams for a specific sport/league |
| `refresh_all_teams_task` | Weekly | Refresh all team data (40+ leagues, all 17 sports) |
| `refresh_daily_scoreboards_task` | Hourly | Refresh today's scoreboards (40+ leagues, all 17 sports) |

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Django secret key | Required in prod |
| `DEBUG` | Debug mode | `False` |
| `DATABASE_URL` | PostgreSQL URL | sqlite for local |
| `CELERY_BROKER_URL` | Redis URL | `redis://localhost:6379/0` |
| `ESPN_TIMEOUT` | API timeout (sec) | `30.0` |
| `ESPN_MAX_RETRIES` | Max retries | `3` |
| `ALLOWED_HOSTS` | Allowed hosts | `localhost,127.0.0.1` |

---

## Project Structure

```
espn_service/
├── config/                # Django configuration
│   ├── settings/
│   │   ├── base.py       # Base settings
│   │   ├── local.py      # Local development
│   │   ├── production.py # Production
│   │   └── test.py       # Test settings
│   ├── celery.py         # Celery config
│   └── urls.py           # URL routing
├── apps/
│   ├── core/             # Core utilities
│   ├── espn/             # ESPN data models & API
│   └── ingest/           # Data ingestion
├── clients/
│   └── espn_client.py    # ESPN API client
├── tests/                # Test suite
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Database Models

| Model | Description |
|-------|-------------|
| `Sport` | Sport types (basketball, football) |
| `League` | Leagues within sports (NBA, NFL) |
| `Team` | Team info with logos, colors |
| `Venue` | Stadium/arena information |
| `Event` | Games with status, scores |
| `Competitor` | Team participation in events |
| `Athlete` | Player information |

---

## Testing

```bash
# All tests with coverage
make test

# Quick tests
make test-fast

# Specific file
pytest tests/test_api.py -v
```

---

## Production Deployment

### Docker Production

```bash
docker compose -f docker-compose.prod.yml up -d
```

### Cloud Platforms

**AWS ECS/Fargate:**
```bash
docker build -t espn-service:latest .
docker push <account>.dkr.ecr.<region>.amazonaws.com/espn-service:latest
```

**Google Cloud Run:**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/espn-service
gcloud run deploy espn-service --image gcr.io/PROJECT_ID/espn-service
```

**Fly.io:**
```bash
fly launch
fly secrets set SECRET_KEY=your-key DATABASE_URL=your-url
fly deploy
```

---

## API Documentation

Once running:
- **Swagger UI**: http://localhost:8000/api/docs/
- **ReDoc**: http://localhost:8000/api/redoc/
- **OpenAPI Schema**: http://localhost:8000/api/schema/

---

## License

MIT License - See LICENSE file
