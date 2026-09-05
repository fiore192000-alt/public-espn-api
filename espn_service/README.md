# ESPN Service API

A production-ready Django REST API for ingesting and querying ESPN sports data.

## Features

- **Data Ingestion**: Fetch and persist data from ESPN's public/undocumented API endpoints
- **REST API**: Clean, paginated endpoints for querying teams, events, and games
- **Match Analysis**: Team form, head-to-head, projected scores and win probabilities from stored history
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
| `home` / `away` | Team, current score, `form` (W-D-L, scoring rates, home/away splits, streak, game log) and open injuries |
| `head_to_head` | Previous meetings, wins per side, combined points per game |
| `league_baseline` | League scoring level, empirical home advantage, margin spread, draw rate |
| `projection` | Expected score per side, margin, and `home_win` / `draw` / `away_win` probabilities |
| `confidence` | `none` / `low` / `medium` / `high`, from the available sample size |
| `insights` | Plain-language notes summarising the above |

Projected scores blend each side's scoring rate with the opponent's concession rate,
using home/away splits once they hold at least three games and falling back to the
overall record otherwise. Win probabilities come from the projected margin against the
league's own margin spread, with the draw share taken from the league's observed draw rate
— so leagues without draws simply get zero.

```bash
curl "http://localhost:8000/api/v1/events/44/analysis/?lookback=10"
curl "http://localhost:8000/api/v1/teams/7/form/?lookback=5"
```

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
