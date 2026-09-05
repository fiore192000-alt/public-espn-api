"""Match analysis computed from stored events.

Everything in this module reads only from the local database, so analysis works
against whatever history has already been ingested — no ESPN calls are made.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from django.db.models import Prefetch

from apps.espn.models import Competitor, Event, Injury, League, Team

DEFAULT_LOOKBACK = 10
MIN_SPLIT_SAMPLE = 3
FALLBACK_MARGIN_SIGMA = 10.0

RESULT_WIN = "win"
RESULT_DRAW = "draw"
RESULT_LOSS = "loss"


class AnalysisNotAvailable(Exception):
    """Raised when an event cannot be analysed (e.g. it is not a two-sided match)."""


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class GameResult:
    """One completed game from a single team's point of view."""

    event_id: int
    espn_id: str
    date: datetime
    name: str
    home_away: str
    opponent: str
    opponent_abbreviation: str
    scored: int
    conceded: int
    result: str

    @property
    def margin(self) -> int:
        return self.scored - self.conceded

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["date"] = self.date.isoformat()
        data["margin"] = self.margin
        return data


@dataclass
class SplitRecord:
    """Win/draw/loss record over a subset of games."""

    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    scored: int = 0
    conceded: int = 0

    def add(self, game: GameResult) -> None:
        self.played += 1
        self.scored += game.scored
        self.conceded += game.conceded
        if game.result == RESULT_WIN:
            self.wins += 1
        elif game.result == RESULT_DRAW:
            self.draws += 1
        else:
            self.losses += 1

    @property
    def avg_scored(self) -> float:
        return round(self.scored / self.played, 2) if self.played else 0.0

    @property
    def avg_conceded(self) -> float:
        return round(self.conceded / self.played, 2) if self.played else 0.0

    @property
    def win_pct(self) -> float:
        if not self.played:
            return 0.0
        return round((self.wins + 0.5 * self.draws) / self.played, 3)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["avg_scored"] = self.avg_scored
        data["avg_conceded"] = self.avg_conceded
        data["win_pct"] = self.win_pct
        return data


@dataclass
class TeamForm:
    """Recent form for one team."""

    team_id: int
    espn_id: str
    abbreviation: str
    display_name: str
    overall: SplitRecord = field(default_factory=SplitRecord)
    home: SplitRecord = field(default_factory=SplitRecord)
    away: SplitRecord = field(default_factory=SplitRecord)
    streak: str = ""
    games: list[GameResult] = field(default_factory=list)

    @property
    def point_differential(self) -> int:
        return self.overall.scored - self.overall.conceded

    def to_dict(self) -> dict[str, Any]:
        return {
            "team_id": self.team_id,
            "espn_id": self.espn_id,
            "abbreviation": self.abbreviation,
            "display_name": self.display_name,
            "overall": self.overall.to_dict(),
            "home": self.home.to_dict(),
            "away": self.away.to_dict(),
            "streak": self.streak,
            "point_differential": self.point_differential,
            "games": [g.to_dict() for g in self.games],
        }


@dataclass
class HeadToHead:
    """Historical record between two teams."""

    played: int = 0
    team_a_wins: int = 0
    team_b_wins: int = 0
    draws: int = 0
    team_a_points: int = 0
    team_b_points: int = 0
    meetings: list[dict[str, Any]] = field(default_factory=list)

    @property
    def avg_total_points(self) -> float:
        if not self.played:
            return 0.0
        return round((self.team_a_points + self.team_b_points) / self.played, 2)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["avg_total_points"] = self.avg_total_points
        return data


@dataclass
class LeagueBaseline:
    """League-wide scoring context used to calibrate projections."""

    sample: int = 0
    avg_home_score: float = 0.0
    avg_away_score: float = 0.0
    home_advantage: float = 0.0
    margin_sigma: float = FALLBACK_MARGIN_SIGMA
    draw_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _score_of(competitor: Competitor) -> int | None:
    return competitor.score_int


def _result_for(scored: int, conceded: int) -> str:
    if scored > conceded:
        return RESULT_WIN
    if scored < conceded:
        return RESULT_LOSS
    return RESULT_DRAW


def _completed_events(league: League, before: datetime | None = None):
    qs = (
        Event.objects.filter(league=league, status=Event.STATUS_FINAL)
        .prefetch_related(
            Prefetch("competitors", queryset=Competitor.objects.select_related("team"))
        )
        .order_by("-date")
    )
    if before is not None:
        qs = qs.filter(date__lt=before)
    return qs


def _sided_competitors(event: Event) -> tuple[Competitor, Competitor] | None:
    """Return (home, away) competitors, or None when the event is not two-sided."""
    competitors = list(event.competitors.all())
    if len(competitors) != 2:
        return None
    home = next((c for c in competitors if c.home_away == Competitor.HOME), None)
    away = next((c for c in competitors if c.home_away == Competitor.AWAY), None)
    if home is None or away is None:
        competitors.sort(key=lambda c: c.order)
        away, home = competitors
    return home, away


def build_team_form(
    team: Team,
    before: datetime | None = None,
    lookback: int = DEFAULT_LOOKBACK,
) -> TeamForm:
    """Summarise a team's most recent completed games."""
    form = TeamForm(
        team_id=team.pk,
        espn_id=team.espn_id,
        abbreviation=team.abbreviation,
        display_name=team.display_name,
    )

    events = _completed_events(team.league, before).filter(competitors__team=team).distinct()

    for event in events:
        if len(form.games) >= lookback:
            break
        sides = _sided_competitors(event)
        if sides is None:
            continue
        home, away = sides
        own, opponent = (home, away) if home.team_id == team.pk else (away, home)
        scored, conceded = _score_of(own), _score_of(opponent)
        if scored is None or conceded is None:
            continue

        game = GameResult(
            event_id=event.pk,
            espn_id=event.espn_id,
            date=event.date,
            name=event.short_name or event.name,
            home_away=own.home_away,
            opponent=opponent.team.display_name,
            opponent_abbreviation=opponent.team.abbreviation,
            scored=scored,
            conceded=conceded,
            result=_result_for(scored, conceded),
        )
        form.games.append(game)
        form.overall.add(game)
        (form.home if own.home_away == Competitor.HOME else form.away).add(game)

    form.streak = _streak_of(form.games)
    return form


def _streak_of(games: list[GameResult]) -> str:
    """Build a streak label such as "W3" from the most recent games."""
    if not games:
        return ""
    latest = games[0].result
    count = 0
    for game in games:
        if game.result != latest:
            break
        count += 1
    return f"{latest[0].upper()}{count}"


def build_head_to_head(
    team_a: Team,
    team_b: Team,
    before: datetime | None = None,
    limit: int = DEFAULT_LOOKBACK,
) -> HeadToHead:
    """Summarise previous meetings between two teams, from team_a's perspective."""
    h2h = HeadToHead()

    events = (
        _completed_events(team_a.league, before)
        .filter(competitors__team=team_a)
        .filter(competitors__team=team_b)
        .distinct()
    )

    for event in events:
        if h2h.played >= limit:
            break
        sides = _sided_competitors(event)
        if sides is None:
            continue
        home, away = sides
        comp_a = home if home.team_id == team_a.pk else away
        comp_b = away if comp_a is home else home
        score_a, score_b = _score_of(comp_a), _score_of(comp_b)
        if score_a is None or score_b is None:
            continue

        h2h.played += 1
        h2h.team_a_points += score_a
        h2h.team_b_points += score_b
        if score_a > score_b:
            h2h.team_a_wins += 1
        elif score_a < score_b:
            h2h.team_b_wins += 1
        else:
            h2h.draws += 1

        h2h.meetings.append(
            {
                "event_id": event.pk,
                "espn_id": event.espn_id,
                "date": event.date.isoformat(),
                "name": event.short_name or event.name,
                "home": home.team.abbreviation,
                "away": away.team.abbreviation,
                "home_score": _score_of(home),
                "away_score": _score_of(away),
            }
        )

    return h2h


def build_league_baseline(
    league: League,
    before: datetime | None = None,
    sample_size: int = 200,
) -> LeagueBaseline:
    """Derive league-wide home advantage, scoring level and margin spread."""
    home_scores: list[int] = []
    away_scores: list[int] = []
    margins: list[int] = []
    draws = 0

    for event in _completed_events(league, before)[:sample_size]:
        sides = _sided_competitors(event)
        if sides is None:
            continue
        home, away = sides
        home_score, away_score = _score_of(home), _score_of(away)
        if home_score is None or away_score is None:
            continue
        home_scores.append(home_score)
        away_scores.append(away_score)
        margins.append(home_score - away_score)
        if home_score == away_score:
            draws += 1

    sample = len(margins)
    if not sample:
        return LeagueBaseline()

    sigma = statistics.pstdev(margins) if sample > 1 else 0.0
    return LeagueBaseline(
        sample=sample,
        avg_home_score=round(_mean(home_scores), 2),
        avg_away_score=round(_mean(away_scores), 2),
        home_advantage=round(_mean(margins), 2),
        margin_sigma=round(sigma, 2) if sigma > 0 else FALLBACK_MARGIN_SIGMA,
        draw_rate=round(draws / sample, 3),
    )


def _expected_score(
    attack: SplitRecord,
    attack_overall: SplitRecord,
    defence: SplitRecord,
    defence_overall: SplitRecord,
) -> float:
    """Blend a team's scoring rate with its opponent's concession rate.

    Home/away splits are used once they hold enough games, otherwise the
    overall record stands in so a thin split does not dominate the estimate.
    """
    scoring = attack if attack.played >= MIN_SPLIT_SAMPLE else attack_overall
    conceding = defence if defence.played >= MIN_SPLIT_SAMPLE else defence_overall
    values = []
    if scoring.played:
        values.append(scoring.avg_scored)
    if conceding.played:
        values.append(conceding.avg_conceded)
    return _mean(values)


def _probabilities(margin: float, baseline: LeagueBaseline) -> dict[str, float]:
    sigma = baseline.margin_sigma or FALLBACK_MARGIN_SIGMA
    home_share = _normal_cdf(margin / sigma)
    draw = baseline.draw_rate
    probs = {
        "home_win": (1 - draw) * home_share,
        "draw": draw,
        "away_win": (1 - draw) * (1 - home_share),
    }
    rounded = {k: round(v, 4) for k, v in probs.items()}
    # Absorb rounding drift into the most likely outcome so the split sums to 1.
    leader = max(rounded, key=lambda k: rounded[k])
    rounded[leader] = round(rounded[leader] + (1 - sum(rounded.values())), 4)
    return rounded


def _confidence(home_form: TeamForm, away_form: TeamForm, baseline: LeagueBaseline) -> str:
    games = min(home_form.overall.played, away_form.overall.played)
    if games == 0 or baseline.sample == 0:
        return "none"
    if games >= 8 and baseline.sample >= 40:
        return "high"
    if games >= 4:
        return "medium"
    return "low"


def _insights(
    home: Team,
    away: Team,
    home_form: TeamForm,
    away_form: TeamForm,
    h2h: HeadToHead,
    projection: dict[str, Any],
) -> list[str]:
    notes: list[str] = []

    for team, form in ((home, home_form), (away, away_form)):
        if not form.overall.played:
            notes.append(f"No completed games on record for {team.display_name}.")
            continue
        notes.append(
            f"{team.abbreviation} last {form.overall.played}: "
            f"{form.overall.wins}-{form.overall.draws}-{form.overall.losses} "
            f"({form.overall.avg_scored} scored / {form.overall.avg_conceded} conceded per game)."
        )
        if form.streak and int(form.streak[1:]) >= 3:
            notes.append(f"{form.abbreviation} is on a {form.streak} run.")

    if h2h.played:
        notes.append(
            f"Head-to-head over {h2h.played} meeting{'s' if h2h.played > 1 else ''}: "
            f"{home.abbreviation} {h2h.team_a_wins} - {h2h.team_b_wins} {away.abbreviation}"
            + (f" ({h2h.draws} drawn)" if h2h.draws else "")
            + f", {h2h.avg_total_points} combined points per game."
        )
    else:
        notes.append("No previous meetings on record between these teams.")

    margin = projection["margin"]
    favourite = home.abbreviation if margin >= 0 else away.abbreviation
    notes.append(
        f"Model leans {favourite} by {abs(margin)} "
        f"({projection['home_score']} - {projection['away_score']} projected)."
    )
    return notes


def _injury_summary(event: Event, team: Team) -> dict[str, Any]:
    counts: dict[str, int] = {}
    names: list[str] = []
    for injury in Injury.objects.filter(league=event.league, team=team).order_by("athlete_name"):
        counts[injury.status] = counts.get(injury.status, 0) + 1
        if len(names) < 10:
            names.append(f"{injury.athlete_name} ({injury.status_display or injury.status})")
    return {"total": sum(counts.values()), "by_status": counts, "players": names}


def analyze_event(event: Event, lookback: int = DEFAULT_LOOKBACK) -> dict[str, Any]:
    """Produce a full analysis payload for a two-sided event."""
    sides = _sided_competitors(event)
    if sides is None:
        raise AnalysisNotAvailable(
            f"Event {event.espn_id} does not have exactly two competitors to compare."
        )
    home_comp, away_comp = sides
    home_team, away_team = home_comp.team, away_comp.team

    home_form = build_team_form(home_team, before=event.date, lookback=lookback)
    away_form = build_team_form(away_team, before=event.date, lookback=lookback)
    h2h = build_head_to_head(home_team, away_team, before=event.date, limit=lookback)
    baseline = build_league_baseline(event.league, before=event.date)

    expected_home = _expected_score(
        home_form.home, home_form.overall, away_form.away, away_form.overall
    )
    expected_away = _expected_score(
        away_form.away, away_form.overall, home_form.home, home_form.overall
    )

    # When splits are too thin to carry it, add the league's home edge explicitly.
    if home_form.home.played < MIN_SPLIT_SAMPLE or away_form.away.played < MIN_SPLIT_SAMPLE:
        expected_home += baseline.home_advantage / 2
        expected_away -= baseline.home_advantage / 2

    expected_home = max(expected_home, 0.0)
    expected_away = max(expected_away, 0.0)
    margin = round(expected_home - expected_away, 2)

    projection = {
        "home_score": round(expected_home, 2),
        "away_score": round(expected_away, 2),
        "total": round(expected_home + expected_away, 2),
        "margin": margin,
        "probabilities": _probabilities(margin, baseline),
    }

    return {
        "event": {
            "id": event.pk,
            "espn_id": event.espn_id,
            "name": event.name,
            "short_name": event.short_name,
            "date": event.date.isoformat(),
            "status": event.status,
            "status_detail": event.status_detail,
            "league": event.league.slug,
            "sport": event.league.sport.slug,
            "venue": event.venue.name if event.venue else None,
        },
        "home": {
            "team": {
                "id": home_team.pk,
                "espn_id": home_team.espn_id,
                "abbreviation": home_team.abbreviation,
                "display_name": home_team.display_name,
                "logo": home_team.primary_logo,
            },
            "score": home_comp.score_int,
            "form": home_form.to_dict(),
            "injuries": _injury_summary(event, home_team),
        },
        "away": {
            "team": {
                "id": away_team.pk,
                "espn_id": away_team.espn_id,
                "abbreviation": away_team.abbreviation,
                "display_name": away_team.display_name,
                "logo": away_team.primary_logo,
            },
            "score": away_comp.score_int,
            "form": away_form.to_dict(),
            "injuries": _injury_summary(event, away_team),
        },
        "head_to_head": h2h.to_dict(),
        "league_baseline": baseline.to_dict(),
        "projection": projection,
        "confidence": _confidence(home_form, away_form, baseline),
        "insights": _insights(home_team, away_team, home_form, away_form, h2h, projection),
        "lookback": lookback,
    }
