"""Situational context around a fixture: rest, congestion and absences.

These are the circumstances a form table cannot see — a side playing its third
match in eight days, or one missing players who actually start. They are reported
alongside the model rather than folded into it: none of them is wired into the
scoreline projection, because nothing here has been validated against real
results yet, and an unvalidated adjustment makes a model worse while looking
sophisticated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

from apps.espn.models import AthleteSeasonStats, Competitor, Event, Injury, League, Team

CONGESTION_WINDOW_DAYS = 14
# A side playing this often in the window is in a genuinely congested run.
CONGESTION_HEAVY = 4

# How much each injury status counts towards the burden. A player who is out is
# a whole absence; a doubtful one is a partial expectation of absence.
STATUS_WEIGHTS = {
    Injury.STATUS_OUT: 1.0,
    Injury.STATUS_IR: 1.0,
    Injury.STATUS_DOUBTFUL: 0.75,
    Injury.STATUS_QUESTIONABLE: 0.4,
    Injury.STATUS_DAY_TO_DAY: 0.3,
    Injury.STATUS_OTHER: 0.2,
}
DEFAULT_STATUS_WEIGHT = 0.2

MAX_LISTED_PLAYERS = 10


@dataclass
class AbsentPlayer:
    """One injured player and how much their absence is judged to matter."""

    name: str
    status: str
    status_display: str
    position: str
    severity: float
    importance: float
    weight: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InjuryBurden:
    """Squad absences, counted and weighted.

    ``count`` is how many players are listed; ``weighted`` scales each by how
    likely they are to miss the match and by how much they play. Importance uses
    stored season appearances where available — ESPN's injury feed does not say
    who starts, so without those stats every absence counts the same and
    ``importance_known`` reports that.
    """

    count: int = 0
    weighted: float = 0.0
    importance_known: bool = False
    by_status: dict[str, int] = field(default_factory=dict)
    players: list[AbsentPlayer] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "weighted": round(self.weighted, 3),
            "importance_known": self.importance_known,
            "by_status": self.by_status,
            "players": [player.to_dict() for player in self.players],
        }


@dataclass
class MatchContext:
    """Everything situational known about one side going into a fixture."""

    rest_days: int | None = None
    matches_in_window: int = 0
    congested: bool = False
    injuries: InjuryBurden = field(default_factory=InjuryBurden)

    def to_dict(self) -> dict:
        return {
            "rest_days": self.rest_days,
            "matches_in_last_14_days": self.matches_in_window,
            "congested": self.congested,
            "injuries": self.injuries.to_dict(),
        }


def _team_events_before(team: Team, before: datetime):
    return (
        Event.objects.filter(
            league=team.league,
            status=Event.STATUS_FINAL,
            date__lt=before,
            competitors__team=team,
        )
        .distinct()
        .order_by("-date")
    )


def rest_days(team: Team, before: datetime) -> int | None:
    """Whole days since this team's previous completed match."""
    previous = _team_events_before(team, before).first()
    if previous is None:
        return None
    return max((before - previous.date).days, 0)


def matches_in_window(
    team: Team,
    before: datetime,
    window_days: int = CONGESTION_WINDOW_DAYS,
) -> int:
    """Completed matches in the window immediately preceding the fixture."""
    return (
        _team_events_before(team, before)
        .filter(date__gte=before - timedelta(days=window_days))
        .count()
    )


def _appearance_share(league: League, athlete_espn_id: str) -> float | None:
    """Fraction of the league-leading appearance count this athlete has played.

    Used as a stand-in for whether someone is a regular starter. Returns None when
    no usable appearance figure is stored, so callers can tell "not important"
    apart from "not known".
    """
    if not athlete_espn_id:
        return None

    record = (
        AthleteSeasonStats.objects.filter(league=league, athlete_espn_id=athlete_espn_id)
        .order_by("-season_year")
        .first()
    )
    if record is None:
        return None

    appearances = _appearances_from(record.stats)
    if appearances is None:
        return None

    league_best = _league_best_appearances(league, record.season_year)
    if not league_best:
        return None
    return min(appearances / league_best, 1.0)


def _appearances_from(stats: dict) -> float | None:
    """Pull an appearance count out of the flexible stats blob."""
    if not isinstance(stats, dict):
        return None
    for key in ("appearances", "gamesPlayed", "games_played", "starts"):
        value = stats.get(key)
        if isinstance(value, int | float) and value >= 0:
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _league_best_appearances(league: League, season_year: int) -> float | None:
    best = 0.0
    for record in AthleteSeasonStats.objects.filter(league=league, season_year=season_year):
        appearances = _appearances_from(record.stats)
        if appearances is not None:
            best = max(best, appearances)
    return best or None


def injury_burden(league: League, team: Team) -> InjuryBurden:
    """Weigh a team's listed absences by severity and, where known, by playing time."""
    burden = InjuryBurden()

    for injury in Injury.objects.filter(league=league, team=team).order_by("athlete_name"):
        severity = STATUS_WEIGHTS.get(injury.status, DEFAULT_STATUS_WEIGHT)
        share = _appearance_share(league, injury.athlete_espn_id)
        if share is not None:
            burden.importance_known = True
        importance = share if share is not None else 1.0

        burden.count += 1
        burden.weighted += severity * importance
        burden.by_status[injury.status] = burden.by_status.get(injury.status, 0) + 1

        if len(burden.players) < MAX_LISTED_PLAYERS:
            burden.players.append(
                AbsentPlayer(
                    name=injury.athlete_name,
                    status=injury.status,
                    status_display=injury.status_display or injury.status,
                    position=injury.position,
                    severity=severity,
                    importance=round(importance, 3),
                    weight=round(severity * importance, 3),
                )
            )

    burden.players.sort(key=lambda player: player.weight, reverse=True)
    return burden


def build_context(event: Event, competitor: Competitor) -> MatchContext:
    """Assemble the situational picture for one side of a fixture."""
    team = competitor.team
    played = matches_in_window(team, event.date)
    return MatchContext(
        rest_days=rest_days(team, event.date),
        matches_in_window=played,
        congested=played >= CONGESTION_HEAVY,
        injuries=injury_burden(event.league, team),
    )
