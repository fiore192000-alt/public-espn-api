"""Views for ESPN data API endpoints."""

from django.db.models import QuerySet
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.request import Request
from rest_framework.response import Response

from apps.espn.analysis import (
    DEFAULT_LOOKBACK,
    AnalysisNotAvailable,
    analyze_event,
    build_team_form,
)
from apps.espn.dixon_coles import DEFAULT_HALF_LIFE_DAYS, NotEnoughData
from apps.espn.filters import EventFilter, TeamFilter
from apps.espn.forecast import forecast_event
from apps.espn.models import (
    AthleteSeasonStats,
    Event,
    Injury,
    League,
    NewsArticle,
    Sport,
    Team,
    Transaction,
)
from apps.espn.serializers import (
    AthleteSeasonStatsSerializer,
    EventListSerializer,
    EventSerializer,
    InjurySerializer,
    LeagueSerializer,
    NewsArticleListSerializer,
    NewsArticleSerializer,
    SportSerializer,
    TeamListSerializer,
    TeamSerializer,
    TransactionSerializer,
)
from apps.espn.value import (
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_KELLY_FRACTION,
    PricedSelection,
    find_value_bets,
)

MAX_LOOKBACK = 50

LOOKBACK_PARAMETER = OpenApiParameter(
    "lookback",
    description=f"Games of history per team, 1-{MAX_LOOKBACK} (default: {DEFAULT_LOOKBACK})",
    type=int,
)


def _lookback_param(request: Request) -> int:
    raw = request.query_params.get("lookback", "")
    if not raw.isdigit():
        return DEFAULT_LOOKBACK
    return max(1, min(int(raw), MAX_LOOKBACK))


def _float_param(request: Request, name: str, default: float) -> float:
    try:
        value = float(request.query_params[name])
    except (KeyError, TypeError, ValueError):
        return default
    return value if value > 0 else default


class SportViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Sport discovery."""

    serializer_class = SportSerializer
    lookup_field = "slug"

    def get_queryset(self) -> QuerySet[Sport]:
        return Sport.objects.prefetch_related("leagues").order_by("name")

    @extend_schema(tags=["Discovery"], summary="List sports")
    def list(self, request: Request, *args, **kwargs) -> Response:
        return super().list(request, *args, **kwargs)

    @extend_schema(tags=["Discovery"], summary="Get sport details")
    def retrieve(self, request: Request, *args, **kwargs) -> Response:
        return super().retrieve(request, *args, **kwargs)


class LeagueViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for League discovery."""

    serializer_class = LeagueSerializer

    def get_queryset(self) -> QuerySet[League]:
        qs = League.objects.select_related("sport").order_by("sport__name", "name")
        sport = self.request.query_params.get("sport")
        if sport:
            qs = qs.filter(sport__slug__iexact=sport)
        return qs

    @extend_schema(
        tags=["Discovery"],
        summary="List leagues",
        parameters=[OpenApiParameter("sport", description="Filter by sport slug", type=str)],
    )
    def list(self, request: Request, *args, **kwargs) -> Response:
        return super().list(request, *args, **kwargs)

    @extend_schema(tags=["Discovery"], summary="Get league details")
    def retrieve(self, request: Request, *args, **kwargs) -> Response:
        return super().retrieve(request, *args, **kwargs)


class TeamViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Team data."""

    filterset_class = TeamFilter
    search_fields = ["display_name", "abbreviation", "location", "name"]
    ordering_fields = ["display_name", "abbreviation", "created_at"]
    ordering = ["display_name"]

    def get_queryset(self) -> QuerySet[Team]:
        return Team.objects.select_related("league", "league__sport").filter(is_active=True)

    def get_serializer_class(self) -> type:
        if self.action == "list":
            return TeamListSerializer
        return TeamSerializer

    @extend_schema(
        tags=["Teams"],
        summary="List teams",
        parameters=[
            OpenApiParameter("sport", description="Filter by sport slug", type=str),
            OpenApiParameter("league", description="Filter by league slug", type=str),
            OpenApiParameter("search", description="Search in name/abbreviation/location", type=str),
        ],
    )
    def list(self, request: Request, *args, **kwargs) -> Response:
        return super().list(request, *args, **kwargs)

    @extend_schema(tags=["Teams"], summary="Get team details")
    def retrieve(self, request: Request, *args, **kwargs) -> Response:
        return super().retrieve(request, *args, **kwargs)

    @extend_schema(
        tags=["Teams"],
        summary="Get team by ESPN ID",
        parameters=[OpenApiParameter("espn_id", location=OpenApiParameter.PATH, type=str)],
    )
    @action(detail=False, methods=["get"], url_path="espn/(?P<espn_id>[^/.]+)")
    def by_espn_id(self, request: Request, espn_id: str) -> Response:  # noqa: ARG002
        team = self.get_queryset().filter(espn_id=espn_id).first()
        if not team:
            return Response({"error": "Team not found"}, status=404)
        return Response(TeamSerializer(team).data)

    @extend_schema(
        tags=["Teams"],
        summary="Get recent form for a team",
        description=(
            "Win/draw/loss record, scoring rates, home and away splits, current streak and "
            "the underlying games, computed from completed events already in the database."
        ),
        parameters=[LOOKBACK_PARAMETER],
        responses={200: OpenApiTypes.OBJECT},
    )
    @action(detail=True, methods=["get"])
    def form(self, request: Request, pk: str | None = None) -> Response:  # noqa: ARG002
        team = self.get_object()
        return Response(build_team_form(team, lookback=_lookback_param(request)).to_dict())


class EventViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Event/Game data."""

    filterset_class = EventFilter
    search_fields = ["name", "short_name"]
    ordering_fields = ["date", "created_at"]
    ordering = ["-date"]

    def get_queryset(self) -> QuerySet[Event]:
        return Event.objects.select_related(
            "league", "league__sport", "venue"
        ).prefetch_related("competitors", "competitors__team")

    def get_serializer_class(self) -> type:
        if self.action == "list":
            return EventListSerializer
        return EventSerializer

    @extend_schema(
        tags=["Events"],
        summary="List events",
        parameters=[
            OpenApiParameter("sport", description="Filter by sport slug", type=str),
            OpenApiParameter("league", description="Filter by league slug", type=str),
            OpenApiParameter("date", description="Filter by date (YYYY-MM-DD)", type=str),
            OpenApiParameter("date_from", description="Filter date >=", type=str),
            OpenApiParameter("date_to", description="Filter date <=", type=str),
            OpenApiParameter("status", description="Filter by status", type=str),
            OpenApiParameter("team", description="Filter by team ESPN ID or abbreviation", type=str),
        ],
    )
    def list(self, request: Request, *args, **kwargs) -> Response:
        return super().list(request, *args, **kwargs)

    @extend_schema(tags=["Events"], summary="Get event details")
    def retrieve(self, request: Request, *args, **kwargs) -> Response:
        return super().retrieve(request, *args, **kwargs)

    @extend_schema(
        tags=["Events"],
        summary="Get event by ESPN ID",
        parameters=[OpenApiParameter("espn_id", location=OpenApiParameter.PATH, type=str)],
    )
    @action(detail=False, methods=["get"], url_path="espn/(?P<espn_id>[^/.]+)")
    def by_espn_id(self, request: Request, espn_id: str) -> Response:  # noqa: ARG002
        event = self.get_queryset().filter(espn_id=espn_id).first()
        if not event:
            return Response({"error": "Event not found"}, status=404)
        return Response(EventSerializer(event).data)

    @extend_schema(
        tags=["Events"],
        summary="Analyse a match",
        description=(
            "Recent form for both sides, head-to-head history, league scoring baseline, and a "
            "projected score with win probabilities. Only history recorded before the event "
            "date is used, so past events can be analysed as they would have looked beforehand."
        ),
        parameters=[LOOKBACK_PARAMETER],
        responses={200: OpenApiTypes.OBJECT, 400: OpenApiTypes.OBJECT},
    )
    @action(detail=True, methods=["get"])
    def analysis(self, request: Request, pk: str | None = None) -> Response:  # noqa: ARG002
        event = self.get_object()
        try:
            return Response(analyze_event(event, lookback=_lookback_param(request)))
        except AnalysisNotAvailable as exc:
            return Response({"error": str(exc)}, status=400)

    @extend_schema(
        tags=["Events"],
        summary="Model-based market probabilities for a match",
        description=(
            "Fits a Dixon-Coles scoreline model to the league history preceding this "
            "event and returns probabilities for 1X2, double chance, over/under, both "
            "teams to score and the most likely correct scores, each with fair odds.\n\n"
            "Where bookmaker odds are stored for the event, selections whose model "
            "probability beats the devigged market price by at least `edge` are also "
            "returned, with a fractional Kelly stake. A positive edge is a hypothesis "
            "about a price, not a prediction of profit; `model.reliable` reports whether "
            "the fit rests on enough history to be worth acting on at all."
        ),
        parameters=[
            OpenApiParameter(
                "half_life",
                description=f"Days after which a past match counts half as much (default: {DEFAULT_HALF_LIFE_DAYS})",
                type=float,
            ),
            OpenApiParameter(
                "edge",
                description=f"Minimum edge over the devigged price (default: {DEFAULT_EDGE_THRESHOLD})",
                type=float,
            ),
            OpenApiParameter(
                "kelly",
                description=f"Fraction of full Kelly to stake (default: {DEFAULT_KELLY_FRACTION})",
                type=float,
            ),
        ],
        responses={200: OpenApiTypes.OBJECT, 400: OpenApiTypes.OBJECT},
    )
    @action(detail=True, methods=["get"])
    def forecast(self, request: Request, pk: str | None = None) -> Response:  # noqa: ARG002
        event = self.get_object()
        try:
            payload = forecast_event(
                event, half_life_days=_float_param(request, "half_life", DEFAULT_HALF_LIFE_DAYS)
            )
        except NotEnoughData as exc:
            return Response({"error": str(exc)}, status=400)

        prices = [
            PricedSelection(
                market=row.market,
                selection=row.selection,
                line=row.line,
                decimal_odds=row.decimal_odds,
                provider_espn_id=row.provider_espn_id,
                provider_name=row.provider_name,
            )
            for row in event.odds.all()
        ]
        bets = find_value_bets(
            payload["markets"],
            prices,
            edge_threshold=_float_param(request, "edge", DEFAULT_EDGE_THRESHOLD),
            kelly_multiplier=_float_param(request, "kelly", DEFAULT_KELLY_FRACTION),
        )
        payload["odds_available"] = len(prices)
        payload["value_bets"] = [bet.to_dict() for bet in bets]
        return Response(payload)


# ---------------------------------------------------------------------------
# New ViewSets — added in audit expansion
# ---------------------------------------------------------------------------


class NewsArticleViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for ESPN news articles."""

    search_fields = ["headline", "description"]
    ordering_fields = ["published", "created_at"]
    ordering = ["-published"]

    def get_queryset(self) -> QuerySet[NewsArticle]:
        qs = NewsArticle.objects.select_related("league", "league__sport")

        sport = self.request.query_params.get("sport")
        if sport:
            qs = qs.filter(league__sport__slug__iexact=sport)

        league = self.request.query_params.get("league")
        if league:
            qs = qs.filter(league__slug__iexact=league)

        date_from = self.request.query_params.get("date_from")
        if date_from:
            qs = qs.filter(published__date__gte=date_from)

        return qs

    def get_serializer_class(self) -> type:
        if self.action == "list":
            return NewsArticleListSerializer
        return NewsArticleSerializer

    @extend_schema(
        tags=["News"],
        summary="List news articles",
        parameters=[
            OpenApiParameter("sport", description="Filter by sport slug", type=str),
            OpenApiParameter("league", description="Filter by league slug", type=str),
            OpenApiParameter("date_from", description="Published on or after (YYYY-MM-DD)", type=str),
            OpenApiParameter("search", description="Search headline/description", type=str),
        ],
    )
    def list(self, request: Request, *args, **kwargs) -> Response:
        return super().list(request, *args, **kwargs)

    @extend_schema(tags=["News"], summary="Get news article detail")
    def retrieve(self, request: Request, *args, **kwargs) -> Response:
        return super().retrieve(request, *args, **kwargs)


class InjuryViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for league injury reports."""

    serializer_class = InjurySerializer
    search_fields = ["athlete_name", "injury_type", "description"]
    ordering_fields = ["updated_at", "athlete_name"]
    ordering = ["-updated_at"]

    def get_queryset(self) -> QuerySet[Injury]:
        qs = Injury.objects.select_related("league", "league__sport", "team")

        sport = self.request.query_params.get("sport")
        if sport:
            qs = qs.filter(league__sport__slug__iexact=sport)

        league = self.request.query_params.get("league")
        if league:
            qs = qs.filter(league__slug__iexact=league)

        status_param = self.request.query_params.get("status")
        if status_param:
            qs = qs.filter(status__iexact=status_param)

        team = self.request.query_params.get("team")
        if team:
            qs = qs.filter(team__abbreviation__iexact=team)

        return qs

    @extend_schema(
        tags=["Injuries"],
        summary="List injury reports",
        parameters=[
            OpenApiParameter("sport", description="Filter by sport slug", type=str),
            OpenApiParameter("league", description="Filter by league slug (e.g., 'nfl')", type=str),
            OpenApiParameter(
                "status",
                description="Filter by status (out, questionable, doubtful, ir, day_to_day)",
                type=str,
            ),
            OpenApiParameter("team", description="Filter by team abbreviation", type=str),
            OpenApiParameter("search", description="Search athlete name / injury type", type=str),
        ],
    )
    def list(self, request: Request, *args, **kwargs) -> Response:
        return super().list(request, *args, **kwargs)

    @extend_schema(tags=["Injuries"], summary="Get injury detail")
    def retrieve(self, request: Request, *args, **kwargs) -> Response:
        return super().retrieve(request, *args, **kwargs)


class TransactionViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for league transaction records."""

    serializer_class = TransactionSerializer
    search_fields = ["description", "athlete_name", "type"]
    ordering_fields = ["date", "created_at"]
    ordering = ["-date"]

    def get_queryset(self) -> QuerySet[Transaction]:
        qs = Transaction.objects.select_related("league", "league__sport", "team")

        sport = self.request.query_params.get("sport")
        if sport:
            qs = qs.filter(league__sport__slug__iexact=sport)

        league = self.request.query_params.get("league")
        if league:
            qs = qs.filter(league__slug__iexact=league)

        date_from = self.request.query_params.get("date_from")
        if date_from:
            qs = qs.filter(date__gte=date_from)

        return qs

    @extend_schema(
        tags=["Transactions"],
        summary="List transactions",
        parameters=[
            OpenApiParameter("sport", description="Filter by sport slug", type=str),
            OpenApiParameter("league", description="Filter by league slug", type=str),
            OpenApiParameter("date_from", description="Transactions on or after (YYYY-MM-DD)", type=str),
            OpenApiParameter("search", description="Search description / athlete name / type", type=str),
        ],
    )
    def list(self, request: Request, *args, **kwargs) -> Response:
        return super().list(request, *args, **kwargs)

    @extend_schema(tags=["Transactions"], summary="Get transaction detail")
    def retrieve(self, request: Request, *args, **kwargs) -> Response:
        return super().retrieve(request, *args, **kwargs)


class AthleteSeasonStatsViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for stored athlete season stats."""

    serializer_class = AthleteSeasonStatsSerializer
    search_fields = ["athlete_name"]
    ordering_fields = ["season_year", "athlete_name"]
    ordering = ["-season_year"]

    def get_queryset(self) -> QuerySet[AthleteSeasonStats]:
        qs = AthleteSeasonStats.objects.select_related("league", "league__sport", "athlete")

        sport = self.request.query_params.get("sport")
        if sport:
            qs = qs.filter(league__sport__slug__iexact=sport)

        league = self.request.query_params.get("league")
        if league:
            qs = qs.filter(league__slug__iexact=league)

        season = self.request.query_params.get("season")
        if season and season.isdigit():
            qs = qs.filter(season_year=int(season))

        athlete_id = self.request.query_params.get("athlete_espn_id")
        if athlete_id:
            qs = qs.filter(athlete_espn_id=athlete_id)

        return qs

    @extend_schema(
        tags=["Athlete Stats"],
        summary="List athlete season stats",
        parameters=[
            OpenApiParameter("sport", description="Filter by sport slug", type=str),
            OpenApiParameter("league", description="Filter by league slug", type=str),
            OpenApiParameter("season", description="Filter by season year (e.g., 2024)", type=int),
            OpenApiParameter("athlete_espn_id", description="Filter by ESPN athlete ID", type=str),
            OpenApiParameter("search", description="Search athlete name", type=str),
        ],
    )
    def list(self, request: Request, *args, **kwargs) -> Response:
        return super().list(request, *args, **kwargs)

    @extend_schema(tags=["Athlete Stats"], summary="Get athlete season stats detail")
    def retrieve(self, request: Request, *args, **kwargs) -> Response:
        return super().retrieve(request, *args, **kwargs)
