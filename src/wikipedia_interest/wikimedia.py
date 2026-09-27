"""Deterministic Wikimedia per-article monthly pageview client.

Methodology is fixed: access=all-access, agent=user, granularity=monthly.
Outcomes are returned, never raised (same style as the contracts).

Missing months: Wikimedia omits months it has no data for. They are reported in
`PageviewSeries.missing_months`, never fabricated as zero. A response with no usable
months at all is treated as ARTICLE_UNAVAILABLE, so "no data" can't read as "no interest".
"""

import calendar
import os
import re
import time
from collections.abc import Callable
from datetime import date
from typing import Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from wikipedia_interest.contracts import (
    _LANGUAGE_RE,
    FieldIssue,
    InvalidInputResult,
    PageviewPoint,
    PageviewSeries,
    Period,
    UnsupportedCode,
    UnsupportedResult,
    UpstreamFailureCode,
    UpstreamFailureResult,
    month_range,
)

BASE_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
USER_AGENT_ENV = "WIKIPEDIA_INTEREST_USER_AGENT"
DEFAULT_USER_AGENT = (
    "wikipedia-interest/0.1 (Agent Skill; set WIKIPEDIA_INTEREST_USER_AGENT "
    "with contact info for production use)"
)
ACCESS = "all-access"
AGENT = "user"
GRANULARITY = "monthly"
# Wikimedia's per-article pageview dataset starts in July 2015.
MIN_SUPPORTED_MONTH = date(2015, 7, 1)

TIMEOUT_SECONDS = 10.0
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (1.0, 2.0)  # wait before attempt 2 and 3
MAX_RETRY_AFTER_SECONDS = 10.0
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_TIMESTAMP_RE = re.compile(r"^(\d{4})(\d{2})0100$")


def _validated_language(language: str) -> str:
    if not _LANGUAGE_RE.match(language):
        raise ValueError(f"malformed Wikipedia language code: {language!r}")
    return language


def wikipedia_domain(language: str) -> str:
    """The MediaWiki host for a Wikipedia language, e.g. "uk" -> "uk.wikipedia.org".

    Used for MediaWiki page verification. Not interchangeable with `pageviews_project`:
    the Wikimedia Analytics API identifies projects without the ".org" suffix.
    """
    return f"{_validated_language(language)}.wikipedia.org"


def pageviews_project(language: str) -> str:
    """The Wikimedia Analytics/Pageviews project identifier, e.g. "uk" -> "uk.wikipedia".

    Used for Pageviews API requests and response validation. See `wikipedia_domain` for the
    (different) MediaWiki domain.
    """
    return f"{_validated_language(language)}.wikipedia"


def api_timestamps(period: Period) -> tuple[str, str]:
    """Inclusive month period -> (start, end) API timestamps.

    End is the last day of the end month so that month's data point is included.
    """
    last_day = calendar.monthrange(period.end.year, period.end.month)[1]
    return period.start.strftime("%Y%m010") + "0", period.end.strftime("%Y%m") + f"{last_day}00"


def build_url(project: str, article: str, period: Period) -> str:
    """Fixed host; article is fully percent-encoded (no '/', '?', '#' can escape its segment)."""
    start, end = api_timestamps(period)
    title = quote(article.replace(" ", "_"), safe="")
    return f"{BASE_URL}/{project}/{ACCESS}/{AGENT}/{title}/{GRANULARITY}/{start}/{end}"


class _Item(BaseModel):
    """One upstream item. Strict; unknown upstream fields are ignored."""

    model_config = ConfigDict(extra="ignore", strict=True)

    project: str
    article: str
    granularity: Literal["monthly"]
    timestamp: str
    access: Literal["all-access"]
    agent: Literal["user"]
    views: int


class _Malformed(Exception):
    pass


def _reason(exc: Exception) -> str:
    """Short, URL-free reason (first error only) that fits the result message limit."""
    if isinstance(exc, ValidationError):
        e = exc.errors()[0]
        return f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
    return str(exc)


def _parse_series(
    payload: object, project: str, article: str, period: Period
) -> PageviewSeries | UnsupportedResult:
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise _Malformed("response is not an object with an 'items' list")
    raw_items: list[object] = payload["items"]
    if not raw_items:
        return UnsupportedResult(
            code=UnsupportedCode.ARTICLE_UNAVAILABLE,
            message=f"Wikimedia returned no pageview data for {article!r} on {project}.",
        )
    wanted = article.replace(" ", "_")
    points: list[PageviewPoint] = []
    for raw in raw_items:
        try:
            item = _Item.model_validate(raw)
            m = _TIMESTAMP_RE.match(item.timestamp)
            if not m:
                raise _Malformed(f"bad timestamp {item.timestamp!r}")
            if item.project != project or item.article != wanted:
                raise _Malformed("item project/article does not match the request")
            points.append(PageviewPoint(month=f"{m[1]}-{m[2]}", views=item.views))
        except (ValidationError, ValueError) as exc:
            raise _Malformed(f"invalid item ({_reason(exc)})") from exc
    present = {p.month.strftime("%Y-%m") for p in points}
    try:
        return PageviewSeries(
            project=project,
            article=article,
            period=period,
            points=tuple(points),
            missing_months=tuple(
                m.strftime("%Y-%m")
                for m in month_range(period)
                if m.strftime("%Y-%m") not in present
            ),
        )
    except ValidationError as exc:
        raise _Malformed(f"inconsistent series ({_reason(exc)})") from exc


def _retry_after(response: httpx.Response) -> float | None:
    """Valid Retry-After in seconds (integer form), capped; None if absent/invalid."""
    raw = response.headers.get("Retry-After", "").strip()
    if not raw.isdigit():
        return None
    return min(float(int(raw)), MAX_RETRY_AFTER_SECONDS)


def _failure(code: UpstreamFailureCode, message: str, retryable: bool) -> UpstreamFailureResult:
    return UpstreamFailureResult(code=code, message=message, retryable=retryable)


class WikimediaClient:
    def __init__(
        self,
        user_agent: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        agent = user_agent or os.environ.get(USER_AGENT_ENV, "").strip() or DEFAULT_USER_AGENT
        self._sleep = sleep
        self._http = httpx.Client(
            headers={"User-Agent": agent, "Accept": "application/json"},
            timeout=httpx.Timeout(TIMEOUT_SECONDS),
            follow_redirects=False,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def fetch_monthly_pageviews(
        self, language: str, article: str, period: Period
    ) -> PageviewSeries | UnsupportedResult | InvalidInputResult | UpstreamFailureResult:
        """Fetch monthly user pageviews. The period is never altered."""
        issues: list[FieldIssue] = []
        if period.start < MIN_SUPPORTED_MONTH:
            issues.append(
                FieldIssue(
                    field="period.start",
                    message="Wikimedia pageview data starts in 2015-07; earlier periods "
                    "are not supported.",
                )
            )
        if not article.strip():
            issues.append(FieldIssue(field="article", message="article must not be empty"))
        try:
            project = pageviews_project(language)
        except ValueError as exc:
            issues.append(FieldIssue(field="language", message=str(exc)))
        if issues:
            return InvalidInputResult(issues=tuple(issues))

        url = build_url(project, article, period)
        response_or_failure = self._get_with_retries(url)
        if isinstance(response_or_failure, UpstreamFailureResult):
            return response_or_failure
        response = response_or_failure
        if response.status_code == 404:
            return UnsupportedResult(
                code=UnsupportedCode.ARTICLE_UNAVAILABLE,
                message=f"No pageview data for {article!r} on {project} (HTTP 404).",
            )
        if response.status_code != 200:
            return _failure(
                UpstreamFailureCode.UNAVAILABLE,
                f"Wikimedia returned HTTP {response.status_code}.",
                retryable=False,
            )
        try:
            return _parse_series(response.json(), project, article, period)
        except (_Malformed, ValueError) as exc:
            return _failure(
                UpstreamFailureCode.MALFORMED_RESPONSE,
                f"Unusable Wikimedia response: {_reason(exc)}"[:190],
                retryable=False,
            )

    def _get_with_retries(self, url: str) -> httpx.Response | UpstreamFailureResult:
        last: UpstreamFailureResult | None = None
        for attempt in range(MAX_ATTEMPTS):
            delay = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
            try:
                response = self._http.get(url)
            except httpx.TimeoutException:
                last = _failure(
                    UpstreamFailureCode.TIMEOUT,
                    f"Wikimedia did not respond within {TIMEOUT_SECONDS:g}s.",
                    retryable=True,
                )
            except httpx.TransportError as exc:
                last = _failure(
                    UpstreamFailureCode.UNAVAILABLE,
                    f"Network failure contacting Wikimedia: {type(exc).__name__}.",
                    retryable=True,
                )
            else:
                if response.status_code not in _RETRYABLE_STATUS:
                    return response
                if response.status_code == 429:
                    last = _failure(
                        UpstreamFailureCode.RATE_LIMITED,
                        "Wikimedia rate limit persisted after retries.",
                        retryable=True,
                    )
                else:
                    last = _failure(
                        UpstreamFailureCode.UNAVAILABLE,
                        f"Wikimedia kept returning HTTP {response.status_code}.",
                        retryable=True,
                    )
                hinted = _retry_after(response)
                delay = delay if hinted is None else hinted
            if attempt < MAX_ATTEMPTS - 1:
                self._sleep(delay)
        assert last is not None
        return last
