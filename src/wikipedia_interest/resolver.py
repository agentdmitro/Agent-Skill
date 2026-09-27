"""Conservative, API-backed Wikidata to Wikipedia topic resolution."""

# The API error messages and union annotations are intentionally kept readable.
# ruff: noqa: E501

import os
import re
import time
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import ValidationError

from wikipedia_interest.contracts import (
    Candidate,
    ClarificationResult,
    InvalidInputResult,
    ResolvedArticle,
    ResolvedTopic,
    TopicResolutionRequest,
    UnavailableReason,
    UnsupportedCode,
    UnsupportedResult,
    UpstreamFailureCode,
    UpstreamFailureResult,
)
from wikipedia_interest.wikimedia import (
    BACKOFF_SECONDS,
    DEFAULT_USER_AGENT,
    MAX_ATTEMPTS,
    MAX_RETRY_AFTER_SECONDS,
    TIMEOUT_SECONDS,
    project_for_language,
)

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
_QID_RE = re.compile(r"^Q[1-9]\d*$")
_TRANSIENT = frozenset({429, 500, 502, 503, 504})


def wikidata_site_for_language(language: str) -> str:
    """Map a language code to the corresponding Wikidata Wikipedia sitelink key."""
    project_for_language(language)
    return f"{language}wiki"


def _normalise_text(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def _failure(code: UpstreamFailureCode, message: str, retryable: bool) -> UpstreamFailureResult:
    return UpstreamFailureResult(code=code, message=message[:200], retryable=retryable)


class TopicResolver:
    """Resolve topics using only fixed Wikimedia hosts and verified API data."""

    def __init__(
        self,
        user_agent: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        agent = (
            user_agent
            or os.environ.get("WIKIPEDIA_INTEREST_USER_AGENT", "").strip()
            or DEFAULT_USER_AGENT
        )
        self._sleep = sleep
        self._http = httpx.Client(
            headers={"User-Agent": agent, "Accept": "application/json"},
            timeout=httpx.Timeout(TIMEOUT_SECONDS),
            follow_redirects=False,
            transport=transport,
        )
        self._projects: dict[str, bool] | None = None

    def close(self) -> None:
        self._http.close()

    def resolve(
        self, request: TopicResolutionRequest
    ) -> (
        ResolvedTopic
        | ClarificationResult
        | UnsupportedResult
        | InvalidInputResult
        | UpstreamFailureResult
    ):
        qid: str
        if request.previous_state is not None:
            if _normalise_text(request.topic) != _normalise_text(
                request.previous_state.topic_label
            ):
                return UnsupportedResult(
                    code=UnsupportedCode.INCONSISTENT_STATE,
                    message="The supplied topic does not match the entity in previous_state.",
                )
            qid = request.previous_state.topic_id
        else:
            direct_qid = self._direct_qid(request.topic)
            if direct_qid is None:
                search = self._search(request.topic, request.query_language)
                if isinstance(search, (UpstreamFailureResult, UnsupportedResult)):
                    return search
                exact = [c for c in search if self._is_exact(c, request.topic)]
                if len(exact) != 1:
                    if len(search) >= 2:
                        return ClarificationResult(topic=request.topic, candidates=tuple(search))
                    return UnsupportedResult(
                        code=UnsupportedCode.TOPIC_NOT_FOUND,
                        message="No safe exact Wikidata match was found for the topic.",
                    )
                qid = exact[0].wikidata_id
            else:
                qid = direct_qid

        hydrated = self._hydrate(qid, request.query_language, request.target_languages)
        if isinstance(hydrated, (UpstreamFailureResult, UnsupportedResult)):
            return hydrated
        label, description, sitelinks = hydrated
        projects = self._site_matrix()
        if isinstance(projects, UpstreamFailureResult):
            return projects

        articles: dict[str, ResolvedArticle] = {}
        unavailable: dict[str, UnavailableReason] = {}
        for language in request.target_languages:
            if not projects.get(language, False):
                unavailable[language] = UnavailableReason.INVALID_PROJECT
                continue
            site = wikidata_site_for_language(language)
            title = sitelinks.get(site)
            if not isinstance(title, str) or not title.strip():
                unavailable[language] = UnavailableReason.NO_SITELINK
                continue
            checked = self._verify_page(language, title)
            if isinstance(checked, UpstreamFailureResult):
                return checked
            if isinstance(checked, UnavailableReason):
                unavailable[language] = checked
            else:
                articles[language] = ResolvedArticle(
                    title=checked, project=project_for_language(language)
                )

        if not articles:
            return UnsupportedResult(
                code=UnsupportedCode.ARTICLE_UNAVAILABLE,
                message="No requested language has a verified main-namespace Wikipedia article.",
            )
        return ResolvedTopic(
            wikidata_id=qid,
            label=label,
            description=description,
            query_language=request.query_language,
            articles=articles,
            unavailable_languages=unavailable,
        )

    @staticmethod
    def _direct_qid(topic: str) -> str | None:
        return topic.strip() if _QID_RE.fullmatch(topic.strip()) else None

    def _search(
        self, topic: str, language: str
    ) -> list[Candidate] | UnsupportedResult | UpstreamFailureResult:
        response = self._request(
            WIKIDATA_API,
            {
                "action": "wbsearchentities",
                "search": " ".join(topic.strip().split()),
                "language": language,
                "uselang": language,
                "type": "item",
                "limit": "5",
                "format": "json",
                "formatversion": "2",
            },
        )
        if isinstance(response, UpstreamFailureResult):
            return response
        results = response.get("search")
        if not isinstance(results, list):
            return _failure(
                UpstreamFailureCode.MALFORMED_RESPONSE,
                "Wikidata search response is malformed.",
                False,
            )
        candidates: list[Candidate] = []
        try:
            for raw in results:
                if (
                    not isinstance(raw, dict)
                    or not isinstance(raw.get("id"), str)
                    or not isinstance(raw.get("label"), str)
                ):
                    return _failure(
                        UpstreamFailureCode.MALFORMED_RESPONSE,
                        "Wikidata search candidate is malformed.",
                        False,
                    )
                description = raw.get("description")
                match = raw.get("match")
                matched_text = match.get("text") if isinstance(match, dict) else None
                candidates.append(
                    Candidate(
                        wikidata_id=raw["id"],
                        label=raw["label"],
                        description=description,
                        matched_text=matched_text,
                    )
                )
        except ValidationError:
            return _failure(
                UpstreamFailureCode.MALFORMED_RESPONSE,
                "Wikidata search candidate is invalid.",
                False,
            )
        if not candidates:
            return UnsupportedResult(
                code=UnsupportedCode.TOPIC_NOT_FOUND,
                message="Wikidata found no candidates for the topic.",
            )
        return candidates

    @staticmethod
    def _is_exact(candidate: Candidate, topic: str) -> bool:
        query = _normalise_text(topic)
        return _normalise_text(candidate.label) == query or (
            candidate.matched_text is not None and _normalise_text(candidate.matched_text) == query
        )

    def _hydrate(
        self, qid: str, query_language: str, target_languages: tuple[str, ...]
    ) -> tuple[str, str | None, dict[str, str]] | UnsupportedResult | UpstreamFailureResult:
        sites = "|".join(wikidata_site_for_language(lang) for lang in target_languages)
        response = self._request(
            WIKIDATA_API,
            {
                "action": "wbgetentities",
                "ids": qid,
                "props": "labels|descriptions|sitelinks",
                "languages": query_language,
                "sitefilter": sites,
                "format": "json",
                "formatversion": "2",
            },
        )
        if isinstance(response, UpstreamFailureResult):
            return response
        entities = response.get("entities")
        if (
            not isinstance(entities, dict)
            or qid not in entities
            or not isinstance(entities[qid], dict)
        ):
            return UnsupportedResult(
                code=UnsupportedCode.TOPIC_NOT_FOUND, message=f"Wikidata item {qid} was not found."
            )
        entity = entities[qid]
        if entity.get("missing") is True:
            return UnsupportedResult(
                code=UnsupportedCode.TOPIC_NOT_FOUND, message=f"Wikidata item {qid} was not found."
            )
        label = self._localized_value(entity.get("labels"), query_language)
        if label is None:
            return _failure(
                UpstreamFailureCode.MALFORMED_RESPONSE,
                "Wikidata entity has no usable label.",
                False,
            )
        description = self._localized_value(entity.get("descriptions"), query_language)
        raw_sitelinks = entity.get("sitelinks", {})
        if not isinstance(raw_sitelinks, dict):
            return _failure(
                UpstreamFailureCode.MALFORMED_RESPONSE, "Wikidata sitelinks are malformed.", False
            )
        sitelinks: dict[str, str] = {}
        for key, value in raw_sitelinks.items():
            if (
                isinstance(key, str)
                and isinstance(value, dict)
                and isinstance(value.get("title"), str)
            ):
                sitelinks[key] = value["title"]
        return label, description, sitelinks

    @staticmethod
    def _localized_value(raw: Any, language: str) -> str | None:
        if not isinstance(raw, dict):
            return None
        preferred = raw.get(language)
        if isinstance(preferred, dict):
            value = preferred.get("value")
            if isinstance(value, str) and value.strip():
                return value
        return None

    def _site_matrix(self) -> dict[str, bool] | UpstreamFailureResult:
        if self._projects is not None:
            return self._projects
        response = self._request(
            WIKIDATA_API, {"action": "sitematrix", "format": "json", "formatversion": "2"}
        )
        if isinstance(response, UpstreamFailureResult):
            return response
        matrix = response.get("sitematrix")
        if not isinstance(matrix, dict):
            return _failure(
                UpstreamFailureCode.MALFORMED_RESPONSE,
                "Wikimedia SiteMatrix response is malformed.",
                False,
            )
        active: dict[str, bool] = {}
        for value in matrix.values():
            if not isinstance(value, dict) or not isinstance(value.get("code"), str):
                continue
            sites = value.get("site")
            active[value["code"]] = isinstance(sites, list) and any(
                isinstance(site, dict)
                and site.get("code") == "wiki"
                and not site.get("closed", False)
                for site in sites
            )
        self._projects = active
        return active

    def _verify_page(
        self, language: str, title: str
    ) -> str | UnavailableReason | UpstreamFailureResult:
        response = self._request(
            f"https://{project_for_language(language)}/w/api.php",
            {
                "action": "query",
                "titles": title,
                "prop": "info",
                "redirects": "1",
                "normalize": "1",
                "format": "json",
                "formatversion": "2",
            },
        )
        if isinstance(response, UpstreamFailureResult):
            return response
        query = response.get("query")
        pages = query.get("pages") if isinstance(query, dict) else None
        if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict):
            return _failure(
                UpstreamFailureCode.MALFORMED_RESPONSE,
                "Wikipedia page response is malformed.",
                False,
            )
        page = pages[0]
        if page.get("missing") is True or not isinstance(page.get("title"), str):
            return UnavailableReason.PAGE_MISSING
        if page.get("ns") != 0:
            return UnavailableReason.NON_ARTICLE_NAMESPACE
        title = page["title"]
        assert isinstance(title, str)
        return title

    def _request(self, url: str, params: dict[str, str]) -> dict[str, Any] | UpstreamFailureResult:
        last: UpstreamFailureResult | None = None
        for attempt in range(MAX_ATTEMPTS):
            delay = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
            try:
                response = self._http.get(url, params=params)
            except httpx.TimeoutException:
                last = _failure(
                    UpstreamFailureCode.TIMEOUT,
                    f"Wikimedia did not respond within {TIMEOUT_SECONDS:g}s.",
                    True,
                )
            except httpx.TransportError as exc:
                last = _failure(
                    UpstreamFailureCode.UNAVAILABLE,
                    f"Network failure contacting Wikimedia: {type(exc).__name__}.",
                    True,
                )
            else:
                if response.status_code not in _TRANSIENT:
                    if response.status_code != 200:
                        return _failure(
                            UpstreamFailureCode.UNAVAILABLE,
                            f"Wikimedia returned HTTP {response.status_code}.",
                            False,
                        )
                    try:
                        payload = response.json()
                    except ValueError:
                        return _failure(
                            UpstreamFailureCode.MALFORMED_RESPONSE,
                            "Wikimedia returned invalid JSON.",
                            False,
                        )
                    if not isinstance(payload, dict):
                        return _failure(
                            UpstreamFailureCode.MALFORMED_RESPONSE,
                            "Wikimedia returned a non-object response.",
                            False,
                        )
                    return payload
                code = (
                    UpstreamFailureCode.RATE_LIMITED
                    if response.status_code == 429
                    else UpstreamFailureCode.UNAVAILABLE
                )
                last = _failure(
                    code, f"Wikimedia kept returning HTTP {response.status_code}.", True
                )
                hinted = response.headers.get("Retry-After", "").strip()
                if hinted.isdigit():
                    delay = min(float(int(hinted)), MAX_RETRY_AFTER_SECONDS)
            if attempt < MAX_ATTEMPTS - 1:
                self._sleep(delay)
        assert last is not None
        return last
