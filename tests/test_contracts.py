from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from wikipedia_interest import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisState,
    ClarificationResult,
    InvalidInputResult,
    invalid_input_from,
)

PERIOD = {"start": "2024-09", "end": "2026-09"}


def make_request(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "topic": "intermittent fasting",
        "languages": ["pl"],
        "period": PERIOD,
        "analysis": "trend",
    }
    return data | overrides


def test_valid_single_language_request() -> None:
    req = AnalysisRequest.model_validate(make_request())
    assert req.languages == ("pl",)
    assert req.previous_state is None


def test_valid_multi_language_comparison() -> None:
    req = AnalysisRequest.model_validate(make_request(languages=["pl", "cs"], analysis="compare"))
    assert req.languages == ("pl", "cs")


def test_compare_with_one_language_rejected() -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest.model_validate(make_request(analysis="compare"))


@pytest.mark.parametrize("topic", ["", "   "])
def test_empty_topic_rejected(topic: str) -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest.model_validate(make_request(topic=topic))


def test_empty_languages_rejected() -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest.model_validate(make_request(languages=[]))


def test_duplicate_languages_deduplicated_in_first_seen_order() -> None:
    req = AnalysisRequest.model_validate(make_request(languages=["cs", "PL", " cs", "pl"]))
    assert req.languages == ("cs", "pl")


def test_compare_of_duplicates_of_one_language_rejected() -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest.model_validate(make_request(languages=["pl", "PL"], analysis="compare"))


@pytest.mark.parametrize("code", ["", "p", "polish1", "pl_PL", "p l", "12", "en/"])
def test_malformed_language_rejected(code: str) -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest.model_validate(make_request(languages=[code]))


@pytest.mark.parametrize("code", ["en", "zh-yue", "simple"])
def test_wikipedia_language_codes_accepted(code: str) -> None:
    assert AnalysisRequest.model_validate(make_request(languages=[code])).languages == (code,)


@pytest.mark.parametrize(
    "period",
    [
        {"start": "2024-13", "end": "2026-09"},
        {"start": "2024-9", "end": "2026-09"},
        {"start": "2024-09-01", "end": "2026-09"},
        {"start": "September 2024", "end": "2026-09"},
        {"start": "2024-09"},
    ],
)
def test_malformed_period_rejected(period: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest.model_validate(make_request(period=period))


@pytest.mark.parametrize(
    "period",
    [
        {"start": "2026-09", "end": "2026-09"},
        {"start": "2026-10", "end": "2026-09"},
    ],
)
def test_start_not_before_end_rejected(period: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest.model_validate(make_request(period=period))


def test_unknown_analysis_intent_rejected() -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest.model_validate(make_request(analysis="forecast"))


def test_unknown_request_field_rejected() -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest.model_validate(make_request(sort_by="views"))


def test_unknown_nested_field_rejected() -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest.model_validate(make_request(period=PERIOD | {"granularity": "day"}))


def test_request_round_trips_through_json() -> None:
    req = AnalysisRequest.model_validate(make_request(languages=["pl", "cs"], analysis="compare"))
    assert AnalysisRequest.model_validate_json(req.model_dump_json()) == req
    assert req.model_dump(mode="json")["period"] == PERIOD


def test_request_is_immutable() -> None:
    req = AnalysisRequest.model_validate(make_request())
    with pytest.raises(ValidationError):
        req.topic = "other"  # type: ignore[misc]


def test_clarification_represents_multiple_candidates() -> None:
    data = {
        "status": "clarification_required",
        "topic": "Mercury",
        "candidates": [
            {"wikidata_id": "Q308", "label": "Mercury", "description": "planet"},
            {"wikidata_id": "Q925", "label": "Mercury", "description": "chemical element"},
            {"wikidata_id": "Q40174", "label": "Mercury", "description": "Roman god"},
        ],
    }
    result: Any = TypeAdapter(AnalysisResult).validate_python(data)
    assert isinstance(result, ClarificationResult)
    assert [c.wikidata_id for c in result.candidates] == ["Q308", "Q925", "Q40174"]


def test_clarification_with_single_candidate_rejected() -> None:
    with pytest.raises(ValidationError):
        ClarificationResult.model_validate(
            {
                "topic": "Mercury",
                "candidates": [
                    {"wikidata_id": "Q308", "label": "Mercury", "description": "planet"}
                ],
            }
        )


def test_result_statuses_are_distinguishable() -> None:
    adapter: TypeAdapter[Any] = TypeAdapter(AnalysisResult)
    unsupported = adapter.validate_python(
        {"status": "unsupported", "code": "article_unavailable", "message": "No cs article."}
    )
    failure = adapter.validate_python(
        {
            "status": "upstream_failure",
            "code": "timeout",
            "message": "Timed out.",
            "retryable": True,
        }
    )
    assert unsupported.status == "unsupported"
    assert failure.status == "upstream_failure"


def test_unknown_result_status_rejected() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(AnalysisResult).validate_python({"status": "maybe"})


def test_success_result_carries_request_and_entities() -> None:
    result: Any = TypeAdapter(AnalysisResult).validate_python(
        {
            "status": "success",
            "request": make_request(),
            "entities": [
                {"wikidata_id": "Q1", "label": "Fasting", "titles": {"pl": "Post przerywany"}}
            ],
        }
    )
    assert result.status == "success"
    assert result.warnings == ()


def state_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "analysis_id": "a-1",
        "version": 1,
        "topic_id": "Q1",
        "topic_label": "Intermittent fasting",
        "languages": ["pl", "cs"],
        "period": PERIOD,
        "analysis": "compare",
    }
    return data | overrides


def test_state_represents_existing_analysis_and_attaches_to_request() -> None:
    state = AnalysisState.model_validate(state_data())
    req = AnalysisRequest.model_validate(
        make_request(languages=["pl", "cs", "uk"], analysis="compare", previous_state=state_data())
    )
    assert req.previous_state == state
    assert state.languages == ("pl", "cs")


@pytest.mark.parametrize(
    "override",
    [
        {"topic_id": "fasting"},
        {"topic_id": "Q0"},
        {"version": 0},
        {"analysis_id": ""},
        {"languages": ["pl"]},  # compare with one language
        {"period": {"start": "2026-09", "end": "2024-09"}},
        {"extra": 1},
    ],
)
def test_invalid_state_rejected(override: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        AnalysisState.model_validate(state_data(**override))


def test_invalid_previous_state_rejects_whole_request() -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest.model_validate(make_request(previous_state=state_data(topic_id="bad")))


def test_validation_error_converts_to_invalid_input_result() -> None:
    with pytest.raises(ValidationError) as exc:
        AnalysisRequest.model_validate(make_request(languages=["x!"], topic=""))
    result = invalid_input_from(exc.value)
    assert isinstance(result, InvalidInputResult)
    assert {i.field for i in result.issues} == {"topic", "languages"}
