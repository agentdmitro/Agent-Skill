"""Regression guards protecting product semantics: Wikipedia interest is never presented as
market size, demand, or a launch recommendation, anywhere in the typed contracts or in the
deterministic report copy."""

import inspect

from wikipedia_interest import contracts, orchestration
from wikipedia_interest.comparison import ComparisonReport, LanguageComparisonSummary
from wikipedia_interest.orchestration import LanguageResult, RunResult
from wikipedia_interest.reporting import _CANDIDATE_TEXT, _LIMITATIONS

FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "market_size",
        "market_score",
        "opportunity_score",
        "revenue",
        "purchase_probability",
        "best_market",
        "launch_recommendation",
    }
)

FORBIDDEN_CLAIMS = (
    "best market",
    "guaranteed demand",
    "users will pay",
    "market size",
)

ALLOWED_CLAIMS = (
    "candidate for further validation",
    "wikipedia interest",
    "evidence reliability",
)


def _field_names(model: type) -> set[str]:
    return set(getattr(model, "model_fields", {}))


def test_no_forbidden_fields_in_result_models() -> None:
    for model in (RunResult, LanguageResult, ComparisonReport, LanguageComparisonSummary):
        assert not (_field_names(model) & FORBIDDEN_FIELD_NAMES), model


def test_no_forbidden_fields_anywhere_in_contracts_module() -> None:
    for _name, obj in inspect.getmembers(contracts):
        if inspect.isclass(obj) and hasattr(obj, "model_fields"):
            assert not (_field_names(obj) & FORBIDDEN_FIELD_NAMES), obj


def test_report_copy_never_makes_forbidden_claims() -> None:
    corpus = " ".join(_CANDIDATE_TEXT.values()).lower() + " " + " ".join(_LIMITATIONS).lower()
    for claim in FORBIDDEN_CLAIMS:
        assert claim not in corpus


def test_report_copy_can_make_allowed_claims() -> None:
    corpus = " ".join(_CANDIDATE_TEXT.values()).lower() + " " + " ".join(_LIMITATIONS).lower()
    assert any(claim in corpus for claim in ALLOWED_CLAIMS)


def test_orchestration_module_defines_no_forbidden_names() -> None:
    names = {name.lower() for name in dir(orchestration)}
    assert not (names & FORBIDDEN_FIELD_NAMES)
