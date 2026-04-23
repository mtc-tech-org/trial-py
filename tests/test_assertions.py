import pytest

import trial
from trial import Trial
from trial.providers.base import BaseProvider
from trial.result import TrialResult


class FakeProvider(BaseProvider):
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, system: str, user: str) -> str:
        return self._response


RESPONSE = "Here is Alice Smith, skills: Python, FastAPI, Docker"
USER_MSG = "Find me a senior Python engineer"


def test_contains_text_pass():
    result = Trial(USER_MSG, RESPONSE).contains_text("Alice").run()
    assert result.passed
    assert result.assertion_failures == []


def test_contains_text_fail():
    result = Trial(USER_MSG, RESPONSE).contains_text("Bob").run()
    assert not result.passed
    assert any("Bob" in f for f in result.assertion_failures)


def test_contains_text_case_insensitive():
    result = Trial(USER_MSG, RESPONSE).contains_text("alice").run()
    assert result.passed


def test_regex_pass():
    result = Trial(USER_MSG, RESPONSE).regex(r"Python").run()
    assert result.passed


def test_regex_fail():
    result = Trial(USER_MSG, RESPONSE).regex(r"\d{4}").run()
    assert not result.passed
    assert any("regex" in f for f in result.assertion_failures)


def test_multiple_checks_all_pass():
    result = (
        Trial(USER_MSG, RESPONSE)
        .contains_text("Alice")
        .regex(r"Python")
        .run()
    )
    assert result.passed
    assert result.score == 1.0


def test_multiple_checks_partial_fail():
    result = (
        Trial(USER_MSG, RESPONSE)
        .contains_text("Alice")
        .contains_text("missing_name")
        .run()
    )
    assert not result.passed
    assert len(result.assertion_failures) == 1


def test_no_checks_returns_pass():
    result = Trial(USER_MSG, RESPONSE).run()
    assert result.passed
    assert result.score == 1.0


def test_from_response_with_string():
    result = Trial.from_response(USER_MSG, RESPONSE).contains_text("Alice").run()
    assert result.passed


def test_from_response_with_object():
    class FakeResponse:
        text = RESPONSE

    result = Trial.from_response(USER_MSG, FakeResponse()).contains_text("Alice").run()
    assert result.passed


def test_from_response_with_unknown_object():
    class Weird:
        def __str__(self):
            return RESPONSE

    result = Trial.from_response(USER_MSG, Weird()).contains_text("Alice").run()
    assert result.passed


def test_result_score_zero_on_fail():
    result = Trial(USER_MSG, RESPONSE).contains_text("NotPresent").run()
    assert result.score == 0.0


def test_judge_uses_fake_provider():
    fake_verdict = '{"pass": true, "score": 0.95, "reason": "Candidate with name and skills found.", "missing": []}'
    provider = FakeProvider(fake_verdict)
    result = (
        Trial(USER_MSG, RESPONSE, provider=provider)
        .passes_judge("Returns a candidate with name and skills")
        .run()
    )
    assert result.passed
    assert result.score == 0.95
    assert result.reason == "Candidate with name and skills found."
    assert result.missing == []


def test_judge_fail_reflected_in_result():
    fake_verdict = '{"pass": false, "score": 0.3, "reason": "No candidate name found.", "missing": ["name"]}'
    provider = FakeProvider(fake_verdict)
    result = (
        Trial(USER_MSG, "Here are some skills: Python", provider=provider)
        .passes_judge("Returns a candidate with name and skills")
        .run()
    )
    assert not result.passed
    assert result.score == 0.3
    assert "name" in result.missing


def test_missing_provider_raises():
    trial.configure.__module__  # ensure module loaded
    import trial.config as cfg
    original = cfg._default_provider
    cfg._default_provider = None

    try:
        with pytest.raises(RuntimeError, match="No provider configured"):
            Trial(USER_MSG, RESPONSE).passes_judge("anything").run()
    finally:
        cfg._default_provider = original
