import os

import pytest

from trial import Trial, configure
from trial.judge import evaluate
from trial.providers.base import BaseProvider
from trial.result import Verdict


class FakeProvider(BaseProvider):
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, system: str, user: str) -> str:
        return self._response


USER_MSG = "Find me a senior Python engineer"
GOOD_RESPONSE = "Here is Alice Smith, skills: Python, FastAPI, Docker"


# --- Deterministic tests (no real LLM) ---

def test_malformed_json_raises():
    provider = FakeProvider("not json at all")
    with pytest.raises(ValueError, match="malformed JSON"):
        evaluate("some criterion", USER_MSG, GOOD_RESPONSE, provider)


def test_missing_fields_raises():
    provider = FakeProvider('{"pass": true, "score": 0.9}')
    with pytest.raises(ValueError, match="missing fields"):
        evaluate("some criterion", USER_MSG, GOOD_RESPONSE, provider)


def test_min_score_overrides_pass():
    # Judge says pass=true, score=0.6, but min_score=0.8 → should fail
    provider = FakeProvider('{"pass": true, "score": 0.6, "reason": "partial", "missing": []}')
    verdict = evaluate("some criterion", USER_MSG, GOOD_RESPONSE, provider, min_score=0.8)
    assert not verdict.passed
    assert verdict.score == 0.6


def test_min_score_allows_pass():
    provider = FakeProvider('{"pass": true, "score": 0.9, "reason": "good", "missing": []}')
    verdict = evaluate("some criterion", USER_MSG, GOOD_RESPONSE, provider, min_score=0.8)
    assert verdict.passed


def test_missing_provider_config_raises():
    import trial.config as cfg
    original = cfg._default_provider
    cfg._default_provider = None

    try:
        with pytest.raises(RuntimeError, match="No provider configured"):
            Trial(USER_MSG, GOOD_RESPONSE).passes_judge("anything").run()
    finally:
        cfg._default_provider = original


def test_verdict_fields():
    provider = FakeProvider(
        '{"pass": true, "score": 1.0, "reason": "All good.", "missing": []}'
    )
    verdict = evaluate("criterion", USER_MSG, GOOD_RESPONSE, provider)
    assert isinstance(verdict, Verdict)
    assert verdict.passed is True
    assert verdict.score == 1.0
    assert verdict.reason == "All good."
    assert verdict.missing == []


# --- Integration tests (real LLM) ---

@pytest.mark.integration
def test_anthropic_happy_path():
    from trial.providers import AnthropicProvider

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    configure(provider=AnthropicProvider(model="claude-haiku-4-5-20251001", api_key=api_key))

    result = (
        Trial(
            user_message=USER_MSG,
            assistant_response=GOOD_RESPONSE,
        )
        .passes_judge("Returns a candidate with a name and list of relevant engineering skills")
        .run()
    )

    assert result.passed
    assert result.score >= 0.7


@pytest.mark.integration
def test_anthropic_strict_fail():
    from trial.providers import AnthropicProvider

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    configure(provider=AnthropicProvider(model="claude-haiku-4-5-20251001", api_key=api_key))

    result = (
        Trial(
            user_message=USER_MSG,
            assistant_response="I cannot help with that.",
        )
        .passes_judge("Returns a candidate with a name and list of relevant engineering skills")
        .run()
    )

    assert not result.passed
    assert result.score < 0.5
