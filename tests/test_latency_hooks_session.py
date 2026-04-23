import time

import pytest

import trial.config as cfg
from trial import Conversation, Trial, Turn
from trial.providers.base import BaseProvider


class FakeProvider(BaseProvider):
    def complete(self, system: str, user: str) -> str:
        return '{"pass": true, "score": 1.0, "reason": "Good.", "missing": []}'


RESPONSE = "Here's spaghetti aglio e olio. Serves 2."
USER_MSG = "Give me a pasta recipe"


# --- completes_within ---

def test_completes_within_pass_with_manual_metrics():
    result = (
        Trial(USER_MSG, RESPONSE, metrics={"elapsed_time": 2.0})
        .completes_within(5.0)
        .run()
    )
    assert result.passed


def test_completes_within_fail_with_manual_metrics():
    result = (
        Trial(USER_MSG, RESPONSE, metrics={"elapsed_time": 8.0})
        .completes_within(5.0)
        .run()
    )
    assert not result.passed
    assert any("8.00s exceeded limit of 5.00s" in f for f in result.assertion_failures)


def test_completes_within_auto_timing():
    def fast_agent(msg: str) -> str:
        return RESPONSE

    cfg._default_provider = FakeProvider()
    cfg._default_agent = fast_agent

    try:
        result = Trial(USER_MSG).completes_within(10.0).run()
        assert result.passed
    finally:
        cfg._default_provider = None
        cfg._default_agent = None


def test_completes_within_no_metric_available():
    result = (
        Trial(USER_MSG, RESPONSE)
        .completes_within(5.0)
        .run()
    )
    assert not result.passed
    assert any("not available" in f for f in result.assertion_failures)


# --- first_token_within ---

def test_first_token_within_pass():
    result = (
        Trial(USER_MSG, RESPONSE, metrics={"first_token_time": 0.8})
        .first_token_within(2.0)
        .run()
    )
    assert result.passed


def test_first_token_within_fail():
    result = (
        Trial(USER_MSG, RESPONSE, metrics={"first_token_time": 5.0})
        .first_token_within(2.0)
        .run()
    )
    assert not result.passed
    assert any("5.00s exceeded limit of 2.00s" in f for f in result.assertion_failures)


def test_first_token_within_not_available():
    result = (
        Trial(USER_MSG, RESPONSE)
        .first_token_within(2.0)
        .run()
    )
    assert not result.passed
    assert any("first_token_time" in f for f in result.assertion_failures)


def test_combined_latency_checks():
    result = (
        Trial(USER_MSG, RESPONSE, metrics={"elapsed_time": 3.0, "first_token_time": 0.5})
        .completes_within(5.0)
        .first_token_within(1.0)
        .run()
    )
    assert result.passed


def test_from_response_passes_metrics():
    result = (
        Trial.from_response(USER_MSG, RESPONSE, metrics={"elapsed_time": 1.0})
        .completes_within(5.0)
        .run()
    )
    assert result.passed


def test_manual_metrics_override_auto():
    def agent(msg: str) -> str:
        time.sleep(0.05)
        return RESPONSE

    cfg._default_provider = FakeProvider()
    cfg._default_agent = agent

    try:
        # Manual metrics override auto-timing — even though agent takes ~50ms,
        # user says elapsed_time=99.0, so threshold of 10s should fail
        result = (
            Trial(USER_MSG, metrics={"elapsed_time": 99.0})
            .completes_within(10.0)
            .run()
        )
        assert not result.passed
    finally:
        cfg._default_provider = None
        cfg._default_agent = None


# --- after hooks ---

def test_after_hook_pass():
    result = (
        Trial(USER_MSG, RESPONSE)
        .after(lambda: True, label="side effect happened")
        .run()
    )
    assert result.passed


def test_after_hook_fail():
    result = (
        Trial(USER_MSG, RESPONSE)
        .after(lambda: False, label="database row created")
        .run()
    )
    assert not result.passed
    assert any("database row created" in f for f in result.assertion_failures)


def test_after_hook_exception():
    def bad_check():
        raise RuntimeError("db connection failed")

    result = (
        Trial(USER_MSG, RESPONSE)
        .after(bad_check, label="db check")
        .run()
    )
    assert not result.passed
    assert any("db check" in f and "db connection failed" in f for f in result.assertion_failures)


def test_after_hook_runs_after_deterministic_checks():
    calls = []

    def check():
        calls.append("after")
        return True

    Trial(USER_MSG, RESPONSE).contains_text("spaghetti").after(check).run()
    assert calls == ["after"]


def test_multiple_after_hooks():
    result = (
        Trial(USER_MSG, RESPONSE)
        .after(lambda: True, label="check 1")
        .after(lambda: False, label="check 2")
        .after(lambda: True, label="check 3")
        .run()
    )
    assert not result.passed
    assert any("check 2" in f for f in result.assertion_failures)
    assert not any("check 1" in f for f in result.assertion_failures)
    assert not any("check 3" in f for f in result.assertion_failures)


# --- Conversation session/metadata ---

def test_conversation_session_id():
    provider = FakeProvider()
    conv = Conversation(
        turns=[Turn(user="Hi", assistant="Hello.")],
        provider=provider,
        session_id="abc-123",
    )
    assert conv.session_id == "abc-123"
    formatted = conv._format()
    assert "abc-123" in formatted


def test_conversation_metadata():
    provider = FakeProvider()
    conv = Conversation(
        turns=[Turn(user="Hi", assistant="Hello.")],
        provider=provider,
        metadata={"user_id": "u1", "locale": "en"},
    )
    formatted = conv._format()
    assert "user_id" in formatted
    assert "u1" in formatted


def test_conversation_session_and_metadata_in_format():
    conv = Conversation(
        turns=[Turn(user="Hi", assistant="Hello.")],
        session_id="sess-1",
        metadata={"env": "prod"},
    )
    formatted = conv._format()
    assert "sess-1" in formatted
    assert "prod" in formatted
    assert "Turn 1:" in formatted


def test_conversation_no_session_no_metadata():
    conv = Conversation(turns=[Turn(user="Hi", assistant="Hello.")])
    formatted = conv._format()
    assert "Session" not in formatted
    assert "Context" not in formatted
    assert "Turn 1:" in formatted
