"""Tests for Trial.from_execution() and .no_errors()."""

import pytest

from trial import Trial
from trial.tools import ToolCall


RESPONSE = "Here's spaghetti aglio e olio. Serves 2."
USER_MSG = "Give me a pasta recipe"


# --- from_execution ---

def test_from_execution_basic_pass():
    result = (
        Trial.from_execution(USER_MSG, RESPONSE)
        .contains_text("spaghetti")
        .run()
    )
    assert result.passed


def test_from_execution_with_metrics():
    result = (
        Trial.from_execution(
            USER_MSG,
            RESPONSE,
            metrics={"elapsed_time": 1.5, "first_token_time": 0.3},
        )
        .completes_within(5.0)
        .first_token_within(1.0)
        .run()
    )
    assert result.passed


def test_from_execution_with_tool_calls():
    tool_calls = [ToolCall(name="search_recipe", input={"query": "pasta"})]
    result = (
        Trial.from_execution(USER_MSG, RESPONSE, tool_calls=tool_calls)
        .called_tool("search_recipe")
        .run()
    )
    assert result.passed


def test_from_execution_tool_calls_not_auto_extracted():
    """from_execution does not auto-extract tool calls — what you pass is what you get."""
    class FakeAnthropicResponse:
        text = RESPONSE

        class _block:
            type = "tool_use"
            name = "auto_extracted_tool"
            input = {}

        content = [_block()]

    # No tool_calls= passed → should NOT auto-extract
    result = (
        Trial.from_execution(USER_MSG, FakeAnthropicResponse())
        .called_tool("auto_extracted_tool")
        .run()
    )
    assert not result.passed


def test_from_execution_explicit_tool_calls_respected():
    explicit = [ToolCall(name="my_custom_tool", input={"k": "v"})]

    class FakeAnthropicResponse:
        text = RESPONSE

        class _block:
            type = "tool_use"
            name = "auto_tool"
            input = {}

        content = [_block()]

    result = (
        Trial.from_execution(USER_MSG, FakeAnthropicResponse(), tool_calls=explicit)
        .called_tool("my_custom_tool")
        .run()
    )
    assert result.passed


def test_from_execution_with_error_no_check():
    """Passing error= without .no_errors() does not fail the trial."""
    result = (
        Trial.from_execution(USER_MSG, RESPONSE, error="timeout")
        .contains_text("spaghetti")
        .run()
    )
    assert result.passed


def test_from_execution_all_params():
    tool_calls = [ToolCall(name="search_recipe", input={"query": "pasta"})]
    result = (
        Trial.from_execution(
            USER_MSG,
            RESPONSE,
            metrics={"elapsed_time": 2.0, "first_token_time": 0.5},
            tool_calls=tool_calls,
            error=None,
        )
        .no_errors()
        .contains_text("spaghetti")
        .called_tool("search_recipe")
        .completes_within(5.0)
        .run()
    )
    assert result.passed


# --- no_errors ---

def test_no_errors_passes_when_no_error():
    result = (
        Trial(USER_MSG, RESPONSE)
        .no_errors()
        .run()
    )
    assert result.passed


def test_no_errors_fails_with_string_error():
    result = (
        Trial(USER_MSG, RESPONSE, error="upstream timeout")
        .no_errors()
        .run()
    )
    assert not result.passed
    assert any("upstream timeout" in f for f in result.assertion_failures)


def test_no_errors_fails_with_exception_error():
    result = (
        Trial(USER_MSG, RESPONSE, error=RuntimeError("connection refused"))
        .no_errors()
        .run()
    )
    assert not result.passed
    assert any("connection refused" in f for f in result.assertion_failures)


def test_no_errors_error_message_format():
    result = (
        Trial(USER_MSG, RESPONSE, error="rate limited")
        .no_errors()
        .run()
    )
    assert any("Expected no errors, but received: rate limited" in f for f in result.assertion_failures)


def test_no_errors_from_execution_pass():
    result = (
        Trial.from_execution(USER_MSG, RESPONSE, error=None)
        .no_errors()
        .run()
    )
    assert result.passed


def test_no_errors_from_execution_fail():
    result = (
        Trial.from_execution(USER_MSG, RESPONSE, error="stream disconnected")
        .no_errors()
        .run()
    )
    assert not result.passed
    assert any("stream disconnected" in f for f in result.assertion_failures)


def test_no_errors_combined_with_other_checks():
    result = (
        Trial(USER_MSG, RESPONSE, error="oops")
        .contains_text("spaghetti")  # passes
        .no_errors()                 # fails
        .run()
    )
    assert not result.passed
    # contains_text passes, so only the no_errors failure
    assert len(result.assertion_failures) == 1
    assert "oops" in result.assertion_failures[0]


def test_no_errors_generic_tool_call():
    """ToolCall works generically without any provider adapter."""
    tc = ToolCall(name="vector_search", input={"query": "engineer", "top_k": 5})
    result = (
        Trial.from_execution(USER_MSG, RESPONSE, tool_calls=[tc])
        .called_tool("vector_search")
        .called_tool_with("vector_search", input_contains={"query": "engineer"})
        .run()
    )
    assert result.passed
