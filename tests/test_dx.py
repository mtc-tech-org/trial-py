import json
import pytest

from trial import Trial, normalize_response
from trial.tools import ToolCall


# --- normalize_response ---

def test_normalize_string():
    assert normalize_response("hello") == "hello"


def test_normalize_text_attribute():
    class R:
        text = "hello"
    assert normalize_response(R()) == "hello"


def test_normalize_dict_response_key():
    assert normalize_response({"response": "hello"}) == "hello"


def test_normalize_dict_text_key():
    assert normalize_response({"text": "hello"}) == "hello"


def test_normalize_dict_content_key():
    assert normalize_response({"content": "hello"}) == "hello"


def test_normalize_dict_unknown_keys_raises():
    with pytest.raises(ValueError, match="Could not extract response text"):
        normalize_response({"unknown_key": "hello"})


def test_normalize_fallback_str():
    class Weird:
        def __str__(self): return "hello"
    assert normalize_response(Weird()) == "hello"


# --- from_response auto tool call extraction ---

def test_from_response_auto_extracts_anthropic_tool_calls():
    class Block:
        type = "tool_use"
        name = "search_recipe"
        input = {"query": "pasta"}

    class AnthropicResponse:
        text = "Here's a recipe."
        content = [Block()]

    t = Trial.from_response(user_message="Give me a recipe", response=AnthropicResponse())
    assert len(t._tool_calls) == 1
    assert t._tool_calls[0].name == "search_recipe"


def test_from_response_auto_extracts_openai_tool_calls():
    class Function:
        name = "search_recipe"
        arguments = json.dumps({"query": "pasta"})

    class ToolCallObj:
        function = Function()

    class Message:
        tool_calls = [ToolCallObj()]
        content = "Here's a recipe."

    class Choice:
        message = Message()

    class OpenAIResponse:
        choices = [Choice()]

    t = Trial.from_response(user_message="Give me a recipe", response=OpenAIResponse())
    assert len(t._tool_calls) == 1
    assert t._tool_calls[0].name == "search_recipe"


def test_from_response_explicit_tool_calls_take_precedence():
    class Block:
        type = "tool_use"
        name = "auto_extracted"
        input = {}

    class AnthropicResponse:
        text = "Here's a recipe."
        content = [Block()]

    explicit = [ToolCall(name="explicit_tool", input={})]
    t = Trial.from_response("msg", AnthropicResponse(), tool_calls=explicit)
    assert t._tool_calls[0].name == "explicit_tool"


def test_from_response_no_tool_calls_returns_empty():
    t = Trial.from_response("msg", "plain string response")
    assert t._tool_calls == []


# --- assert_passed ---

def test_assert_passed_does_not_raise_on_pass():
    from trial.result import TrialResult
    result = TrialResult(passed=True, score=1.0, reason="All good.", missing=[])
    result.assert_passed()  # should not raise


def test_assert_passed_raises_on_fail():
    from trial.result import TrialResult
    result = TrialResult(
        passed=False,
        score=0.3,
        reason="Missing ingredients.",
        missing=["ingredients"],
        assertion_failures=["Expected response to contain text: 'spaghetti'"],
    )
    with pytest.raises(AssertionError) as exc:
        result.assert_passed()

    msg = str(exc.value)
    assert "0.30" in msg
    assert "Missing ingredients." in msg
    assert "spaghetti" in msg
    assert "ingredients" in msg


def test_assert_passed_message_format():
    from trial.result import TrialResult
    result = TrialResult(
        passed=False,
        score=0.5,
        reason="Partial.",
        missing=["name", "skills"],
        assertion_failures=["Expected response to match regex: 'r\\d+'"],
    )
    with pytest.raises(AssertionError) as exc:
        result.assert_passed()

    msg = str(exc.value)
    assert "Failures:" in msg
    assert "Missing:" in msg
    assert "name" in msg
    assert "skills" in msg


def test_trial_assert_passed_integration():
    result = (
        Trial("Give me a pasta recipe", assistant_response="Here's spaghetti. Serves 2.")
        .contains_text("spaghetti")
        .run()
    )
    result.assert_passed()  # should not raise


def test_trial_assert_passed_fails_with_clear_message():
    result = (
        Trial("Give me a pasta recipe", assistant_response="Here's risotto.")
        .contains_text("spaghetti")
        .run()
    )
    with pytest.raises(AssertionError) as exc:
        result.assert_passed()
    assert "spaghetti" in str(exc.value)
