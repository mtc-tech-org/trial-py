import pytest

from trial import Conversation, Trial, Turn
from trial.providers.base import BaseProvider
from trial.tools import ToolCall


class FakeProvider(BaseProvider):
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, system: str, user: str) -> str:
        return self._response


USER_MSG = "Find me a senior Python engineer"
RESPONSE = "Here is Alice Smith, skills: Python, FastAPI, Docker"

VALID_JSON = '{"name": "Alice Smith", "skills": ["Python", "FastAPI"]}'
INVALID_JSON = "not json"

VALID_PYTHON = "def greet(name):\n    return f'Hello, {name}'"
INVALID_PYTHON = "def greet(name)\n    return name"  # missing colon


# --- Tool call assertions ---

def test_called_tool_pass():
    calls = [ToolCall(name="search_candidates", input={"query": "Python"})]
    result = Trial(USER_MSG, RESPONSE, tool_calls=calls).called_tool("search_candidates").run()
    assert result.passed


def test_called_tool_fail():
    result = Trial(USER_MSG, RESPONSE, tool_calls=[]).called_tool("search_candidates").run()
    assert not result.passed
    assert any("search_candidates" in f for f in result.assertion_failures)


def test_called_tool_with_pass():
    calls = [ToolCall(name="search_candidates", input={"query": "Python", "level": "senior"})]
    result = (
        Trial(USER_MSG, RESPONSE, tool_calls=calls)
        .called_tool_with("search_candidates", input_contains={"query": "Python"})
        .run()
    )
    assert result.passed


def test_called_tool_with_wrong_input():
    calls = [ToolCall(name="search_candidates", input={"query": "Java"})]
    result = (
        Trial(USER_MSG, RESPONSE, tool_calls=calls)
        .called_tool_with("search_candidates", input_contains={"query": "Python"})
        .run()
    )
    assert not result.passed


def test_called_tool_with_missing_tool():
    result = (
        Trial(USER_MSG, RESPONSE, tool_calls=[])
        .called_tool_with("search_candidates", input_contains={"query": "Python"})
        .run()
    )
    assert not result.passed


def test_toolcall_from_anthropic():
    class FakeBlock:
        name = "search"
        input = {"query": "Python"}

    tc = ToolCall.from_anthropic(FakeBlock())
    assert tc.name == "search"
    assert tc.input == {"query": "Python"}


def test_toolcall_from_openai():
    import json

    class FakeFunction:
        name = "search"
        arguments = json.dumps({"query": "Python"})

    class FakeToolCall:
        function = FakeFunction()

    tc = ToolCall.from_openai(FakeToolCall())
    assert tc.name == "search"
    assert tc.input == {"query": "Python"}


# --- JSON schema assertions ---

def test_json_schema_pass():
    schema = {"type": "object", "required": ["name", "skills"]}
    result = Trial(USER_MSG, VALID_JSON).json_schema(schema).run()
    assert result.passed


def test_json_schema_fail_missing_field():
    schema = {"type": "object", "required": ["name", "skills", "email"]}
    result = Trial(USER_MSG, VALID_JSON).json_schema(schema).run()
    assert not result.passed
    assert any("schema" in f for f in result.assertion_failures)


def test_json_schema_invalid_json():
    schema = {"type": "object"}
    result = Trial(USER_MSG, INVALID_JSON).json_schema(schema).run()
    assert not result.passed
    assert any("not valid JSON" in f for f in result.assertion_failures)


# --- JSON path assertions ---

def test_json_path_contains_pass():
    result = Trial(USER_MSG, VALID_JSON).json_path("$.name", contains="Alice").run()
    assert result.passed


def test_json_path_contains_fail():
    result = Trial(USER_MSG, VALID_JSON).json_path("$.name", contains="Bob").run()
    assert not result.passed


def test_json_path_equals_pass():
    result = Trial(USER_MSG, VALID_JSON).json_path("$.name", equals="Alice Smith").run()
    assert result.passed


def test_json_path_equals_fail():
    result = Trial(USER_MSG, VALID_JSON).json_path("$.name", equals="Bob Jones").run()
    assert not result.passed


def test_json_path_missing_key():
    result = Trial(USER_MSG, VALID_JSON).json_path("$.missing_key", contains="x").run()
    assert not result.passed
    assert any("could not be resolved" in f for f in result.assertion_failures)


def test_json_path_invalid_json():
    result = Trial(USER_MSG, INVALID_JSON).json_path("$.name", contains="Alice").run()
    assert not result.passed


# --- Syntax validity ---

def test_syntactically_valid_python_pass():
    result = Trial(USER_MSG, VALID_PYTHON).syntactically_valid("python").run()
    assert result.passed


def test_syntactically_valid_python_fail():
    result = Trial(USER_MSG, INVALID_PYTHON).syntactically_valid("python").run()
    assert not result.passed
    assert any("syntax" in f.lower() for f in result.assertion_failures)


def test_syntactically_valid_unsupported_language():
    with pytest.raises(ValueError, match="Unsupported language"):
        Trial(USER_MSG, RESPONSE).syntactically_valid("javascript")


# --- Conversation ---

def test_conversation_passes_judge_fake():
    fake_verdict = '{"pass": true, "score": 0.9, "reason": "Followed up correctly.", "missing": []}'
    provider = FakeProvider(fake_verdict)

    result = (
        Conversation(
            turns=[
                Turn(user="Find me a Python engineer", assistant="Here is Alice Smith."),
                Turn(user="Show me her GitHub", assistant="Her GitHub is github.com/alice."),
            ],
            provider=provider,
        )
        .passes_judge("Agent correctly follows up with a GitHub link")
        .run()
    )
    assert result.passed
    assert result.score == 0.9


def test_conversation_fail():
    fake_verdict = '{"pass": false, "score": 0.2, "reason": "No GitHub link provided.", "missing": ["GitHub link"]}'
    provider = FakeProvider(fake_verdict)

    result = (
        Conversation(
            turns=[
                Turn(user="Find me a Python engineer", assistant="Here is Alice Smith."),
                Turn(user="Show me her GitHub", assistant="I don't know."),
            ],
            provider=provider,
        )
        .passes_judge("Agent correctly follows up with a GitHub link")
        .run()
    )
    assert not result.passed
    assert "GitHub link" in result.missing


def test_conversation_with_tool_calls_in_turn():
    fake_verdict = '{"pass": true, "score": 1.0, "reason": "Correct.", "missing": []}'
    provider = FakeProvider(fake_verdict)

    result = (
        Conversation(
            turns=[
                Turn(
                    user="Find me a Python engineer",
                    assistant="Here is Alice Smith.",
                    tool_calls=[ToolCall(name="search_candidates", input={"query": "Python"})],
                ),
            ],
            provider=provider,
        )
        .passes_judge("Used search tool and returned a candidate")
        .run()
    )
    assert result.passed


def test_conversation_no_checks():
    result = Conversation(turns=[Turn(user="hi", assistant="hello")]).run()
    assert result.passed
    assert result.score == 1.0


# --- from_response with tool_calls ---

def test_from_response_with_tool_calls():
    calls = [ToolCall(name="search_candidates", input={"query": "Python"})]
    result = (
        Trial.from_response(USER_MSG, RESPONSE, tool_calls=calls)
        .called_tool("search_candidates")
        .run()
    )
    assert result.passed
