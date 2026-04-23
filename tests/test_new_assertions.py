import pytest

from trial import Conversation, Trial, Turn
from trial.providers.base import BaseProvider
from trial.tools import ToolCall


class FakeProvider(BaseProvider):
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, system: str, user: str) -> str:
        return self._response


USER_MSG = "Give me a quick pasta recipe for two people"
RESPONSE = "Here's spaghetti aglio e olio: 200g spaghetti, 4 garlic cloves, olive oil, chili flakes, parsley. Serves 2."

VALID_JSON = '{"title": "Spaghetti Aglio e Olio", "ingredients": ["spaghetti", "garlic", "olive oil"]}'
INVALID_JSON = "not json"

VALID_PYTHON = "def scale_recipe(servings, base=2):\n    return servings / base"
INVALID_PYTHON = "def scale_recipe(servings, base=2)\n    return servings / base"  # missing colon


# --- Tool call assertions ---

def test_called_tool_pass():
    calls = [ToolCall(name="search_recipe", input={"query": "pasta"})]
    result = Trial(USER_MSG, RESPONSE, tool_calls=calls).called_tool("search_recipe").run()
    assert result.passed


def test_called_tool_fail():
    result = Trial(USER_MSG, RESPONSE, tool_calls=[]).called_tool("search_recipe").run()
    assert not result.passed
    assert any("search_recipe" in f for f in result.assertion_failures)


def test_called_tool_with_pass():
    calls = [ToolCall(name="search_recipe", input={"query": "pasta", "servings": 2})]
    result = (
        Trial(USER_MSG, RESPONSE, tool_calls=calls)
        .called_tool_with("search_recipe", input_contains={"query": "pasta"})
        .run()
    )
    assert result.passed


def test_called_tool_with_wrong_input():
    calls = [ToolCall(name="search_recipe", input={"query": "pizza"})]
    result = (
        Trial(USER_MSG, RESPONSE, tool_calls=calls)
        .called_tool_with("search_recipe", input_contains={"query": "pasta"})
        .run()
    )
    assert not result.passed


def test_called_tool_with_missing_tool():
    result = (
        Trial(USER_MSG, RESPONSE, tool_calls=[])
        .called_tool_with("search_recipe", input_contains={"query": "pasta"})
        .run()
    )
    assert not result.passed


def test_toolcall_from_anthropic():
    class FakeBlock:
        name = "search_recipe"
        input = {"query": "pasta"}

    tc = ToolCall.from_anthropic(FakeBlock())
    assert tc.name == "search_recipe"
    assert tc.input == {"query": "pasta"}


def test_toolcall_from_openai():
    import json

    class FakeFunction:
        name = "search_recipe"
        arguments = json.dumps({"query": "pasta"})

    class FakeToolCall:
        function = FakeFunction()

    tc = ToolCall.from_openai(FakeToolCall())
    assert tc.name == "search_recipe"
    assert tc.input == {"query": "pasta"}


# --- JSON schema assertions ---

def test_json_schema_pass():
    schema = {"type": "object", "required": ["title", "ingredients"]}
    result = Trial(USER_MSG, VALID_JSON).json_schema(schema).run()
    assert result.passed


def test_json_schema_fail_missing_field():
    schema = {"type": "object", "required": ["title", "ingredients", "calories"]}
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
    result = Trial(USER_MSG, VALID_JSON).json_path("$.title", contains="Aglio").run()
    assert result.passed


def test_json_path_contains_fail():
    result = Trial(USER_MSG, VALID_JSON).json_path("$.title", contains="Carbonara").run()
    assert not result.passed


def test_json_path_equals_pass():
    result = Trial(USER_MSG, VALID_JSON).json_path("$.title", equals="Spaghetti Aglio e Olio").run()
    assert result.passed


def test_json_path_equals_fail():
    result = Trial(USER_MSG, VALID_JSON).json_path("$.title", equals="Pizza Margherita").run()
    assert not result.passed


def test_json_path_missing_key():
    result = Trial(USER_MSG, VALID_JSON).json_path("$.missing_key", contains="x").run()
    assert not result.passed
    assert any("could not be resolved" in f for f in result.assertion_failures)


def test_json_path_invalid_json():
    result = Trial(USER_MSG, INVALID_JSON).json_path("$.title", contains="Spaghetti").run()
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
    fake_verdict = '{"pass": true, "score": 0.9, "reason": "Calorie estimate provided.", "missing": []}'
    provider = FakeProvider(fake_verdict)

    result = (
        Conversation(
            turns=[
                Turn(user="What's in spaghetti carbonara?", assistant="Eggs, pecorino, guanciale, black pepper."),
                Turn(user="How many calories does it have?", assistant="Around 600 calories per serving."),
            ],
            provider=provider,
        )
        .passes_judge("Agent correctly provides a calorie estimate when asked")
        .run()
    )
    assert result.passed
    assert result.score == 0.9


def test_conversation_fail():
    fake_verdict = '{"pass": false, "score": 0.2, "reason": "No calorie estimate given.", "missing": ["calorie count"]}'
    provider = FakeProvider(fake_verdict)

    result = (
        Conversation(
            turns=[
                Turn(user="What's in spaghetti carbonara?", assistant="Eggs, pecorino, guanciale, black pepper."),
                Turn(user="How many calories does it have?", assistant="I'm not sure."),
            ],
            provider=provider,
        )
        .passes_judge("Agent correctly provides a calorie estimate when asked")
        .run()
    )
    assert not result.passed
    assert "calorie count" in result.missing


def test_conversation_with_tool_calls_in_turn():
    fake_verdict = '{"pass": true, "score": 1.0, "reason": "Correct.", "missing": []}'
    provider = FakeProvider(fake_verdict)

    result = (
        Conversation(
            turns=[
                Turn(
                    user="Give me a pasta recipe",
                    assistant="Here's spaghetti aglio e olio.",
                    tool_calls=[ToolCall(name="search_recipe", input={"query": "pasta"})],
                ),
            ],
            provider=provider,
        )
        .passes_judge("Used search tool and returned a recipe")
        .run()
    )
    assert result.passed


def test_conversation_no_checks():
    result = Conversation(turns=[Turn(user="hi", assistant="hello")]).run()
    assert result.passed
    assert result.score == 1.0


# --- from_response with tool_calls ---

def test_from_response_with_tool_calls():
    calls = [ToolCall(name="search_recipe", input={"query": "pasta"})]
    result = (
        Trial.from_response(USER_MSG, RESPONSE, tool_calls=calls)
        .called_tool("search_recipe")
        .run()
    )
    assert result.passed
