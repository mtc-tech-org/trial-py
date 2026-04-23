import pytest

import trial.config as cfg
from trial import Trial, configure
from trial.providers.base import BaseProvider


class FakeProvider(BaseProvider):
    def __init__(self, verdict: str) -> None:
        self._verdict = verdict

    def complete(self, system: str, user: str) -> str:
        return self._verdict


GOOD_VERDICT = '{"pass": true, "score": 0.95, "reason": "Recipe with ingredients.", "missing": []}'


def setup_function():
    cfg._default_provider = None
    cfg._default_agent = None


def teardown_function():
    cfg._default_provider = None
    cfg._default_agent = None


def test_callable_agent_string_return():
    def my_agent(user_message: str) -> str:
        return "Here's spaghetti aglio e olio with garlic and olive oil. Serves 2."

    configure(provider=FakeProvider(GOOD_VERDICT), agent=my_agent)

    result = (
        Trial(user_message="Give me a pasta recipe")
        .contains_text("spaghetti")
        .run()
    )
    assert result.passed


def test_callable_agent_object_with_text():
    class AgentResponse:
        def __init__(self, text: str) -> None:
            self.text = text

    def my_agent(user_message: str) -> AgentResponse:
        return AgentResponse("Spaghetti carbonara: eggs, pecorino, guanciale. Serves 2.")

    configure(provider=FakeProvider(GOOD_VERDICT), agent=my_agent)

    result = Trial(user_message="Give me a pasta recipe").contains_text("carbonara").run()
    assert result.passed


def test_callable_agent_used_for_judge():
    def my_agent(user_message: str) -> str:
        return "Spaghetti aglio e olio: garlic, olive oil, parsley. Serves 2."

    configure(provider=FakeProvider(GOOD_VERDICT), agent=my_agent)

    result = (
        Trial(user_message="Give me a pasta recipe")
        .passes_judge("Returns a recipe with ingredients")
        .run()
    )
    assert result.passed
    assert result.score == 0.95


def test_explicit_response_overrides_agent():
    def my_agent(user_message: str) -> str:
        raise AssertionError("agent should not be called when response is provided")

    configure(provider=FakeProvider(GOOD_VERDICT), agent=my_agent)

    result = (
        Trial(
            user_message="Give me a pasta recipe",
            assistant_response="Spaghetti with tomato sauce.",
        )
        .contains_text("spaghetti")
        .run()
    )
    assert result.passed


def test_no_agent_no_response_raises():
    configure(provider=FakeProvider(GOOD_VERDICT))  # no agent

    with pytest.raises(RuntimeError, match="no agent configured"):
        Trial(user_message="Give me a pasta recipe").contains_text("x").run()


def test_no_checks_no_agent_no_response_raises():
    configure(provider=FakeProvider(GOOD_VERDICT))

    with pytest.raises(RuntimeError, match="no agent configured"):
        Trial(user_message="Give me a pasta recipe").passes_judge("anything").run()
