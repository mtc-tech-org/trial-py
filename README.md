# trial

**Put your AI agents on trial.**

A Python framework for testing AI behavior with deterministic assertions, tool-call checks, and an LLM judge.

---

## Install

```bash
pip install trial
```

With providers:

```bash
pip install "trial[anthropic]"
pip install "trial[openai]"
pip install "trial[json]"
pip install "trial[all]"
```

---

## 30-second example

```python
from trial import Trial, configure
from trial.providers import AnthropicProvider

configure(
    provider=AnthropicProvider(model="claude-sonnet-4-6"),
    agent=my_agent.run,
)

result = (
    Trial("Give me a pasta recipe for two")
    .contains_text("spaghetti")
    .regex(r"Serves \d+")
    .passes_judge("Returns a pasta recipe with a name and list of ingredients")
    .run()
)

assert result.passed
```

---

## Why trial

Most AI eval tools:
- are vague
- are hard to trust
- or are tied to a specific framework

Trial is different:
- deterministic checks for exact behavior
- strict LLM judge for semantic quality
- tool-call assertions for agent correctness
- multi-turn testing for real conversations
- works with any AI system

No lock-in. Just input → output → verdict.

---

## Core idea

Every test has three layers:

**1. Evidence** — fast, deterministic, no LLM

```python
.contains_text("spaghetti")
.regex(r"Serves \d+")
.called_tool("search_recipe")
.json_schema({"type": "object", "required": ["title", "ingredients"]})
.syntactically_valid("python")
```

**2. Judgement** — LLM evaluation against a strict criterion

```python
.passes_judge("Returns a pasta recipe with a name and list of ingredients")
```

Strict by design:
- does not infer missing information
- partial answers fail
- returns structured output

**3. Verdict**

```python
result.passed    # True / False
result.score     # 0.0 – 1.0
result.reason    # short explanation
result.missing   # what was absent
```

---

## Works with any AI system

Trial evaluates plain inputs and outputs.

**Auto-call your agent**

```python
configure(
    provider=AnthropicProvider(model="claude-sonnet-4-6"),
    agent=my_agent.run,
)

Trial("Give me a pasta recipe").passes_judge("Returns a recipe").run()
```

**HTTP endpoint**

```python
configure(
    provider=AnthropicProvider(model="claude-sonnet-4-6"),
    agent="http://localhost:8000/chat",   # POST {message: ...}, expects {response: ...}
)
```

**Manual response**

```python
response = agent.run("Give me a pasta recipe")

Trial.from_response(
    user_message="Give me a pasta recipe",
    response=response,
).passes_judge("Returns a recipe with ingredients").run()
```

---

## Tool call assertions

Verify your agent actually did the right thing:

```python
Trial(..., tool_calls=tool_calls)
    .called_tool("search_recipe")
    .called_tool_with("get_nutrition", input_contains={"dish": "spaghetti"})
    .run()
```

Normalize tool calls from any provider:

```python
# Anthropic
tool_calls = [ToolCall.from_anthropic(b) for b in response.content if b.type == "tool_use"]

# OpenAI
tool_calls = [ToolCall.from_openai(tc) for tc in response.choices[0].message.tool_calls]
```

---

## JSON assertions

```python
Trial(user_msg, response)
    .json_schema({"type": "object", "required": ["title", "ingredients"]})
    .json_path("$.title", contains="Spaghetti")
    .run()
```

Requires: `pip install "trial[json]"`

---

## Code validation

```python
Trial(user_msg, code)
    .syntactically_valid("python")
    .run()
```

---

## Multi-turn conversations

```python
from trial import Conversation, Turn

Conversation([
    Turn(user="Give me a recipe", assistant="..."),
    Turn(user="How many calories?", assistant="..."),
])
.passes_judge("Agent follows up correctly and answers both questions")
.run()
```

---

## Strict judge

Trial's judge is intentionally strict:
- no guessing
- no partial credit for missing key information
- fails incomplete answers

```
Score:
  1.0   correct
  0.7   minor gaps
  0.4   major gaps
  0.0   incorrect
```

---

## Configuration

```python
configure(provider=AnthropicProvider(model="claude-sonnet-4-6"))
```

Override per test:

```python
Trial(..., provider=OpenAIProvider(model="gpt-4o"))
```

---

## Testing

```bash
pytest tests/ -m "not integration"

ANTHROPIC_API_KEY=... pytest -m integration
```

---

## Roadmap

- Latency assertions (TTFT, completion time)
- CI integration
- TypeScript SDK

---

## Philosophy

Trial is not about generating responses.

It's about verifying behavior.

---

## Limits

Trial helps you verify and improve agent behavior, but it does not guarantee correctness.

- Passing tests means the agent satisfied your defined criteria — not that it is universally correct
- LLM judges are probabilistic and can be wrong
- Coverage depends on the quality and breadth of your tests

Trial is best used to catch regressions, enforce expected behavior, and improve consistency over time. Not to prove absolute correctness.

---

## License

MIT
