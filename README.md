# Trial

**Put your AI agents on trial.**

A Python framework for testing AI responses with deterministic assertions and an LLM judge.

---

## Why Trial

Most AI eval tools are vague, hard to trust, or tied to a specific framework.

Trial lets you:
- Assert exact behavior with text, regex, tool call, and JSON checks
- Enforce quality with a strict LLM judge
- Test multi-turn conversations end-to-end
- Fail fast on regressions
- Work with any AI system — Claude, OpenAI, LangChain, or your own

No lock-in. Just input → output → verdict.

---

## Install

```bash
pip install trial
```

With providers:

```bash
pip install "trial[anthropic]"
pip install "trial[openai]"
pip install "trial[json]"      # JSON schema validation
pip install "trial[all]"
```

---

## Quick Example

```python
import trial
from trial.providers import AnthropicProvider

trial.configure(provider=AnthropicProvider(model="claude-sonnet-4-6"))

result = (
    trial.Trial(
        user_message="Give me a quick pasta recipe for two people",
        assistant_response="Here's spaghetti aglio e olio: 200g spaghetti, 4 garlic cloves, olive oil, chili flakes, parsley. Serves 2.",
    )
    .contains_text("spaghetti")
    .regex(r"Serves \d+")
    .passes_judge("Returns a pasta recipe with a name and list of ingredients")
    .run()
)

assert result.passed
print(result.score)    # 0.95
print(result.reason)   # "Recipe includes name and full ingredient list"
print(result.missing)  # []
```

---

## How It Works

Each test puts your AI system on trial:

**1. Evidence** — deterministic checks run first (fast, cheap, no LLM)

```python
.contains_text("spaghetti")
.regex(r"Serves \d+")
.called_tool("search_recipe")
.json_schema({"type": "object", "required": ["title", "ingredients"]})
.syntactically_valid("python")
```

**2. Judgement** — an LLM evaluates semantic quality against your criterion

```python
.passes_judge("Returns a pasta recipe with a name and list of ingredients")
```

**3. Verdict** — a structured result you can assert on

```python
result.passed    # True / False
result.score     # 0.0 – 1.0
result.reason    # short explanation
result.missing   # what was absent
```

---

## Assertions

### Text and regex

```python
Trial(user_msg, response)
    .contains_text("spaghetti")       # case-insensitive substring
    .regex(r"Serves \d+")             # re.search pattern
    .run()
```

### Tool calls

Assert that your agent called the right tools with the right inputs:

```python
from trial import Trial, ToolCall

result = Trial(
    user_message="Give me a pasta recipe for two",
    assistant_response="Here's spaghetti aglio e olio...",
    tool_calls=[
        ToolCall(name="search_recipe", input={"query": "pasta", "servings": 2}),
        ToolCall(name="get_nutrition", input={"dish": "spaghetti-aglio"}),
    ],
)
.called_tool("search_recipe")
.called_tool_with("get_nutrition", input_contains={"dish": "spaghetti-aglio"})
.run()
```

Normalize tool calls from any provider:

```python
# From Anthropic response
tool_calls = [ToolCall.from_anthropic(block) for block in response.content if block.type == "tool_use"]

# From OpenAI response
tool_calls = [ToolCall.from_openai(tc) for tc in response.choices[0].message.tool_calls]
```

### JSON structure

Validate that the response is valid, structured JSON:

```python
Trial(user_msg, response)
    .json_schema({"type": "object", "required": ["title", "ingredients"]})
    .json_path("$.title", contains="Spaghetti")
    .json_path("$.title", equals="Spaghetti Aglio e Olio")
    .run()
```

Requires: `pip install "trial[json]"`

### Code validity

Assert that generated code is syntactically correct:

```python
Trial(user_msg, generated_code)
    .syntactically_valid("python")
    .passes_judge("Implements a function that scales a recipe by number of servings")
    .run()
```

### LLM Judge

```python
.passes_judge("Returns a pasta recipe with a name and list of ingredients")
.passes_judge("Response is in the same language as the question")
.passes_judge("Includes preparation time and serving size", min_score=0.9)
```

The judge is intentionally strict:
- Does not infer missing information
- Partial answers fail
- Returns a score, reason, and list of missing elements

```
Scoring:
  1.0   Fully satisfies the criterion
  0.7+  Minor gaps
  0.4+  Important gaps
  0.0+  Incorrect or irrelevant
```

---

## Multi-Turn Conversations

Test full conversations end-to-end:

```python
from trial import Conversation, Turn

result = (
    Conversation([
        Turn(
            user="Give me a pasta recipe",
            assistant="Here's spaghetti aglio e olio.",
            tool_calls=[ToolCall(name="search_recipe", input={"query": "pasta"})],
        ),
        Turn(
            user="How many calories does it have?",
            assistant="Around 400 calories per serving.",
        ),
    ])
    .passes_judge("Agent used search tool and correctly provided a calorie estimate when asked")
    .run()
)

assert result.passed
```

---

## Works With Any Framework

Trial evaluates plain strings. It doesn't care where your response came from.

**Claude Agents SDK**
```python
response = agent.run("Give me a pasta recipe")

Trial.from_response(
    user_message="Give me a pasta recipe",
    response=response,     # accepts str or object with .text
    tool_calls=[ToolCall.from_anthropic(tc) for tc in response.tool_uses],
).passes_judge("Returns a recipe with ingredients").run()
```

**LangChain**
```python
response = chain.invoke(input_text)

Trial(user_message=input_text, assistant_response=response)
    .contains_text("ingredients")
    .run()
```

**Anything else**
```python
Trial(
    user_message="...",
    assistant_response=my_app.ask("..."),
).run()
```

---

## Configuration

Configure once at the top of your test file or `conftest.py`:

```python
trial.configure(provider=AnthropicProvider(model="claude-sonnet-4-6"))
```

Override per test if needed:

```python
Trial(..., provider=OpenAIProvider(model="gpt-4o"))
```

---

## Running Tests

```bash
# Unit tests (no API key needed)
pytest tests/ -m "not integration"

# Integration tests
ANTHROPIC_API_KEY=sk-... pytest tests/ -m integration
```

With Docker:

```bash
docker compose run eval pytest tests/ -m "not integration"
```

---

## Roadmap

- Latency assertions (TTFT, completion time)
- CI integrations
- TypeScript SDK

---

## Philosophy

Trial is not about generating responses.

It's about verifying behavior.

---

## License

MIT
