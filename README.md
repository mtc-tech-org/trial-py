# Trial

**Put your AI agents on trial.**

A Python framework for testing AI responses with deterministic assertions and an LLM judge.

---

## Why Trial

Most AI eval tools are vague, hard to trust, or tied to a specific framework.

Trial lets you:
- Assert exact behavior with text and regex checks
- Enforce quality with a strict LLM judge
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
        user_message="Find me a senior Python engineer",
        assistant_response="Here is Alice Smith, skills: Python, FastAPI, Docker",
    )
    .contains_text("Alice")
    .regex(r"Python")
    .passes_judge("Returns a candidate with a name and relevant engineering skills")
    .run()
)

assert result.passed
print(result.score)    # 0.95
print(result.reason)   # "Includes candidate name and relevant engineering skills"
print(result.missing)  # []
```

---

## How It Works

Each test puts your AI system on trial:

**1. Evidence** — deterministic checks run first (fast, cheap, no LLM)

```python
.contains_text("Alice")
.regex(r"Python")
```

**2. Judgement** — an LLM evaluates semantic quality against your criterion

```python
.passes_judge("Returns a candidate with a name and relevant engineering skills")
```

**3. Verdict** — a structured result you can assert on

```python
result.passed    # True / False
result.score     # 0.0 – 1.0
result.reason    # short explanation
result.missing   # what was absent
```

---

## Works With Any Framework

Trial evaluates plain strings. It doesn't care where your response came from.

**Claude Agents SDK**
```python
response = agent.run("Find me a senior Python engineer")

trial.Trial(
    user_message="Find me a senior Python engineer",
    assistant_response=response.text,
).passes_judge("Returns a candidate with relevant skills").run()
```

**LangChain**
```python
response = chain.invoke(input_text)

trial.Trial(
    user_message=input_text,
    assistant_response=response,
).contains_text("Python").run()
```

**Anything else**
```python
trial.Trial(
    user_message="...",
    assistant_response=my_app.ask("..."),
).run()
```

Or use the helper for responses with a `.text` attribute:

```python
Trial.from_response(user_message="...", response=agent_response)
```

---

## Strict by Design

Trial's judge does not infer missing information.
Partial answers fail.
This makes it reliable for CI pipelines and regression testing.

```
Scoring:
  1.0   Fully satisfies the criterion
  0.7+  Minor gaps
  0.4+  Important gaps
  0.0+  Incorrect or irrelevant
```

If a response scores below `min_score` (default: `0.7`), it fails — even if the judge says `pass`.

```python
.passes_judge("Returns structured JSON with all required fields", min_score=0.9)
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

- Multi-turn conversation testing
- Tool call assertions
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
