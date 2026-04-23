from __future__ import annotations

import ast
import json
import re
import time
from typing import Any, Callable

from . import judge as _judge
from .config import get_agent, get_provider
from .providers.base import BaseProvider
from .result import TrialResult
from .tools import ToolCall


def _resolve_json_path(data: Any, path: str) -> Any:
    parts = path.lstrip("$").lstrip(".").split(".")
    current = data
    for part in parts:
        current = current[part]
    return current


def normalize_response(response: Any) -> str:
    """Extract a plain string from any agent response object."""
    if isinstance(response, str):
        return response
    if hasattr(response, "text") and isinstance(response.text, str):
        return response.text
    if isinstance(response, dict):
        for key in ("response", "text", "content", "message", "output"):
            if key in response and isinstance(response[key], str):
                return response[key]
        raise ValueError(
            f"Could not extract response text from dict. "
            f"Expected a key 'response', 'text', 'content', 'message', or 'output'. "
            f"Got: {list(response.keys())}"
        )
    return str(response)


def _extract_tool_calls(response: Any) -> list:
    """Auto-detect and extract tool calls from Anthropic or OpenAI response objects."""
    # Anthropic: response.content is a list of blocks with .type == "tool_use"
    if hasattr(response, "content") and isinstance(response.content, list):
        try:
            tools = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            if tools:
                return [ToolCall.from_anthropic(b) for b in tools]
        except Exception:
            pass

    # OpenAI: response.choices[0].message.tool_calls
    if hasattr(response, "choices") and response.choices:
        try:
            tcs = response.choices[0].message.tool_calls
            if tcs:
                return [ToolCall.from_openai(tc) for tc in tcs]
        except Exception:
            pass

    return []


class Trial:
    def __init__(
        self,
        user_message: str,
        assistant_response: str | None = None,
        provider: BaseProvider | None = None,
        tool_calls: list[ToolCall] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        self._user_message = user_message
        self._assistant_response = assistant_response
        self._provider = provider
        self._tool_calls = tool_calls or []
        self._metrics: dict[str, float] = metrics or {}
        self._auto_metrics: dict[str, float] = {}
        self._text_checks: list[str] = []
        self._regex_checks: list[str] = []
        self._tool_checks: list[tuple[str, dict | None]] = []
        self._json_schema_checks: list[dict] = []
        self._json_path_checks: list[tuple[str, str | None, Any]] = []
        self._syntax_checks: list[str] = []
        self._metric_checks: list[tuple[str, str, float]] = []
        self._after_checks: list[tuple[str, Callable[[], bool]]] = []
        self._judge_checks: list[tuple[str, float]] = []

    @classmethod
    def from_response(
        cls,
        user_message: str,
        response: str | Any,
        provider: BaseProvider | None = None,
        tool_calls: list[ToolCall] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> Trial:
        text = normalize_response(response)
        if tool_calls is None:
            tool_calls = _extract_tool_calls(response)
        return cls(
            user_message=user_message,
            assistant_response=text,
            provider=provider,
            tool_calls=tool_calls,
            metrics=metrics,
        )

    def contains_text(self, text: str) -> Trial:
        self._text_checks.append(text)
        return self

    def regex(self, pattern: str) -> Trial:
        self._regex_checks.append(pattern)
        return self

    def called_tool(self, name: str) -> Trial:
        self._tool_checks.append((name, None))
        return self

    def called_tool_with(self, name: str, input_contains: dict) -> Trial:
        self._tool_checks.append((name, input_contains))
        return self

    def json_schema(self, schema: dict) -> Trial:
        self._json_schema_checks.append(schema)
        return self

    def json_path(
        self,
        path: str,
        *,
        contains: str | None = None,
        equals: Any | None = None,
    ) -> Trial:
        self._json_path_checks.append((path, contains, equals))
        return self

    def syntactically_valid(self, language: str) -> Trial:
        if language != "python":
            raise ValueError(
                f"Unsupported language: {language!r}. Currently supported: 'python'"
            )
        self._syntax_checks.append(language)
        return self

    def completes_within(self, seconds: float) -> Trial:
        self._metric_checks.append(("elapsed_time", "completes_within", seconds))
        return self

    def first_token_within(self, seconds: float) -> Trial:
        self._metric_checks.append(("first_token_time", "first_token_within", seconds))
        return self

    def after(self, check: Callable[[], bool], label: str = "post-run check") -> Trial:
        self._after_checks.append((label, check))
        return self

    def passes_judge(self, criterion: str, min_score: float = 0.7) -> Trial:
        self._judge_checks.append((criterion, min_score))
        return self

    def _effective_metrics(self) -> dict[str, float]:
        # User-provided metrics take precedence over auto-collected
        return {**self._auto_metrics, **self._metrics}

    def _resolve_response(self) -> str:
        agent = get_agent()
        if agent is None:
            raise RuntimeError(
                "No assistant_response provided and no agent configured. "
                "Pass assistant_response= or call trial.configure(agent=...)."
            )
        if isinstance(agent, str):
            return self._call_endpoint(agent)
        start = time.monotonic()
        raw = agent(self._user_message)
        self._auto_metrics["elapsed_time"] = time.monotonic() - start
        return normalize_response(raw)

    def _call_endpoint(self, url: str) -> str:
        import urllib.request

        data = json.dumps({"message": self._user_message}).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        start = time.monotonic()
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read())
        self._auto_metrics["elapsed_time"] = time.monotonic() - start
        if isinstance(body, str):
            return body
        for key in ("response", "text", "content", "message"):
            if key in body:
                return body[key]
        raise ValueError(
            f"Could not extract response text from endpoint payload. "
            f"Expected a key 'response', 'text', 'content', or 'message'. Got: {list(body.keys())}"
        )

    def run(self) -> TrialResult:
        response = self._assistant_response
        if response is None:
            response = self._resolve_response()

        failures: list[str] = []

        for text in self._text_checks:
            if text.lower() not in response.lower():
                failures.append(f"Expected response to contain text: {text!r}")

        for pattern in self._regex_checks:
            if not re.search(pattern, response):
                failures.append(f"Expected response to match regex: {pattern!r}")

        for name, input_contains in self._tool_checks:
            matching = [tc for tc in self._tool_calls if tc.name == name]
            if not matching:
                failures.append(f"Expected tool to be called: {name!r}")
            elif input_contains is not None:
                match = any(
                    all(tc.input.get(k) == v for k, v in input_contains.items())
                    for tc in matching
                )
                if not match:
                    failures.append(
                        f"Tool {name!r} not called with expected input: {input_contains}"
                    )

        for schema in self._json_schema_checks:
            try:
                import jsonschema
            except ImportError:
                raise ImportError(
                    "jsonschema is not installed. Install it with: pip install 'trial[json]'"
                )
            try:
                data = json.loads(response)
                jsonschema.validate(data, schema)
            except json.JSONDecodeError as e:
                failures.append(f"Response is not valid JSON: {e}")
            except jsonschema.ValidationError as e:
                failures.append(f"JSON schema validation failed: {e.message}")

        for path, contains, equals in self._json_path_checks:
            try:
                data = json.loads(response)
                value = _resolve_json_path(data, path)
                if contains is not None and contains not in str(value):
                    failures.append(
                        f"JSON path {path!r} value {value!r} does not contain {contains!r}"
                    )
                if equals is not None and value != equals:
                    failures.append(
                        f"JSON path {path!r}: expected {equals!r}, got {value!r}"
                    )
            except json.JSONDecodeError as e:
                failures.append(f"Response is not valid JSON: {e}")
            except (KeyError, TypeError) as e:
                failures.append(f"JSON path {path!r} could not be resolved: {e}")

        for language in self._syntax_checks:
            try:
                ast.parse(response)
            except SyntaxError as e:
                failures.append(f"Python syntax error: {e}")

        metrics = self._effective_metrics()
        for metric_key, label, threshold in self._metric_checks:
            value = metrics.get(metric_key)
            if value is None:
                failures.append(
                    f"{label}({threshold}s): metric '{metric_key}' not available — "
                    f"pass metrics= or use configure(agent=...) for auto-timing"
                )
            elif value > threshold:
                failures.append(f"{label}: {value:.2f}s exceeded limit of {threshold:.2f}s")

        if not self._judge_checks:
            passed = len(failures) == 0
            result = TrialResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                reason="All deterministic checks passed." if passed else f"{len(failures)} check(s) failed.",
                missing=[],
                assertion_failures=failures,
            )
        else:
            provider = self._provider or get_provider()
            last_verdict = None

            for criterion, min_score in self._judge_checks:
                verdict = _judge.evaluate(
                    criterion=criterion,
                    user_message=self._user_message,
                    assistant_response=response,
                    provider=provider,
                    min_score=min_score,
                )
                if not verdict.passed:
                    failures.append(f"Judge failed: {verdict.reason}")
                last_verdict = verdict

            passed = len(failures) == 0
            result = TrialResult(
                passed=passed,
                score=last_verdict.score,
                reason=last_verdict.reason,
                missing=last_verdict.missing,
                assertion_failures=failures,
            )

        for label, check in self._after_checks:
            try:
                ok = check()
                if not ok:
                    result.passed = False
                    result.assertion_failures.append(f"Post-run check failed: {label!r}")
            except Exception as e:
                result.passed = False
                result.assertion_failures.append(f"Post-run check raised: {label!r} — {e}")

        return result
