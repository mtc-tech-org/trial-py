from __future__ import annotations

import re
from typing import Any

from . import judge as _judge
from .config import get_provider
from .providers.base import BaseProvider
from .result import TrialResult


class Trial:
    def __init__(
        self,
        user_message: str,
        assistant_response: str,
        provider: BaseProvider | None = None,
    ) -> None:
        self._user_message = user_message
        self._assistant_response = assistant_response
        self._provider = provider
        self._text_checks: list[str] = []
        self._regex_checks: list[str] = []
        self._judge_checks: list[tuple[str, float]] = []

    @classmethod
    def from_response(
        cls,
        user_message: str,
        response: str | Any,
        provider: BaseProvider | None = None,
    ) -> Trial:
        if isinstance(response, str):
            text = response
        elif hasattr(response, "text"):
            text = response.text
        else:
            text = str(response)
        return cls(user_message=user_message, assistant_response=text, provider=provider)

    def contains_text(self, text: str) -> Trial:
        self._text_checks.append(text)
        return self

    def regex(self, pattern: str) -> Trial:
        self._regex_checks.append(pattern)
        return self

    def passes_judge(self, criterion: str, min_score: float = 0.7) -> Trial:
        self._judge_checks.append((criterion, min_score))
        return self

    def run(self) -> TrialResult:
        failures: list[str] = []

        for text in self._text_checks:
            if text.lower() not in self._assistant_response.lower():
                failures.append(f"Expected response to contain text: {text!r}")

        for pattern in self._regex_checks:
            if not re.search(pattern, self._assistant_response):
                failures.append(f"Expected response to match regex: {pattern!r}")

        if not self._judge_checks:
            passed = len(failures) == 0
            return TrialResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                reason="All deterministic checks passed." if passed else f"{len(failures)} check(s) failed.",
                missing=[],
                assertion_failures=failures,
            )

        provider = self._provider or get_provider()
        last_verdict = None

        for criterion, min_score in self._judge_checks:
            verdict = _judge.evaluate(
                criterion=criterion,
                user_message=self._user_message,
                assistant_response=self._assistant_response,
                provider=provider,
                min_score=min_score,
            )
            if not verdict.passed:
                failures.append(f"Judge failed: {verdict.reason}")
            last_verdict = verdict

        passed = len(failures) == 0
        return TrialResult(
            passed=passed,
            score=last_verdict.score,
            reason=last_verdict.reason,
            missing=last_verdict.missing,
            assertion_failures=failures,
        )
