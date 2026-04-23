from __future__ import annotations

from dataclasses import dataclass, field

from . import judge as _judge
from .config import get_provider
from .providers.base import BaseProvider
from .result import TrialResult
from .tools import ToolCall


@dataclass
class Turn:
    user: str
    assistant: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class Conversation:
    def __init__(
        self,
        turns: list[Turn],
        provider: BaseProvider | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._turns = turns
        self._provider = provider
        self.session_id = session_id
        self.metadata = metadata or {}
        self._judge_checks: list[tuple[str, float]] = []

    def passes_judge(self, criterion: str, min_score: float = 0.7) -> Conversation:
        self._judge_checks.append((criterion, min_score))
        return self

    def run(self) -> TrialResult:
        if not self._judge_checks:
            return TrialResult(
                passed=True,
                score=1.0,
                reason="No checks defined.",
                missing=[],
                assertion_failures=[],
            )

        formatted = self._format()
        provider = self._provider or get_provider()
        failures: list[str] = []
        last_verdict = None

        for criterion, min_score in self._judge_checks:
            verdict = _judge.evaluate(
                criterion=criterion,
                user_message="(multi-turn conversation — see full context below)",
                assistant_response=formatted,
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

    def _format(self) -> str:
        lines: list[str] = []
        if self.session_id:
            lines.append(f"Session ID: {self.session_id}")
        if self.metadata:
            lines.append(f"Context: {self.metadata}")
        if lines:
            lines.append("")
        for i, turn in enumerate(self._turns, 1):
            lines.append(f"Turn {i}:")
            lines.append(f"  User: {turn.user}")
            if turn.tool_calls:
                for tc in turn.tool_calls:
                    lines.append(f"  Tool call: {tc.name}({tc.input})")
                    if tc.output is not None:
                        lines.append(f"  Tool output: {tc.output}")
            lines.append(f"  Assistant: {turn.assistant}")
        return "\n".join(lines)
