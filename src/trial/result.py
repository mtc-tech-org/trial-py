from dataclasses import dataclass, field


@dataclass
class Verdict:
    passed: bool
    score: float
    reason: str
    missing: list[str]


@dataclass
class TrialResult:
    passed: bool
    score: float
    reason: str
    missing: list[str]
    assertion_failures: list[str] = field(default_factory=list)

    def assert_passed(self) -> None:
        if not self.passed:
            lines = [f"Trial failed (score: {self.score:.2f}): {self.reason}"]
            if self.assertion_failures:
                lines.append("Failures:")
                for f in self.assertion_failures:
                    lines.append(f"  - {f}")
            if self.missing:
                lines.append("Missing:")
                for m in self.missing:
                    lines.append(f"  - {m}")
            raise AssertionError("\n".join(lines))
