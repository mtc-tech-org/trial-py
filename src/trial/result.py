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
