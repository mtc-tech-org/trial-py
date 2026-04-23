import json

from .providers.base import BaseProvider
from .result import Verdict

_SYSTEM_PROMPT = """You are an expert evaluator of AI system behavior.

Be strict, objective, and critical. Do not be lenient. Do not assume intent.
Only evaluate what is explicitly present in the response.
Do NOT infer missing information.
If the criterion is only partially satisfied, the result must be FAIL.
If key details are missing, the result must be FAIL.

Return JSON only — no extra text, no markdown fences:
{"pass": bool, "score": float, "reason": string, "missing": string[]}

Scoring guidelines:
- 1.0: Fully satisfies the criterion
- 0.7-0.9: Mostly correct but missing minor details
- 0.4-0.6: Partially correct but important gaps
- 0.0-0.3: Incorrect or irrelevant"""

_USER_TEMPLATE = """Criterion:
{criterion}

User message:
{user_message}

Assistant response:
{assistant_response}

Evaluate whether the assistant response satisfies the criterion."""


def evaluate(
    criterion: str,
    user_message: str,
    assistant_response: str,
    provider: BaseProvider,
    min_score: float = 0.7,
) -> Verdict:
    user_prompt = _USER_TEMPLATE.format(
        criterion=criterion,
        user_message=user_message,
        assistant_response=assistant_response,
    )

    raw = provider.complete(system=_SYSTEM_PROMPT, user=user_prompt)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Judge returned malformed JSON: {e}\nRaw response: {raw!r}")

    required = {"pass", "score", "reason", "missing"}
    missing_fields = required - data.keys()
    if missing_fields:
        raise ValueError(f"Judge response missing fields: {missing_fields}\nRaw: {raw!r}")

    if not isinstance(data["score"], (int, float)):
        raise ValueError(f"Judge 'score' must be a number, got: {type(data['score'])}")
    if not isinstance(data["missing"], list):
        raise ValueError(f"Judge 'missing' must be a list, got: {type(data['missing'])}")

    passed = bool(data["pass"]) and float(data["score"]) >= min_score

    return Verdict(
        passed=passed,
        score=float(data["score"]),
        reason=str(data["reason"]),
        missing=[str(m) for m in data["missing"]],
    )
