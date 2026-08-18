"""The validation scorer — the scarce skill, kept explicit.

This is NOT the judge. This compares the judge's verdict against the controlled
ground-truth label and records everything needed for the downstream reliability
analysis (which failure mode, which difficulty, whether the verdict parsed).

We keep only the correct/incorrect signal + rich metadata here; the interesting
metrics (recall on catching ungrounded answers, Cohen's kappa, per-failure-mode
breakdowns, judge-vs-judge significance tests) are computed in `fiel.analysis`
from the logged scores, because Inspect's built-in metrics don't cover them.
"""

from __future__ import annotations

import re

from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from fiel.schema import Label

# Match the required final "VERDICT: grounded|ungrounded" line; tolerate ':' or '-'.
_VERDICT_RE = re.compile(r"verdict\s*[:\-]?\s*(grounded|ungrounded)", re.IGNORECASE)


def parse_verdict(completion: str) -> tuple[Label | None, bool]:
    """Extract the judge's verdict from its output. Returns (verdict, parse_failed).

    Uses the LAST verdict-like line so a stray mention earlier in the rationale
    doesn't override the model's final decision.
    """
    matches = _VERDICT_RE.findall(completion or "")
    if not matches:
        return None, True
    return Label(matches[-1].lower()), False


@scorer(metrics=[accuracy(), stderr()])
def grounding_agreement() -> Scorer:
    """Score = does the judge's verdict agree with the ground-truth label?"""

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion if state.output else ""
        verdict, parse_failed = parse_verdict(completion)
        gold = Label(target.text.strip().lower())

        # A verdict we can't parse is treated as a miss (conservative) but flagged,
        # so the analysis can report judge output-format reliability separately.
        correct = (not parse_failed) and (verdict == gold)

        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=(verdict.value if verdict else "PARSE_FAILED"),
            explanation=completion,
            metadata={
                "judge_verdict": verdict.value if verdict else None,
                "gold_label": gold.value,
                "parse_failed": parse_failed,
                "failure_mode": state.metadata.get("failure_mode"),
                "difficulty": state.metadata.get("difficulty"),
                "complexity": state.metadata.get("complexity"),
                "passage_id": state.metadata.get("passage_id"),
            },
        )

    return score
